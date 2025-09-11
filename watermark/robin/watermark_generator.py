from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from evaluation.dataset import BaseDataset
from typing import Optional, Union, List, Callable, Tuple
import torch
import math
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from accelerate import Accelerator
from transformers.models.clip.modeling_clip import CLIPTextModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler
import logging
from utils.utils import set_random_seed
from utils.media_utils import *
import copy
from diffusers.utils import BaseOutput
import PIL
import time

logging.basicConfig(
    level=logging.INFO,  # seg logger level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # set logger format
    handlers=[
        logging.StreamHandler(),  # output to terminal
        # logging.FileHandler('logs/output.log', mode='a', encoding='utf-8')  # output to file
    ]
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class OptimizedDataset(Dataset):
    def __init__(
        self,
        data_root,
        custom_dataset: BaseDataset,
        size=512,
        repeats=10,
        interpolation="bicubic",
        set="train",
        center_crop=False,
    ):

        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop

        file_list = os.listdir(self.data_root)
        file_list.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))  # ori-lg7.5-xx.jpg
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in file_list]
        self.dataset = custom_dataset
        
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        text = self.dataset[i % self.num_images] # __getitem__ of BaseDataset: return prompt[idx]
        example["prompt"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        example["pixel_values"] = pil_to_torch(image, normalize=False) # scale to [0, 1]
        
        return example
    

def circle_mask(size=64, r_max=10, r_min=0, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return (((x - x0)**2 + (y-y0)**2)<= r_max**2) & (((x - x0)**2 + (y-y0)**2) > r_min**2)
    
def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r_max=args.w_up_radius, r_min=args.w_low_radius)

        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask
    
def get_watermarking_pattern(pipe, args, device, shape=None):
    set_random_seed(args.w_seed)
    # set_random_seed(10)  # test weak high freq watermark
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)#.type(torch.complex32)
    else:
        gt_init = get_random_latents(pipe=pipe)

    if 'seed_ring' in args.w_pattern:  # spacial
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_up_radius, args.w_low_radius, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r_max=args.w_up_radius, r_min=args.w_low_radius)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_up_radius, args.w_low_radius, -1):  
            tmp_mask = circle_mask(gt_init.shape[-1],r_max=i,r_min=args.w_low_radius)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        
    return gt_patch    

def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    gt_patch = gt_patch.to(init_latents_w_fft.dtype)
    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()  # complexhalf = complexfloat
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w
    

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def to_ring(latent_fft, args):
    # Calculate mean value for each ring
    num_rings = args.w_up_radius - args.w_low_radius
    r_max = args.w_up_radius
    for i in range(num_rings):
        # ring_mask = mask[..., (radii[i * 2] <= distances) & (distances < radii[i * 2 + 1])]
        ring_mask = circle_mask(latent_fft.shape[-1], r_max=r_max, r_min=r_max-1)
        ring_mean = latent_fft[:, args.w_channel,ring_mask].real.mean().item()
        # print(f'ring mean: {ring_mean}')
        latent_fft[:, args.w_channel,ring_mask] = ring_mean
        r_max = r_max - 1

    return latent_fft

def optimizer_wm_prompt(pipe: StableDiffusionPipeline,
                        dataloader: OptimizedDataset,
                        hyperparameters: dict, 
                        mask: torch.Tensor,
                        opt_wm: torch.Tensor,
                        save_path: str,
                        args: dict,
                        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                        eta: float = 0.0,) -> tuple[torch.Tensor, torch.Tensor]:
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    gradient_checkpointing = hyperparameters["gradient_checkpointing"]
    original_guidance_scale = hyperparameters["guidance_scale"]
    optimized_guidance_scale = hyperparameters["optimized_guidance_scale"]

    # Check if checkpoint exists
    checkpoint_path = os.path.join(save_path, f"optimized_wm5-30_embedding-step-{max_train_steps}.pt")
    # checkpoint_path = "/workspace/panleyi/gs/ROBIN/ckpts/optimized_wm5-30_embedding-step-2000.pt"
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        opt_wm = checkpoint['opt_wm'].to(pipe.device)
        opt_wm_embedding = checkpoint['opt_acond'].to(pipe.device)
        return opt_wm, opt_wm_embedding

    text_encoder: CLIPTextModel = pipe.text_encoder
    unet: UNet2DConditionModel = pipe.unet
    vae: AutoencoderKL = pipe.vae
    scheduler: DPMSolverMultistepScheduler = pipe.scheduler

    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=hyperparameters["mixed_precision"]
    )

    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    # null text, text_embedding.dtype = torch.float16
    do_classifier_free_guidance = False  # guidance_scale = 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=tester_prompt, 
        device=pipe.device, 
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_images_per_prompt=1,
    )
    
    text_embeddings = prompt_embeds

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    unet, text_encoder, dataloader,text_embeddings = accelerator.prepare(
        unet, text_encoder, dataloader, text_embeddings
    ) 

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    scaler = GradScaler(device=accelerator.device)
    # pipe.scheduler.set_timesteps(1000)  # need for compute the next state
    
    do_classifier_free_guidance = False  # guidance_scale = 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt='', 
        device=pipe.device, 
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_images_per_prompt=1,
    )
    
    opt_wm_embedding = prompt_embeds
    null_embedding = opt_wm_embedding.clone()
    total_time = 0
    with autocast(device_type=accelerator.device.type):
        for epoch in range(num_train_epochs):
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    gt_tensor = batch["pixel_values"]
                    image = 2.0 * gt_tensor - 1.0
                    latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * 0.18215
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    ori_timesteps = torch.randint(200, 300, (bsz,), device=latents.device).long()  # 35～40steps
                    timesteps = len(scheduler) - 1 - ori_timesteps

                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    opt_wm = opt_wm.to(noisy_latents.device).to(torch.complex64)  # add wm to latents


                    ### detailed the inject_watermark function for fft.grad
                    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(noisy_latents), dim=(-1, -2))
                    init_latents_w_fft[mask] = opt_wm[mask].clone()
                    init_latents_w_fft.requires_grad = True
                    noisy_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real
                    ### Get the text embedding for conditioning CFG 
                    prompt = batch["prompt"]
                    do_classifier_free_guidance = False  # guidance_scale = 1.0
                    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                        prompt=prompt, 
                        device=pipe.device, 
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        num_images_per_prompt=1,
                    )
                    
                    cond_embedding = prompt_embeds
                    text_embeddings = torch.cat([opt_wm_embedding, cond_embedding, null_embedding]) 
                    text_embeddings.requires_grad = True

                    ### Predict the noise residual with CFG 
                    latent_model_input = torch.cat([noisy_latents] * 3)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)
                    noise_pred = unet(latent_model_input, ori_timesteps, encoder_hidden_states=text_embeddings).sample
                    noise_pred_wm, noise_pred_text, noise_pred_null = noise_pred.chunk(3)
                    noise_pred = noise_pred_null + original_guidance_scale * (noise_pred_text - noise_pred_null) + optimized_guidance_scale * (noise_pred_wm - noise_pred_null)   # different guidance scale
                    
                    
                    ### get the predicted x0 tensor
                    scheduler._init_step_index(timesteps)
                    x0_latents = scheduler.convert_model_output(model_output=noise_pred, sample=noisy_latents)  #predict x0 in one-step
                    x0_tensor = decode_media_latents(pipe=pipe, latents=x0_latents)
                    
                    loss_noise = F.mse_loss(x0_tensor.float(), gt_tensor.float(), reduction="mean")  # pixel alignment
                    loss_wm = torch.mean(torch.abs(opt_wm[mask].real))
                    loss_constrain = F.mse_loss(noise_pred_wm.float(), noise_pred_null.float(), reduction="mean")  # prompt constraint

                    ### optimize wm pattern and uncond prompt alternately
                    if (global_step // 500) % 2 == 0:
                        loss = 10 * loss_noise + loss_constrain - 0.00001 * loss_wm  # opt wm pattern
                        accelerator.backward(loss)
                        with torch.no_grad():  
                            grads = init_latents_w_fft.grad
                            init_latents_w_fft = init_latents_w_fft - 1.0 * grads  # update wm pattern
                            init_latents_w_fft = to_ring(init_latents_w_fft, args)
                            opt_wm = init_latents_w_fft.detach()
                    else:
                        loss = 10 * loss_noise + loss_constrain  # opt prompt
                        accelerator.backward(loss)
                        with torch.no_grad():  
                            grads = text_embeddings.grad
                            text_embeddings = text_embeddings - 5e-04 * grads  
                            opt_wm_embedding = text_embeddings[0].unsqueeze(0).detach()  # update acond embedding


                    print(f'global_step: {global_step}, loss_mse: {loss_noise}, loss_wm: {loss_wm}, loss_cons: {loss_constrain},loss: {loss}')

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % hyperparameters["save_steps"] == 0:
                        path = os.path.join(save_path, f"optimized_wm5-30_embedding-step-{global_step}.pt")
                        torch.save({'opt_acond': opt_wm_embedding, 'opt_wm': opt_wm.cpu()}, path)

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break

            accelerator.wait_for_everyone()

    return opt_wm, opt_wm_embedding

class ROBINStableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]
    latents: Optional[torch.FloatTensor]
    inner_latents: Optional[List[torch.FloatTensor]]

@torch.no_grad()
def ROBINWatermarkedImageGeneration(
    pipe: StableDiffusionPipeline,
    prompt: Union[str, List[str]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    optimized_guidance_scale: float = 3.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
    watermarking_mask: Optional[torch.BoolTensor] = None,
    watermarking_step: int = None,
    args = None,
    gt_patch = None,
    opt_acond = None
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        original_guidance_scale (`float`, *optional*, defaults to 3.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `original_guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `original_guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        optimized_guidance_scale (`float`, *optional*, defaults to 3.5):
            TODO: add description
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `original_guidance_scale` is less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    # print('got new version')
    inner_latents = []
    # 0. Default height and width to unet
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(prompt, height, width, callback_steps)

    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = pipe._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        text_embeddings.dtype,
        device,
        generator,
        latents,
    )

    init_latents = copy.deepcopy(latents)

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    inner_latents.append(init_latents)

    # 7. Denoising loop
    max_train_steps=1  #100
    latents_wm = None
    text_embeddings_opt = None
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    
    start_time = time.time()
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if (watermarking_step is not None) and (i >= watermarking_step):
                mask = watermarking_mask  # mask from outside
                if i == watermarking_step:
                    latents_wm = inject_watermark(latents, mask,gt_patch, args)  # inject latent watermark
                    inner_latents[-1] = latents_wm  
                    if opt_acond is not None:
                        uncond, cond = text_embeddings.chunk(2)
                        opt_acond = opt_acond.to(cond.dtype)
                        text_embeddings_opt = torch.cat([uncond, opt_acond, cond])  # opt as another cond
                    else:
                        text_embeddings_opt = text_embeddings.clone()
                    # if lguidance is not None:
                    #     guidance_scale = lguidance  

                latents_wm, _ = xn1_latents_3(pipe,latents_wm,do_classifier_free_guidance,t
                                                        ,text_embeddings_opt,guidance_scale,optimized_guidance_scale,**extra_step_kwargs)

            if (watermarking_step is None) or (watermarking_step is not None and i < watermarking_step):
                latents, _ = xn1_latents(pipe,latents,do_classifier_free_guidance,t
                                                        ,text_embeddings,guidance_scale,**extra_step_kwargs)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
            
            if (watermarking_step is not None and i < watermarking_step) or (watermarking_step is None):
                inner_latents.append(latents)   # save for memory
            else: 
                inner_latents.append(latents_wm)

            if watermarking_step is not None and watermarking_step == 50:
                latents_wm = inject_watermark(latents, watermarking_mask,gt_patch, args)  # inject latent watermark
                inner_latents[-1] = latents_wm  

    end_time = time.time()
    execution_time = end_time - start_time
    # 8. Post-processing
    if latents_wm is not None:
        # Convert latents to the same dtype as VAE
        latents_wm = latents_wm.to(dtype=pipe.vae.dtype)
        image = pipe.decode_latents(latents_wm)
    else:
        # Convert latents to the same dtype as VAE
        latents = latents.to(dtype=pipe.vae.dtype)
        image = pipe.decode_latents(latents)

    # 9. Run safety checker
    image, has_nsfw_concept = pipe.run_safety_checker(image, device, text_embeddings.dtype)

    # 10. Convert to PIL
    if output_type == "pil":
        image = pipe.numpy_to_pil(image)

    if not return_dict:
        return (image, has_nsfw_concept)
    if text_embeddings_opt is not None:
        return ROBINStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents, latents=latents, inner_latents=inner_latents,gt_patch=gt_patch,opt_acond=text_embeddings_opt[0],time=execution_time)
    else:
        return ROBINStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents, latents=latents, inner_latents=inner_latents,gt_patch=gt_patch,time=execution_time)

def xn1_latents_3(pipe,latents,do_classifier_free_guidance,t
                        ,text_embeddings,original_guidance_scale,optimized_guidance_scale,**extra_step_kwargs):
    latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text1, noise_pred_text2 = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + original_guidance_scale * (noise_pred_text1 - noise_pred_uncond) + optimized_guidance_scale * (noise_pred_text2 - noise_pred_uncond)
    latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    return latents, noise_pred

def xn1_latents(pipe,latents,do_classifier_free_guidance,t
                    ,text_embeddings,guidance_scale,**extra_step_kwargs):
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    return latents, noise_pred  # Make sure to return both values