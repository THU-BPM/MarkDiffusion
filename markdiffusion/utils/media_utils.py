import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import cv2
from diffusers import StableDiffusionPipeline, TextToVideoSDPipeline, StableVideoDiffusionPipeline
from transformers import get_cosine_schedule_with_warmup
from typing import Optional, Callable, Union, List, Tuple, Dict, Any
from tqdm import tqdm
import copy
from markdiffusion.utils.pipeline_utils import (
    get_pipeline_type, 
    PIPELINE_TYPE_IMAGE, 
    PIPELINE_TYPE_TEXT_TO_VIDEO, 
    PIPELINE_TYPE_IMAGE_TO_VIDEO
)

# ===== Common Utility Functions =====

def torch_to_numpy(tensor) -> np.ndarray:
    """Convert tensor to numpy array with proper scaling."""
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    if tensor.dim() == 4:  # Image: B, C, H, W
        return tensor.cpu().permute(0, 2, 3, 1).numpy()
    elif tensor.dim() == 5:  # Video: B, C, F, H, W
        return tensor.cpu().permute(0, 2, 3, 4, 1).numpy()
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

def pil_to_torch(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL image to torch tensor."""
    tensor = pil_to_tensor(image) / 255.0
    if normalize:
        tensor = 2.0 * tensor - 1.0  # Normalize to [-1, 1]
    return tensor

def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL image."""
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert cv2 image (numpy array) to PIL image."""
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to cv2 format (numpy array)."""
    return np.asarray(pil_img) / 255.0

def transform_to_model_format(media: Union[Image.Image, List[Image.Image], np.ndarray, torch.Tensor],
                              target_size: Optional[int] = None) -> torch.Tensor:
    """
    Transform image or video frames to model input format.
    For image, `media` is a PIL image that will be resized to `target_size`(if provided) and then normalized to [-1, 1] and permuted to [C, H, W] from [H, W, C].
    For video, `media` is a list of frames (PIL images or numpy arrays) that will be normalized to [-1, 1] and permuted to [F, C, H, W] from [F, H, W, C].
    
    Args:
        media: PIL image or list of frames or video tensor
        target_size: Target size for resize operations (for images)
    
    Returns:
        torch.Tensor: Normalized tensor ready for model input
    """
    if isinstance(media, Image.Image):
        # Single image
        if target_size is not None:
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.ToTensor()
        return 2.0 * transform(media) - 1.0
    
    elif isinstance(media, list):
        # List of frames (PIL images or numpy arrays)
        if all(isinstance(frame, Image.Image) for frame in media):
            return torch.stack([2.0 * transforms.ToTensor()(frame) - 1.0 for frame in media])
        elif all(isinstance(frame, np.ndarray) for frame in media):
            return torch.stack([2.0 * transforms.ToTensor()(numpy_to_pil(frame)) - 1.0 for frame in media])
        else:
            raise ValueError("All frames must be either PIL images or numpy arrays")
    
    elif isinstance(media, np.ndarray) and media.ndim >= 3:
        # Video numpy array
        if media.ndim == 3:  # Single frame: H, W, C
            return 2.0 * transforms.ToTensor()(media) - 1.0
        elif media.ndim == 4:  # Multiple frames: F, H, W, C
            return torch.stack([2.0 * transforms.ToTensor()(frame) - 1.0 for frame in media])
        else:
            raise ValueError(f"Unsupported numpy array shape: {media.shape}")
    
    else:
        raise ValueError(f"Unsupported media type: {type(media)}")

# ===== Latent Processing Functions =====

def set_inversion(pipe: Union[StableDiffusionPipeline, TextToVideoSDPipeline, StableVideoDiffusionPipeline], inversion_type: str):
    """Set the inversion for the given pipe."""
    from markdiffusion.inversions import DDIMInversion, ExactInversion
    
    if inversion_type == "ddim":
        return DDIMInversion(pipe.scheduler, pipe.unet, pipe.device)
    elif inversion_type == "exact":
        return ExactInversion(pipe.scheduler, pipe.unet, pipe.device)
    else:
        raise ValueError(f"Invalid inversion type: {inversion_type}")

def get_random_latents(pipe: Union[StableDiffusionPipeline, TextToVideoSDPipeline, StableVideoDiffusionPipeline], 
                      latents=None, num_frames=None, height=512, width=512, generator=None) -> torch.Tensor:
    """Get random latents for the given pipe."""
    pipeline_type = get_pipeline_type(pipe)
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    batch_size = 1
    device = pipe._execution_device
    num_channels_latents = pipe.unet.config.in_channels

    # Handle different pipeline types
    if pipeline_type == PIPELINE_TYPE_IMAGE or num_frames is None:
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            pipe.text_encoder.dtype,
            device,
            generator,
            latents,
        )
    else:
        # Video pipelines with frames
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            num_frames,
            height,
            width,
            pipe.text_encoder.dtype,
            device,
            generator,
            latents,
        )
        
    return latents

# ===== Image-Specific Functions =====

def _get_image_latents(pipe: StableDiffusionPipeline, image: torch.Tensor, 
                     sample: bool = True, rng_generator: Optional[torch.Generator] = None, 
                     decoder_inv: bool = False) -> torch.Tensor:
    """Get the image latents for the given image."""
    encoding_dist = pipe.vae.encode(image).latent_dist
    if sample:
        encoding = encoding_dist.sample(generator=rng_generator)
    else:
        encoding = encoding_dist.mode()
    latents = encoding * 0.18215
    if decoder_inv:
        latents = decoder_inv_optimization(pipe, latents, image)
    return latents

def _decode_image_latents(pipe: StableDiffusionPipeline, latents: torch.FloatTensor) -> torch.Tensor:
    """Decode the image from the given latents."""
    scaled_latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(scaled_latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def decoder_inv_optimization(pipe: StableDiffusionPipeline, latents: torch.FloatTensor, 
                           image: torch.FloatTensor, num_steps: int = 100) -> torch.Tensor:
    """
    Optimize latents to better reconstruct the input image by minimizing the error between
    decoded latents and original image.
    
    Args:
        pipe: The diffusion pipeline
        latents: Initial latents
        image: Target image
        num_steps: Number of optimization steps
        
    Returns:
        torch.Tensor: Optimized latents
    """
    input_image = image.clone().float()
    z = latents.clone().float().detach()
    z.requires_grad_(True)

    loss_function = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam([z], lr=0.1)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=num_steps)

    for i in tqdm(range(num_steps)):
        # Decode without normalization to match original implementation
        scaled_latents = 1 / 0.18215 * z
        x_pred = pipe.vae.decode(scaled_latents, return_dict=False)[0]
        
        loss = loss_function(x_pred, input_image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    
    return z.detach()

# ===== Video-Specific Functions =====

def _get_video_latents(pipe: Union[TextToVideoSDPipeline, StableVideoDiffusionPipeline], 
                     video_frames: torch.Tensor, sample: bool = True, 
                     rng_generator: Optional[torch.Generator] = None, 
                     permute: bool = True,
                     decoder_inv: bool = False) -> torch.Tensor:
    """
    Encode video frames to latents.
    
    Args:
        pipe: Video diffusion pipeline
        video_frames: Tensor of video frames [F, C, H, W]
        sample: Whether to sample from the latent distribution
        rng_generator: Random generator for sampling
        permute: Whether to permute the latents to [B, C, F, H, W] format
        decoder_inv: Whether to decode the latents
        
    Returns:
        torch.Tensor: Video latents
    """
    encoding_dist = pipe.vae.encode(video_frames).latent_dist
    if sample:
        encoding = encoding_dist.sample(generator=rng_generator)
    else:
        encoding = encoding_dist.mode()
    latents = (encoding * 0.18215).unsqueeze(0)
    if permute:
        latents = latents.permute(0, 2, 1, 3, 4)
    if decoder_inv: # TODO: Implement decoder inversion for video latents
        raise NotImplementedError("Decoder inversion is not implemented for video latents")
    return latents

def tensor2vid(video: torch.Tensor, processor, output_type: str = "np"):
    """
    Convert video tensor to desired output format.
    
    Args:
        video: Video tensor [B, C, F, H, W]
        processor: Video processor from the diffusion pipeline
        output_type: Output type - 'np', 'pt', or 'pil'
        
    Returns:
        Video in requested format
    """
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)
    elif output_type == "pt":
        outputs = torch.stack(outputs)
    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs

def _decode_video_latents(pipe: Union[TextToVideoSDPipeline, StableVideoDiffusionPipeline], 
                        latents: torch.Tensor, 
                        num_frames: Optional[int] = None) -> np.ndarray:
    """
    Decode latents to video frames.
    
    Args:
        pipe: Video diffusion pipeline
        latents: Video latents
        num_frames: Number of frames to decode
        
    Returns:
        np.ndarray: Video frames
    """
    if num_frames is None:
        video_tensor = pipe.decode_latents(latents)
    else:
        video_tensor = pipe.decode_latents(latents, num_frames)
    video = tensor2vid(video_tensor, pipe.video_processor)
    return video

def convert_video_frames_to_images(frames: List[Union[np.ndarray, Image.Image]]) -> List[Image.Image]:
    """
    Convert video frames to a list of PIL.Image objects.
    
    Args:
        frames: List of video frames (numpy arrays or PIL images)
        
    Returns:
        List[Image.Image]: List of PIL images
    """
    pil_frames = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            # Convert numpy array to PIL
            pil_frames.append(numpy_to_pil(frame))
        elif isinstance(frame, Image.Image):
            # Already a PIL image
            pil_frames.append(frame)
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")
    return pil_frames

def save_video_frames(frames: List[Union[np.ndarray, Image.Image]], save_dir: str) -> None:
    """
    Save video frames to a directory.
    
    Args:
        frames: List of video frames (numpy arrays or PIL images)
        save_dir: Directory to save frames
    """
    if isinstance(frames[0], np.ndarray):
        frames = [(frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame for frame in frames]
    elif isinstance(frames[0], Image.Image):
        frames = [np.array(frame) for frame in frames]

    for i, frame in enumerate(frames):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{save_dir}/{i:02d}.png', img)

# ===== Utility Functions for Working with Different Pipeline Types =====

def get_media_latents(pipe: Union[StableDiffusionPipeline, TextToVideoSDPipeline, StableVideoDiffusionPipeline],
                     media: Union[torch.Tensor, List[torch.Tensor]],
                     sample: bool = True,
                     rng_generator: Optional[torch.Generator] = None,
                     decoder_inv: bool = False) -> torch.Tensor:
    """
    Get latents from media (either image or video) based on pipeline type.
    
    Args:
        pipe: Diffusion pipeline
        media: Image tensor or video frames tensor
        sample: Whether to sample from the latent distribution
        rng_generator: Random generator for sampling
        decoder_inv: Whether to use decoder inversion optimization
    Returns:
        torch.Tensor: Media latents
    """
    pipeline_type = get_pipeline_type(pipe)
    
    if pipeline_type == PIPELINE_TYPE_IMAGE:
        return _get_image_latents(pipe, media, sample, rng_generator, decoder_inv)
    elif pipeline_type in [PIPELINE_TYPE_TEXT_TO_VIDEO, PIPELINE_TYPE_IMAGE_TO_VIDEO]:
        permute = pipeline_type == PIPELINE_TYPE_TEXT_TO_VIDEO
        return _get_video_latents(pipe, media, sample, rng_generator, permute, decoder_inv)
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

def decode_media_latents(pipe: Union[StableDiffusionPipeline, TextToVideoSDPipeline, StableVideoDiffusionPipeline],
                        latents: torch.Tensor,
                        num_frames: Optional[int] = None) -> Union[torch.Tensor, np.ndarray]:
    """
    Decode latents to media (either image or video) based on pipeline type.
    
    Args:
        pipe: Diffusion pipeline
        latents: Media latents
        num_frames: Number of frames (for video)
        
    Returns:
        Union[torch.Tensor, np.ndarray]: Decoded media
    """
    pipeline_type = get_pipeline_type(pipe)
    
    if pipeline_type == PIPELINE_TYPE_IMAGE:
        return _decode_image_latents(pipe, latents)
    elif pipeline_type in [PIPELINE_TYPE_TEXT_TO_VIDEO, PIPELINE_TYPE_IMAGE_TO_VIDEO]:
        return _decode_video_latents(pipe, latents, num_frames)
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")