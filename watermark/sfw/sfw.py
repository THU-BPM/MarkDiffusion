from ..base import BaseWatermark, BaseConfig
from utils.media_utils import *
import torch
from typing import Dict, Optional
from utils.utils import set_random_seed, inherit_docstring
from utils.diffusion_config import DiffusionConfig
import numpy as np
from PIL import Image
from visualize.data_for_visualization import DataForVisualization
from detection.sfw.sfw_detection import SFWDetector
import torchvision.transforms as tforms
import qrcode
import os

class SFWConfig(BaseConfig):
    """Config class for SFW algorithm, load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.w_seed = self.config_dict['w_seed']
        self.delta=self.config_dict['delta']
        self.wm_type=self.config_dict['wm_type']
        self.threshold = self.config_dict['threshold']
        self.w_channel = self.config_dict['w_channel']
        
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'SFW'

class SFWUtils:
    """Utility class for SFW algorithm, contains helper functions."""

    def __init__(self, config: SFWConfig, *args, **kwargs) -> None:
        """
            Initialize the SFW watermarking algorithm.

            Parameters:
                config (SFWConfig): Configuration for the SFW algorithm.
        """
        self.config = config
        self.gt_patch = self._get_watermarking_pattern()
        self.watermarking_mask = self._get_watermarking_mask(self.config.init_latents)

    # [Fourier transforms]
    @staticmethod
    def fft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim=(-1, -2))

    @staticmethod
    def ifft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim=(-1, -2)))
    
    @staticmethod
    @torch.no_grad()
    def rfft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.fftshift(torch.fft.rfft2(input_tensor, dim=(-2,-1)), dim=-2)
    
    @staticmethod
    @torch.no_grad()
    def irfft(input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.irfft2(torch.fft.ifftshift(input_tensor, dim=-2), dim=(-2,-1), s=(input_tensor.shape[-2],input_tensor.shape[-2]))
    
    def circle_mask(self,size: int, r=16, x_offset=0, y_offset=0):
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        return ((x - x0)**2 + (y-y0)**2)<= r**2
    
    @torch.no_grad()
    def enforce_hermitian_symmetry(self,freq_tensor):
        B, C, H, W = freq_tensor.shape # fftshifted frequency (complex tensor) - center (32,32)
        assert H == W, "H != W"
        freq_tensor = freq_tensor.clone()
        freq_tensor_tmp = freq_tensor.clone()
        # DC point (no imaginary)
        freq_tensor[:, :, H//2, W//2] = torch.real(freq_tensor_tmp[:, :, H//2, W//2])
        if H % 2 == 0: # Even
            # Nyquist Points (no imaginary)
            freq_tensor[:, :, 0, 0] = torch.real(freq_tensor_tmp[:, :, 0, 0])
            freq_tensor[:, :, H//2, 0] = torch.real(freq_tensor_tmp[:, :, H//2, 0])  # (32, 0)
            freq_tensor[:, :, 0, W//2] = torch.real(freq_tensor_tmp[:, :, 0, W//2])  # (0, 32)
    
            # Nyquist axis - conjugate
            freq_tensor[:, :, 0, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 0, W//2+1:], dims=[2]))
            freq_tensor[:, :, H//2, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2, W//2+1:], dims=[2]))
            freq_tensor[:, :, 1:H//2, 0] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, 0], dims=[2]))
            freq_tensor[:, :, 1:H//2, W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2], dims=[2]))
            # Square quadrants - conjugate
            freq_tensor[:, :, 1:H//2, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2+1:], dims=[2, 3]))
            freq_tensor[:, :, H//2+1:, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 1:H//2, W//2+1:], dims=[2, 3]))
        else: # Odd
            # Nyquist axis - conjugate
            freq_tensor[:, :, H//2, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2, W//2+1:], dims=[2]))
            freq_tensor[:, :, 0:H//2, W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2], dims=[2]))
            # Square quadrants - conjugate
            freq_tensor[:, :, 0:H//2, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2+1:], dims=[2, 3]))
            freq_tensor[:, :, H//2+1:, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 0:H//2, W//2+1:], dims=[2, 3]))
        return freq_tensor

    @torch.no_grad()
    def make_Fourier_treering_pattern(self,pipe, shape, w_seed=999999, resolution=512):
        assert shape[-1] == shape[-2] # 64==64
        g = torch.Generator(device=self.config.device).manual_seed(w_seed)
        gt_init = pipe.prepare_latents(1, pipe.unet.in_channels, resolution, resolution, pipe.unet.dtype, torch.device(self.config.device), g) # (1,4,64,64)
        # [HSTR] center-aware design
        watermarked_latents_fft = SFWUtils.fft(torch.zeros(shape, device=self.config.device)) # (1,4,64,64) complex64
        start = 10
        end = 54 # 64-10 = hw_latent-start
        center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))
        gt_patch_tmp = SFWUtils.fft(gt_init[center_slice]).clone().detach() # (1,4,44,44) complex64
        center_len = gt_patch_tmp.shape[-1] // 2 # 22
        for radius in range(center_len-1, 0, -1): # [21,20,...,1]
            tmp_mask = torch.tensor(self.circle_mask(size=shape[-1], r=radius)) # (64,64)
            for j in range(watermarked_latents_fft.shape[1]): # GT : all channel Tree-Ring
                watermarked_latents_fft[:, j, tmp_mask] = gt_patch_tmp[0, j, center_len, center_len + radius].item() # Use (22,22+radius) element.
        # Gaussian noise key (Heterogenous watermark in RingID)
        watermarked_latents_fft[:,[0], start:end, start:end] = gt_patch_tmp[:, [0]] # (1,1,44,44) complex64
        # [Hermitian Symmetric Fourier] HSTR 
        return self.enforce_hermitian_symmetry(watermarked_latents_fft)

    # HSQR - hermitian symmetric QR pattern
    class QRCodeGenerator:
        def __init__(self, box_size=2, border=1, qr_version=1):
            self.qr = qrcode.QRCode(version=qr_version, box_size=box_size, border=border,
            error_correction = qrcode.constants.ERROR_CORRECT_H)
    
        def make_qr_tensor(self, data, filename='qrcode.png', save_img=False):
            self.qr.add_data(data)
            self.qr.make(fit=True)
            img = self.qr.make_image(fill_color="black", back_color="white")
            if save_img:
                img.save(filename)
            self.clear()
            img_array = np.array(img)
            tensor = torch.from_numpy(img_array)
            return tensor.clone().detach() # boolean (h,w)
    
        def clear(self):
            self.qr.clear()

    @torch.no_grad()
    def make_hsqr_pattern(self,idx: int):
        qr_generator = self.QRCodeGenerator(box_size=2, border=0, qr_version=1)
        data = f"HSQR{idx % 10000}"
        qr_tensor = qr_generator.make_qr_tensor(data=data) # (42,42) boolean tensor
        qr_tensor = qr_tensor.repeat(len([self.config.w_channel]), 1, 1) # (c_wm,42,42) boolean tensor
        return qr_tensor # (c_wm,42,42) boolean tensor

    def _get_watermarking_pattern(self) -> torch.Tensor:
        """Get the ground truth watermarking pattern."""
        set_random_seed(self.config.w_seed)
        shape = (1, 4, 64, 64)
        if self.config.wm_type == "HSQR":
            Fourier_watermark_pattern_list = [self.make_hsqr_pattern(idx=self.config.w_seed)]
        else:
            Fourier_watermark_pattern_list = [self.make_Fourier_treering_pattern(self.config.pipe, shape, self.config.w_seed)]
        pattern_gt_batch = [Fourier_watermark_pattern_list[0]]
        # adjust dims of pattern_gt_batch
        if len(pattern_gt_batch[0].shape) == 4:
            pattern_gt_batch = torch.cat(pattern_gt_batch, dim=0) # (N,4,64,64) for HSTR
        elif len(pattern_gt_batch[0].shape) == 3:
            pattern_gt_batch = torch.stack(pattern_gt_batch, dim=0) # (N,c_wm,42,42) for HSQR
        else:
            raise ValueError(f"Unexpected pattern_gt_batch shape: {pattern_gt_batch[0].shape}")
        assert len(pattern_gt_batch.shape) == 4

        return pattern_gt_batch

    # ring mask
    class RounderRingMask:
        def __init__(self, size=65, r_out=14):
            assert size >= 3
            self.size = size
            self.r_out = r_out

            num_rings = r_out
            zero_bg_freq = torch.zeros(size, size)
            center = size // 2
            center_x, center_y = center, center

            ring_vector = torch.tensor([(200 - i*4) * (-1)**i for i in range(num_rings)])
            zero_bg_freq[center_x, center_y:center_y+num_rings] = ring_vector
            zero_bg_freq = zero_bg_freq[None, None, ...]
            self.ring_vector_np = ring_vector.numpy()

            res = torch.zeros(360, size, size)
            res[0] = zero_bg_freq
            for angle in range(1, 360):
                zero_bg_freq_rot = tforms.functional.rotate(zero_bg_freq, angle=angle)
                res[angle] = zero_bg_freq_rot

            res = res.numpy()
            self.res = res
            self.pure_bg = np.zeros((size, size))
            for x in range(size):
                for y in range(size):
                    values, count = np.unique(res[:, x, y],  return_counts=True)
                    if len(count) > 2:
                        self.pure_bg[x, y] = values[count == max(count[values!=0])][0]
                    elif len(count) == 2:
                        self.pure_bg[x, y] = values[values!=0][0]
        
        def get_ring_mask(self, r_out, r_in):
            # get mask from pure_bg
            assert r_out <= self.r_out
            if r_in - 1 < 0:
                right_end = 0  # None, to take the center
            else:
                right_end = r_in - 1
            cand_list = self.ring_vector_np[r_out-1:right_end:-1]
            mask = np.isin(self.pure_bg, cand_list)
            if self.size % 2:
                mask = mask[:self.size-1, :self.size-1]  # [64, 64]
            return mask
        
    def _get_watermarking_mask(self, init_latents: torch.Tensor) -> torch.Tensor:
        """Get the watermarking mask."""
        shape=(1,4,64,64)
        tree_masks = torch.zeros(shape, dtype=torch.bool) # (1,4,64,64)
        single_channel_tree_watermark_mask = torch.tensor(self.circle_mask(size=shape[-1], r=14)) # (64,64)
        tree_masks[:, [self.config.w_channel]] = single_channel_tree_watermark_mask # (64,64)
        masks = tree_masks
        mask_obj = self.RounderRingMask(size=65, r_out=14)
        single_channel_heter_watermark_mask = torch.tensor(mask_obj.get_ring_mask(r_out=14, r_in=3) ) # (64,64)
        masks[:, [0]] = single_channel_heter_watermark_mask # (64,64) RounderRingMask for Hetero Watermark (noise)
        return masks

    @torch.no_grad()
    def inject_wm(self,init_latents: torch.Tensor):
        # for HSTR
        assert len(self.gt_patch.shape) == 4
        assert len(self.watermarking_mask.shape) == 4
        batch_size = init_latents.shape[0]
        self.watermarking_mask = self.watermarking_mask.repeat(batch_size, 1, 1, 1)

        init_latents =init_latents.to(self.config.device)
        self.gt_patch = self.gt_patch.to(self.config.device)
        self.watermarking_mask = self.watermarking_mask.to(self.config.device)

        # inject watermarks in fourier space
        start = 10
        end = 54 # 64-10 = hw_latent-start
        center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))
        assert len(init_latents[center_slice].shape) == 4
        center_latent_fft=torch.fft.fftshift(torch.fft.fft2(init_latents[center_slice]), dim=(-1, -2))# (N,4,44,44) complex64
        #injection
        temp_mask = self.watermarking_mask[center_slice] # (N,4,44,44) boolean
        temp_pattern = self.gt_patch[center_slice] # (N,4,44,44) complex64
        center_latent_fft[temp_mask] = temp_pattern[temp_mask].clone() # (N,4,44,44) complex64
        # IFFT
        assert len(center_latent_fft.shape) == 4
        center_latent_ifft=torch.fft.ifft2(torch.fft.ifftshift(center_latent_fft, dim=(-1, -2)))# (N,4,44,44)
        center_latent_ifft = center_latent_ifft.real if center_latent_ifft.imag.abs().max() < 1e-3 else center_latent_ifft

        init_latents = init_latents.clone()
        init_latents[center_slice] = center_latent_ifft
        init_latents[init_latents == float("Inf")] = 4
        init_latents[init_latents == float("-Inf")] = -4
        return init_latents # float32

    @torch.no_grad()
    def inject_hsqr(self,inverted_latent): # (N,4,64,64) -> (N,4,64,64)
        assert len(self.gt_patch.shape) == 4 # (N,c_wm,42,42)
        inverted_latent = inverted_latent.to(self.config.device)
        self.gt_patch = self.gt_patch.to(self.config.device)
        qr_pix_len = self.gt_patch.shape[-1]    # 42
        qr_pix_half = (qr_pix_len + 1) // 2 # 21
        qr_left = self.gt_patch[:, :, :, :qr_pix_half]    # (N,c_wm,42,21) boolean
        qr_right = self.gt_patch[:, :, :, qr_pix_half:]   # (N,c_wm,42,21) boolean
        # rfft
        start = 10
        end = 54 # 64-10 = hw_latent-start
        center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))
        center_latent_rfft = SFWUtils.rfft(inverted_latent[center_slice]) # (N,4,44,44) -> # (N,4,44,23) complex64
        center_real_batch = center_latent_rfft.real # (N,4,44,23) f32
        center_imag_batch = center_latent_rfft.imag # (N,4,44,23) f32
        real_slice = (slice(None),[self.config.w_channel], slice(1, 1+qr_pix_len), slice(1, 1+qr_pix_half))
        imag_slice = (slice(None), [self.config.w_channel], slice(1, 1+qr_pix_len), slice(1, 1+qr_pix_half))
        center_real_batch[real_slice] = torch.where(qr_left, center_real_batch[real_slice].abs() + self.config.delta, -center_real_batch[real_slice].abs() - self.config.delta)
        center_imag_batch[imag_slice] = torch.where(qr_right, center_imag_batch[imag_slice].abs() + self.config.delta, -center_imag_batch[imag_slice].abs() - self.config.delta)
        center_latent_ifft = SFWUtils.irfft(torch.complex(center_real_batch, center_imag_batch)) # (N,4,44,44) f32
        inverted_latent = inverted_latent.clone()
        inverted_latent[center_slice] = center_latent_ifft
        return inverted_latent # (N,4,64,64)
   
@inherit_docstring
class SFW(BaseWatermark):
    def __init__(self,
                 watermark_config: SFWConfig,
                 *args, **kwargs):
        """
            Initialize the SFW watermarking algorithm.
            
            Parameters:
                watermark_config (SFWConfig): Configuration instance of the SFW algorithm.
        """
        self.config = watermark_config
        self.utils = SFWUtils(self.config)
        self.detector = SFWDetector(
            watermarking_mask=self.utils.watermarking_mask,
            gt_patch=self.utils.gt_patch,
            w_channel=self.config.w_channel,
            threshold=self.config.threshold,
            device=self.config.device,
            wm_type=self.config.wm_type
        )
    
    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        """Internal method to generate a watermarked image."""
        if(self.config.wm_type=="HSQR"):
            watermarked_latents = self.utils.inject_hsqr(self.config.init_latents)
        else:
            watermarked_latents = self.utils.inject_wm(self.config.init_latents)
        
        # save watermarked latents
        self.set_orig_watermarked_latents(watermarked_latents)
        
        # Construct generation parameters
        generation_params = {
            "num_images_per_prompt": self.config.num_images,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "height": self.config.image_size[0],
            "width": self.config.image_size[1],
            "latents": watermarked_latents,
        }
        
        # Add parameters from config.gen_kwargs
        if hasattr(self.config, "gen_kwargs") and self.config.gen_kwargs:
            for key, value in self.config.gen_kwargs.items():
                if key not in generation_params:
                    generation_params[key] = value
                    
        # Use kwargs to override default parameters
        for key, value in kwargs.items():
            generation_params[key] = value
            
        # Ensure latents parameter is not overridden
        generation_params["latents"] = watermarked_latents
        
        return self.config.pipe(
            prompt,
            **generation_params
        ).images[0]
    
    def _detect_watermark_in_image(self, 
                         image: Image.Image, 
                         prompt: str = "",
                         *args, 
                         **kwargs) -> Dict[str, float]:
        """Detect the watermark in the image."""
        # Use config values as defaults if not explicitly provided
        guidance_scale_to_use = kwargs.get('guidance_scale', self.config.guidance_scale)
        num_steps_to_use = kwargs.get('num_inference_steps', self.config.num_inference_steps)
        
        # Step 1: Get Text Embeddings
        do_classifier_free_guidance = (guidance_scale_to_use > 1.0)
        prompt_embeds, negative_prompt_embeds = self.config.pipe.encode_prompt(
            prompt=prompt, 
            device=self.config.device, 
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=1, # TODO: Multiple image generation to be supported
        )
        
        if do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            text_embeddings = prompt_embeds
        
        # Step 2: Preprocess Image
        image = transform_to_model_format(image, target_size=self.config.image_size[0]).unsqueeze(0).to(text_embeddings.dtype).to(self.config.device)
        
        # Step 3: Get Image Latents
        image_latents = get_media_latents(pipe=self.config.pipe, media=image, sample=False, decoder_inv=kwargs.get('decoder_inv', False))
        
        # Step 4: Reverse Image Latents
        # Pass only known parameters to forward_diffusion, and let kwargs handle any additional parameters
        inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder_inv', 'guidance_scale', 'num_inference_steps']}
        
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale_to_use,
            num_inference_steps=num_steps_to_use,
            **inversion_kwargs
        )[-1]
        # Step 5: Evaluate Watermark
        if 'detector_type' in kwargs:
            return self.detector.eval_watermark(reversed_latents, detector_type=kwargs['detector_type'])
        else:
            return self.detector.eval_watermark(reversed_latents)
        
    def get_data_for_visualize(self, 
                               image: Image.Image,
                               prompt: str="",
                               guidance_scale: Optional[float]=None,
                               decoder_inv: bool=False,
                               *args,
                               **kwargs) -> DataForVisualization:
        """Get data for visualization including detection inversion"""
        # Use config values as defaults if not explicitly provided
        guidance_scale_to_use = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        
        # Step 1: Generate watermarked latents (generation process)
        set_random_seed(self.config.gen_seed)
        if (self.config.wm_type=="HSQR"):
            watermarked_latents = self.utils.inject_hsqr(self.config.init_latents)
        else:
            watermarked_latents = self.utils.inject_wm(self.config.init_latents)
        
        # Step 2: Generate actual watermarked image using the same process as _generate_watermarked_image
        generation_params = {
            "num_images_per_prompt": self.config.num_images,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "height": self.config.image_size[0],
            "width": self.config.image_size[1],
            "latents": watermarked_latents,
        }
        
        # Add parameters from config.gen_kwargs
        if hasattr(self.config, "gen_kwargs") and self.config.gen_kwargs:
            for key, value in self.config.gen_kwargs.items():
                if key not in generation_params:
                    generation_params[key] = value
        
        # Generate the actual watermarked image
        watermarked_image = self.config.pipe(
            prompt,
            **generation_params
        ).images[0]
        
        # Step 3: Perform watermark detection to get inverted latents (detection process)
        inverted_latents = None
        try:
            # Get Text Embeddings for detection
            do_classifier_free_guidance = (guidance_scale_to_use > 1.0)
            prompt_embeds, negative_prompt_embeds = self.config.pipe.encode_prompt(
                prompt=prompt, 
                device=self.config.device, 
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_images_per_prompt=1, # TODO: Multiple image generation to be supported
            )
            
            if do_classifier_free_guidance:
                text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                text_embeddings = prompt_embeds
            
            # Preprocess watermarked image for detection
            processed_image = transform_to_model_format(
                watermarked_image, 
                target_size=self.config.image_size[0]
            ).unsqueeze(0).to(text_embeddings.dtype).to(self.config.device)
            
            # Get Image Latents
            image_latents = get_media_latents(
                pipe=self.config.pipe, 
                media=processed_image, 
                sample=False, 
                decoder_inv=decoder_inv
            )
            
            # Reverse Image Latents to get inverted noise
            inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['prompt', 'decoder_inv', 'guidance_scale', 'num_inference_steps']}
            
            reversed_latents_list = self.config.inversion.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale_to_use,
                num_inference_steps=self.config.num_inference_steps,
                **inversion_kwargs
            )
            
            inverted_latents = reversed_latents_list[-1]
            
        except Exception as e:
            print(f"Warning: Could not perform inversion for visualization: {e}")
            inverted_latents = None
        
        # Step 4: Prepare visualization data  
        return DataForVisualization(
            config=self.config,
            utils=self.utils,
            reversed_latents=reversed_latents_list,
            orig_watermarked_latents=self.orig_watermarked_latents,
            image=image
        )
