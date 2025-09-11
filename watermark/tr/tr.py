from ..base import BaseWatermark, BaseConfig
from utils.media_utils import *
import torch
from typing import Dict, Union, List, Optional
from utils.utils import set_random_seed, inherit_docstring
from utils.diffusion_config import DiffusionConfig
import copy
import numpy as np
from PIL import Image
from visualize.data_for_visualization import DataForVisualization
from detection.tr.tr_detection import TRDetector

class TRConfig(BaseConfig):
    """Config class for TR algorithm, load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.w_seed = self.config_dict['w_seed']
        self.w_channel = self.config_dict['w_channel']
        self.w_pattern = self.config_dict['w_pattern']
        self.w_mask_shape = self.config_dict['w_mask_shape']
        self.w_radius = self.config_dict['w_radius']
        self.w_pattern_const = self.config_dict['w_pattern_const']
        self.threshold = self.config_dict['threshold']
        
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'TR'
        
class TRUtils:
    """Utility class for TR algorithm, contains helper functions."""

    def __init__(self, config: TRConfig, *args, **kwargs) -> None:
        """
            Initialize the Tree-Ring watermarking algorithm.
            
            Parameters:
                config (TRConfig): Configuration for the Tree-Ring algorithm.
        """
        self.config = config
        self.gt_patch = self._get_watermarking_pattern()
        self.watermarking_mask = self._get_watermarking_mask(self.config.init_latents)
        
    def _circle_mask(self, size: int=64, r: int=10, x_offset: int=0, y_offset: int=0) -> np.ndarray:
        """Generate a circular mask."""
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]

        return ((x - x0)**2 + (y-y0)**2)<= r**2
        
    def _get_watermarking_pattern(self) -> torch.Tensor:
        """Get the ground truth watermarking pattern."""
        set_random_seed(self.config.w_seed)
        
        gt_init = get_random_latents(pipe=self.config.pipe, height=self.config.image_size[0], width=self.config.image_size[1])

        if 'seed_ring' in self.config.w_pattern:
            gt_patch = gt_init

            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.config.w_radius, 0, -1):
                tmp_mask = self._circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(self.config.device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif 'seed_zeros' in self.config.w_pattern:
            gt_patch = gt_init * 0
        elif 'seed_rand' in self.config.w_pattern:
            gt_patch = gt_init
        elif 'rand' in self.config.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
        elif 'zeros' in self.config.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif 'const' in self.config.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
            gt_patch += self.config.w_pattern_const
        elif 'ring' in self.config.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.config.w_radius, 0, -1):
                tmp_mask = self._circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(self.config.device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

        return gt_patch
    
    def _get_watermarking_mask(self, init_latents: torch.Tensor) -> torch.Tensor:
        """Get the watermarking mask."""
        watermarking_mask = torch.zeros(init_latents.shape, dtype=torch.bool).to(self.config.device)

        if self.config.w_mask_shape == 'circle':
            np_mask = self._circle_mask(init_latents.shape[-1], r=self.config.w_radius)
            torch_mask = torch.tensor(np_mask).to(self.config.device)

            if self.config.w_channel == -1:
                # all channels
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.config.w_channel] = torch_mask
        elif self.config.w_mask_shape == 'square':
            anchor_p = init_latents.shape[-1] // 2
            if self.config.w_channel == -1:
                # all channels
                watermarking_mask[:, :, anchor_p-self.config.w_radius:anchor_p+self.config.w_radius, anchor_p-self.config.w_radius:anchor_p+self.config.w_radius] = True
            else:
                watermarking_mask[:, self.config.w_channel, anchor_p-self.config.w_radius:anchor_p+self.config.w_radius, anchor_p-self.config.w_radius:anchor_p+self.config.w_radius] = True
        elif self.config.w_mask_shape == 'no':
            pass
        else:
            raise NotImplementedError(f'w_mask_shape: {self.config.w_mask_shape}')

        return watermarking_mask
    
    def inject_watermark(self, init_latents: torch.Tensor) -> torch.Tensor:
        init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        
        init_latents_w_fft[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
        
        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

        return init_latents_w

@inherit_docstring
class TR(BaseWatermark):
    def __init__(self,
                 watermark_config: TRConfig,
                 *args, **kwargs):
        """
            Initialize the TR watermarking algorithm.
            
            Parameters:
                watermark_config (TRConfig): Configuration instance of the Tree-Ring algorithm.
        """
        self.config = watermark_config
        self.utils = TRUtils(self.config)
        
        self.detector = TRDetector(
            watermarking_mask=self.utils.watermarking_mask,
            gt_patch=self.utils.gt_patch,
            threshold=self.config.threshold,
            device=self.config.device
        )
    
    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        """Internal method to generate a watermarked image."""
        watermarked_latents = self.utils.inject_watermark(self.config.init_latents)
        
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
        """Get data for visualization including detection inversion - similar to GS logic."""
        # Use config values as defaults if not explicitly provided
        guidance_scale_to_use = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        
        # Step 1: Generate watermarked latents (generation process)
        set_random_seed(self.config.gen_seed)
        watermarked_latents = self.utils.inject_watermark(self.config.init_latents)
        
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
            image=image,
        )
