# Copyright 2025 THU-BPM MarkDiffusion.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import hashlib
import numpy as np
import logging
from typing import Dict, Any, Union, List, Optional
from PIL import Image
from utils.media_utils import *
from utils.utils import load_config_file, set_random_seed
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import transform_to_model_format, get_media_latents
from watermark.base import BaseConfig, BaseWatermark
from exceptions.exceptions import AlgorithmNameMismatchError
from detection.wind.wind_detection import WINDetector
from visualize.data_for_visualization import DataForVisualization

logger = logging.getLogger(__name__)

class WINDConfig(BaseConfig):
    
    def initialize_parameters(self) -> None:
        
        self.w_seed = self.config_dict['w_seed']
        self.N = self.config_dict['num_noises']
        self.M = self.config_dict['num_groups']
        self.secret_salt = self.config_dict['secret_salt'].encode()
        self.hash_func = getattr(hashlib, self.config_dict['hash_function'])
        self.group_radius = self.config_dict['group_radius']
        self.threshold = self.config_dict['threshold']
        self.current_index = self.config_dict['current_index']
        self.noise_groups = self._precompute_noise_groups() 
        
    def _precompute_noise_groups(self):
        groups = {}
        for i in range(self.N):
            g = i % self.M
            if g not in groups:
                groups[g] = []
            seed = self._generate_seed(i)
            groups[g].append(self._generate_noise(seed))
        return groups

    def _generate_seed(self, index: int) -> bytes:
        """Generate seed"""
        return self.hash_func(f"{index}{self.secret_salt}".encode()).digest()

    def _generate_noise(self, seed: bytes) -> torch.Tensor:
        """Generate noises from seeds"""
        rng = np.random.RandomState(int.from_bytes(seed[:4], 'big'))
        return torch.from_numpy(rng.randn(4, 64, 64)).float().to(self.device)
    
    @property
    def algorithm_name(self) -> str:
        return 'WIND'

class WINDUtils:
    
    def __init__(self, config: WINDConfig):
        self.config = config
        self.group_patterns = self._generate_group_patterns()
        self.original_noise = None
    
    def _generate_group_patterns(self) -> Dict[int, torch.Tensor]:
        set_random_seed(self.config.w_seed)
        patterns = {}
        for g in range(self.config.M):
            pattern = torch.fft.fftshift(
                torch.fft.fft2(torch.randn(4, 64, 64).to(self.config.device)),
                dim=(-1, -2)
            )
            mask = self._circle_mask(64, self.config.group_radius)
            pattern *= mask
            patterns[g] = pattern
        return patterns

    def _circle_mask(self, size: int, r: int) -> torch.Tensor:
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
        center = size // 2
        dist = (x - center)**2 + (y - center)**2
        return ((dist >= (r-2)**2) & (dist <= r**2)).float().to(self.config.device)

    def inject_watermark(self, index: int) -> torch.Tensor:
        seed = self.config._generate_seed(index)
        z_i = self.config._generate_noise(seed)
        self.original_noise = z_i
        g = index % self.config.M
        z_fft = torch.fft.fftshift(torch.fft.fft2(z_i), dim=(-1, -2))
    
        mask = self._circle_mask(64, self.config.group_radius)
        z_fft = z_fft + self.group_patterns[g] * mask  
    
        z_combined = torch.fft.ifft2(torch.fft.ifftshift(z_fft)).real
        return z_combined

class WIND(BaseWatermark):
    
    def __init__(self, watermark_config: WINDConfig, *args, **kwargs):
        """
            Initialize the WIND algorithm.

            Parameters:
                watermark_config (WINDConfig): Configuration instance of the WIND algorithm.
        """
        self.config = watermark_config
        self.utils = WINDUtils(self.config)
        
        self.detector = WINDetector(
            noise_groups=self.config.noise_groups,
            group_patterns=self.utils.group_patterns,
            threshold=self.config.threshold,
            device=self.config.device,
            group_radius=self.config.group_radius
        )

    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        """Generate a watermarked image."""
        index = self.config.current_index % self.config.M
        
        watermarked_z = self.utils.inject_watermark(index).unsqueeze(0) # [1, 4, 64, 64]
        self.set_orig_watermarked_latents(watermarked_z)
        set_random_seed(self.config.gen_seed)
        
        generation_params = {
            "num_images_per_prompt": self.config.num_images,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "height": self.config.image_size[0],
            "width": self.config.image_size[1],
            "latents": watermarked_z,
        }
        
        if hasattr(self.config, "gen_kwargs") and self.config.gen_kwargs:
            for key, value in self.config.gen_kwargs.items():
                if key not in generation_params:
                    generation_params[key] = value
                    
        # Use kwargs to override default parameters
        for key, value in kwargs.items():
            generation_params[key] = value
            
        # Ensure latents parameter is not overridden
        generation_params["latents"] = watermarked_z
                    
        result = self.config.pipe(
            prompt,
            **generation_params
        )
    
        if isinstance(result, tuple):
            return result[0].images[0]
        else:
            return result.images[0]

    def _detect_watermark_in_image(self, 
                             image: Image.Image, 
                             prompt: str = "",
                             *args, 
                             **kwargs) -> Dict[str, Any]:

        guidance_scale_to_use = kwargs.get('guidance_scale', self.config.guidance_scale)
        num_inference_steps = kwargs.get('num_inference_steps', 50)
    
        do_classifier_free_guidance = (guidance_scale_to_use > 1.0)
        prompt_embeds, negative_prompt_embeds = self.config.pipe.encode_prompt(
            prompt=prompt, 
            device=self.config.device, 
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=1,
        )
        
        if do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            text_embeddings = prompt_embeds

        processed_img = transform_to_model_format(
            image, 
            target_size=self.config.image_size[0]
        ).unsqueeze(0).to(text_embeddings.dtype).to(self.config.device)
    
        image_latents = get_media_latents(
            pipe=self.config.pipe,
            media=processed_img,
            sample=False,
            decoder_inv = kwargs.get('decoder_inv',False)
        )
        inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder_inv','guidance_scale','num_inference_steps']}
        
        reversed_latents = self.config.inversion.forward_diffusion(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale_to_use,
            latents=image_latents,
            text_embeddings=text_embeddings,
            **inversion_kwargs
        )[-1]
        if 'detector_type' in kwargs:
            return self.detector.eval_watermark(reversed_latents, detector_type=kwargs['detector_type']) 
        else:
            return self.detector.eval_watermark(reversed_latents)
    
    def get_data_for_visualize(self, 
                            image: Image.Image,
                            prompt: str = "",
                            guidance_scale: Optional[float] = None,
                            decoder_inv: bool = False,
                            *args,
                            **kwargs):
        
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = kwargs.get('num_inference_steps', 50)
    
        do_classifier_free_guidance = (guidance_scale > 1.0)
        prompt_embeds, negative_prompt_embeds = self.config.pipe.encode_prompt(
            prompt=prompt, 
            device=self.config.device, 
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=1,
        )
        
        if do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            text_embeddings = prompt_embeds
    
        processed_img = transform_to_model_format(
            image, 
            target_size=self.config.image_size[0]
        ).unsqueeze(0).to(text_embeddings.dtype).to(self.config.device)
    
        image_latents = get_media_latents(
            pipe=self.config.pipe,
            media=processed_img,
            sample=False,
            decoder_inv=decoder_inv
        )
    
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            reverse=True,
            **kwargs
        )
        
        data = DataForVisualization(
            config=self.config,
            utils=self.utils,
            image=image,
            reversed_latents=reversed_latents,
            orig_watermarked_latents=self.orig_watermarked_latents,
        )
        
        return data
