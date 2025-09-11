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


import itertools

import torch
from ..base import BaseWatermark, BaseConfig
import numpy as np
from typing import Dict
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
from detection.ri.ri_detection import RIDetector
from utils.media_utils import *
from utils.utils import set_random_seed
from visualize.data_for_visualization import DataForVisualization

class RIConfig(BaseConfig):
    """Configuration class for the RI algorithm."""

    def initialize_parameters(self) -> None:
        """Initialize parameters for the RI algorithm."""
        self.ring_width = self.config_dict['ring_width']
        self.quantization_levels = self.config_dict['quantization_levels']
        self.ring_value_range = self.config_dict['ring_value_range']

        self.fix_gt = self.config_dict['fix_gt']
        self.time_shift = self.config_dict['time_shift']
        self.time_shift_factor = self.config_dict['time_shift_factor']
        self.assigned_keys = self.config_dict['assigned_keys']
        self.channel_min = self.config_dict['channel_min']

        self.radius = self.config_dict['radius']
        self.anchor_x_offset = self.config_dict['anchor_x_offset']
        self.anchor_y_offset = self.config_dict['anchor_y_offset']
        self.radius_cutoff = self.config_dict['radius_cutoff']

        self.heter_watermark_channel = self.config_dict["heter_watermark_channel"]
        self.ring_watermark_channel = self.config_dict["ring_watermark_channel"]
        self.watermark_channel = sorted(self.heter_watermark_channel + self.ring_watermark_channel)
        self.threshold = self.config_dict['threshold']
        
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "RI"

class RIUtils:
    """Utility class for the Ring-ID algorithm."""

    def __init__(self, config: RIConfig, *args, **kwargs) -> None:
        """Initialize the Ring-ID watermarking algorithm."""
        self.config = config
        
        self.latents, self.pattern, self.mask, self.pattern_list = self._prepare_fourier_pattern_and_mask()
        
    def fft(self, input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim=(-1, -2))

    def ifft(self, input_tensor):
        assert len(input_tensor.shape) == 4
        return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim=(-1, -2)))
    
    
    def _ring_mask(self, size=65, r_out=16, r_in=8, x_offset=0, y_offset=0, mode='full'):
        """
        Construct a rotationally symmetric ring mask (fully replace the logic of RounderRingMask class)
        """
        assert size >= 3
        assert mode == 'full', f"mode '{mode}' not implemented"

        # Step 1: Initialize the frequency domain image and ring vector
        num_rings = r_out
        zero_bg_freq = torch.zeros(size, size)
        center = size // 2
        center_x, center_y = center + x_offset, center - y_offset

        ring_vector = torch.tensor([(200 - i * 4) * (-1) ** i for i in range(num_rings)])
        zero_bg_freq[center_x, center_y:center_y + num_rings] = ring_vector
        zero_bg_freq = zero_bg_freq[None, None, ...]
        ring_vector_np = ring_vector.numpy()

        # Step 2: Rotate the frequency domain image to generate the rotationally invariant background pure_bg
        res = torch.zeros(360, size, size)
        res[0] = zero_bg_freq
        for angle in range(1, 360):
            res[angle] = F.rotate(zero_bg_freq, angle=angle)

        res = res.numpy()
        pure_bg = np.zeros((size, size))

        for x in range(size):
            for y in range(size):
                values, count = np.unique(res[:, x, y], return_counts=True)
                if len(count) > 2:
                    nonzero_values = values[values != 0]
                    max_value = nonzero_values[np.argmax(count[values != 0])]
                    pure_bg[x, y] = max_value
                elif len(count) == 2:
                    pure_bg[x, y] = values[values != 0][0]

        # Step 3: Extract the specified ring interval mask from pure_bg
        right_end = 0 if r_in - 1 < 0 else r_in - 1
        cand_list = ring_vector_np[r_out - 1:right_end:-1]
        mask = np.isin(pure_bg, cand_list)

        # Step 4: Crop the odd dimension → 64×64
        if size % 2:
            mask = mask[:size - 1, :size - 1]
        return mask
    
    def _make_Fourier_ringid_pattern(
            self,
            device,
            key_value_combination,
            no_watermark_latents,
            radius,
            radius_cutoff,
            ring_watermark_channel,
            heter_watermark_channel,
            heter_watermark_region_mask=None,
            ring_width=1,
    ):
        if ring_width != 1:
            raise NotImplementedError(f'Proposed watermark generation only implemented for ring width = 1.')

        if len(key_value_combination) != (self.config.radius - self.config.radius_cutoff):
            raise ValueError('Mismatch between #key values and #slots')

        shape = no_watermark_latents.shape
        if len(shape) != 4:
            raise ValueError(f'Invalid shape for initial latent: {shape}')

        latents_fft = self.fft(no_watermark_latents)
        # watermarked_latents_fft = copy.deepcopy(latents_fft)
        watermarked_latents_fft = torch.zeros_like(latents_fft)

        radius_list = [this_radius for this_radius in range(radius, radius_cutoff, -1)]

        # put ring
        for radius_index in range(len(radius_list)):
            this_r_out = radius_list[radius_index]
            this_r_in = this_r_out - ring_width
            mask = torch.tensor(self._ring_mask(size=shape[-1], r_out=this_r_out, r_in=this_r_in)).to(device).to(
                torch.float64)  # sector_idx default to -1
            for batch_index in range(shape[0]):
                for channel_index in range(len(ring_watermark_channel)):
                    watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real = (1 - mask) * \
                                                                                                       watermarked_latents_fft[
                                                                                                           batch_index,
                                                                                                           ring_watermark_channel[
                                                                                                               channel_index]].real + mask * \
                                                                                                       key_value_combination[
                                                                                                           radius_index][
                                                                                                           channel_index]
                    watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag = (1 - mask) * \
                                                                                                       watermarked_latents_fft[
                                                                                                           batch_index,
                                                                                                           ring_watermark_channel[
                                                                                                               channel_index]].imag + mask * \
                                                                                                       key_value_combination[
                                                                                                           radius_index][
                                                                                                           channel_index]

        # put noise or zeros
        if len(heter_watermark_channel) > 0:
            assert len(heter_watermark_channel) == len(heter_watermark_region_mask)
            heter_watermark_region_mask = heter_watermark_region_mask.to(torch.float64)
            w_type = 'noise'

            if w_type == 'noise':
                w_content = self.fft(torch.randn(*shape, device=device))  # [N, c, h, w]
            elif w_type == 'zeros':
                w_content = self.fft(torch.zeros(*shape, device=device))  # [N, c, h, w]
            else:
                raise NotImplementedError

            for batch_index in range(shape[0]):
                for channel_id, channel_mask in zip(heter_watermark_channel, heter_watermark_region_mask):
                    watermarked_latents_fft[batch_index, channel_id].real = \
                        (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].real + channel_mask * \
                        w_content[batch_index][channel_id].real
                    watermarked_latents_fft[batch_index, channel_id].imag = \
                        (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].imag + channel_mask * \
                        w_content[batch_index][channel_id].imag

        return watermarked_latents_fft
    
    def _prepare_fourier_pattern_and_mask(self):
        # if self.pattern is not None and self.mask is not None:
        #     return self.latents, self.pattern, self.mask
        # get latent shape
        base_latents = get_random_latents(pipe=self.config.pipe, height=self.config.image_size[0], width=self.config.image_size[1])
        original_latents_shape = base_latents.shape
        base_latents = base_latents.to(torch.float64)
        # self.latents = base_latents

        sing_channel_ring_watermark_mask = torch.tensor(
            self._ring_mask(
                size=original_latents_shape[-1],
                r_out=self.config.radius,
                r_in=self.config.radius_cutoff)
        )

        # get heterogeneous watermark mask
        if len(self.config.heter_watermark_channel) > 0:
            single_channel_heter_watermark_mask = torch.tensor(
                self._ring_mask(
                    size=original_latents_shape[-1],
                    r_out=self.config.radius,
                    r_in=self.config.radius_cutoff)  # TODO: change to whole mask
            )
            heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(
                len(self.config.heter_watermark_channel), 1, 1).to(self.config.device)

        watermark_region_mask = []
        for channel_idx in self.config.watermark_channel:
            if channel_idx in self.config.ring_watermark_channel:
                watermark_region_mask.append(sing_channel_ring_watermark_mask)
            else:
                watermark_region_mask.append(single_channel_heter_watermark_mask)
        watermark_region_mask = torch.stack(watermark_region_mask).to(self.config.device)  # [C, 64, 64]
        # self.mask = watermark_region_mask

        single_channel_num_slots = self.config.radius - self.config.radius_cutoff
        key_value_list = [[list(combo) for combo in itertools.product(
            np.linspace(-self.config.ring_value_range, self.config.ring_value_range, self.config.quantization_levels).tolist(),
            repeat=len(self.config.ring_watermark_channel))] for _ in range(single_channel_num_slots)]
        key_value_combinations = list(itertools.product(*key_value_list))

        # random select from all possible value combinations, then generate patterns for selected ones.
        if self.config.assigned_keys > 0:
            assert self.config.assigned_keys <= len(key_value_combinations)
            key_value_combinations = random.sample(key_value_combinations, k=self.config.assigned_keys)
        Fourier_watermark_pattern_list = [self._make_Fourier_ringid_pattern(self.config.device, list(combo), base_latents,
                                                                      radius=self.config.radius, radius_cutoff=self.config.radius_cutoff,
                                                                      ring_watermark_channel=self.config.ring_watermark_channel,
                                                                      heter_watermark_channel=self.config.heter_watermark_channel,
                                                                      heter_watermark_region_mask=heter_watermark_region_mask if len(
                                                                          self.config.heter_watermark_channel) > 0 else None)
                                          for _, combo in enumerate(key_value_combinations)]
        ring_capacity = len(Fourier_watermark_pattern_list)
        print(ring_capacity)

        if self.config.fix_gt:
            Fourier_watermark_pattern_list = [self.fft(self.ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in
                                              Fourier_watermark_pattern_list]

        if self.config.time_shift:
            for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
                # Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * args.time_shift_factor)
                Fourier_watermark_pattern[:, self.config.ring_watermark_channel, ...] = self.fft(
                    torch.fft.fftshift(self.ifft(Fourier_watermark_pattern[:, self.config.ring_watermark_channel, ...]), dim=(-1, -2)))

        # self.pattern_list = Fourier_watermark_pattern_list
        # Use a single ring pattern for verification
        Fourier_watermark_pattern = Fourier_watermark_pattern_list[
            -1]  # [64, -64, 64, -64, 64...], select this ring pattern
        # self.pattern = Fourier_watermark_pattern
        return base_latents, Fourier_watermark_pattern, watermark_region_mask, Fourier_watermark_pattern_list

            
    def generate_Fourier_watermark_latents(self, device, radius, radius_cutoff, watermark_region_mask, watermark_channel,
                                           original_latents=None, watermark_pattern=None):

        # set_random_seed(seed)

        if original_latents is None:
            # original_latents = torch.randn(*shape, device = device)
            raise NotImplementedError('Original latents should be provided.')

        if watermark_pattern is None:
            raise NotImplementedError('Fourier watermark pattern should be provided.')

        # circular_mask = torch.tensor(_ring_mask(size = original_latents.shape[-1], r_out = radius, r_in = radius_cutoff)).to(device)
        watermarked_latents_fft = torch.fft.fftshift(torch.fft.fft2(original_latents), dim=(-1, -2))

        # for channel in watermark_channel:
        #     watermarked_latents_fft[:, channel] = watermarked_latents_fft[:, channel] * ~circular_mask + watermark_pattern[:, channel] * circular_mask

        assert len(watermark_channel) == len(watermark_region_mask)
        for channel, channel_mask in zip(watermark_channel, watermark_region_mask):
            watermarked_latents_fft[:, channel] = watermarked_latents_fft[:,
                                                  channel] * ~channel_mask + watermark_pattern[:,
                                                                             channel] * channel_mask

        return torch.fft.ifft2(torch.fft.ifftshift(watermarked_latents_fft, dim=(-1, -2))).real


class RI(BaseWatermark):
    """RI watermarking algorithm."""

    def __init__(self,
                 watermark_config: RIConfig,
                 *args, **kwargs):
        """
            Initialize the RI algorithm.

            Parameters:
                watermark_config (RIConfig): Configuration instance of the RI algorithm.
        """
        self.config = watermark_config
        self.utils = RIUtils(self.config)
        
        self.detector = RIDetector(
            watermarking_mask=self.utils.mask,
            ring_watermark_channel=self.config.ring_watermark_channel,
            heter_watermark_channel=self.config.heter_watermark_channel,
            pattern_list=self.utils.pattern_list,
            threshold=self.config.threshold,
            device=self.config.device
        )
        
    def _generate_watermarked_image(self, prompt: str, *args,
                    **kwargs) -> Image.Image:
        """Generate an image with a watermarked latent representation."""
        watermarked_latents = self.utils.generate_Fourier_watermark_latents(
            device=self.config.device,
            radius=self.config.radius,
            radius_cutoff=self.config.radius_cutoff,
            original_latents= self.utils.latents,
            watermark_pattern= self.utils.pattern,
            watermark_channel=self.config.watermark_channel,
            watermark_region_mask=self.utils.mask,
        ).to(torch.float32)
        
        # save watermarked latents
        self.set_orig_watermarked_latents(watermarked_latents)
        
        # Set gen seed
        set_random_seed(self.config.gen_seed)

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
            num_images_per_prompt=1,
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
        
        if 'detector_type' in kwargs:
            return self.detector.eval_watermark(reversed_latents, detector_type=kwargs['detector_type'])
        else:
            return self.detector.eval_watermark(reversed_latents)
        
    def get_data_for_visualize(self,
                               image: Image.Image = None,
                               prompt: str = "",
                               guidance_scale: float = 1,
                               decoder_inv: bool = False,
                               *args,
                               **kwargs) -> DataForVisualization:
        """
        Collect data for visualization of the RingID watermarking process.
        
        Returns a DataForVisualization object containing all necessary data for RIVisualizer.
        """
        # Use config values as defaults if not explicitly provided
        guidance_scale_to_use = kwargs.get('guidance_scale', self.config.guidance_scale)
        num_steps_to_use = kwargs.get('num_inference_steps', self.config.num_inference_steps)
        
        # Step 1: Get Text Embeddings
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
        )
        
        # Step 4: Create DataForVisualization object with extended attributes for RI
        data = DataForVisualization(
            config=self.config,
            utils=self.utils,
            image=image,
            reversed_latents=reversed_latents,
            orig_watermarked_latents=self.orig_watermarked_latents,
        )
        
        return data
