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


from ..base import BaseWatermark, BaseConfig
import torch
from typing import List
from utils.utils import set_random_seed
from utils.diffusion_config import DiffusionConfig
from visualize.data_for_visualization import DataForVisualization
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import math
from detection.seal.seal_detection import SEALDetector
from utils.media_utils import *

class SEALConfig(BaseConfig):
    """Config class for SEAL algorithm."""
    
    def initialize_parameters(self) -> None:
        """Initialize parameters for SEAL algorithm."""
        self.k_value = self.config_dict['k_value']
        self.b_value = self.config_dict['b_value']
        self.patch_distance_threshold = self.config_dict['patch_distance_threshold']
        self.theta_mid = self.config_dict['theta_mid']
        self.cap_processor = Blip2Processor.from_pretrained(self.config_dict['cap_processor'])
        self.cap_model = Blip2ForConditionalGeneration.from_pretrained(self.config_dict['cap_processor'], torch_dtype=torch.float16).to(self.device)
        self.sentence_model = SentenceTransformer(self.config_dict['sentence_model']).to(self.device)
        
        self.secret_salt = self.config_dict['secret_salt']
        
    @property
    def algorithm_name(self) -> str:
        """Return the name of the algorithm."""
        return "SEAL"
        
class SEALUtils:
    """Utility class for SEAL algorithm."""
    
    def __init__(self, config: SEALConfig, *args, **kwargs) -> None:
        """Initialize SEAL utility."""
        self.config = config
        
    def _simhash(self, v: torch.Tensor, k: int, b: int, seed: int) -> List[int]:
        """
            SimHash algorithm to generate hash keys for an embedding vector.
            
            Args:
                v: Input embedding vector
                k: Number of patches
                b: Number of bits per patch
                seed: Random seed
                
            Returns:
                List of hash keys
        """
        
        keys = []
        set_random_seed(seed)
        for j in range(k):
            bits = [0] * b
            for i in range(b):
                r_i = torch.randn_like(v)
                bits[i] = 1 if torch.dot(r_i, v) > 0 else 0
                bits[i] = (bits[i] + i + j * 1e4)
            hash_value = hash(tuple(bits))
            keys.append(hash_value)
        return keys
        
    def generate_caption(self, image: Image.Image) -> str:
        """
            Generate caption for an image.
            
            Args:
                image: PIL Image object
                
            Returns:
                Caption string
        """
        raw_image = image.convert('RGB')
        inputs = self.config.cap_processor(raw_image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        out = self.config.cap_model.generate(**inputs)
        return self.config.cap_processor.decode(out[0], skip_special_tokens=True)
    
    def generate_initial_noise(self, embedding: torch.Tensor, k: int, b: int, seed: int) -> torch.Tensor:
        """
        Generates initial noise using improved simhash approach.
        
        Args:
            embedding: Input embedding vector(Nomalized)
            k: k_value(Patch number)
            b: b_value(Bit number per patch)
            seed: Random seed(secret_salt)
        Returns:
            Noise tensor with shape [1, 4, 64, 64]
        """
        
        # Calculate patch grid dimensions
        patch_per_side = int(math.ceil(math.sqrt(k)))
        
        # Generate hash keys for the embedding
        keys = self._simhash(embedding, k, b, seed)
        
        # Create empty noise tensor
        initial_noise = torch.zeros(1, 4, 64, 64, device=self.config.device)
        
        # Calculate patch dimensions
        patch_height = 64 // patch_per_side
        patch_width = 64 // patch_per_side
        
        # Fill noise tensor with random patches based on hash keys
        patch_count = 0
        hash_mapping = {}
        
        for i in range(patch_per_side):
            for j in range(patch_per_side):
                if patch_count >= k:
                    break
                
                # Get hash key for this patch
                hash_key = keys[patch_count]
                hash_mapping[patch_count] = hash_key
                
                # Set random seed based on hash key
                set_random_seed(hash_key)
                
                # Calculate patch coordinates
                y_start = i * patch_height
                x_start = j * patch_width
                y_end = min(y_start + patch_height, 64)
                x_end = min(x_start + patch_width, 64)
                
                # Generate random noise for this patch
                initial_noise[:, :, y_start:y_end, x_start:x_end] = torch.randn(
                    (1, 4, y_end - y_start, x_end - x_start), 
                    device=self.config.device
                )
                
                patch_count += 1
        
        return initial_noise
    
class SEAL(BaseWatermark):
    """SEAL watermarking algorithm."""
    
    def __init__(self, watermark_config: SEALConfig, *args, **kwargs) -> None:
        """
            Initialize the SEAL algorithm.

            Parameters:
                watermark_config (SEALConfig): Configuration instance of the SEAL algorithm.
        """
        self.config = watermark_config
        self.utils = SEALUtils(self.config)
        self.original_embedding = None
        
        self.detector = SEALDetector(self.config.k_value, self.config.b_value, self.config.theta_mid, self.config.cap_processor, self.config.cap_model, self.config.sentence_model, self.config.patch_distance_threshold, self.config.device)
        
    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        """Generate watermarked image."""
        
        ## Step 1: Generate original image
        image = self.config.pipe(prompt).images[0]
        
        ## Step 2: Caption the original image
        image_caption = self.utils.generate_caption(image)
        
        ## Step 3: Get the embedding of the caption
        embedding = self.config.sentence_model.encode(image_caption, convert_to_tensor=True).to(self.config.device)
        embedding = embedding / torch.norm(embedding)
        self.original_embedding = embedding
        
        ## Step 4: Get the watermarked initial latents
        watermarked_latents = self.utils.generate_initial_noise(embedding, self.config.k_value, self.config.b_value, self.config.secret_salt)
        
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
        
        ## Step 5: Generate watermarked image
        watermarked_image = self.config.pipe(prompt, **generation_params).images[0]
        
        return watermarked_image
    
    def _detect_watermark_in_image(self, 
                         image: Image.Image, 
                         prompt: str="", 
                         *args, 
                         **kwargs) -> bool:
        """Detect watermark in the image."""
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
        image_tensor = transform_to_model_format(image, target_size=self.config.image_size[0]).unsqueeze(0).to(text_embeddings.dtype).to(self.config.device)
        image_tensor = image_tensor.to(dtype=self.config.pipe.vae.dtype)
        
        # Step 3: Get Image Latents
        image_latents = get_media_latents(pipe=self.config.pipe, media=image_tensor, sample=False, decoder_inv=kwargs.get('decoder_inv', False))
        
        # Pass only known parameters to forward_diffusion, and let kwargs handle any additional parameters
        inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder_inv', 'guidance_scale', 'num_inference_steps']}
        
        # Step 4: Reverse Image Latents
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale_to_use,
            num_inference_steps=num_steps_to_use,
            **inversion_kwargs
        )[-1]
        
        # Step 5: Generate noise z(~) again from the inspected image's caption
        inspected_caption = self.utils.generate_caption(image)
        inspected_embedding = self.config.sentence_model.encode(inspected_caption, convert_to_tensor=True).to(self.config.device)
        inspected_embedding = inspected_embedding / torch.norm(inspected_embedding)
        
        inspected_noise = self.utils.generate_initial_noise(inspected_embedding, self.config.k_value, self.config.b_value, self.config.secret_salt)
        
        # Detect the watermark
        if 'detector_type' in kwargs:
            return self.detector.eval_watermark(reversed_latents, inspected_noise, detector_type=kwargs['detector_type'])
        else:
            return self.detector.eval_watermark(reversed_latents, inspected_noise)
        
        
    def get_data_for_visualize(self, 
                               image: Image.Image, 
                               prompt: str="", 
                               guidance_scale: float=1, 
                               decoder_inv: bool=False,
                               *args, **kwargs) -> DataForVisualization:
        """
        Get data for visualization of SEAL watermarking process.
        
        Args:
            image: Input image for visualization
            prompt: Text prompt used for generation
            guidance_scale: Guidance scale for diffusion process
            decoder_inv: Whether to use decoder inversion
            
        Returns:
            DataForVisualization object with SEAL-specific data
        """
        # Step 1: Perform detection process for comparison
        inspected_caption = self.utils.generate_caption(image)
        inspected_embedding = self.config.sentence_model.encode(inspected_caption, convert_to_tensor=True).to(self.config.device)
        normalized_inspected_embedding = inspected_embedding / torch.norm(inspected_embedding)
        
        # Generate inspected noise for detection
        inspected_noise = self.utils.generate_initial_noise(normalized_inspected_embedding, self.config.k_value, self.config.b_value, self.config.secret_salt)
        
        # Step 2: Get inverted latents for detection visualization
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
        image_tensor = transform_to_model_format(image, target_size=self.config.image_size[0]).unsqueeze(0).to(text_embeddings.dtype).to(self.config.device)
        image_tensor = image_tensor.to(dtype=self.config.pipe.vae.dtype)
        image_latents = get_media_latents(pipe=self.config.pipe, media=image_tensor, sample=False, decoder_inv=decoder_inv)
        
        inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder_inv', 'guidance_scale', 'num_inference_steps']}
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            **inversion_kwargs
        )
        
        # Get original watermarked latents 
        orig_watermarked_latents = self.get_orig_watermarked_latents()
        
        # Get original embedding 
        original_embedding = self.original_embedding

        # Create DataForVisualization object with SEAL-specific data
        data_for_visualization = DataForVisualization(
            config=self.config,
            utils=self.utils,
            orig_watermarked_latents=orig_watermarked_latents,
            reversed_latents=reversed_latents,
            inspected_embedding=normalized_inspected_embedding,
            original_embedding=original_embedding,
            reference_noise=inspected_noise,
            image=image,
        )
        
        return data_for_visualization
        
        
