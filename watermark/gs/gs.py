from ..base import BaseWatermark, BaseConfig
import torch
from typing import Dict
from PIL import Image
from utils.diffusion_config import DiffusionConfig
import numpy as np
from Crypto.Cipher import ChaCha20
import random
from scipy.stats import norm,truncnorm
from functools import reduce
from visualize.data_for_visualization import DataForVisualization
from detection.gs.gs_detection import GSDetector
from utils.media_utils import *
from utils.utils import set_random_seed

class GSConfig(BaseConfig):
    """Config class for Gaussian Shading algorithm."""
    
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.channel_copy = self.config_dict['channel_copy']
        self.hw_copy = self.config_dict['hw_copy']
        self.chacha = self.config_dict['chacha']
        self.wm_key = self.config_dict['wm_key']
        self.chacha_key_seed = self.config_dict['chacha_key_seed']
        self.chacha_nonce_seed = self.config_dict['chacha_nonce_seed']
        self.threshold = self.config_dict['threshold']
        self.vote_threshold = 1 if self.hw_copy == 1 and self.channel_copy == 1 else self.channel_copy * self.hw_copy * self.hw_copy // 2
        
        self.latents_height = self.image_size[0] // self.pipe.vae_scale_factor
        self.latents_width = self.image_size[1] // self.pipe.vae_scale_factor
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.wm_key)
        self.watermark = torch.randint(0, 2, [1, 4 // self.channel_copy, self.latents_height // self.hw_copy, self.latents_width // self.hw_copy], generator=generator, device=self.device)
        if not self.chacha:
            self.key = torch.randint(0, 2, [1, 4, self.latents_height, self.latents_width], generator=generator, device=self.device)

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'GS'

class GSUtils:
    """Utility class for Gaussian Shading algorithm."""
    
    def __init__(self, config: GSConfig, *args, **kwargs) -> None:
        """
            Initialize the Gaussian Shading watermarking utility.
            
            Parameters:
                config (GSConfig): Configuration for the Gaussian Shading watermarking algorithm.
        """
        self.config = config
        self.chacha_key = self._get_bytes_with_seed(self.config.chacha_key_seed, 32)
        self.chacha_nonce = self._get_bytes_with_seed(self.config.chacha_nonce_seed, 12)
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.config.channel_copy * self.config.hw_copy * self.config.hw_copy)
    
    def _get_bytes_with_seed(self, seed: int, n: int) -> bytes:
        random.seed(seed)
        return bytes(random.getrandbits(8) for _ in range(n))
    
    def _stream_key_encrypt(self, sd):
        """Encrypt the watermark using ChaCha20 cipher."""
        cipher = ChaCha20.new(key=self.chacha_key, nonce=self.chacha_nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit    
    
    def _truncSampling(self, message):
        """Truncated Gaussian sampling for watermarking."""
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).float()
        return z.cuda()
    
    def _create_watermark(self) -> torch.Tensor:
        """Create watermark pattern without encryption."""
        sd = self.config.watermark.repeat(1,self.config.channel_copy,self.config.hw_copy,self.config.hw_copy)
        m = ((sd + self.config.key) % 2).flatten().cpu().numpy()
        w = self._truncSampling(m)
        return w
    
    def _create_watermark_chacha(self) -> torch.Tensor:
        """Create watermark pattern using ChaCha20 cipher."""
        sd = self.config.watermark.repeat(1,self.config.channel_copy,self.config.hw_copy,self.config.hw_copy)
        m = self._stream_key_encrypt(sd.flatten().cpu().numpy())
        w = self._truncSampling(m)
        return w
        
    def inject_watermark(self) -> torch.Tensor:
        """Inject watermark into latent space."""
        if self.config.chacha:
            watermarked = self._create_watermark_chacha()
        else:
            watermarked = self._create_watermark()
        return watermarked

class GS(BaseWatermark):
    """Main class for Gaussian Shading watermarking algorithm."""
    
    def __init__(self,
                 watermark_config: GSConfig,
                 *args, **kwargs):
        """
            Initialize the Gaussian Shading watermarking algorithm.
            
            Parameters:
                watermark_config (GSConfig): Configuration instance of the GS algorithm.
        """
        self.config = watermark_config
        self.utils = GSUtils(self.config)
        
        self.detector = GSDetector(
            watermarking_mask=self.config.watermark,
            chacha=self.config.chacha,
            wm_key=(self.utils.chacha_key, self.utils.chacha_nonce) if self.config.chacha else self.config.key,
            channel_copy=self.config.channel_copy,
            hw_copy=self.config.hw_copy,
            vote_threshold=self.config.vote_threshold,
            threshold=self.config.threshold,
            device=self.config.device
        )
    
    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        """Generate image with Gaussian Shading watermark."""
        set_random_seed(self.config.gen_seed)
        watermarked_latents = self.utils.inject_watermark()
        
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
                                   prompt: str="", 
                                   *args, 
                                   **kwargs) -> Dict[str, float]:
        """Detect Gaussian Shading watermark."""
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
        image_latents = get_media_latents(pipe=self.config.pipe, media=image, sample=False, decoder_inv=kwargs.get("decoder_inv", False))
        
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
        
        if 'detector_type' in kwargs:
            return self.detector.eval_watermark(reversed_latents, detector_type=kwargs['detector_type'])
        else:
            return self.detector.eval_watermark(reversed_latents)
        
    def get_data_for_visualize(self, 
                           image: Image.Image,
                           prompt: str="",
                           guidance_scale: float=1,
                           decoder_inv: bool=False,
                           *args,
                           **kwargs) -> DataForVisualization:
        """Get Gaussian Shading visualization data"""
        # 1. Generate watermarked latents
        set_random_seed(self.config.gen_seed)
        watermarked_latents = self.utils.inject_watermark()
        
        # 2. Generate actual watermarked image using the same process as _generate_watermarked_image
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
        
        # 3. Perform watermark detection to get inverted latents (for comparison)
        inverted_latents = None
        try:
            # Use the same detection process as _detect_watermark_in_image
            guidance_scale_to_use = kwargs.get('guidance_scale', self.config.guidance_scale)
            num_steps_to_use = kwargs.get('num_inference_steps', self.config.num_inference_steps)
            
            # Get Text Embeddings
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
            inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder_inv', 'guidance_scale', 'num_inference_steps']}
            
            reversed_latents = self.config.inversion.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale_to_use,
                num_inference_steps=num_steps_to_use,
                **inversion_kwargs
            )
            
        except Exception as e:
            raise ValueError(f"Warning: Could not perform inversion for visualization: {e}")
    
        # 4. Prepare visualization data
        return DataForVisualization(
            config=self.config,
            utils=self.utils,
            orig_watermarked_latents=self.orig_watermarked_latents,
            reversed_latents=reversed_latents,
            image=image
            )
