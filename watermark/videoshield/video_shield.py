from ..base import BaseWatermark, BaseConfig
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from PIL import Image
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
import logging
from scipy.stats import norm, truncnorm
from functools import reduce
from visualize.data_for_visualization import DataForVisualization
from detection.videoshield.videoshield_detection import VideoShieldDetector
from utils.media_utils import *
from utils.utils import set_random_seed
from utils.pipeline_utils import is_video_pipeline, is_t2v_pipeline, is_i2v_pipeline
from utils.callbacks import DenoisingLatentsCollector
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VAE_DOWNSAMPLE_FACTOR = 8
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class VideoShieldConfig(BaseConfig):
    """Config class for VideoShield algorithm."""
    
    def initialize_parameters(self) -> None:
        """Initialize VideoShield configuration."""
        # Repetition factors for video watermarking
        self.k_f: int = self.config_dict['k_f']  # Frame repetition factor
        self.k_c: int = self.config_dict['k_c']  # Channel repetition factor  
        self.k_h: int = self.config_dict['k_h']  # Height repetition factor
        self.k_w: int = self.config_dict['k_w']  # Width repetition factor
        
        # Temporal threshold for localization
        self.t_temp: float = self.config_dict['t_temp']
        
        # HSTR (Hierarchical Spatio-Temporal Refinement) parameters
        self.hstr_levels: int = self.config_dict['hstr_levels']
        self.t_wm: List[float] = self.config_dict['t_wm']
        self.t_orig: List[float] = self.config_dict['t_orig']
        
        # Watermark generation parameters
        self.wm_key: int = self.config_dict.get('wm_key', 42)
        self.chacha_key_seed: int = self.config_dict.get('chacha_key_seed', 123456)
        self.chacha_nonce_seed: int = self.config_dict.get('chacha_nonce_seed', 789012)
        
        # Detection threshold
        self.threshold: float = self.config_dict.get('threshold', 0.6)
        
        # Calculate latent dimensions
        self.latents_height = self.image_size[0] // VAE_DOWNSAMPLE_FACTOR
        self.latents_width = self.image_size[1] // VAE_DOWNSAMPLE_FACTOR
        
        # Generate watermark pattern
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.wm_key)
        
        # For video, we need frame dimension
        if hasattr(self, 'num_frames') and self.num_frames > 0:
            self.watermark = torch.randint(
                0, 2, 
                [1, 4 // self.k_c, self.num_frames // self.k_f, 
                 self.latents_height // self.k_h, self.latents_width // self.k_w], 
                generator=generator, device=self.device
            )
        else:
            # Fallback for image-only case
            self.watermark = torch.randint(
                0, 2,
                [1, 4 // self.k_c, self.latents_height // self.k_h, self.latents_width // self.k_w],
                generator=generator, device=self.device
            )
    
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'VideoShield'


class VideoShieldUtils:
    """Utility class for VideoShield algorithm."""
    
    def __init__(self, config: VideoShieldConfig, *args, **kwargs) -> None:
        """Initialize the VideoShield watermarking utility."""
        self.config = config
        
        # Generate deterministic cryptographic keys using seeds
        self.chacha_key = self._get_bytes_with_seed(self.config.chacha_key_seed, 32)
        self.chacha_nonce = self._get_bytes_with_seed(self.config.chacha_nonce_seed, 12)
        
        # Calculate latent space dimensions
        if hasattr(self.config, 'num_frames') and self.config.num_frames > 0:
            # Video case: include frame dimension
            self.latentlength = 4 * self.config.num_frames * self.config.latents_height * self.config.latents_width
        else:
            # Image case: no frame dimension
            self.latentlength = 4 * self.config.latents_height * self.config.latents_width
            
        # Calculate watermark length based on repetition factors
        self.marklength = self.latentlength // (self.config.k_f * self.config.k_c * self.config.k_h * self.config.k_w)
        
        # Voting threshold for watermark extraction
        if self.config.k_f == 1 and self.config.k_c == 1 and self.config.k_h == 1 and self.config.k_w == 1:
            self.vote_threshold = 1
        else:
            self.vote_threshold = (self.config.k_f * self.config.k_c * self.config.k_h * self.config.k_w) // 2
    
    def _get_bytes_with_seed(self, seed: int, n: int) -> bytes:
        """Generate deterministic bytes using a seed."""
        random.seed(seed)
        return bytes(random.getrandbits(8) for _ in range(n))
    
    def _stream_key_encrypt(self, sd: np.ndarray) -> np.ndarray:
        """Encrypt the watermark using ChaCha20 cipher."""
        try:
            cipher = ChaCha20.new(key=self.chacha_key, nonce=self.chacha_nonce)
            m_byte = cipher.encrypt(np.packbits(sd).tobytes())
            m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
            return m_bit[:len(sd)]  # Ensure same length as input
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise RuntimeError("Encryption failed") from e
    
    def _truncated_sampling(self, message: np.ndarray) -> torch.Tensor:
        """Truncated Gaussian sampling for watermarking.
        
        Args:
            message: Binary message as a numpy array of 0s and 1s
            
        Returns:
            Watermarked latents tensor
        """
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        
        # Reshape based on whether this is video or image
        if hasattr(self.config, 'num_frames') and self.config.num_frames > 0:
            # Video: (batch, channels, frames, height, width)
            z = torch.from_numpy(z).reshape(
                1, 4, self.config.num_frames, self.config.latents_height, self.config.latents_width
            ).float()
        else:
            # Image: (batch, channels, height, width)
            z = torch.from_numpy(z).reshape(
                1, 4, self.config.latents_height, self.config.latents_width
            ).float()
            
        return z.to(self.config.device)
    
    def create_watermark_and_return_w(self) -> torch.Tensor:
        """Create watermark pattern and return watermarked initial latents."""
        # Create repeated watermark pattern
        if hasattr(self.config, 'num_frames') and self.config.num_frames > 0:
            # Video case: repeat along all dimensions including frames
            sd = self.config.watermark.repeat(1, self.config.k_c, self.config.k_f, self.config.k_h, self.config.k_w)
        else:
            # Image case: repeat along spatial dimensions only
            sd = self.config.watermark.repeat(1, self.config.k_c, self.config.k_h, self.config.k_w)
            
        # Encrypt the repeated watermark
        m = self._stream_key_encrypt(sd.flatten().cpu().numpy())
        
        # Generate watermarked latents using truncated sampling
        w = self._truncated_sampling(m)
        
        return w

class VideoShieldWatermark(BaseWatermark):
    """Main class for VideoShield watermarking algorithm."""
    
    def __init__(self, watermark_config: VideoShieldConfig, *args, **kwargs) -> None:
        """Initialize the VideoShield watermarking algorithm.
        
        Args:
            watermark_config: Configuration instance of the VideoShield algorithm
        """
        self.config = watermark_config
        self.utils = VideoShieldUtils(self.config)
        
        # Initialize detector with encryption keys from utils
        self.detector = VideoShieldDetector(
            watermark=self.config.watermark,
            threshold=self.config.threshold,
            device=self.config.device,
            chacha_key=self.utils.chacha_key,
            chacha_nonce=self.utils.chacha_nonce,
            height=self.config.image_size[0],
            width=self.config.image_size[1],
            num_frames=self.config.num_frames,
            k_f=self.config.k_f,
            k_c=self.config.k_c,
            k_h=self.config.k_h,
            k_w=self.config.k_w
        )
    
    def _generate_watermarked_video(self, prompt: str, num_frames: Optional[int] = None, *args, **kwargs) -> List[Image.Image]:
        """Generate watermarked video using VideoShield algorithm.
        
        Args:
            prompt: The input prompt for video generation
            num_frames: Number of frames to generate (uses config value if None)
            
        Returns:
            List of generated watermarked video frames
        """
        if not is_video_pipeline(self.config.pipe):
            raise ValueError(f"This pipeline ({self.config.pipe.__class__.__name__}) does not support video generation.")
        
        # Set random seed for reproducibility
        set_random_seed(self.config.gen_seed)
        
        # Use config frames if not specified
        frames_to_generate = num_frames if num_frames is not None else self.config.num_frames
        
        # Set num_frames in config for watermark generation
        original_num_frames = getattr(self.config, 'num_frames', None)
        self.config.num_frames = frames_to_generate
        
        try:
            # Generate watermarked latents
            watermarked_latents = self.utils.create_watermark_and_return_w().to(self.config.pipe.unet.dtype)
            
            # Save watermarked latents for visualization
            self.set_orig_watermarked_latents(watermarked_latents)
            
            # Construct video generation parameters
            generation_params = {
                "num_inference_steps": self.config.num_inference_steps,
                "guidance_scale": self.config.guidance_scale,
                "height": self.config.image_size[0],
                "width": self.config.image_size[1],
                "num_frames": frames_to_generate,
                "latents": watermarked_latents
            }
            
            # Add parameters from config.gen_kwargs
            if hasattr(self.config, "gen_kwargs") and self.config.gen_kwargs:
                for key, value in self.config.gen_kwargs.items():
                    if key not in generation_params:
                        generation_params[key] = value
            
            # Use kwargs to override default parameters
            for key, value in kwargs.items():
                if key != "num_frames":  # Prevent overriding processed parameters
                    generation_params[key] = value
            
            # Handle I2V pipelines that need dimension permutation (like SVD)
            final_latents = watermarked_latents
            if is_i2v_pipeline(self.config.pipe):
                logger.info("I2V pipeline detected, permuting latent dimensions.")
                final_latents = final_latents.permute(0, 2, 1, 3, 4)  # (b,c,f,h,w) -> (b,f,c,h,w)
            
            generation_params["latents"] = final_latents
            
            # Generate video
            output = self.config.pipe(
                prompt,
                **generation_params
            )
            
            # Extract frames from output
            if hasattr(output, 'frames'):
                frames = output.frames[0]
            elif hasattr(output, 'videos'):
                frames = output.videos[0]
            else:
                frames = output[0] if isinstance(output, tuple) else output
            
            # Convert frames to PIL Images
            frame_list = []
            for frame in frames:
                if not isinstance(frame, Image.Image):
                    if isinstance(frame, np.ndarray):
                        if frame.dtype == np.uint8:
                            frame_pil = Image.fromarray(frame)
                        else:
                            frame_scaled = (frame * 255).astype(np.uint8)
                            frame_pil = Image.fromarray(frame_scaled)
                    elif isinstance(frame, torch.Tensor):
                        if frame.dim() == 3 and frame.shape[-1] in [1, 3]:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).byte()
                            frame_np = frame.cpu().numpy()
                            frame_pil = Image.fromarray(frame_np)
                        else:
                            raise ValueError(f"Unexpected tensor shape for frame: {frame.shape}")
                    else:
                        raise TypeError(f"Unexpected type for frame: {type(frame)}")
                else:
                    frame_pil = frame
                
                frame_list.append(frame_pil)
            
            return frame_list
            
        finally:
            # Restore original num_frames
            if original_num_frames is not None:
                self.config.num_frames = original_num_frames
            elif hasattr(self.config, 'num_frames'):
                delattr(self.config, 'num_frames')
    
    def _detect_watermark_in_image(self, image: Image.Image, prompt: str = "", 
                                   *args, **kwargs) -> Dict[str, float]:
        """Detect VideoShield watermark in image.
        
        Args:
            image: Input PIL image
            prompt: Text prompt used for generation
            
        Returns:
            Dictionary containing detection results
        """
        # Use config values as defaults if not explicitly provided
        guidance_scale_to_use = kwargs.get('guidance_scale', self.config.guidance_scale)
        num_steps_to_use = kwargs.get('num_inference_steps', self.config.num_inference_steps)
        
        # Get text embeddings
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
        
        # Preprocess image
        image_tensor = transform_to_model_format(
            image, target_size=self.config.image_size[0]
        ).unsqueeze(0).to(text_embeddings.dtype).to(self.config.device)
        
        # Get image latents
        image_latents = get_media_latents(
            pipe=self.config.pipe, 
            media=image_tensor, 
            sample=False, 
            decoder_inv=kwargs.get("decoder_inv", False)
        )
        
        # Perform DDIM inversion
        inversion_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['decoder_inv', 'guidance_scale', 'num_inference_steps']}
        
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale_to_use,
            num_inference_steps=num_steps_to_use,
            **inversion_kwargs
        )[-1]
        
        # Use detector or utils for evaluation
        if 'detector_type' in kwargs:
            return self.detector.eval_watermark(reversed_latents, detector_type=kwargs['detector_type'])
        else:
            return self.utils.eval_watermark(reversed_latents)
    
    def _get_video_latents(self, vae, video_frames, sample=True, rng_generator=None, permute=True):
        encoding_dist = vae.encode(video_frames).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = (encoding * 0.18215).unsqueeze(0)
        if permute:
            latents = latents.permute(0, 2, 1, 3, 4)
        return latents
    
    def _detect_watermark_in_video(self, 
                                   video_frames: Union[torch.Tensor, List[Image.Image]], 
                                   prompt: str = "", 
                                   detector_type: str = 'bit_acc',
                                   *args, **kwargs) -> Dict[str, float]:
        """Detect VideoShield watermark in video.
        
        Args:
            video_frames: Input video frames as tensor or list of PIL images
            prompt: Text prompt used for generation
            
        Returns:
            Dictionary containing detection results
        """
        # Use config values as defaults if not explicitly provided
        guidance_scale_to_use = kwargs.get('guidance_scale', self.config.guidance_scale)
        num_steps_to_use = kwargs.get('num_inference_steps', self.config.num_inference_steps)
        
        # Convert frames to tensor if needed
        if isinstance(video_frames, list):
            from torchvision import transforms
            frames_tensor = torch.stack([transforms.ToTensor()(frame) for frame in video_frames])
            video_frames = 2.0 * frames_tensor - 1.0  # Normalize to [-1, 1]
        
        video_frames = video_frames.to(self.config.device).to(self.config.pipe.vae.dtype)
        
        # Get video latents
        with torch.no_grad():
            # TODO: Add support for I2V pipeline
            video_latents = self._get_video_latents(self.config.pipe.vae, video_frames, sample=False)
        
        # Perform DDIM inversion
        inversion_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['guidance_scale', 'num_inference_steps']}
        
        from diffusers import DDIMInverseScheduler
        original_scheduler = self.config.pipe.scheduler
        inverse_scheduler = DDIMInverseScheduler.from_config(original_scheduler.config)
        self.config.pipe.scheduler = inverse_scheduler
        
        final_reversed_latents = self.config.pipe(
            prompt=prompt,
            latents=video_latents,
            num_inference_steps=num_steps_to_use,
            guidance_scale=guidance_scale_to_use,
            output_type='latent',
            **inversion_kwargs
        ).frames # [B, F, H, W, C](T2V)
        self.config.pipe.scheduler = original_scheduler
        
        # Use detector for evaluation
        return self.detector.eval_watermark(final_reversed_latents, detector_type=detector_type)
       
    
    def get_data_for_visualize(self, 
                               video_frames: List[Image.Image], 
                               prompt: str = "", 
                               guidance_scale: float = 1, 
                               *args, **kwargs) -> DataForVisualization:
        """Get VideoShield visualization data.
        
        This method generates the necessary data for visualizing VideoShield watermarks,
        including original watermarked latents and reversed latents from inversion.
        
        Args:
            image: The image to visualize watermarks for (can be None for generation only)
            prompt: The text prompt used for generation
            guidance_scale: Guidance scale for generation and inversion
            
        Returns:
            DataForVisualization object containing visualization data
        """
        # Use config values as defaults if not explicitly provided
        guidance_scale_to_use = kwargs.get('guidance_scale', self.config.guidance_scale)
        num_steps_to_use = kwargs.get('num_inference_steps', self.config.num_inference_steps)
        
        # Convert frames to tensor if needed
        if isinstance(video_frames, list):
            from torchvision import transforms
            frames_tensor = torch.stack([transforms.ToTensor()(frame) for frame in video_frames])
            video_frames = 2.0 * frames_tensor - 1.0  # Normalize to [-1, 1]
        
        video_frames = video_frames.to(self.config.device).to(self.config.pipe.vae.dtype)
        
        # Get video latents
        with torch.no_grad():
            # TODO: Add support for I2V pipeline
            video_latents = self._get_video_latents(self.config.pipe.vae, video_frames, sample=False)
        
        # Perform DDIM inversion
        inversion_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['guidance_scale', 'num_inference_steps']}
        
        from diffusers import DDIMInverseScheduler
        original_scheduler = self.config.pipe.scheduler
        inverse_scheduler = DDIMInverseScheduler.from_config(original_scheduler.config)
        self.config.pipe.scheduler = inverse_scheduler
        collector = DenoisingLatentsCollector(save_every_n_steps=1, to_cpu=True)
        
        final_reversed_latents = self.config.pipe(
            prompt=prompt,
            latents=video_latents,
            num_inference_steps=num_steps_to_use,
            guidance_scale=guidance_scale_to_use,
            output_type='latent',
            callback=collector,
            callback_steps=1,
            **inversion_kwargs
        ).frames # [B, F, H, W, C](T2V)
        self.config.pipe.scheduler = original_scheduler
        
        reversed_latents = collector.latents_list # List[Tensor]
        
        return DataForVisualization(
            config=self.config,
            utils=self.utils,
            orig_watermarked_latents=self.get_orig_watermarked_latents(),
            reversed_latents=reversed_latents,
            video_frames=video_frames,
        )