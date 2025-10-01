from ..base import BaseWatermark, BaseConfig
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from PIL import Image
import galois
from scipy.sparse import csr_matrix
from scipy.special import binom
import logging
from functools import reduce
from visualize.data_for_visualization import DataForVisualization
from detection.videomark.videomark_detection import VideoMarkDetector
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


class VideoMarkConfig(BaseConfig):
    """Config class for VideoMark algorithm."""
    
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.fpr = self.config_dict['fpr']
        self.t = self.config_dict['prc_t']
        self.var = self.config_dict['var']
        self.threshold = self.config_dict['threshold']
        self.sequence_length = self.config_dict['sequence_length'] # Length of the watermark sequence
        self.message_length = self.config_dict['message_length'] # Number of bits in each sequence
        self.message_sequence = np.random.randint(0, 2, size=(self.sequence_length, self.message_length)) # <= 512 bits for robustness
        self.shift = np.random.default_rng().integers(0, self.sequence_length - self.num_frames)
        self.message = self.message_sequence[self.shift : self.shift + self.num_frames]
        self.latents_height = self.image_size[0] // self.pipe.vae_scale_factor
        self.latents_width = self.image_size[1] // self.pipe.vae_scale_factor
        self.latents_channel = self.pipe.unet.config.in_channels
        self.n = self.latents_height * self.latents_width * self.latents_channel # Dimension of the latent space
        self.GF = galois.GF(2)
        
        # Seeds for key generation
        self.gen_matrix_seed = self.config_dict['keygen']['gen_matrix_seed']
        self.indice_seed = self.config_dict['keygen']['indice_seed']
        self.one_time_pad_seed = self.config_dict['keygen']['one_time_pad_seed']
        self.test_bits_seed = self.config_dict['keygen']['test_bits_seed']
        self.permute_bits_seed = self.config_dict['keygen']['permute_bits_seed']
        
        # Seeds for encoding
        self.payload_seed = self.config_dict['encode']['payload_seed']
        self.error_seed = self.config_dict['encode']['error_seed']
        self.pseudogaussian_seed = self.config_dict['encode']['pseudogaussian_seed']
    
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'VideoMark'

    def _get_message(length: int, window: int, seed=None) -> int:
        """Return a random start index for a subarray of size `window` in array of size `length`."""
        rng = np.random.default_rng()
        return rng.integers(0, length - window)

class VideoMarkUtils:
    """Utility class for VideoMark algorithm."""
    
    def __init__(self, config: VideoMarkConfig, *args, **kwargs) -> None:
        """Initialize PRC utility."""
        self.config = config
        self.encoding_key, self.decoding_key = self._generate_encoding_key(self.config.message_length)
    
    def _generate_encoding_key(self, message_length: int) -> Tuple[tuple, tuple]:
        """Generate encoding key for PRC algorithm."""
        # Set basic scheme parameters
        num_test_bits = int(np.ceil(np.log2(1 / self.config.fpr)))
        secpar = int(np.log2(binom(self.config.n, self.config.t)))
        g = secpar
        k = message_length + g + num_test_bits
        r = self.config.n - k - secpar
        noise_rate = 1 - 2 ** (-secpar / g ** 2)
        
        # Sample n by k generator matrix (all but the first n-r of these will be over-written)
        generator_matrix = self.config.GF.Random(shape=(self.config.n, k), seed=self.config.gen_matrix_seed)
        
        # Sample scipy.sparse parity-check matrix together with the last n-r rows of the generator matrix
        row_indices = []
        col_indices = []
        data = []
        for i, row in enumerate(range(r)):
            np.random.seed(self.config.indice_seed + i)
            chosen_indices = np.random.choice(self.config.n - r + row, self.config.t - 1, replace=False)
            chosen_indices = np.append(chosen_indices, self.config.n - r + row)
            row_indices.extend([row] * self.config.t)
            col_indices.extend(chosen_indices)
            data.extend([1] * self.config.t)
            generator_matrix[self.config.n - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
        parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))
        
        # Compute scheme parameters
        max_bp_iter = int(np.log(self.config.n) / np.log(self.config.t))
        
        # Sample one-time pad and test bits
        one_time_pad = self.config.GF.Random(self.config.n, seed=self.config.one_time_pad_seed)
        test_bits = self.config.GF.Random(num_test_bits, seed=self.config.test_bits_seed)
        
        # Permute bits
        np.random.seed(self.config.permute_bits_seed)
        permutation = np.random.permutation(self.config.n)
        generator_matrix = generator_matrix[permutation]
        one_time_pad = one_time_pad[permutation]
        parity_check_matrix = parity_check_matrix[:, permutation]
        
        return (generator_matrix, one_time_pad, test_bits, g, noise_rate), (generator_matrix, parity_check_matrix, one_time_pad, self.config.fpr, noise_rate, test_bits, g, max_bp_iter, self.config.t)

    def _encode_message(self, encoding_key: tuple, message: np.ndarray = None) -> np.ndarray:
        """Encode a message using PRC algorithm."""
        generator_matrix, one_time_pad, test_bits, g, noise_rate = encoding_key
        n, k = generator_matrix.shape

        if message is None:
            payload = np.concatenate((test_bits, self.config.GF.Random(k - len(test_bits), seed=self.config.payload_seed)))
        else:
            assert len(message) <= k-len(test_bits)-g, "Message is too long"
            payload = np.concatenate((test_bits, self.config.GF.Random(g, seed=self.config.payload_seed), self.config.GF(message), self.config.GF.Zeros(k-len(test_bits)-g-len(message))))

        np.random.seed(self.config.error_seed)
        error = self.config.GF(np.random.binomial(1, noise_rate, n))

        return 1 - 2 * torch.tensor(payload @ generator_matrix.T + one_time_pad + error, dtype=float)
    

    def _sample_prc_codeword(self, codeword: torch.Tensor, basis: torch.Tensor = None) -> torch.Tensor:
        """Sample a PRC codeword."""
        codeword_np = codeword.numpy()
        np.random.seed(self.config.pseudogaussian_seed)
        pseudogaussian_np = codeword_np * np.abs(np.random.randn(*codeword_np.shape))
        pseudogaussian = torch.from_numpy(pseudogaussian_np).to(dtype=torch.float32)
        if basis is None:
            return pseudogaussian
        return pseudogaussian @ basis.T

    def inject_watermark(self) -> torch.Tensor:
        """Generate watermarked latents from PRC codeword."""
        # Step 1: Encode message
        prc_codeword = torch.stack([self._encode_message(self.encoding_key, self.config.message[frame_index]) for frame_index in range(self.config.num_frames)])
        # Step 2: Sample PRC codeword and get watermarked latents
        watermarked_latents = self._sample_prc_codeword(prc_codeword).reshape(self.config.num_frames, 1, self.config.latents_channel, self.config.latents_height, self.config.latents_width).to(self.config.device)
        
        return watermarked_latents.permute(1, 2, 0, 3, 4) # (b, c, f, h, w)
    


class VideoMarkWatermark(BaseWatermark):
    """Main class for VideoMark watermarking algorithm."""
    
    def __init__(self, watermark_config: VideoMarkConfig, *args, **kwargs) -> None:
        """Initialize the VideoShield watermarking algorithm.
        
        Args:
            watermark_config: Configuration instance of the VideoMark algorithm
        """
        self.config = watermark_config
        self.utils = VideoMarkUtils(self.config)
        
        # Initialize detector with encryption keys from utils
        self.detector = VideoMarkDetector(
            message_sequence=self.config.message_sequence,
            watermark=self.config.message,
            num_frames=self.config.num_frames,
            var=self.config.var,
            decoding_key=self.utils.decoding_key,
            GF=self.config.GF,
            threshold=self.config.threshold,
            device=self.config.device
        )
    
    def _generate_watermarked_video(self, prompt: str, num_frames: Optional[int] = None, *args, **kwargs) -> List[Image.Image]:
        """Generate watermarked video using VideoMark algorithm.
        
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
            watermarked_latents = self.utils.inject_watermark().to(self.config.pipe.unet.dtype)
            
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
        """Detect VideoMark watermark in image.
        
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
        """Detect VideoMark watermark in video.
        
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
        """Get VideoMark visualization data.
        
        This method generates the necessary data for visualizing VideoMark watermarks,
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