from ..base import BaseWatermark, BaseConfig
import torch
from typing import Dict, Tuple
from markdiffusion.utils.diffusion_config import DiffusionConfig
import numpy as np
import galois
from scipy.sparse import csr_matrix
from scipy.special import binom
from markdiffusion.visualize.data_for_visualization import DataForVisualization
from markdiffusion.detection.prc.prc_detection import PRCDetector
from markdiffusion.utils.media_utils import *
from markdiffusion.utils.utils import set_random_seed
from PIL import Image

class PRCConfig(BaseConfig):
    """Config class for PRC algorithm."""
    
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.fpr = self.config_dict['fpr']
        self.t = self.config_dict['prc_t']
        self.var = self.config_dict['var']
        self.threshold = self.config_dict['threshold']
        self.message = self._str_to_binary_array(self.config_dict['message'])
        self.message_length = len(self.message) # 8-bit for each character, <= 512 bits for robustness
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
        return 'PRC'
        
    def _str_to_binary_array(self, s: str) -> np.ndarray:
        """Convert string to binary array."""
        # Convert string to binary string
        binary_str = ''.join(format(ord(c), '08b') for c in s)
        
        # Convert binary string to NumPy array
        binary_array = np.array([int(bit) for bit in binary_str])
        
        return binary_array
    
    
class PRCUtils:
    """Utility class for PRC algorithm."""
    
    def __init__(self, config: PRCConfig, *args, **kwargs) -> None:
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

    def _encode_message(self, encoding_key: tuple, message: str = None) -> np.ndarray:
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
        prc_codeword = self._encode_message(self.encoding_key, self.config.message)
        # Step 2: Sample PRC codeword and get watermarked latents
        watermarked_latents = self._sample_prc_codeword(prc_codeword).reshape(1, self.config.latents_channel, self.config.latents_height, self.config.latents_width).to(self.config.device)

        return watermarked_latents
    
class PRC(BaseWatermark):
    """PRC watermark class."""
    
    def __init__(self, 
                 watermark_config: PRCConfig,
                 *args, **kwargs):
        """
            Initialize PRC watermarking algorithm.
            
            Parameters:
                watermark_config (PRCConfig): Configuration instance of the PRC algorithm.
        """
        self.config = watermark_config
        self.utils = PRCUtils(self.config)
        
        self.detector = PRCDetector(
            var=self.config.var,
            decoding_key=self.utils.decoding_key,
            GF=self.config.GF,
            threshold=self.config.threshold,
            device=self.config.device
        )
        
    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> torch.Tensor:
        """Generate watermarked image."""
        watermarked_latents = self.utils.inject_watermark()
        
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
                         prompt: str="", 
                         *args,
                         **kwargs) -> Dict[str, float]:
        """Detect watermark in image."""
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
        # 1. Generate watermarked latents and collect intermediate data
        set_random_seed(self.config.gen_seed)
        
        # Step 1: Encode message
        prc_codeword = self.utils._encode_message(self.utils.encoding_key, self.config.message)
        
        # Step 2: Sample PRC codeword
        pseudogaussian_noise = self.utils._sample_prc_codeword(prc_codeword)
        
        # Step 3: Generate watermarked latents
        watermarked_latents = pseudogaussian_noise.reshape(1, self.config.latents_channel, self.config.latents_height, self.config.latents_width).to(self.config.device)
        
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
            )[-1]
            
            inverted_latents = reversed_latents
            
        except Exception as e:
            print(f"Warning: Could not perform inversion for visualization: {e}")
            inverted_latents = None
        
        # 3.5. Run actual detection to get recovered PRC codeword
        recovered_prc = None
        try:
            if inverted_latents is not None:
                # Use the detector to recover the PRC codeword
                detection_result = self.detector.eval_watermark(inverted_latents)
                # The detector should have recovered_prc attribute or return it
                if hasattr(self.detector, 'recovered_prc') and self.detector.recovered_prc is not None:
                    recovered_prc = self.detector.recovered_prc
                elif 'recovered_prc' in detection_result:
                    recovered_prc = detection_result['recovered_prc']
                else:
                    print("Warning: Detector did not provide recovered_prc")
        except Exception as e:
            print(f"Warning: Could not recover PRC codeword for visualization: {e}")
            recovered_prc = None
    
        # 4. Prepare PRC-specific data
        # Convert message to binary
        message_bits = torch.tensor(self.config._str_to_binary_array(self.config.config_dict['message']), dtype=torch.float32)
        
        # Get generator matrix
        generator_matrix = torch.tensor(np.array(self.utils.encoding_key[0], dtype=float), dtype=torch.float32)
        
        # Get parity check matrix
        parity_check_matrix = self.utils.decoding_key[1] 
        
        # PRC parameters for visualization
        prc_params = {
            'FPR': self.config.fpr,
            'Parameter t': self.config.t,
            'Variance': self.config.var,
            'Threshold': self.config.threshold,
            'Message Length': self.config.message_length,
            'Latent Dimension': self.config.n
        }
        
        # 5. Prepare visualization data
        # Convert inverted_latents to list format to match base class expectations
        reversed_latents_list = [inverted_latents] if inverted_latents is not None else [None]
        
        return DataForVisualization(
            config=self.config,                    
            utils=self.utils,                      
            latent_lists=[watermarked_latents],
            orig_latents=watermarked_latents,
            orig_watermarked_latents=watermarked_latents,
            watermarked_latents=watermarked_latents,
            watermarked_image=watermarked_image,    
            image=image,                            
            reversed_latents=reversed_latents_list,  
            inverted_latents=inverted_latents,      
            # PRC-specific data
            message_bits=message_bits,
            prc_codeword=torch.tensor(prc_codeword, dtype=torch.float32),
            pseudogaussian_noise=torch.tensor(pseudogaussian_noise, dtype=torch.float32),
            generator_matrix=generator_matrix,
            parity_check_matrix=parity_check_matrix,
            prc_params=prc_params,
            threshold=self.config.threshold,
            recovered_prc=torch.tensor(recovered_prc, dtype=torch.float32) if recovered_prc is not None else None
        )
    
