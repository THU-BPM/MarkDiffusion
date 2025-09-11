from ..base import BaseWatermark, BaseConfig
from utils.media_utils import *
import os
import types
import torch
from typing import Dict, Union, List, Optional
from utils.utils import set_random_seed, inherit_docstring
from utils.diffusion_config import DiffusionConfig
import copy
import numpy as np
from PIL import Image
from visualize.data_for_visualization import DataForVisualization
from evaluation.dataset import StableDiffusionPromptsDataset
from utils.media_utils import get_random_latents
from .watermark_generator import OptimizedDataset, get_watermarking_mask, get_watermarking_pattern, inject_watermark, optimizer_wm_prompt, ROBINWatermarkedImageGeneration
from detection.robin.robin_detection import ROBINDetector

class ROBINConfig(BaseConfig):
    """Config class for ROBIN algorithm, load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        ## Watermarking-Specific Parameters
        self.w_seed = self.config_dict['w_seed']
        self.w_channel = self.config_dict['w_channel']
        self.w_pattern = self.config_dict['w_pattern']
        self.w_mask_shape = self.config_dict['w_mask_shape']
        self.w_up_radius = self.config_dict['w_up_radius']
        self.w_low_radius = self.config_dict['w_low_radius']
        self.w_injection = self.config_dict['w_injection']
        self.w_pattern_const = self.config_dict['w_pattern_const']
        self.threshold = self.config_dict['threshold']
        
        self.watermarking_step = self.config_dict['watermarking_step']
        
        ## Training-Specific Parameters
        self.learning_rate = self.config_dict['learning_rate'] # learning rate for watermark optimization
        self.scale_lr = self.config_dict['scale_lr'] # if True, learning_rate will be multiplied by gradient_accumulation_steps * train_batch_size * num_processes
        self.max_train_steps = self.config_dict['max_train_steps'] # maximum number of training steps for watermark optimization
        self.save_steps = self.config_dict['save_steps'] # save steps for watermark optimization
        self.train_batch_size = self.config_dict['train_batch_size'] # batch size for watermark optimization
        self.gradient_accumulation_steps = self.config_dict['gradient_accumulation_steps'] # gradient accumulation steps for watermark optimization
        self.gradient_checkpointing = self.config_dict['gradient_checkpointing'] # if True, use gradient checkpointing for watermark optimization
        self.mixed_precision = self.config_dict['mixed_precision'] # fp16, fp32, bf16
        self.train_seed = self.config_dict['train_seed'] # seed for watermark optimization

        self.optimized_guidance_scale = self.config_dict['optimized_guidance_scale'] # guidance scale for optimized prompt signal
        self.data_guidance_scale = self.config_dict['data_guidance_scale'] # guidance scale for data prompt signal
        self.train_guidance_scale = self.config_dict['train_guidance_scale'] # guidance scale for training prompt signal
        
        # self.output_img_dir = 'watermark/robin/generated_images'
        self.output_img_dir = "watermark/robin/generated_images"
        self.ckpt_dir = 'watermark/robin/ckpts'
        
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'ROBIN'
        
class ROBINUtils:
    """Utility class for ROBIN algorithm, contains helper functions."""

    def __init__(self, config: ROBINConfig, *args, **kwargs) -> None:
        """
            Initialize the ROBIN watermarking algorithm.
            
            Parameters:
                config (ROBINConfig): Configuration for the ROBIN algorithm.
        """
        self.config = config
        
    def build_generation_params(self, **kwargs) -> Dict:
        """Build generation parameters from config and kwargs."""
        generation_params = {
            "num_images_per_prompt": self.config.num_images,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "height": self.config.image_size[0],
            "width": self.config.image_size[1],
            "latents": self.config.init_latents,
        }
        
        # Add parameters from config.gen_kwargs
        if hasattr(self.config, "gen_kwargs") and self.config.gen_kwargs:
            for key, value in self.config.gen_kwargs.items():
                if key not in generation_params:
                    generation_params[key] = value
                    
        # Use kwargs to override default parameters
        for key, value in kwargs.items():
            generation_params[key] = value
            
        return generation_params
    
    def generate_clean_images(self, dataset: StableDiffusionPromptsDataset, **kwargs) -> List[Image.Image]:
        """Generate clean images for optimization."""
        generation_params = self.build_generation_params(**kwargs, guidance_scale=self.config.data_guidance_scale)
        
        clean_images = []
        for i, prompt in enumerate(dataset):
            formatted_img_filename = f"ori-lg{generation_params['guidance_scale']}-{i}.jpg"
            if os.path.exists(os.path.join(self.config.output_img_dir, formatted_img_filename)):
                clean_images.append(Image.open(os.path.join(self.config.output_img_dir, formatted_img_filename)))
            else:            
                no_watermarked_image = self.config.pipe(
                    prompt,
                    **generation_params,
                ).images[0]
                clean_images.append(no_watermarked_image)
                
                os.makedirs(self.config.output_img_dir, exist_ok=True)
                no_watermarked_image.save(os.path.join(self.config.output_img_dir, f"ori-lg{generation_params['guidance_scale']}-{i}.jpg"))
                
        return clean_images
    
    def build_watermarking_args(self) -> types.SimpleNamespace:
        """Build watermarking arguments from config."""
        watermarking_args = {
            "w_seed": self.config.w_seed,
            "w_channel": self.config.w_channel,
            "w_pattern": self.config.w_pattern,
            "w_mask_shape": self.config.w_mask_shape,
            "w_up_radius": self.config.w_up_radius,
            "w_low_radius": self.config.w_low_radius,
            "w_pattern_const": self.config.w_pattern_const,
            "w_injection": self.config.w_injection,
        }
        return types.SimpleNamespace(**watermarking_args)
    
    def build_hyperparameters(self) -> Dict:
        """Build hyperparameters for optimization from config."""
        return {
            "learning_rate": self.config.learning_rate,
            "scale_lr": self.config.scale_lr,
            "max_train_steps": self.config.max_train_steps,
            "save_steps": self.config.save_steps,
            "train_batch_size": self.config.train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "guidance_scale": self.config.train_guidance_scale,
            "optimized_guidance_scale": self.config.optimized_guidance_scale,
            "mixed_precision": self.config.mixed_precision,
            "seed": self.config.train_seed,
            "output_dir": self.config.ckpt_dir,
        }
    
    def optimize_watermark(self, dataset: StableDiffusionPromptsDataset, watermarking_args: types.SimpleNamespace) -> tuple:
        """Optimize watermark and watermarking signal."""
        init_latents_w = get_random_latents(pipe=self.config.pipe)
        watermarking_mask = get_watermarking_mask(init_latents_w, self.config, self.config.device).detach().cpu()
        
        # Build hyperparameters
        hyperparameters = self.build_hyperparameters()
        checkpoint_path = os.path.join(self.config.ckpt_dir, f"optimized_wm5-30_embedding-step-{hyperparameters['max_train_steps']}.pt")

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            optimized_watermark = checkpoint['opt_wm'].to(self.config.device)
            optimized_watermarking_signal = checkpoint['opt_acond'].to(self.config.device)

            return watermarking_mask, optimized_watermark, optimized_watermarking_signal
        else:
            print(f"No checkpoint found at {checkpoint_path}, training from scratch")
            # Generate clean images
            clean_images = self.generate_clean_images(dataset)
            # Create training dataset
            train_dataset = OptimizedDataset(
                data_root=self.config.output_img_dir,
                custom_dataset=dataset,
                size=512,
                repeats=10,
                interpolation="bicubic",
            )
            
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.train_batch_size, shuffle=True)
            
            opt_watermark = get_watermarking_pattern(pipe=self.config.pipe, args=watermarking_args, device=self.config.device)
            
            optimized_watermark, optimized_watermarking_signal = optimizer_wm_prompt(
                pipe=self.config.pipe, 
                dataloader=train_dataloader,
                hyperparameters=hyperparameters,
                mask=watermarking_mask,
                opt_wm=opt_watermark,
                save_path=self.config.ckpt_dir,
                args=watermarking_args,
            )
            
            return watermarking_mask, optimized_watermark, optimized_watermarking_signal
    
    def initialize_detector(self, watermarking_mask, optimized_watermark) -> ROBINDetector:
        """Initialize the ROBIN detector."""
        return ROBINDetector(
            watermarking_mask=watermarking_mask,
            gt_patch=optimized_watermark,
            threshold=self.config.threshold,
            device=self.config.device
        )
    
    def preprocess_image_for_detection(self, image: Image.Image, prompt: str, guidance_scale: float) -> tuple:
        """Preprocess image and get text embeddings for detection."""
        # Get Text Embeddings
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
        
        # Preprocess Image
        processed_image = transform_to_model_format(
            image, 
            target_size=self.config.image_size[0]
        ).unsqueeze(0).to(text_embeddings.dtype).to(self.config.device)
        
        return text_embeddings, processed_image
    
    def extract_latents_for_detection(self, 
                                    image: Image.Image, 
                                    prompt: str, 
                                    guidance_scale: float, 
                                    num_inference_steps: int,
                                    extract_latents_step: int,
                                    **kwargs) -> torch.Tensor:
        """Extract and reverse latents for watermark detection."""
        # Preprocess image and get text embeddings
        text_embeddings, processed_image = self.preprocess_image_for_detection(image, prompt, guidance_scale)
        
        # Get Image Latents
        image_latents = get_media_latents(
            pipe=self.config.pipe, 
            media=processed_image, 
            sample=False, 
            decoder_inv=kwargs.get('decoder_inv', False)
        )
        
        # Reverse Image Latents
        inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder_inv', 'guidance_scale', 'num_inference_steps']}
        
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **inversion_kwargs
        )[extract_latents_step]
        
        return reversed_latents

@inherit_docstring
class ROBIN(BaseWatermark):
    def __init__(self,
                 watermarking_config: ROBINConfig,
                 *args, **kwargs):
        """
            Initialize the ROBIN watermarking algorithm.
            
            Parameters:
                watermarking_config (ROBINConfig): Configuration for the ROBIN algorithm.
        """
        #super().__init__(algorithm_config, diffusion_config)
        self.config = watermarking_config
        self.utils = ROBINUtils(self.config)
        
        # === Get the optimized watermark & watermarking signal before generation ===
        self.dataset = StableDiffusionPromptsDataset()
        
        # Build watermarking arguments
        self.watermarking_args = self.utils.build_watermarking_args()
        
        # Optimize watermark and get components
        self.watermarking_mask, self.optimized_watermark, self.optimized_watermarking_signal = self.utils.optimize_watermark(
            self.dataset, 
            self.watermarking_args
        )
                
        # Initialize detector
        self.detector = self.utils.initialize_detector(self.watermarking_mask, self.optimized_watermark)
    
    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        """Internal method to generate a watermarked image."""
        self.set_orig_watermarked_latents(self.config.init_latents)
        
        # Build generation parameters using utils
        generation_params = self.utils.build_generation_params(**kwargs)
        # Override guidance_scale for watermarked generation
        generation_params["guidance_scale"] = self.config.guidance_scale
        # Ensure latents parameter is not overridden
        generation_params["latents"] = self.config.init_latents
        
        # Filter out parameters not supported by ROBINWatermarkedImageGeneration
        supported_params = {
            'height', 'width', 'num_inference_steps', 'guidance_scale', 'optimized_guidance_scale',
            'negative_prompt', 'num_images_per_prompt', 'eta', 'generator', 'latents', 
            'output_type', 'return_dict', 'callback', 'callback_steps'
        }
        filtered_params = {k: v for k, v in generation_params.items() if k in supported_params}
        
        # Ensure watermarking components are on the correct device
        watermarking_mask = self.watermarking_mask.to(self.config.device)
        optimized_watermark = self.optimized_watermark.to(self.config.device)
        optimized_watermarking_signal = self.optimized_watermarking_signal.to(self.config.device) if self.optimized_watermarking_signal is not None else None
        
        # Generate watermarked image
        set_random_seed(self.config.gen_seed)
        result = ROBINWatermarkedImageGeneration(
            pipe=self.config.pipe,
            prompt=prompt,
            watermarking_mask=watermarking_mask,
            gt_patch=optimized_watermark,
            opt_acond=optimized_watermarking_signal,
            watermarking_step=self.config.watermarking_step,
            args=self.watermarking_args,
            **filtered_params,
        )
        return result.images[0]
        
    
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
        
        # Pass only known parameters to forward_diffusion, and let kwargs handle any additional parameters
        inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['decoder_inv', 'guidance_scale', 'num_inference_steps']}

        # Extract and reverse latents for detection using utils
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale_to_use,
            num_inference_steps=num_steps_to_use,
            **inversion_kwargs
        )[num_steps_to_use - 1 - self.config.watermarking_step]
        
        # Evaluate watermark
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
         # Use config values as defaults if not explicitly provided
        guidance_scale_to_use = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        
        # Step 1: Generate watermarked latents (generation process)
        set_random_seed(self.config.gen_seed)
        # For ROBIN, the watermarked latents are the init_latents (watermark is applied during generation)
        watermarked_latents = self.config.init_latents
        
        # Step 2: Generate actual watermarked image using the same process as _generate_watermarked_image
        generation_params = self.utils.build_generation_params()
        generation_params["guidance_scale"] = self.config.guidance_scale
        generation_params["latents"] = self.config.init_latents
        
        # Generate the actual watermarked image with ROBIN watermarking
        watermarked_image = ROBINWatermarkedImageGeneration(
            pipe=self.config.pipe,
            prompt=prompt,
            watermarking_mask=self.watermarking_mask,
            gt_patch=self.optimized_watermark,
            opt_acond=self.optimized_watermarking_signal,
            watermarking_step=self.config.watermarking_step,
            args=self.watermarking_args,
            **generation_params,
        ).images[0]
        
        # Step 3: Perform watermark detection to get inverted latents (detection process)
        reversed_latents_list = None
        
        # Get Text Embeddings for detection
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
        inversion_kwargs = {k: v for k, v in kwargs.items() if k not in ['prompt', 'decoder_inv', 'guidance_scale', 'num_inference_steps']}
        
        reversed_latents_list = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale_to_use,
            num_inference_steps=self.config.num_inference_steps,
            **inversion_kwargs
        )
        
        # Step 4: Prepare visualization data  
        return DataForVisualization(
            config=self.config,
            utils=self.utils,
            reversed_latents=reversed_latents_list,
            orig_watermarked_latents=self.orig_watermarked_latents,
            image=image,
            # ROBIN-specific data
            watermarking_mask=self.watermarking_mask,
            optimized_watermark=self.optimized_watermark,
        )
