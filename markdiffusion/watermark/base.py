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


from abc import ABC, abstractmethod
import torch
from typing import Dict, List, Union, Optional, Any, Tuple
from utils.diffusion_config import DiffusionConfig
from utils.utils import load_config_file, set_random_seed
from utils.media_utils import *
from utils.pipeline_utils import (
    get_pipeline_type,
    is_image_pipeline, 
    is_video_pipeline,
    is_t2v_pipeline,
    is_i2v_pipeline,
    PIPELINE_TYPE_IMAGE, 
    PIPELINE_TYPE_TEXT_TO_VIDEO, 
    PIPELINE_TYPE_IMAGE_TO_VIDEO
)
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    TextToVideoSDPipeline, 
    StableVideoDiffusionPipeline,
    DDIMInverseScheduler
)

class BaseConfig(ABC):
    """Base configuration class for diffusion watermarking methods."""
    
    def __init__(self, algorithm_config: str, diffusion_config: DiffusionConfig, *args, **kwargs) -> None:
        """Initialize base configuration with common parameters."""
        
        # Load config file
        self.config_dict = load_config_file(f'config/{self.algorithm_name()}.json') if algorithm_config is None else load_config_file(algorithm_config)
        
        # Diffusion model parameters
        if diffusion_config is None:
            raise ValueError("diffusion_config cannot be None for BaseConfig initialization")
        
        if kwargs:
            self.config_dict.update(kwargs)
        
        self.pipe = diffusion_config.pipe
        self.scheduler = diffusion_config.scheduler
        self.device = diffusion_config.device
        self.guidance_scale = diffusion_config.guidance_scale
        self.num_images = diffusion_config.num_images
        self.num_inference_steps = diffusion_config.num_inference_steps
        self.num_inversion_steps = diffusion_config.num_inversion_steps
        self.image_size = diffusion_config.image_size
        self.dtype = diffusion_config.dtype
        self.gen_seed = diffusion_config.gen_seed
        self.init_latents_seed = diffusion_config.init_latents_seed
        self.inversion_type = diffusion_config.inversion_type
        self.num_frames = diffusion_config.num_frames
        
        # Set inversion module
        self.inversion = set_inversion(self.pipe, self.inversion_type)
        # Set generation kwargs
        self.gen_kwargs = diffusion_config.gen_kwargs
        
        # Get initial latents
        init_latents_rng = torch.Generator(device=self.device)
        init_latents_rng.manual_seed(self.init_latents_seed)
        if self.num_frames < 1:
            self.init_latents = get_random_latents(self.pipe, height=self.image_size[0], width=self.image_size[1], generator=init_latents_rng)
        else:
            self.init_latents = get_random_latents(self.pipe, num_frames=self.num_frames, height=self.image_size[0], width=self.image_size[1], generator=init_latents_rng)

        # Initialize algorithm-specific parameters
        self.initialize_parameters()

    @abstractmethod
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters. Should be overridden by subclasses."""
        raise NotImplementedError

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        raise NotImplementedError

class BaseWatermark(ABC):
    """Base class for diffusion watermarking methods."""
    
    def __init__(self, 
                 config: BaseConfig,
                 *args, **kwargs) -> None:
        """Initialize the watermarking algorithm."""
        self.config = config
        self.orig_watermarked_latents = None
        
        # Determine pipeline type
        self.pipeline_type = self._detect_pipeline_type()
        
        # Validate pipeline configuration
        self._validate_pipeline_config()
    
    def _detect_pipeline_type(self) -> str:
        """Detect the type of pipeline being used."""
        pipeline_type = get_pipeline_type(self.config.pipe)
        if pipeline_type is None:
            raise ValueError(f"Unsupported pipeline type: {type(self.config.pipe)}")
        return pipeline_type
    
    def _validate_pipeline_config(self) -> None:
        """Validate that the pipeline configuration is correct for the pipeline type."""
        # For image-to-video pipelines, ensure num_frames is set correctly
        if self.pipeline_type == PIPELINE_TYPE_IMAGE_TO_VIDEO or self.pipeline_type == PIPELINE_TYPE_TEXT_TO_VIDEO:
            if self.config.num_frames < 1:
                raise ValueError(f"For {self.pipeline_type} pipelines, num_frames must be >= 1, got {self.config.num_frames}")
        # For image pipelines, ensure num_frames is -1
        elif self.pipeline_type == PIPELINE_TYPE_IMAGE:
            if self.config.num_frames >= 1:
                raise ValueError(f"For {self.pipeline_type} pipelines, num_frames should be -1, got {self.config.num_frames}")
    
    def get_orig_watermarked_latents(self) -> torch.Tensor:
        """Get the original watermarked latents."""
        return self.orig_watermarked_latents
        
    def set_orig_watermarked_latents(self, value: torch.Tensor) -> None:
        """Set the original watermarked latents."""
        self.orig_watermarked_latents = value

    def generate_watermarked_media(self, 
                                 input_data: Union[str, Image.Image], 
                                 *args, 
                                 **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate watermarked media (image or video) based on pipeline type.
        
        This is the main interface for generating watermarked content with any
        watermarking algorithm. It automatically routes to the appropriate generation
        method based on the pipeline type (image or video).
        
        Args:
            input_data: Text prompt (for T2I or T2V) or input image (for I2V)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments, including:
                - guidance_scale: Guidance scale for generation
                - num_inference_steps: Number of inference steps
                - height, width: Dimensions of generated media
                - seed: Random seed for generation
        
        Returns:
            Union[Image.Image, List[Image.Image]]: Generated watermarked media
            - For image pipelines: Returns a single PIL Image
            - For video pipelines: Returns a list of PIL Images (frames)
        
        Examples:
            ```python
            # Image watermarking
            watermark = AutoWatermark.load('TR', diffusion_config=config)
            image = watermark.generate_watermarked_media(
                input_data="A beautiful landscape",
                guidance_scale=7.5,
                num_inference_steps=50
            )
            
            # Video watermarking (T2V)
            watermark = AutoWatermark.load('VideoShield', diffusion_config=config)
            frames = watermark.generate_watermarked_media(
                input_data="A dog running in a park",
                num_frames=16
            )
            
            # Video watermarking (I2V)
            watermark = AutoWatermark.load('VideoShield', diffusion_config=config)
            frames = watermark.generate_watermarked_media(
                input_data=reference_image,
                num_frames=16
            )
            ```
        """
        # Route to the appropriate generation method based on pipeline type
        if is_image_pipeline(self.config.pipe):
            if not isinstance(input_data, str):
                raise ValueError("For image generation, input_data must be a text prompt (string)")
            return self._generate_watermarked_image(input_data, *args, **kwargs)
        elif is_video_pipeline(self.config.pipe):
            return self._generate_watermarked_video(input_data, *args, **kwargs)
    
    def generate_unwatermarked_media(self, 
                                   input_data: Union[str, Image.Image], 
                                   *args, 
                                   **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate unwatermarked media (image or video) based on pipeline type.
        
        Args:
            input_data: Text prompt (for T2I or T2V) or input image (for I2V)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments, including:
                - save_path: Path to save the generated media
        
        Returns:
            Union[Image.Image, List[Image.Image]]: Generated unwatermarked media
        """
        # Route to the appropriate generation method based on pipeline type
        if is_image_pipeline(self.config.pipe):
            if not isinstance(input_data, str):
                raise ValueError("For image generation, input_data must be a text prompt (string)")
            return self._generate_unwatermarked_image(input_data, *args, **kwargs)
        elif is_video_pipeline(self.config.pipe):
            return self._generate_unwatermarked_video(input_data, *args, **kwargs)
    
    def detect_watermark_in_media(self, 
                                  media: Union[Image.Image, List[Image.Image], np.ndarray, torch.Tensor],
                                  *args,
                                  **kwargs) -> Dict[str, Any]:
        """
        Detect watermark in media (image or video).
        
        Args:
            media: The media to detect watermark in (can be PIL image, list of frames, numpy array, or tensor)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments, including:
                - prompt: Optional text prompt used to generate the media (for some algorithms)
                - num_inference_steps: Optional number of inference steps
                - guidance_scale: Optional guidance scale
                - num_frames: Optional number of frames
                - decoder_inv: Optional decoder inversion
                - inv_order: Inverse order for Exact Inversion
                - detector_type: Type of detector to use
        
        Returns:
            Dict[str, Any]: Detection results with metrics and possibly visualizations
        """
        # Process the input media into the right format based on pipeline type
        processed_media = self._preprocess_media_for_detection(media)
        
        # Route to the appropriate detection method
        if is_image_pipeline(self.config.pipe):
            return self._detect_watermark_in_image(
                processed_media, 
                *args,
                **kwargs
            )
        else:
            return self._detect_watermark_in_video(
                processed_media,
                *args,
                **kwargs
            )
    
    def _preprocess_media_for_detection(self, 
                                       media: Union[Image.Image, List[Image.Image], np.ndarray, torch.Tensor]
                                       ) -> Union[Image.Image, List[Image.Image], torch.Tensor]:
        """
        Preprocess media for detection based on its type and the pipeline type.
        
        Args:
            media: The media to preprocess
            
        Returns:
            Union[Image.Image, List[Image.Image], torch.Tensor]: Preprocessed media
        """
        if is_image_pipeline(self.config.pipe):
            if isinstance(media, Image.Image):
                return media
            elif isinstance(media, np.ndarray):
                return cv2_to_pil(media)
            elif isinstance(media, torch.Tensor):
                # Convert tensor to PIL image
                if media.dim() == 3:  # C, H, W
                    media = media.unsqueeze(0)  # Add batch dimension
                img_np = torch_to_numpy(media)[0]  # Take first image
                return cv2_to_pil(img_np)
            elif isinstance(media, list): # Compatible for detection pipeline
                return media[0]
            else:
                raise ValueError(f"Unsupported media type for image pipeline: {type(media)}")
        else:
            # Video pipeline
            if isinstance(media, list):
                # List of frames
                if all(isinstance(frame, Image.Image) for frame in media):
                    return media
                elif all(isinstance(frame, np.ndarray) for frame in media):
                    return [cv2_to_pil(frame) for frame in media]
                else:
                    raise ValueError("All frames must be either PIL images or numpy arrays")
            elif isinstance(media, np.ndarray):
                # Convert numpy video to list of PIL images
                if media.ndim == 4:  # F, H, W, C
                    return [cv2_to_pil(frame) for frame in media]
                else:
                    raise ValueError(f"Unsupported numpy array shape for video: {media.shape}")
            elif isinstance(media, torch.Tensor):
                # Convert tensor to list of PIL images
                if media.dim() == 5:  # B, C, F, H, W
                    video_np = torch_to_numpy(media)[0]  # Take first batch
                    return [cv2_to_pil(frame) for frame in video_np]
                elif media.dim() == 4 and media.shape[0] > 3:  # F, C, H, W (assuming F > 3)
                    frames = []
                    for i in range(media.shape[0]):
                        frame_np = torch_to_numpy(media[i].unsqueeze(0))[0]
                        frames.append(cv2_to_pil(frame_np))
                    return frames
                else:
                    raise ValueError(f"Unsupported tensor shape for video: {media.shape}")
            else:
                raise ValueError(f"Unsupported media type for video pipeline: {type(media)}")
    
    def _generate_watermarked_image(self, 
                   prompt: str,
                   *args,
                   **kwargs) -> Image.Image:
        """
            Generate watermarked image from text prompt.
            
            Parameters:
                prompt (str): The input prompt.
            
            Returns:
                Image.Image: The generated watermarked image.
                
            Raises:
                ValueError: If the pipeline doesn't support image generation.
        """
        if self.pipeline_type != PIPELINE_TYPE_IMAGE:
            raise ValueError(f"This pipeline ({self.pipeline_type}) does not support image generation. Use generate_watermarked_video instead.")
        
        # The implementation depends on the specific watermarking algorithm
        # This method should be implemented by subclasses
        raise NotImplementedError("This method is not implemented for this watermarking algorithm.")

    def _generate_watermarked_video(self, 
                                   input_data: Union[str, Image.Image],
                                   *args,
                                   **kwargs) -> Union[List[Image.Image], Image.Image]:
        """
            Generate watermarked video based on text prompt or input image.
            
            Parameters:
                input_data (Union[str, Image.Image]): Either a text prompt (for T2V) or an input image (for I2V).
                    - If the pipeline is T2V, input_data should be a string prompt.
                    - If the pipeline is I2V, input_data should be an Image object or can be passed as kwargs['input_image'].
                kwargs:
                    - 'input_image': The input image for I2V pipelines.
                    - 'prompt': The text prompt for T2V pipelines.
                    - 'image_path': The path to the input image for I2V pipelines.
            
            Returns:
                Union[List[Image.Image], Image.Image]: The generated watermarked video frames.
                
            Raises:
                ValueError: If the pipeline doesn't support video generation or if input type is incompatible.
        """
        if not is_video_pipeline(self.config.pipe):
            raise ValueError(f"This pipeline ({self.pipeline_type}) does not support video generation. Use generate_watermarked_image instead.")
        
        # The implementation depends on the specific watermarking algorithm
        # This method should be implemented by subclasses
        raise NotImplementedError("This method is not implemented for this watermarking algorithm.")

    def _generate_unwatermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        """
            Generate unwatermarked image from text prompt.
            
            Parameters:
                prompt (str): The input prompt.
            
            Returns:
                Image.Image: The generated unwatermarked image.
                
            Raises:
                ValueError: If the pipeline doesn't support image generation.
        """
        if not is_image_pipeline(self.config.pipe):
            raise ValueError(f"This pipeline ({self.pipeline_type}) does not support image generation. Use generate_unwatermarked_video instead.")
        
        # Construct generation parameters
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
        
        set_random_seed(self.config.gen_seed)
        return self.config.pipe(
            prompt,
            **generation_params
        ).images[0]
        
    def _generate_unwatermarked_video(self, input_data: Union[str, Image.Image], *args, **kwargs) -> List[Image.Image]:
        """
            Generate unwatermarked video based on text prompt or input image.
            
            Parameters:
                input_data (Union[str, Image.Image]): Either a text prompt (for T2V) or an input image (for I2V).
                    - If the pipeline is T2V, input_data should be a string prompt.
                    - If the pipeline is I2V, input_data should be an Image object or can be passed as kwargs['input_image'].
                kwargs:
                    - 'input_image': The input image for I2V pipelines.
                    - 'prompt': The text prompt for T2V pipelines.
                    - 'image_path': The path to the input image for I2V pipelines.
            
            Returns:
                List[Image.Image]: The generated unwatermarked video frames.
                
            Raises:
                ValueError: If the pipeline doesn't support video generation or if input type is incompatible.
        """
        if not is_video_pipeline(self.config.pipe):
            raise ValueError(f"This pipeline ({self.pipeline_type}) does not support video generation. Use generate_unwatermarked_image instead.")
        
        # Handle Text-to-Video pipeline
        if is_t2v_pipeline(self.config.pipe):
            # For T2V, input should be a text prompt
            if not isinstance(input_data, str):
                raise ValueError("Text-to-Video pipeline requires a text prompt as input_data")
            
            # Construct generation parameters
            generation_params = {
                "latents": self.config.init_latents,
                "num_frames": self.config.num_frames,
                "height": self.config.image_size[0],
                "width": self.config.image_size[1],
                "num_inference_steps": self.config.num_inference_steps,
                "guidance_scale": self.config.guidance_scale,
            }
            
            # Add parameters from config.gen_kwargs
            if hasattr(self.config, "gen_kwargs") and self.config.gen_kwargs:
                for key, value in self.config.gen_kwargs.items():
                    if key not in generation_params:
                        generation_params[key] = value
                        
            # Use kwargs to override default parameters
            for key, value in kwargs.items():
                generation_params[key] = value

            # Generate the video
            set_random_seed(self.config.gen_seed)
            output = self.config.pipe(
                input_data,  # Use prompt
                **generation_params
            )
            
            # 根据测试结果，我们知道 TextToVideoSDPipeline 的输出有 frames 属性
            if hasattr(output, 'frames'):
                frames = output.frames[0]
            elif hasattr(output, 'videos'):
                frames = output.videos[0]
            else:
                frames = output[0] if isinstance(output, tuple) else output
            
            # Convert frames to PIL images
            frame_list = [cv2_to_pil(frame) for frame in frames]
            return frame_list
            
        # Handle Image-to-Video pipeline
        elif is_i2v_pipeline(self.config.pipe):
            # For I2V, input should be an image, text prompt is optional
            input_image = None
            text_prompt = None

            # Check if input_data is an image passed via kwargs
            if "input_image" in kwargs and isinstance(kwargs["input_image"], Image.Image):
                input_image = kwargs["input_image"]

            # Check if input_data is an image
            elif isinstance(input_data, Image.Image):                
                input_image = input_data

            # If input_data is a string but we need an image, check if an image path was provided
            elif isinstance(input_data, str):
                import os
                from PIL import Image as PILImage

                if os.path.exists(input_data):
                    try:
                        input_image = PILImage.open(input_data).convert("RGB")
                    except Exception as e:
                        raise ValueError(f"Input data is neither an Image object nor a valid image path. Failed to load image from path: {e}")
                else:
                    # Treat as text prompt if no valid image path
                    text_prompt = input_data
            if input_image is None:
                raise ValueError("Input image is required for Image-to-Video pipeline")
                
            # Construct generation parameters
            generation_params = {
                "image": input_image,
                "height": self.config.image_size[0],
                "width": self.config.image_size[1],
                "num_frames": self.config.num_frames,
                "latents": self.config.init_latents,
                "num_inference_steps": self.config.num_inference_steps,
                "max_guidance_scale": self.config.guidance_scale,
                "output_type": "np",
            }
            # In I2VGen-XL, the text prompt is needed
            if text_prompt is not None:
                generation_params["prompt"] = text_prompt
            
            # Add parameters from config.gen_kwargs
            if hasattr(self.config, "gen_kwargs") and self.config.gen_kwargs:
                for key, value in self.config.gen_kwargs.items():
                    if key not in generation_params:
                        generation_params[key] = value
                        
            # Use kwargs to override default parameters
            for key, value in kwargs.items():
                generation_params[key] = value
                
            # Generate the video
            set_random_seed(self.config.gen_seed)
            video = self.config.pipe(
                **generation_params
            ).frames[0]
            
            # Convert frames to PIL images
            frame_list = [cv2_to_pil(frame) for frame in video]
            return frame_list
        
        # This should never happen since we already checked pipeline type
        raise NotImplementedError(f"Unsupported video pipeline type: {self.pipeline_type}")
    
    def _detect_watermark_in_video(self, 
                                 video_frames: List[Image.Image],
                                 *args,
                                 **kwargs) -> Dict[str, Any]:
        """
        Detect watermark in video frames.
        
        Args:
            video_frames: List of video frames as PIL images
            kwargs:
                - 'prompt': Optional text prompt used for generation (for T2V pipelines)
                - 'reference_image': Optional reference image (for I2V pipelines)
                - 'guidance_scale': The guidance scale for the detector (optional)
                - 'detector_type': The type of detector to use (optional)
                - 'num_inference_steps': Number of inference steps for inversion (optional)
                - 'num_frames': Number of frames to use for detection (optional for I2V pipelines)
                - 'decoder_inv': Whether to use decoder inversion (optional)
                - 'inv_order': Inverse order for Exact Inversion (optional)
            
        Returns:
            Dict[str, Any]: Detection results
            
        Raises:
            NotImplementedError: If the watermarking algorithm doesn't support video watermark detection
        """
        raise NotImplementedError("Video watermark detection is not implemented for this algorithm")

    def _detect_watermark_in_image(self, 
                         image: Image.Image, 
                         prompt: str = "", 
                         *args, 
                         **kwargs) -> Dict[str, float]:
        """
            Detect watermark in image.
            
            Args:
                image (Image.Image): The input image.
                prompt (str): The prompt used for generation.
                kwargs:
                    - 'guidance_scale': The guidance scale for the detector.
                    - 'detector_type': The type of detector to use.
                    - 'num_inference_steps': Number of inference steps for inversion.
                    - 'decoder_inv': Whether to use decoder inversion.
                    - 'inv_order': Inverse order for Exact Inversion.
                    
            Returns:
                Dict[str, float]: The detection result.
        """
        raise NotImplementedError("Watermark detection in image is not implemented for this algorithm")

    @abstractmethod
    def get_data_for_visualize(self, media, *args, **kwargs):
        """Get data for visualization."""
        pass
