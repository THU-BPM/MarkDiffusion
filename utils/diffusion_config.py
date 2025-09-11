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


from dataclasses import dataclass
from typing import Optional, Union, Any, Dict
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, TextToVideoSDPipeline, StableVideoDiffusionPipeline
from utils.pipeline_utils import (
    get_pipeline_type, 
    PIPELINE_TYPE_IMAGE, 
    PIPELINE_TYPE_TEXT_TO_VIDEO, 
    PIPELINE_TYPE_IMAGE_TO_VIDEO
)

@dataclass
class DiffusionConfig:
    """Configuration class for diffusion models and parameters."""
    
    def __init__(
        self,
        scheduler: DPMSolverMultistepScheduler,
        pipe: Union[StableDiffusionPipeline, TextToVideoSDPipeline, StableVideoDiffusionPipeline],
        device: str,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        num_inference_steps: int = 50,
        num_inversion_steps: Optional[int] = None,
        image_size: tuple = (512, 512),
        dtype: torch.dtype = torch.float16,
        gen_seed: int = 0,
        init_latents_seed: int = 0,
        inversion_type: str = "ddim",
        num_frames: int = -1, # -1 means image generation; >=1 means video generation
        **kwargs
    ):
        self.device = device
        self.scheduler = scheduler
        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.num_images = num_images
        self.num_inference_steps = num_inference_steps
        self.num_inversion_steps = num_inversion_steps or num_inference_steps
        self.image_size = image_size
        self.dtype = dtype
        self.gen_seed = gen_seed
        self.init_latents_seed = init_latents_seed
        self.inversion_type = inversion_type
        self.num_frames = num_frames
        # Store additional kwargs
        self.gen_kwargs = kwargs
        
        ## Assertions
        assert self.inversion_type in ["ddim", "exact"], f"Invalid inversion type: {self.inversion_type}"
        
        ## Validate pipeline type and parameter compatibility
        self._validate_pipeline_config()
    
    def _validate_pipeline_config(self) -> None:
        """Validate pipeline type and parameter compatibility."""
        pipeline_type = get_pipeline_type(self.pipe)
        
        if pipeline_type is None:
            raise ValueError(f"Unsupported pipeline type: {type(self.pipe)}")
        
        # Validate num_frames setting based on pipeline type
        if pipeline_type == PIPELINE_TYPE_IMAGE:
            if self.num_frames >= 1:
                # Auto-correct for image pipelines
                self.num_frames = -1
        elif pipeline_type in [PIPELINE_TYPE_TEXT_TO_VIDEO, PIPELINE_TYPE_IMAGE_TO_VIDEO]:
            if self.num_frames < 1:
                raise ValueError(f"For {pipeline_type} pipelines, num_frames must be >= 1, got {self.num_frames}")
    
    @property
    def pipeline_type(self) -> str:
        """Get the pipeline type."""
        return get_pipeline_type(self.pipe)
    
    @property
    def is_video_pipeline(self) -> bool:
        """Check if this is a video pipeline."""
        return self.pipeline_type in [PIPELINE_TYPE_TEXT_TO_VIDEO, PIPELINE_TYPE_IMAGE_TO_VIDEO]
    
    @property
    def is_image_pipeline(self) -> bool:
        """Check if this is an image pipeline."""
        return self.pipeline_type == PIPELINE_TYPE_IMAGE
    
    @property
    def pipeline_requirements(self) -> Dict[str, Any]:
        """Get the requirements for this pipeline type."""
        if self.pipeline_type == PIPELINE_TYPE_IMAGE:
            return {
                "required_params": [],
                "optional_params": ["height", "width", "num_images_per_prompt"]
            }
        elif self.pipeline_type == PIPELINE_TYPE_TEXT_TO_VIDEO:
            return {
                "required_params": ["num_frames"],
                "optional_params": ["height", "width", "fps"]
            }
        elif self.pipeline_type == PIPELINE_TYPE_IMAGE_TO_VIDEO:
            return {
                "required_params": ["input_image", "num_frames"],
                "optional_params": ["height", "width", "fps"]
            }
        else:
            return {"required_params": [], "optional_params": []}