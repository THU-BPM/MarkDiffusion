from diffusers import StableDiffusionPipeline, TextToVideoSDPipeline, StableVideoDiffusionPipeline
from typing import Any, Dict, List, Optional, Union

# Pipeline type constants
PIPELINE_TYPE_IMAGE = "image"
PIPELINE_TYPE_TEXT_TO_VIDEO = "t2v"
PIPELINE_TYPE_IMAGE_TO_VIDEO = "i2v"

def get_pipeline_type(pipeline) -> Optional[str]:
    """
    Determine the type of diffusion pipeline.
    
    Args:
        pipeline: The diffusion pipeline object
        
    Returns:
        str: One of the pipeline type constants or None if not recognized
    """
    if isinstance(pipeline, StableDiffusionPipeline):
        return PIPELINE_TYPE_IMAGE
    elif isinstance(pipeline, TextToVideoSDPipeline):
        return PIPELINE_TYPE_TEXT_TO_VIDEO
    elif isinstance(pipeline, StableVideoDiffusionPipeline):
        return PIPELINE_TYPE_IMAGE_TO_VIDEO
    else:
        return None

def is_video_pipeline(pipeline) -> bool:
    """
    Check if the pipeline is a video generation pipeline.
    
    Args:
        pipeline: The diffusion pipeline object
        
    Returns:
        bool: True if the pipeline is a video generation pipeline, False otherwise
    """
    pipeline_type = get_pipeline_type(pipeline)
    return pipeline_type in [PIPELINE_TYPE_TEXT_TO_VIDEO, PIPELINE_TYPE_IMAGE_TO_VIDEO]

def is_image_pipeline(pipeline) -> bool:
    """
    Check if the pipeline is an image generation pipeline.
    
    Args:
        pipeline: The diffusion pipeline object
        
    Returns:
        bool: True if the pipeline is an image generation pipeline, False otherwise
    """
    return get_pipeline_type(pipeline) == PIPELINE_TYPE_IMAGE

def is_t2v_pipeline(pipeline) -> bool:
    """
    Check if the pipeline is a text-to-video pipeline.
    
    Args:
        pipeline: The diffusion pipeline object
        
    Returns:
        bool: True if the pipeline is a text-to-video pipeline, False otherwise
    """
    return get_pipeline_type(pipeline) == PIPELINE_TYPE_TEXT_TO_VIDEO

def is_i2v_pipeline(pipeline) -> bool:
    """
    Check if the pipeline is an image-to-video pipeline.
    
    Args:
        pipeline: The diffusion pipeline object
        
    Returns:
        bool: True if the pipeline is an image-to-video pipeline, False otherwise
    """
    return get_pipeline_type(pipeline) == PIPELINE_TYPE_IMAGE_TO_VIDEO

def get_pipeline_requirements(pipeline_type: str) -> Dict[str, Any]:
    """
    Get the requirements for a specific pipeline type (required parameters, etc.)
    
    Args:
        pipeline_type: The pipeline type string
        
    Returns:
        Dict: A dictionary containing the pipeline requirements
    """
    if pipeline_type == PIPELINE_TYPE_IMAGE:
        return {
            "required_params": [],
            "optional_params": ["height", "width", "num_images_per_prompt"]
        }
    elif pipeline_type == PIPELINE_TYPE_TEXT_TO_VIDEO:
        return {
            "required_params": ["num_frames"],
            "optional_params": ["height", "width", "fps"]
        }
    elif pipeline_type == PIPELINE_TYPE_IMAGE_TO_VIDEO:
        return {
            "required_params": ["input_image", "num_frames"], 
            "optional_params": ["height", "width", "fps"]
        }
    else:
        return {"required_params": [], "optional_params": []} 