import importlib
from watermark.base import BaseWatermark
from typing import Union, Optional
from utils.pipeline_utils import (
    get_pipeline_type, 
    PIPELINE_TYPE_IMAGE, 
    PIPELINE_TYPE_TEXT_TO_VIDEO, 
    PIPELINE_TYPE_IMAGE_TO_VIDEO
)
from watermark.auto_config import AutoConfig

WATERMARK_MAPPING_NAMES={
    'TR': 'watermark.tr.TR',
    'GS': 'watermark.gs.GS',
    'PRC': 'watermark.prc.PRC',
    'VideoShield': 'watermark.videoshield.VideoShieldWatermark',
    "VideoMark": 'watermark.videomark.VideoMarkWatermark',
    'RI': 'watermark.ri.RI',
    'SEAL': 'watermark.seal.SEAL',
    'ROBIN': 'watermark.robin.ROBIN',
    'WIND': 'watermark.wind.WIND',
    'GM': 'watermark.gm.GM'
}

# Dictionary mapping pipeline types to supported watermarking algorithms
PIPELINE_SUPPORTED_WATERMARKS = {
    PIPELINE_TYPE_IMAGE: ["TR", "GS", "PRC", "RI", "SEAL", "ROBIN", "WIND", "GM"],
    PIPELINE_TYPE_TEXT_TO_VIDEO: ["VideoShield", "VideoMark"],
    PIPELINE_TYPE_IMAGE_TO_VIDEO: ["VideoShield", "VideoMark"]
}

def watermark_name_from_alg_name(name: str) -> Optional[str]:
    """Get the watermark class name from the algorithm name."""
    for algorithm_name, watermark_name in WATERMARK_MAPPING_NAMES.items():
        if name.lower() == algorithm_name.lower():
            return watermark_name
    return None

class AutoWatermark:
    """
        This is a generic watermark class that will be instantiated as one of the watermark classes of the library when
        created with the [`AutoWatermark.load`] class method.

        This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, diffusion_config)` method."
        )

    @staticmethod
    def _check_pipeline_compatibility(pipeline_type: str, algorithm_name: str) -> bool:
        """Check if the pipeline type is compatible with the watermarking algorithm."""
        if pipeline_type is None:
            return False

        if algorithm_name not in WATERMARK_MAPPING_NAMES:
            return False

        return algorithm_name in PIPELINE_SUPPORTED_WATERMARKS.get(pipeline_type, [])

    @classmethod
    def load(cls, algorithm_name, algorithm_config=None, diffusion_config=None, *args, **kwargs) -> BaseWatermark:
        """Load the watermark algorithm instance based on the algorithm name."""
        # Check if the algorithm exists
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        if watermark_name is None:
            supported_algs = list(WATERMARK_MAPPING_NAMES.keys())
            raise ValueError(f"Invalid algorithm name: {algorithm_name}. Please use one of the supported algorithms: {', '.join(supported_algs)}")

        # Check pipeline compatibility
        if diffusion_config and diffusion_config.pipe:
            pipeline_type = get_pipeline_type(diffusion_config.pipe)
            if not cls._check_pipeline_compatibility(pipeline_type, algorithm_name):
                supported_algs = PIPELINE_SUPPORTED_WATERMARKS.get(pipeline_type, [])
                raise ValueError(
                    f"The algorithm '{algorithm_name}' is not compatible with the {pipeline_type} pipeline type. "
                    f"Supported algorithms for this pipeline type are: {', '.join(supported_algs)}"
                )

        # Load the watermark module
        module_name, class_name = watermark_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)
        watermark_config = AutoConfig.load(algorithm_name, diffusion_config, algorithm_config_path=algorithm_config, **kwargs)
        watermark_instance = watermark_class(watermark_config)
        return watermark_instance

    @classmethod
    def list_supported_algorithms(cls, pipeline_type: Optional[str] = None):
        """List all supported watermarking algorithms, optionally filtered by pipeline type."""
        if pipeline_type is None:
            return list(WATERMARK_MAPPING_NAMES.keys())
        else:
            if pipeline_type not in PIPELINE_SUPPORTED_WATERMARKS:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}. Supported types are: {', '.join(PIPELINE_SUPPORTED_WATERMARKS.keys())}")
            return PIPELINE_SUPPORTED_WATERMARKS[pipeline_type]

