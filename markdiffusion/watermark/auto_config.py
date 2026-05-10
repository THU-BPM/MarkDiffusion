import importlib
from typing import Dict, Optional, Any
from markdiffusion.utils.diffusion_config import DiffusionConfig
from markdiffusion.utils.utils import default_algorithm_config_path

CONFIG_MAPPING_NAMES = {
    'TR': 'markdiffusion.watermark.tr.TRConfig',
    'GS': 'markdiffusion.watermark.gs.GSConfig',
    'PRC': 'markdiffusion.watermark.prc.PRCConfig',
    'VideoShield': 'markdiffusion.watermark.videoshield.VideoShieldConfig',
    "VideoMark": 'markdiffusion.watermark.videomark.VideoMarkConfig',
    'RI': 'markdiffusion.watermark.ri.RIConfig',
    'SEAL': 'markdiffusion.watermark.seal.SEALConfig',
    'ROBIN': 'markdiffusion.watermark.robin.ROBINConfig',
    'WIND': 'markdiffusion.watermark.wind.WINDConfig',
    'SFW': 'markdiffusion.watermark.sfw.SFWConfig',
    'GM': 'markdiffusion.watermark.gm.GMConfig',
}

def config_name_from_alg_name(name: str) -> Optional[str]:
    """Get the config class name from the algorithm name."""
    if name in CONFIG_MAPPING_NAMES:
        return CONFIG_MAPPING_NAMES[name]
    else:
        raise ValueError(f"Invalid algorithm name: {name}")

class AutoConfig:
    """
    A generic configuration class that will be instantiated as one of the configuration classes
    of the library when created with the [`AutoConfig.load`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.load(algorithm_name, **kwargs)` method."
        )

    @classmethod
    def load(cls, algorithm_name: str, diffusion_config: DiffusionConfig, algorithm_config_path=None, **kwargs) -> Any:
        """
        Load the configuration class for the specified watermark algorithm.

        Args:
            algorithm_name (str): The name of the watermark algorithm
            diffusion_config (DiffusionConfig): Configuration for the diffusion model
            algorithm_config_path (str): Path to the algorithm configuration file
            **kwargs: Additional keyword arguments to pass to the configuration class

        Returns:
            The instantiated configuration class for the specified algorithm
        """
        config_name = config_name_from_alg_name(algorithm_name)
        if config_name is None:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")
            
        module_name, class_name = config_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        if algorithm_config_path is None:
            algorithm_config_path = default_algorithm_config_path(algorithm_name)
        config_instance = config_class(algorithm_config_path, diffusion_config, **kwargs)
        return config_instance