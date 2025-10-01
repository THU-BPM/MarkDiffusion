import torch
from PIL import Image
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from visualize.data_for_visualization import DataForVisualization
import importlib
from visualize.base import BaseVisualizer

# Mapping of algorithm names to visualization data classes
VISUALIZATION_DATA_MAPPING = {
    'TR': 'visualize.tr.TreeRingVisualizer',
    'GS': 'visualize.gs.GaussianShadingVisualizer', 
    'PRC': 'visualize.prc.PRCVisualizer',
    'RI': 'visualize.ri.RingIDVisualizer',
    'WIND': 'visualize.wind.WINDVisualizer',
    'SEAL': 'visualize.seal.SEALVisualizer',
    'ROBIN': 'visualize.robin.ROBINVisualizer',
    'VideoShield': 'visualize.videoshield.VideoShieldVisualizer',
    'VideoMark': 'visualize.videomark.VideoMarkVisualizer',
    'GM': 'visualize.gm.GaussMarkerVisualizer',
}

class AutoVisualizer:
    """
    Factory class for creating visualization data instances.
    
    This is a generic visualization data factory that will instantiate the appropriate
    visualization data class based on the algorithm name.
    
    This class cannot be instantiated directly using __init__() (throws an error).
    """
    
    def __init__(self):
        raise EnvironmentError(
            "AutoVisualizer is designed to be instantiated "
            "using the `AutoVisualizer.load(algorithm_name, **kwargs)` method."
        )
    
    @staticmethod
    def _get_visualization_class_name(algorithm_name: str) -> Optional[str]:
        """Get the visualization data class name from the algorithm name."""
        for alg_name, class_path in VISUALIZATION_DATA_MAPPING.items():
            if algorithm_name.lower() == alg_name.lower():
                return class_path
        return None
    
    @classmethod
    def load(cls, algorithm_name: str, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1) -> BaseVisualizer:
        """
        Load the visualization data instance based on the algorithm name.
        
        Args:
            algorithm_name: Name of the watermarking algorithm (e.g., 'TR', 'GS', 'PRC')
            data_for_visualization: DataForVisualization instance
            
        Returns:
            BaseVisualizer: Instance of the appropriate visualization data class
            
        Raises:
            ValueError: If the algorithm name is not supported
        """
        # Check if the algorithm exists
        class_path = cls._get_visualization_class_name(algorithm_name)
        
        if algorithm_name != data_for_visualization.algorithm_name:
            raise ValueError(f"Algorithm name mismatch: {algorithm_name} != {data_for_visualization.algorithm_name}")
        
        if class_path is None:
            supported_algs = list(VISUALIZATION_DATA_MAPPING.keys())
            raise ValueError(
                f"Invalid algorithm name: {algorithm_name}. "
                f"Supported algorithms: {', '.join(supported_algs)}"
            )
        
        # Load the visualization data module and class
        module_name, class_name = class_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            visualization_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load visualization data class '{class_name}' "
                f"from module '{module_name}': {e}"
            )
        
        # Create and validate the instance
        instance = visualization_class(data_for_visualization=data_for_visualization, dpi=dpi, watermarking_step=watermarking_step)
        return instance