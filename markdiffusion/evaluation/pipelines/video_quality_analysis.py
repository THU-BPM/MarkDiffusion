from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Union, Dict, Tuple
from PIL import Image
from markdiffusion.evaluation.tools.image_editor import ImageEditor
from markdiffusion.evaluation.tools.video_editor import VideoEditor
from markdiffusion.evaluation.tools.video_quality_analyzer import VideoQualityAnalyzer
from markdiffusion.evaluation.dataset import BaseDataset
from markdiffusion.watermark.base import BaseWatermark
import os
import numpy as np
from tqdm import tqdm


class SilentProgressBar:
    """A silent progress bar wrapper that supports set_description but shows no output."""

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, desc):
        """No-op for silent mode."""
        pass

class QualityPipelineReturnType(Enum):
    """Return type of the image quality analysis pipeline."""
    FULL = auto()
    SCORES = auto()
    MEAN_SCORES = auto()

@dataclass
class DatasetForEvaluation:
    """Dataset for evaluation."""
    watermarked_videos: List[List[Image.Image]] = field(default_factory=list)
    unwatermarked_videos: List[List[Image.Image]] = field(default_factory=list)
    reference_videos: List[List[Image.Image]] = field(default_factory=list)
    indexes: List[int] = field(default_factory=list)

class QualityComparisonResult:
    """Result of quality comparison."""

    def __init__(self, 
                 store_path: str, 
                 watermarked_quality_scores: Dict[str, List[float]], 
                 unwatermarked_quality_scores: Dict[str, List[float]],
                 prompts: List[str],
                 ) -> None:
        """
        Initialize the image quality comparison result.

        Parameters:
            store_path: The path to store the results.
            watermarked_quality_scores: The quality scores of the watermarked image.
            unwatermarked_quality_scores: The quality scores of the unwatermarked image.
            prompts: The prompts used to generate the images.
        """
        self.store_path = store_path
        self.watermarked_quality_scores = watermarked_quality_scores
        self.unwatermarked_quality_scores = unwatermarked_quality_scores
        self.prompts = prompts
        

class VideoQualityAnalysisPipeline:
    """Pipeline for video quality analysis."""

    def __init__(self, 
                 dataset: BaseDataset, 
                 watermarked_video_editor_list: List[VideoEditor] = [],
                 unwatermarked_video_editor_list: List[VideoEditor] = [], 
                 watermarked_frame_editor_list: List[ImageEditor] = [],
                 unwatermarked_frame_editor_list: List[ImageEditor] = [],
                 analyzers: List[VideoQualityAnalyzer] = None, 
                 show_progress: bool = True, 
                 store_path: str = None,
                 return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        """Initialize the image quality analysis pipeline.

        Args:
            dataset (BaseDataset): The dataset for evaluation.
            watermarked_video_editor_list (List[VideoEditor], optional): The list of video editors for watermarked videos. Defaults to [].
            unwatermarked_video_editor_list (List[VideoEditor], optional): List of quality analyzers for videos. Defaults to [].
            watermarked_frame_editor_list (List[ImageEditor], optional): List of image editors for editing individual watermarked frames. Defaults to [].
            unwatermarked_frame_editor_list (List[ImageEditor], optional): List of image editors for editing individual unwatermarked frames. Defaults to [].
            analyzers (List[VideoQualityAnalyzer], optional): Whether to show progress. Defaults to None.
            show_progress (bool, optional): The path to store the results. Defaults to True.
            store_path (str, optional): The path to store the results. Defaults to None.
            return_type (QualityPipelineReturnType, optional): The return type of the pipeline. Defaults to QualityPipelineReturnType.MEAN_SCORES.
        """
        self.dataset = dataset
        self.watermarked_video_editor_list = watermarked_video_editor_list
        self.unwatermarked_video_editor_list = unwatermarked_video_editor_list
        self.watermarked_frame_editor_list = watermarked_frame_editor_list
        self.unwatermarked_frame_editor_list = unwatermarked_frame_editor_list
        self.analyzers = analyzers or []
        self.show_progress = show_progress
        self.store_path = store_path
        self.return_type = return_type

    def _check_compatibility(self):
        """Check if the pipeline is compatible with the dataset."""
        pass

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        pass

    def _get_progress_bar(self, iterable):
        """Return an iterable possibly wrapped with a progress bar."""
        if self.show_progress:
            return tqdm(iterable, desc="Processing", leave=True)
        return SilentProgressBar(iterable)
    
    def _get_prompt(self, index: int) -> str:
        """Get prompt from dataset."""
        return self.dataset.get_prompt(index)
    
    def _get_watermarked_video(self, watermark: BaseWatermark, index: int, **generation_kwargs) -> List[Image.Image]:
        """Generate watermarked image from dataset."""
        prompt = self._get_prompt(index)
        frames = watermark.generate_watermarked_media(input_data=prompt, **generation_kwargs)
        return frames
    
    def _get_unwatermarked_video(self, watermark: BaseWatermark, index: int, **generation_kwargs) -> List[Image.Image]:
        """Generate or retrieve unwatermarked image from dataset."""
        prompt = self._get_prompt(index)
        frames = watermark.generate_unwatermarked_media(input_data=prompt, **generation_kwargs)
        return frames
    
    def _edit_watermarked_video(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Edit watermarked image using image editors."""
        # Step 1: Edit all frames using video editors
        for video_editor in self.watermarked_video_editor_list:
            frames = video_editor.edit(frames)
        # Step 2: Edit individual frames using image editors
        for frame_editor in self.watermarked_frame_editor_list:
            frames = [frame_editor.edit(frame) for frame in frames]
        return frames
    
    def _edit_unwatermarked_video(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Edit unwatermarked image using image editors."""
        # Step 1: Edit all frames using video editors
        for video_editor in self.unwatermarked_video_editor_list:
            frames = video_editor.edit(frames)
        # Step 2: Edit individual frames using image editors
        for frame_editor in self.unwatermarked_frame_editor_list:
            frames = [frame_editor.edit(frame) for frame in frames]
        return frames
    
    def _prepare_dataset(self, watermark: BaseWatermark, **generation_kwargs) -> DatasetForEvaluation:
        """
        Prepare and generate all necessary data for quality analysis.
        
        This method should be overridden by subclasses to implement specific
        data preparation logic based on the analysis requirements.
        
        Parameters:
            watermark: The watermark algorithm instance.
            generation_kwargs: Additional generation parameters.
            
        Returns:
            DatasetForEvaluation object containing all prepared data.
        """
        dataset_eval = DatasetForEvaluation()
        
        # Generate all videos
        bar = self._get_progress_bar(self._get_iterable())
        bar.set_description("Generating videos for quality analysis")
        for index in bar:
            # Generate and edit watermarked image
            watermarked_frames = self._get_watermarked_video(watermark, index, **generation_kwargs)
            watermarked_frames = self._edit_watermarked_video(watermarked_frames)
            
            # Generate and edit unwatermarked image
            unwatermarked_frames = self._get_unwatermarked_video(watermark, index, **generation_kwargs)
            unwatermarked_frames = self._edit_unwatermarked_video(unwatermarked_frames)
            
            dataset_eval.watermarked_videos.append(watermarked_frames)
            dataset_eval.unwatermarked_videos.append(unwatermarked_frames)
            dataset_eval.indexes.append(index)
            
            if self.dataset.num_references > 0:
                reference_frames = self.dataset.get_reference(index)
                dataset_eval.reference_videos.append(reference_frames)
            
        return dataset_eval
    
    def _prepare_input_for_quality_analyzer(self, 
                                          watermarked_videos: List[List[Image.Image]], 
                                          unwatermarked_videos: List[List[Image.Image]], 
                                          reference_videos: List[List[Image.Image]]):
        """        Prepare input for quality analyzer.
        
        Args:
            watermarked_videos (List[List[Image.Image]]): Watermarked video(s)
            unwatermarked_videos (List[List[Image.Image]]): Unwatermarked video(s)
            reference_videos (List[List[Image.Image]]): Reference video if available
        """
        pass
    
    def _store_results(self, prepared_dataset: DatasetForEvaluation):
        """Store results."""
        os.makedirs(self.store_path, exist_ok=True)
        dataset_name = self.dataset.name

        for (index, watermarked_video, unwatermarked_video) in zip(prepared_dataset.indexes, prepared_dataset.watermarked_videos, prepared_dataset.unwatermarked_videos):
            # unwatermarked/watermarked_video is List[Image.Image], so first make a video from the frames
            save_dir = os.path.join(self.store_path, f"{self.__class__.__name__}_{dataset_name}_watermarked_prompt{index}")
            os.makedirs(save_dir, exist_ok=True)
            for i, frame in enumerate(watermarked_video):
                frame.save(os.path.join(save_dir, f"frame_{i}.png"))
            
            save_dir = os.path.join(self.store_path, f"{self.__class__.__name__}_{dataset_name}_unwatermarked_prompt{index}")
            os.makedirs(save_dir, exist_ok=True)
            for i, frame in enumerate(unwatermarked_video):
                frame.save(os.path.join(save_dir, f"frame_{i}.png"))
            
            if self.dataset.num_references > 0:
                reference_frames = self.dataset.get_reference(index)
                save_dir = os.path.join(self.store_path, f"{self.__class__.__name__}_{dataset_name}_reference_prompt{index}")
                os.makedirs(save_dir, exist_ok=True)
                for i, frame in enumerate(reference_frames):
                    frame.save(os.path.join(save_dir, f"frame_{i}.png"))

    def analyze_quality(self, prepared_data, analyzer):
        """Analyze quality of watermarked and unwatermarked images."""
        pass

    def evaluate(self, watermark: BaseWatermark, generation_kwargs={}):
        """Conduct evaluation utilizing the pipeline."""
        # Check compatibility
        self._check_compatibility()
        
        # Prepare dataset
        prepared_dataset = self._prepare_dataset(watermark, **generation_kwargs)
        
        # Store results
        if self.store_path:
            self._store_results(prepared_dataset)
        
        # Prepare input for quality analyzer
        prepared_data = self._prepare_input_for_quality_analyzer(
            prepared_dataset.watermarked_videos, 
            prepared_dataset.unwatermarked_videos, 
            prepared_dataset.reference_videos
        )
        
        # Analyze quality
        all_scores = {}
        for analyzer in self.analyzers:
            w_scores, u_scores = self.analyze_quality(prepared_data, analyzer)
            analyzer_name = analyzer.__class__.__name__
            all_scores[analyzer_name] = (w_scores, u_scores)
        
        # Get prompts and indexes
        prompts = []
        
        # For other pipelines
        for idx in prepared_dataset.indexes:
            prompts.append(self._get_prompt(idx))
        
        # Create result
        watermarked_scores = {}
        unwatermarked_scores = {}
        
        for analyzer_name, (w_scores, u_scores) in all_scores.items():
            watermarked_scores[analyzer_name] = w_scores
            unwatermarked_scores[analyzer_name] = u_scores
        
        result = QualityComparisonResult(
            store_path=self.store_path,
            watermarked_quality_scores=watermarked_scores,
            unwatermarked_quality_scores=unwatermarked_scores,
            prompts=prompts,
        )
        
        # Format results based on return_type
        return self._format_results(result)
    
    def _format_results(self, result: QualityComparisonResult):
        """Format results based on return_type."""
        if self.return_type == QualityPipelineReturnType.FULL:
            return result
        elif self.return_type == QualityPipelineReturnType.SCORES:
            return {
                'watermarked': result.watermarked_quality_scores,
                'unwatermarked': result.unwatermarked_quality_scores,
                'prompts': result.prompts
            }
        elif self.return_type == QualityPipelineReturnType.MEAN_SCORES:
            # Calculate mean scores for each analyzer
            mean_watermarked = {}
            mean_unwatermarked = {}
            
            for analyzer_name, scores in result.watermarked_quality_scores.items():
                if isinstance(scores, list) and len(scores) > 0:
                    mean_watermarked[analyzer_name] = np.mean(scores)
                else:
                    mean_watermarked[analyzer_name] = scores
            
            for analyzer_name, scores in result.unwatermarked_quality_scores.items():
                if isinstance(scores, list) and len(scores) > 0:
                    mean_unwatermarked[analyzer_name] = np.mean(scores)
                else:
                    mean_unwatermarked[analyzer_name] = scores
            
            return {
                'watermarked': mean_watermarked,
                'unwatermarked': mean_unwatermarked
            }

class DirectVideoQualityAnalysisPipeline(VideoQualityAnalysisPipeline):
    """Pipeline for direct video quality analysis."""

    def __init__(self, 
                 dataset: BaseDataset, 
                 watermarked_video_editor_list: List[VideoEditor] = [], 
                 unwatermarked_video_editor_list: List[VideoEditor] = [], 
                 watermarked_frame_editor_list: List[ImageEditor] = [],
                 unwatermarked_frame_editor_list: List[ImageEditor] = [],
                 analyzers: List[VideoQualityAnalyzer] = None, 
                 show_progress: bool = True, 
                 store_path: str = None, 
                 return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        """Initialize the video quality analysis pipeline.

        Args:
            dataset (BaseDataset): The dataset for evaluation.
            watermarked_video_editor_list (List[VideoEditor], optional): The list of video editors for watermarked videos. Defaults to [].
            unwatermarked_video_editor_list (List[VideoEditor], optional): List of quality analyzers for videos. Defaults to [].
            watermarked_frame_editor_list (List[ImageEditor], optional): List of image editors for editing individual watermarked frames. Defaults to [].
            unwatermarked_frame_editor_list (List[ImageEditor], optional): List of image editors for editing individual unwatermarked frames. Defaults to [].
            analyzers (List[VideoQualityAnalyzer], optional): Whether to show progress. Defaults to None.
            show_progress (bool, optional): Whether to show progress. Defaults to True.
            store_path (str, optional): The path to store the results. Defaults to None.
            return_type (QualityPipelineReturnType, optional): The return type of the pipeline. Defaults to QualityPipelineReturnType.MEAN_SCORES.
        """
        super().__init__(dataset, watermarked_video_editor_list, unwatermarked_video_editor_list, watermarked_frame_editor_list, unwatermarked_frame_editor_list, analyzers, show_progress, store_path, return_type)

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        return range(self.dataset.num_samples)
    
    def _get_prompt(self, index: int) -> str:
        """Get prompt from dataset."""
        return self.dataset.get_prompt(index)
    
    def _get_watermarked_video(self, watermark: BaseWatermark, index: int, **generation_kwargs) -> List[Image.Image]:
        """Generate watermarked video from dataset."""
        prompt = self._get_prompt(index)
        frames = watermark.generate_watermarked_media(input_data=prompt, **generation_kwargs)
        return frames
    
    def _get_unwatermarked_video(self, watermark: BaseWatermark, index: int, **generation_kwargs) -> List[Image.Image]:
        """Generate or retrieve unwatermarked video from dataset."""
        prompt = self._get_prompt(index)
        frames = watermark.generate_unwatermarked_media(input_data=prompt, **generation_kwargs)
        return frames
    
    def _prepare_input_for_quality_analyzer(self, 
                                          watermarked_videos: List[List[Image.Image]], 
                                          unwatermarked_videos: List[List[Image.Image]], 
                                          reference_videos: List[List[Image.Image]]):
        """Prepare input for quality analyzer."""
        # Group videos by prompt
        return watermarked_videos, unwatermarked_videos
    
    def analyze_quality(self, 
                        prepared_data: Tuple[List[List[Image.Image]], List[List[Image.Image]], List[List[Image.Image]]], 
                        analyzer: VideoQualityAnalyzer):
        """Analyze quality of watermarked and unwatermarked videos."""
        watermarked_videos, unwatermarked_videos = prepared_data
        
        # Create pairs of watermarked and unwatermarked videos
        video_pairs = list(zip(watermarked_videos, unwatermarked_videos))
        
        bar = self._get_progress_bar(video_pairs)
        bar.set_description(f"Analyzing quality for {analyzer.__class__.__name__}")
        w_scores, u_scores = [], []
        for watermarked_video, unwatermarked_video in bar:
            w_score = analyzer.analyze(watermarked_video)
            u_score = analyzer.analyze(unwatermarked_video)
            w_scores.append(w_score)
            u_scores.append(u_score)
        return w_scores, u_scores