from markdiffusion.watermark.base import BaseWatermark
from markdiffusion.evaluation.dataset import BaseDataset
from tqdm import tqdm
from enum import Enum, auto
from PIL import Image
from markdiffusion.evaluation.tools.image_editor import ImageEditor
from markdiffusion.evaluation.tools.video_editor import VideoEditor
from typing import List, Union

class DetectionPipelineReturnType(Enum):
    FULL = auto()
    SCORES = auto()
    IS_WATERMARKED = auto()
    
class WatermarkDetectionResult:
    
    def __init__(self,
                 generated_or_retrieved_media,
                 edited_media,
                 detect_result) -> None:
        self.generated_or_retrieved_media = generated_or_retrieved_media
        self.edited_media = edited_media
        self.detect_result = detect_result
        pass
        
    def __str__(self):
        return f"WatermarkDetectionResult(generated_or_retrieved_media={self.generated_or_retrieved_media}, edited_media={self.edited_media}, detect_result={self.detect_result})"
    
class WatermarkDetectionPipeline:
    def __init__(self, 
                 dataset: BaseDataset, 
                 media_editor_list: List[Union[ImageEditor, VideoEditor]] = [],
                 show_progress: bool = True,
                 detector_type: str = "l1_distance",
                 return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES):
        self.dataset = dataset
        self.media_editor_list = media_editor_list
        self.show_progress = show_progress
        self.return_type = return_type
        self.detector_type = detector_type
    
    def _edit_media(self, media: List[Image.Image]) -> List[Image.Image]:
        """
        Edit the media using the media editor list.

        Args:
            media (List[Image.Image]): The media to edit.

        Raises:
            ValueError: If the editor type is not supported.

        Returns:
            List[Image.Image]: The edited media.
        """
        results = media
        
        for editor in self.media_editor_list:
            if isinstance(editor, ImageEditor): 
                for i in range(len(results)):
                    results[i] = editor.edit(results[i]) # return single edited image
            elif isinstance(editor, VideoEditor): 
                results = editor.edit(results) # return a list of edited videos
            else:
                raise ValueError(f"Invalid media type: {type(media)}")
        
        return results
    
    def _detect_watermark(self, media:List[Image.Image], watermark: BaseWatermark, **kwargs):
        detect_result = watermark.detect_watermark_in_media(media, detector_type=self.detector_type, **kwargs)
        print(detect_result)
        return detect_result
            
    def _get_iterable(self):
        pass
            
    def _get_progress_bar(self, iterable):
        if self.show_progress:
            return tqdm(iterable, desc="Processing")
        return iterable

    def _generate_or_retrieve_media(self, index: int, watermark: BaseWatermark, **kwargs) -> List[Image.Image]:
        pass
    
    def evaluate(self, watermark: BaseWatermark, detection_kwargs={}, generation_kwargs={}):
        evaluation_results = []
        bar = self._get_progress_bar(self._get_iterable())
        
        for index in bar:
            generated_or_retrieved_media = self._generate_or_retrieve_media(index, watermark,**generation_kwargs)
            edited_media = self._edit_media(generated_or_retrieved_media)
            
            detect_result = self._detect_watermark(edited_media, watermark, **detection_kwargs)
            evaluation_results.append(WatermarkDetectionResult(generated_or_retrieved_media, edited_media, detect_result))
            
        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_results
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result[self.detector_type] for result in evaluation_results]
        elif self.return_type == DetectionPipelineReturnType.IS_WATERMARKED:
            return [result.detect_result['is_watermarked'] for result in evaluation_results]
    
class WatermarkedMediaDetectionPipeline(WatermarkDetectionPipeline):
    def __init__(self, dataset: BaseDataset, media_editor_list: List[Union[ImageEditor, VideoEditor]], 
                 show_progress: bool = True, 
                 detector_type: str = "l1_distance",
                 return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES,
                 *args, **kwargs):
        super().__init__(dataset, media_editor_list, show_progress, detector_type, return_type, *args, **kwargs)
        
    def _get_iterable(self):
        return range(self.dataset.num_samples)
    
    def _generate_or_retrieve_media(self, index: int, watermark: BaseWatermark, **generation_kwargs):
        prompt = self.dataset.get_prompt(index)
        generated_media =  watermark.generate_watermarked_media(input_data=prompt, **generation_kwargs)
        if isinstance(generated_media, Image.Image):
            return [generated_media]
        elif isinstance(generated_media, list):
            return generated_media
        else:
            raise ValueError(f"Invalid media type: {type(generated_media)}")
    
class UnWatermarkedMediaDetectionPipeline(WatermarkDetectionPipeline):
    def __init__(self, dataset: BaseDataset, media_editor_list: List[Union[ImageEditor, VideoEditor]], media_source_mode : str ="ground_truth",
                 show_progress: bool = True, 
                 detector_type: str = "l1_distance",
                 return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES,
                 *args, **kwargs):
        super().__init__(dataset, media_editor_list, show_progress, detector_type, return_type, *args, **kwargs)
        self.media_source_mode = media_source_mode
        
    def _get_iterable(self):
        if self.media_source_mode == "ground_truth":
            assert self.dataset.num_references != 0, "This dataset does not have ground truth images or videos"
            return range(self.dataset.num_references)
        elif self.media_source_mode == "generated":
            return range(self.dataset.num_samples)
        else:
            raise ValueError(f"Invalid media source mode: {self.media_source_mode}")
        
    def _generate_or_retrieve_media(self, index: int, watermark: BaseWatermark, **generation_kwargs):
        if self.media_source_mode == "ground_truth":
            return [self.dataset.get_reference(index)]
        elif self.media_source_mode == "generated":
            prompt = self.dataset.get_prompt(index)
            generated_media = watermark.generate_unwatermarked_media(input_data=prompt, **generation_kwargs)
            if isinstance(generated_media, Image.Image):
                return [generated_media]
            elif isinstance(generated_media, list):
                return generated_media
            else:
                raise ValueError(f"Invalid media type: {type(generated_media)}")
        else:
            raise ValueError(f"Invalid media source mode: {self.media_source_mode}")
        
    
    