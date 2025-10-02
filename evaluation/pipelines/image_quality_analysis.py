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


from watermark.base import BaseWatermark
from evaluation.dataset import BaseDataset
from tqdm import tqdm
from enum import Enum, auto
from PIL import Image
from evaluation.tools.image_editor import ImageEditor
from typing import List, Dict, Union, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass, field
import os
import random
from evaluation.tools.image_quality_analyzer import (
    ImageQualityAnalyzer
)
import lpips

class QualityPipelineReturnType(Enum):
    """Return type of the image quality analysis pipeline."""
    FULL = auto()
    SCORES = auto()
    MEAN_SCORES = auto()

@dataclass
class DatasetForEvaluation:
    """Dataset for evaluation."""
    watermarked_images: List[Union[Image.Image, List[Image.Image]]] = field(default_factory=list)
    unwatermarked_images: List[Union[Image.Image, List[Image.Image]]] = field(default_factory=list)
    reference_images: List[Image.Image] = field(default_factory=list)
    indexes: List[int] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)

class QualityComparisonResult:
    """Result of image quality comparison."""

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


class ImageQualityAnalysisPipeline:
    """Pipeline for image quality analysis."""

    def __init__(self, 
                 dataset: BaseDataset, 
                 watermarked_image_editor_list: List[ImageEditor] = [],
                 unwatermarked_image_editor_list: List[ImageEditor] = [], 
                 analyzers: List[ImageQualityAnalyzer] = None, 
                 unwatermarked_image_source: str = 'generated', 
                 reference_image_source: str = 'natural',
                 show_progress: bool = True, 
                 store_path: str = None,
                 return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        """
        Initialize the image quality analysis pipeline.

        Parameters:
            dataset: The dataset for evaluation.
            watermarked_image_editor_list: The list of image editors for watermarked images.
            unwatermarked_image_editor_list: The list of image editors for unwatermarked images.
            analyzers: List of quality analyzers for images.
            unwatermarked_image_source: The source of unwatermarked images ('natural' or 'generated').
            reference_image_source: The source of reference images ('natural' or 'generated').
            show_progress: Whether to show progress.
            store_path: The path to store the results. If None, the generated images will not be stored.
            return_type: The return type of the pipeline.
        """
        if unwatermarked_image_source not in ['natural', 'generated']:
            raise ValueError(f"Invalid unwatermarked_image_source: {unwatermarked_image_source}")
        
        self.dataset = dataset
        self.watermarked_image_editor_list = watermarked_image_editor_list
        self.unwatermarked_image_editor_list = unwatermarked_image_editor_list
        self.analyzers = analyzers or []
        self.unwatermarked_image_source = unwatermarked_image_source
        self.reference_image_source = reference_image_source
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
        return iterable
    
    def _get_prompt(self, index: int) -> str:
        """Get prompt from dataset."""
        return self.dataset.get_prompt(index)
    
    def _get_watermarked_image(self, watermark: BaseWatermark, index: int, **generation_kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate watermarked image from dataset."""
        prompt = self._get_prompt(index)
        image = watermark.generate_watermarked_media(input_data=prompt, **generation_kwargs)
        return image
    
    def _get_unwatermarked_image(self, watermark: BaseWatermark, index: int, **generation_kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate or retrieve unwatermarked image from dataset."""
        if self.unwatermarked_image_source == 'natural':
            return self.dataset.get_reference(index)
        elif self.unwatermarked_image_source == 'generated':
            prompt = self._get_prompt(index)
            image = watermark.generate_unwatermarked_media(input_data=prompt, **generation_kwargs)
            return image
    
    def _edit_watermarked_image(self, image: Union[Image.Image, List[Image.Image]]) -> Union[Image.Image, List[Image.Image]]:
        """Edit watermarked image using image editors."""
        if isinstance(image, list):
            edited_images = []
            for img in image:
                for image_editor in self.watermarked_image_editor_list:
                    img = image_editor.edit(img)
                edited_images.append(img)
            return edited_images
        else:
            for image_editor in self.watermarked_image_editor_list:
                image = image_editor.edit(image)
            return image
    
    def _edit_unwatermarked_image(self, image: Union[Image.Image, List[Image.Image]]) -> Union[Image.Image, List[Image.Image]]:
        """Edit unwatermarked image using image editors."""
        if isinstance(image, list):
            edited_images = []
            for img in image:
                for image_editor in self.unwatermarked_image_editor_list:
                    img = image_editor.edit(img)
                edited_images.append(img)
            return edited_images
        else:
            for image_editor in self.unwatermarked_image_editor_list:
                image = image_editor.edit(image)
            return image
    
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
        
        # Generate all images
        bar = self._get_progress_bar(self._get_iterable())
        bar.set_description("Generating images for quality analysis")
        for index in bar:
            # Generate and edit watermarked image
            watermarked_image = self._get_watermarked_image(watermark, index, **generation_kwargs)
            watermarked_image = self._edit_watermarked_image(watermarked_image)
            
            # Generate and edit unwatermarked image
            unwatermarked_image = self._get_unwatermarked_image(watermark, index, **generation_kwargs)
            unwatermarked_image = self._edit_unwatermarked_image(unwatermarked_image)
            
            dataset_eval.watermarked_images.append(watermarked_image)
            dataset_eval.unwatermarked_images.append(unwatermarked_image)
            dataset_eval.indexes.append(index)
            dataset_eval.prompts.append(self._get_prompt(index))
            
            if self.reference_image_source == 'natural':
                if self.dataset.num_references > 0:
                    reference_image = self.dataset.get_reference(index)
                    dataset_eval.reference_images.append(reference_image)
                else:
                    # For text-based analyzers, add None placeholder
                    dataset_eval.reference_images.append(None)
            else:
                dataset_eval.reference_images.append(unwatermarked_image)
            
        return dataset_eval
    
    def _prepare_input_for_quality_analyzer(self, 
                                          prepared_dataset: DatasetForEvaluation):
        """
        Prepare input for quality analyzer.
        
        Parameters:
            prepared_dataset: The prepared dataset.
        """
        pass
    
    def _store_results(self, prepared_dataset: DatasetForEvaluation):
        """Store results."""
        os.makedirs(self.store_path, exist_ok=True)
        dataset_name = self.dataset.name

        for (index, watermarked_image, unwatermarked_image, prompt) in enumerate(zip(prepared_dataset.indexes, prepared_dataset.watermarked_images, prepared_dataset.unwatermarked_images, prepared_dataset.prompts)):
            watermarked_image.save(os.path.join(self.store_path, f"{self.__class__.__name__}_{dataset_name}_watermarked_prompt_{prompt}_{index}.png"))
            unwatermarked_image.save(os.path.join(self.store_path, f"{self.__class__.__name__}_{dataset_name}_unwatermarked_prompt_{prompt}_{index}.png"))

    def analyze_quality(self, prepared_data, analyzer):
        """Analyze quality of watermarked and unwatermarked images."""
        pass

    def evaluate(self, watermark: BaseWatermark, generation_kwargs={}):
        """Conduct evaluation utilizing the pipeline."""
        # Check compatibility
        self._check_compatibility()
        print(self.store_path)
        
        # Prepare dataset
        prepared_dataset = self._prepare_dataset(watermark, **generation_kwargs)
        
        # Store results
        if self.store_path:
            self._store_results(prepared_dataset)
        
        # Prepare input for quality analyzer
        prepared_data = self._prepare_input_for_quality_analyzer(
            prepared_dataset
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
                'prompts': result.prompt
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

class DirectImageQualityAnalysisPipeline(ImageQualityAnalysisPipeline):
    """
    Pipeline for direct image quality analysis.
    
    This class analyzes the quality of images by directly comparing the characteristics 
    of watermarked images with unwatermarked images. It evaluates metrics such as PSNR, 
    SSIM, LPIPS, FID without the need for any external reference image.
    
    Use this pipeline to assess the impact of watermarking on image quality directly.
    """

    def __init__(self, 
                 dataset: BaseDataset, 
                 watermarked_image_editor_list: List[ImageEditor] = [], 
                 unwatermarked_image_editor_list: List[ImageEditor] = [],
                 analyzers: List[ImageQualityAnalyzer] = None, 
                 unwatermarked_image_source: str = 'generated', 
                 reference_image_source: str = 'natural',
                 show_progress: bool = True, 
                 store_path: str = None,
                 return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        
        super().__init__(dataset, watermarked_image_editor_list, unwatermarked_image_editor_list, 
                        analyzers, unwatermarked_image_source, reference_image_source, show_progress, store_path, return_type)

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        return range(self.dataset.num_samples)
    
    def _prepare_input_for_quality_analyzer(self, 
                                          prepared_dataset: DatasetForEvaluation):
        """Prepare input for quality analyzer."""
        return [(watermarked_image, unwatermarked_image) for watermarked_image, unwatermarked_image in zip(prepared_dataset.watermarked_images, prepared_dataset.unwatermarked_images)]
    
    def analyze_quality(self, 
                        prepared_data: List[Tuple[Image.Image, Image.Image]], 
                        analyzer: ImageQualityAnalyzer):
        """Analyze quality of watermarked and unwatermarked images."""
        bar = self._get_progress_bar(prepared_data)
        bar.set_description(f"Analyzing quality for {analyzer.__class__.__name__}")
        w_scores, u_scores = [], []
        # For direct analyzers, we analyze each image independently
        for watermarked_image, unwatermarked_image in bar:
             # watermarked score
            try:
                w_score = analyzer.analyze(watermarked_image)
            except TypeError:
                # analyzer expects a reference -> use unwatermarked_image as reference
                w_score = analyzer.analyze(watermarked_image, unwatermarked_image)
            # unwatermarked score
            try:
                u_score = analyzer.analyze(unwatermarked_image)
            except TypeError:
                u_score = analyzer.analyze(unwatermarked_image, watermarked_image)
            w_scores.append(w_score)
            u_scores.append(u_score)
        
        return w_scores, u_scores


class ReferencedImageQualityAnalysisPipeline(ImageQualityAnalysisPipeline):
    """
    Pipeline for referenced image quality analysis.

    This pipeline assesses image quality by comparing both watermarked and unwatermarked 
    images against a common reference image. It measures the degree of similarity or 
    deviation from the reference.
    
    Ideal for scenarios where the impact of watermarking on image quality needs to be 
    assessed, particularly in relation to specific reference images or ground truth.
    """

    def __init__(self, 
                 dataset: BaseDataset, 
                 watermarked_image_editor_list: List[ImageEditor] = [], 
                 unwatermarked_image_editor_list: List[ImageEditor] = [],
                 analyzers: List[ImageQualityAnalyzer] = None, 
                 unwatermarked_image_source: str = 'generated', 
                 reference_image_source: str = 'natural',
                 show_progress: bool = True, 
                 store_path: str = None,
                 return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
                    
        super().__init__(dataset, watermarked_image_editor_list, unwatermarked_image_editor_list, 
                        analyzers, unwatermarked_image_source, reference_image_source, show_progress, store_path, return_type)

    def _check_compatibility(self):
        """Check if the pipeline is compatible with the dataset."""
        # Check if we have analyzers that use text as reference
        has_text_analyzer = any(hasattr(analyzer, 'reference_source') and analyzer.reference_source == 'text' 
                               for analyzer in self.analyzers)
        
        # If all analyzers use text reference, we don't need reference images
        if not has_text_analyzer and self.dataset.num_references == 0:
            raise ValueError(f"Reference images are required for referenced image quality analysis. Dataset {self.dataset.name} has no reference images.")

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        return range(self.dataset.num_samples)
    
    def _prepare_input_for_quality_analyzer(self, 
                                          prepared_dataset: DatasetForEvaluation):
        """Prepare input for quality analyzer."""
        return [(watermarked_image, unwatermarked_image, reference_image, prompt) 
                for watermarked_image, unwatermarked_image, reference_image, prompt in 
                zip(prepared_dataset.watermarked_images, prepared_dataset.unwatermarked_images, prepared_dataset.reference_images, prepared_dataset.prompts)
                ]
    
    def analyze_quality(self, 
                        prepared_data: List[Tuple[Image.Image, Image.Image, Image.Image, str]], 
                        analyzer: ImageQualityAnalyzer):
        """Analyze quality of watermarked and unwatermarked images."""
        bar = self._get_progress_bar(prepared_data)
        bar.set_description(f"Analyzing quality for {analyzer.__class__.__name__}")
        w_scores, u_scores = [], []
        # For referenced analyzers, we compare against the reference
        for watermarked_image, unwatermarked_image, reference_image, prompt in bar:
            if analyzer.reference_source == "image":
                w_score = analyzer.analyze(watermarked_image, reference_image)
                u_score = analyzer.analyze(unwatermarked_image, reference_image)
            elif analyzer.reference_source == "text":
                w_score = analyzer.analyze(watermarked_image, prompt)
                u_score = analyzer.analyze(unwatermarked_image, prompt)
            else:
                raise ValueError(f"Invalid reference source: {analyzer.reference_source}")
            w_scores.append(w_score)
            u_scores.append(u_score)
        return w_scores, u_scores
        

class GroupImageQualityAnalysisPipeline(ImageQualityAnalysisPipeline):
    """
    Pipeline for group-based image quality analysis.
    
    This pipeline analyzes quality metrics that require comparing distributions
    of multiple images (e.g., FID). It generates all images upfront and then
    performs a single analysis on the entire collection.
    """

    def __init__(self, 
                 dataset: BaseDataset, 
                 watermarked_image_editor_list: List[ImageEditor] = [], 
                 unwatermarked_image_editor_list: List[ImageEditor] = [],
                 analyzers: List[ImageQualityAnalyzer] = None, 
                 unwatermarked_image_source: str = 'generated', 
                 reference_image_source: str = 'natural',
                 show_progress: bool = True, 
                 store_path: str = None,
                 return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        
        super().__init__(dataset, watermarked_image_editor_list, unwatermarked_image_editor_list, 
                        analyzers, unwatermarked_image_source, reference_image_source, show_progress, store_path, return_type)

    def _get_iterable(self):
        """Return an iterable for analyzers instead of dataset indices."""
        return range(self.dataset.num_samples)
    
    def _prepare_input_for_quality_analyzer(self, 
                                          prepared_dataset: DatasetForEvaluation):
        """Prepare input for group analyzer."""
        return [(prepared_dataset.watermarked_images, prepared_dataset.unwatermarked_images, prepared_dataset.reference_images)]
    
    def analyze_quality(self, 
                        prepared_data: List[Tuple[List[Image.Image], List[Image.Image], List[Image.Image]]], 
                        analyzer: ImageQualityAnalyzer):
        """Analyze quality of image groups."""
        bar = self._get_progress_bar(prepared_data)
        bar.set_description(f"Analyzing quality for {analyzer.__class__.__name__}")
        w_scores, u_scores = [], []
        # For group analyzers, we pass the entire collection
        for watermarked_images, unwatermarked_images, reference_images in bar:
            w_score = analyzer.analyze(watermarked_images, reference_images)
            u_score = analyzer.analyze(unwatermarked_images, reference_images)
            w_scores.append(w_score)
            u_scores.append(u_score)
        return w_scores, u_scores


class RepeatImageQualityAnalysisPipeline(ImageQualityAnalysisPipeline):
    """
    Pipeline for repeat-based image quality analysis.
    
    This pipeline analyzes diversity metrics by generating multiple images
    for each prompt (e.g., LPIPS diversity). It generates multiple versions
    per prompt and analyzes the diversity within each group.
    """

    def __init__(self, 
                 dataset: BaseDataset,
                 prompt_per_image: int = 20,
                 watermarked_image_editor_list: List[ImageEditor] = [], 
                 unwatermarked_image_editor_list: List[ImageEditor] = [],
                 analyzers: List[ImageQualityAnalyzer] = None, 
                 unwatermarked_image_source: str = 'generated', 
                 reference_image_source: str = 'natural',
                 show_progress: bool = True, 
                 store_path: str = None,
                 return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        
        super().__init__(dataset, watermarked_image_editor_list, unwatermarked_image_editor_list, 
                        analyzers, unwatermarked_image_source, reference_image_source, show_progress, store_path, return_type)
        
        self.prompt_per_image = prompt_per_image

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        return range(self.dataset.num_samples * self.prompt_per_image)
    
    def _get_prompt(self, index: int) -> str:
        """Get prompt from dataset."""
        prompt_index = index // self.prompt_per_image
        return self.dataset.get_prompt(prompt_index)
    
    def _get_watermarked_image(self, watermark: BaseWatermark, index: int, **generation_kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Get watermarked image."""
        prompt = self._get_prompt(index)
        # Randomly select a generation seed
        generation_kwargs['gen_seed'] = random.randint(0, 1000000)
        return watermark.generate_watermarked_media(input_data=prompt, **generation_kwargs)
    
    def _get_unwatermarked_image(self, watermark: BaseWatermark, index: int, **generation_kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Get unwatermarked image."""
        prompt = self._get_prompt(index)
        # Randomly select a generation seed
        generation_kwargs['gen_seed'] = random.randint(0, 1000000)
        if self.unwatermarked_image_source == 'natural':
            return self.dataset.get_reference(index)
        else:
            return watermark.generate_unwatermarked_media(input_data=prompt, **generation_kwargs)
    
    def _prepare_input_for_quality_analyzer(self, 
                                          watermarked_images: List[Image.Image], 
                                          unwatermarked_images: List[Image.Image], 
                                          reference_images: List[Image.Image]):
        """Prepare input for diversity analyzer."""
        # Group images by prompt
        grouped_images = []
        for i in range(0, len(watermarked_images), self.prompt_per_image):
            grouped_images.append((watermarked_images[i:i+self.prompt_per_image], 
                                  unwatermarked_images[i:i+self.prompt_per_image]))
        return grouped_images
    
    def analyze_quality(self, 
                        prepared_data: List[Tuple[List[Image.Image], List[Image.Image]]], 
                        analyzer: ImageQualityAnalyzer):
        """Analyze diversity of image batches."""   
        bar = self._get_progress_bar(prepared_data)
        bar.set_description(f"Analyzing diversity for {analyzer.__class__.__name__}")
        w_scores, u_scores = [], []
        # For diversity analyzers, we analyze each batch
        for watermarked_images, unwatermarked_images in bar:
            w_score = analyzer.analyze(watermarked_images)
            u_score = analyzer.analyze(unwatermarked_images)
            w_scores.append(w_score)
            u_scores.append(u_score)
        
        return w_scores, u_scores


class ComparedImageQualityAnalysisPipeline(ImageQualityAnalysisPipeline):
    """
    Pipeline for compared image quality analysis.
    
    This pipeline directly compares watermarked and unwatermarked images
    to compute metrics like PSNR, SSIM, and MS-SSIM. The analyzer receives
    both images and outputs a single comparison score.
    """

    def __init__(self, 
                 dataset: BaseDataset, 
                 watermarked_image_editor_list: List[ImageEditor] = [], 
                 unwatermarked_image_editor_list: List[ImageEditor] = [],
                 analyzers: List[ImageQualityAnalyzer] = None, 
                 unwatermarked_image_source: str = 'generated',
                 reference_image_source: str = 'natural',
                 show_progress: bool = True, 
                 store_path: str = None,
                 return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        
        super().__init__(dataset, watermarked_image_editor_list, unwatermarked_image_editor_list, 
                        analyzers, unwatermarked_image_source, reference_image_source, show_progress, store_path, return_type)

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        return range(self.dataset.num_samples)
    
    def _prepare_input_for_quality_analyzer(self, 
                                          prepared_dataset: DatasetForEvaluation):
        """Prepare input for comparison analyzer."""
        return [(watermarked_image, unwatermarked_image) for watermarked_image, unwatermarked_image in zip(prepared_dataset.watermarked_images, prepared_dataset.unwatermarked_images)]
    
    def analyze_quality(self, 
                        prepared_data: List[Tuple[Image.Image, Image.Image]], 
                        analyzer: ImageQualityAnalyzer):
        """Analyze quality by comparing watermarked and unwatermarked images."""
        bar = self._get_progress_bar(prepared_data)
        bar.set_description(f"Analyzing quality for {analyzer.__class__.__name__}")
        w_scores, u_scores = [], []
        # For comparison analyzers, we compute similarity/difference
        for watermarked_image, unwatermarked_image in bar:
            # Compare watermarked images with unwatermarked images
            w_score = analyzer.analyze(watermarked_image, unwatermarked_image)
            w_scores.append(w_score)
        return w_scores, u_scores # u_scores is not used for comparison analyzers
