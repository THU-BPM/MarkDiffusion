# Copyright 2024 THU-BPM MarkLLM.
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

# ==========================================================================
# assess_quality.py
# Description: Assess the impact on text quality of a watermarking algorithm
# ==========================================================================

import torch
import torch
from watermark.auto_watermark import AutoWatermark
from evaluation.dataset import StableDiffusionPromptsDataset, MSCOCODataset, VBenchDataset
from evaluation.pipelines.image_quality_analysis import (
    DirectImageQualityAnalysisPipeline, 
    ReferencedImageQualityAnalysisPipeline, 
    GroupImageQualityAnalysisPipeline, 
    RepeatImageQualityAnalysisPipeline, 
    ComparedImageQualityAnalysisPipeline, 
    QualityPipelineReturnType
)
from evaluation.tools.image_quality_analyzer import (
NIQECalculator, 
CLIPScoreCalculator, FIDCalculator, InceptionScoreCalculator, LPIPSAnalyzer, PSNRAnalyzer,SSIMAnalyzer)
from utils.diffusion_config import DiffusionConfig
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import dotenv
import os
import dotenv
dotenv.load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "/data2/shared_model/fuzheyu/stable-diffusion-2-1-base/"
"""
    DirectImageQualityAnalysisPipeline: PSNRAnalyzer, SSIMAnalyzer
    ReferencedImageQualityAnalysisPipeline: CLIPScoreCalculator
    GroupImageQualityAnalysisPipeline: FIDCalculator, InceptionScoreCalculator
    RepeatImageQualityAnalysisPipeline: LPIPSAnalyzer
    ComparedImageQualityAnalysisPipeline: PSNRAnalyzer, SSIMAnalyzer

"""

def assess_image_quality(algorithm_name, metric, max_samples=10):
    print(model_path)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    diffusion_config = DiffusionConfig(
            scheduler = scheduler,
            pipe = pipe,
            device = device,
            image_size = (512, 512),
            num_inference_steps = 50,
            guidance_scale = 3.5,
            gen_seed = 42,
            inversion_type = "ddim"
        )
    if metric == 'NIQE':
        my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
        pipeline = DirectImageQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_image_editor_list=[],
                                                     unwatermarked_image_editor_list=[],
                                                     analyzers=[NIQECalculator()],
                                                    show_progress=True, 
                                                    return_type=QualityPipelineReturnType.MEAN_SCORES)

    elif metric == 'CLIP-T':
        my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
        pipeline = ReferencedImageQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_image_editor_list=[],
                                                     unwatermarked_image_editor_list=[],
                                                     analyzers=[CLIPScoreCalculator(reference_source="text")],
                                                     unwatermarked_image_source='generated',
                                                     reference_image_source='generated',  
                                                     show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
        
    elif metric == 'CLIP-I':
        my_dataset = MSCOCODataset(max_samples=max_samples)
        pipeline = ReferencedImageQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_image_editor_list=[],
                                                     unwatermarked_image_editor_list=[],
                                                     analyzers=[CLIPScoreCalculator(reference_source="image")],
                                                     unwatermarked_image_source='generated',
                                                     reference_image_source='natural',
                                                     show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
        
    elif metric == 'FID':
        my_dataset = MSCOCODataset(max_samples=max_samples)
        pipeline = GroupImageQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_image_editor_list=[],
                                                     unwatermarked_image_editor_list=[],
                                                     analyzers=[FIDCalculator()],
                                                     unwatermarked_image_source='generated',
                                                     reference_image_source='natural',
                                                     show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
    elif metric == 'IS':
        my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
        pipeline = GroupImageQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_image_editor_list=[],
                                                     unwatermarked_image_editor_list=[],
                                                     analyzers=[InceptionScoreCalculator()],
                                                     show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)

    elif metric == 'LPIPS':
        my_dataset = StableDiffusionPromptsDataset(max_samples=10)
        pipeline = RepeatImageQualityAnalysisPipeline(dataset=my_dataset, 
                                                     prompt_per_image=20,
                                                     watermarked_image_editor_list=[],
                                                     unwatermarked_image_editor_list=[],
                                                     analyzers=[LPIPSAnalyzer()],
                                                     show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)

    elif metric == 'PSNR':
        my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
        pipeline = ComparedImageQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_image_editor_list=[],
                                                     unwatermarked_image_editor_list=[],
                                                     analyzers=[PSNRAnalyzer()],
                                                     show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
        
    elif metric == 'SSIM':
        my_dataset = StableDiffusionPromptsDataset(max_samples=max_samples)
        pipeline = DirectImageQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_image_editor_list=[],
                                                     unwatermarked_image_editor_list=[],
                                                     analyzers=[SSIMAnalyzer()],
                                                     show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
    else:
        raise ValueError('Invalid metric')
    
    
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    diffusion_config=diffusion_config)
    print(pipeline.evaluate(my_watermark))

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='TR')
    parser.add_argument('--metric', type=str, default='FID')
    parser.add_argument('--max_samples', type=int, default=10)
    args = parser.parse_args()

    assess_image_quality(args.algorithm, args.metric, args.max_samples)
