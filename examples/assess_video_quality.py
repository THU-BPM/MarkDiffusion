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
# assess_video_quality.py
# Description: Assess the impact on video quality of a watermarking algorithm
# ==========================================================================

import torch
import os
import dotenv
from watermark.auto_watermark import AutoWatermark
from evaluation.dataset import VBenchDataset
from evaluation.pipelines.video_quality_analysis import (
    DirectVideoQualityAnalysisPipeline,
    QualityPipelineReturnType
)
from evaluation.tools.video_quality_analyzer import (
    SubjectConsistencyAnalyzer,
    MotionSmoothnessAnalyzer, 
    DynamicDegreeAnalyzer,
    BackgroundConsistencyAnalyzer,
    ImagingQualityAnalyzer
)
from utils.diffusion_config import DiffusionConfig
from diffusers import DDIMScheduler, TextToVideoSDPipeline

# Load environment variables
dotenv.load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.getenv("T2V_MODEL_PATH")

"""
Video Quality Analysis Pipeline and Metrics:
    DirectVideoQualityAnalysisPipeline: 
        - SubjectConsistencyAnalyzer: Measures subject consistency across frames using DINO features
        - MotionSmoothnessAnalyzer: Evaluates motion smoothness using AMT-S frame interpolation
        - DynamicDegreeAnalyzer: Analyzes motion intensity using RAFT optical flow
        - BackgroundConsistencyAnalyzer: Measures background consistency using CLIP features
        - ImagingQualityAnalyzer: Evaluates overall imaging quality using MUSIQ
"""

def assess_video_quality(algorithm_name: str = "VideoShield", metric: str = "subject_consistency", dimension: str = "subject_consistency"):
    """
    Assess video quality using specified metric and VBench dataset.
    
    Args:
        algorithm_name (str): Name of the watermarking algorithm
        metric (str): Quality metric to evaluate ('subject_consistency', 'motion_smoothness', 
                     'dynamic_degree', 'background_consistency', 'imaging_quality')
        dimension (str): VBench dimension to use for evaluation
    """
    
    # Load VBench dataset
    my_dataset = VBenchDataset(max_samples=200, dimension=dimension)
    
    # Initialize analyzer based on metric
    if metric == 'subject_consistency':
        analyzer = SubjectConsistencyAnalyzer(device=device)
    elif metric == 'motion_smoothness':
        analyzer = MotionSmoothnessAnalyzer(device=device)
    elif metric == 'dynamic_degree':
        analyzer = DynamicDegreeAnalyzer(device=device)
    elif metric == 'background_consistency':
        analyzer = BackgroundConsistencyAnalyzer(device=device)
    elif metric == 'imaging_quality':
        analyzer = ImagingQualityAnalyzer(device=device)
    else:
        raise ValueError(f'Invalid metric: {metric}. Supported metrics: subject_consistency, motion_smoothness, dynamic_degree, background_consistency, imaging_quality')
    
    # Create video quality analysis pipeline
    pipeline = DirectVideoQualityAnalysisPipeline(
        dataset=my_dataset,
        watermarked_video_editor_list=[],
        unwatermarked_video_editor_list=[],
        watermarked_frame_editor_list=[],
        unwatermarked_frame_editor_list=[],
        analyzers=[analyzer],
        show_progress=True,
        return_type=QualityPipelineReturnType.MEAN_SCORES
    )
    
    # Create diffusion config for video generation
    if model_path is None:
        raise ValueError("T2V_MODEL_PATH environment variable is not set")
    
    # Load video generation pipeline (placeholder - adapt based on your T2V model)
    # For example, if using VideoCrafter or similar models:
    try:
        # This is a placeholder - replace with actual T2V pipeline initialization
        # scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        # pipe = YourVideoGenerationPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe = TextToVideoSDPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
        diffusion_config = DiffusionConfig(
            scheduler=scheduler,
            pipe=pipe,
            device=device,
            # Video-specific parameters
            num_frames=16,  # Number of frames
            width=512,
            height=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            gen_seed=42,
            inversion_type="ddim"
        )
    except Exception as e:
        print(f"Warning: Could not load T2V model from {model_path}. Using default config. Error: {e}")
        diffusion_config = DiffusionConfig(
            device=device,
            num_frames=16,
            width=512,
            height=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            gen_seed=42,
        )
    
    # Load watermark algorithm
    my_watermark = AutoWatermark.load(
        f'{algorithm_name}', 
        algorithm_config=f'config/{algorithm_name}.json',
        diffusion_config=diffusion_config
    )
    
    # Run evaluation
    print(f"Evaluating {algorithm_name} with {metric} metric on VBench {dimension} dimension...")
    result = pipeline.evaluate(my_watermark)
    print(f"Results: {result}")
    
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Assess video quality impact of watermarking algorithms')
    parser.add_argument('--algorithm', type=str, default='VideoShield', 
                       help='Watermarking algorithm name')
    parser.add_argument('--metric', type=str, default='subject_consistency',
                       choices=['subject_consistency', 'motion_smoothness', 'dynamic_degree', 
                               'background_consistency', 'imaging_quality'],
                       help='Quality metric to evaluate')
    parser.add_argument('--dimension', type=str, default='subject_consistency',
                       choices=['subject_consistency', 'background_consistency', 'imaging_quality', 
                               'motion_smoothness', 'dynamic_degree'],
                       help='VBench dimension to use for evaluation')
    
    args = parser.parse_args()
    
    assess_video_quality(args.algorithm, args.metric, args.dimension)
