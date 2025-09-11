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


import torch
from watermark.auto_watermark import AutoWatermark
from evaluation.dataset import StableDiffusionPromptsDataset
from evaluation.pipelines.detection import WatermarkMediaDetectionPipeline, UnWatermarkMediaDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.image_editor import JPEGCompression, Rotation, CrSc, GaussianBlurring, GaussianNoise, Brightness
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from utils.diffusion_config import DiffusionConfig
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import dotenv
import os

dotenv.load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.getenv("MODEL_PATH")

def assess_robustness(algorithm_name, attack_name):
    my_dataset = StableDiffusionPromptsDataset(max_samples=200)
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

    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    diffusion_config=diffusion_config)
    if attack_name == 'JPEG':
        attack = JPEGCompression(quality=25)
    elif attack_name == 'Rotation':
        attack = Rotation(angle=75, expand=False)
    elif attack_name == 'CrSc':
        attack = CrSc(crop_ratio=0.75)
    elif attack_name == 'Blur':
        attack = GaussianBlurring(radius=8)
    elif attack_name == 'Noise':
        attack = GaussianNoise(sigma=0.1)
    elif attack_name == 'Brightness':
        attack = Brightness(factor=0.6)

    pipline1 = WatermarkedMediaDetectionPipeline(dataset=my_dataset, media_editor_list=[attack],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 

    pipline2 = UnWatermarkedMediaDetectionPipeline(dataset=my_dataset, media_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='best')
    print(calculator.calculate(pipline1.evaluate(my_watermark), pipline2.evaluate(my_watermark)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='TR')
    parser.add_argument('--attack', type=str, default='JPEG')
    args = parser.parse_args()

    assess_image_robustness(args.algorithm, args.attack)