import torch
from watermark.auto_watermark import AutoWatermark
from evaluation.dataset import StableDiffusionPromptsDataset
from evaluation.pipelines.detection import WatermarkedMediaDetectionPipeline, UnWatermarkedMediaDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.image_editor import JPEGCompression
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator, FundamentalSuccessRateCalculator
from utils.diffusion_config import DiffusionConfig
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import dotenv
import os

dotenv.load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.getenv("MODEL_PATH")

def assess_numerical_detectability(algorithm_name, labels, rules, target_fpr):
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

    pipeline1 = WatermarkedMediaDetectionPipeline(dataset=my_dataset, media_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 

    pipeline2 = UnWatermarkedMediaDetectionPipeline(dataset=my_dataset, media_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

    detection_kwargs = {
        "num_inference_steps": 50,
        "guidance_scale": 1.0,
    }

    calculator = DynamicThresholdSuccessRateCalculator(labels=labels, rule=rules, target_fpr=target_fpr)
    print(calculator.calculate(pipeline1.evaluate(my_watermark, detection_kwargs=detection_kwargs), pipeline2.evaluate(my_watermark, detection_kwargs=detection_kwargs)))

def assess_binary_detectability(algorithm_name, labels, rules, target_fpr):
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
        inversion_type = "exact"
    )
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    diffusion_config=diffusion_config)

    # Use IS_WATERMARKED return type for fixed threshold evaluation
    pipeline1 = WatermarkedMediaDetectionPipeline(
        dataset=my_dataset, 
        media_editor_list=[],
        show_progress=True, 
        detector_type="is_watermark",
        return_type=DetectionPipelineReturnType.IS_WATERMARKED
    )
    
    pipeline2 = UnWatermarkedMediaDetectionPipeline(
        dataset=my_dataset, 
        media_editor_list=[], 
        media_source_mode="generated",
        show_progress=True, 
        detector_type="is_watermark",
        return_type=DetectionPipelineReturnType.IS_WATERMARKED
    )

   # Use FundamentalSuccessRateCalculator for fixed threshold evaluation
    calculator = FundamentalSuccessRateCalculator(labels=['F1', 'TPR', 'TNR', 'FPR', 'P', 'R', 'ACC'])
    
    detection_kwargs = {
        "num_inference_steps": 50,
        "guidance_scale": 1.0,
        "decoder_inv": False,
        "inv_order": 0
    }
    
    # Get detection results
    watermarked_results = pipeline1.evaluate(my_watermark, detection_kwargs=detection_kwargs)
    non_watermarked_results = pipeline2.evaluate(my_watermark, detection_kwargs=detection_kwargs)
    print(calculator.calculate(watermarked_results, non_watermarked_results))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PRC')
    parser.add_argument('--labels', nargs='+', default=['TPR', 'F1'])
    parser.add_argument('--rules', type=str, default='best')
    parser.add_argument('--target_fpr', type=float, default=0.01)
    args = parser.parse_args()

    # assess_numerical_detectability(args.algorithm, args.labels, args.rules, args.target_fpr)
    assess_binary_detectability(args.algorithm, args.labels, args.rules, args.target_fpr)