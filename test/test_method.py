import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model path
model_path = "/mnt/ckpt/stable-diffusion-2-1-base"

def test_algorithm_for_img(algorithm_name):
    # Configure diffusion pipeline
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    diffusion_config = DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=(512, 512),
        num_inference_steps=50,
        guidance_scale=7.5,
        gen_seed=42,
        inversion_type="ddim"
    )

    # Load watermark algorithm
    watermark = AutoWatermark.load(algorithm_name, 
                                algorithm_config=f'config/{algorithm_name}.json',
                                diffusion_config=diffusion_config)

    # Generate watermarked media
    prompt = "A beautiful sunset over the ocean"
    watermarked_image = watermark.generate_watermarked_media(prompt)
    unwatermarked_image = watermark.generate_unwatermarked_media(prompt)

    # Detect watermark
    detection_result = watermark.detect_watermark_in_media(watermarked_image)
    print(f"Detection Result: {detection_result}")

    detection_result_unwatermarked = watermark.detect_watermark_in_media(unwatermarked_image)
    print(f"Detection Result: {detection_result_unwatermarked}")

    # Save images
    watermarked_image.save("watermarked_image.png")
    unwatermarked_image.save("unwatermarked_image.png")


if __name__ == '__main__':
    test_algorithm_for_img('TR')