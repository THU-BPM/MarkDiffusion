import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig
from diffusers import StableDiffusionPipeline, TextToVideoSDPipeline, DPMSolverMultistepScheduler

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model path
model_path = "/mnt/ckpt/stable-diffusion-2-1-base"
video_model_path = "/mnt/ckpt/text-to-video-ms-1.7b/"

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


def test_algorithm_for_video(algorithm_name):
    # Configure text-to-video diffusion pipeline
    scheduler = DPMSolverMultistepScheduler.from_pretrained(video_model_path, subfolder="scheduler")
    pipe = TextToVideoSDPipeline.from_pretrained(
        video_model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    ).to(device)

    diffusion_config = DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=(512, 512),
        num_inference_steps=50,
        guidance_scale=7.5,
        gen_seed=123,
        inversion_type="ddim",
        num_frames=16
    )

    # Load VideoShield algorithm
    watermark = AutoWatermark.load(algorithm_name,
                                   algorithm_config=f'config/{algorithm_name}.json',
                                   diffusion_config=diffusion_config)

    prompt = "A cinematic timelapse of city lights at night"
    watermarked_frames = watermark.generate_watermarked_media(prompt, num_frames=diffusion_config.num_frames)
    unwatermarked_frames = watermark.generate_unwatermarked_media(prompt, num_frames=diffusion_config.num_frames)

    detection_result = watermark.detect_watermark_in_media(
        watermarked_frames,
        prompt=prompt,
        num_frames=diffusion_config.num_frames
    )
    print(f"Video detection result (watermarked): {detection_result}")

    detection_result_unwatermarked = watermark.detect_watermark_in_media(
        unwatermarked_frames,
        prompt=prompt,
        num_frames=diffusion_config.num_frames
    )
    print(f"Video detection result (unwatermarked): {detection_result_unwatermarked}")

    # Save first frame for quick inspection
    if watermarked_frames:
        watermarked_frames[0].save("watermarked_frame_0.png")
    if unwatermarked_frames:
        unwatermarked_frames[0].save("unwatermarked_frame_0.png")


if __name__ == '__main__':
    test_algorithm_for_img('TR')
    test_algorithm_for_video('VideoShield')