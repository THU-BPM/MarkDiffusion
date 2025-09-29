"""Integration-style tests for MarkDiffusion watermark algorithms.

This module iterates over every image and video watermark currently exposed through
`AutoWatermark`. For each method we verify that:

1. A watermarked sample can be generated from a shared Stable Diffusion
   pipeline configuration.
2. An unwatermarked baseline can also be produced.
3. The detector returns structured metrics for both watermarked and
   unwatermarked media.

Given the heavy dependency on text-to-image diffusion pipelines, the tests are
guarded behind the environment variable ``MARKDIFFUSION_RUN_FULL_TESTS``.
Set it to ``"1"`` to enable the suite once the required model weights are
available locally.

For example, you can run it in your terminal by:
```
cd ./MarkDiffusion
MARKDIFFUSION_RUN_FULL_TESTS=1 \
MARKDIFFUSION_TEST_MODEL=/path/to/your/t2i/model/ \
MARKDIFFUSION_TEST_VIDEO_MODEL=/path/to/your/t2v/model/ \
MARKDIFFUSION_LOCAL_ONLY=1 \
pytest test/test_method.py -s
```
"""

from __future__ import annotations

import os
from typing import Generator

import pytest
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

try:
	from diffusers import TextToVideoSDPipeline
except ImportError:  # pragma: no cover - optional dependency
	TextToVideoSDPipeline = None

from utils.diffusion_config import DiffusionConfig
from utils.pipeline_utils import PIPELINE_TYPE_IMAGE, PIPELINE_TYPE_TEXT_TO_VIDEO
from watermark.auto_watermark import AutoWatermark
from utils.media_utils import convert_video_frames_to_images


pytestmark = pytest.mark.skipif(
	os.getenv("MARKDIFFUSION_RUN_FULL_TESTS") != "1",
	reason=(
		"MarkDiffusion watermark integration tests require Stable Diffusion "
		"weights and can be resource intensive. Set MARKDIFFUSION_RUN_FULL_TESTS=1 "
		"to enable them."
	),
)

MODEL_ID = os.getenv("MARKDIFFUSION_TEST_MODEL", "runwayml/stable-diffusion-v1-5")
TEST_PROMPT = "A small watercolor painting of a lighthouse by the sea"
VIDEO_MODEL_ID = os.getenv("MARKDIFFUSION_TEST_VIDEO_MODEL")
VIDEO_PROMPT = os.getenv(
	"MARKDIFFUSION_TEST_VIDEO_PROMPT",
	"A time-lapse of aurora borealis dancing across the night sky",
)
VIDEO_NUM_FRAMES = int(os.getenv("MARKDIFFUSION_TEST_VIDEO_FRAMES", "16"))


@pytest.fixture(scope="session")
def device() -> str:
	return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def pipeline(device: str) -> Generator[StableDiffusionPipeline, None, None]:
	try:
		pipe = StableDiffusionPipeline.from_pretrained(
			MODEL_ID,
			torch_dtype=torch.float32,
			local_files_only=os.getenv("MARKDIFFUSION_LOCAL_ONLY", "0") == "1",
		)
	except OSError as exc:
		pytest.skip(
			f"Stable Diffusion weights for '{MODEL_ID}' are unavailable. "
			"Download them or set MARKDIFFUSION_LOCAL_ONLY=0 to allow fetching."
		)

	pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
	pipe = pipe.to(device)
	pipe.set_progress_bar_config(disable=True)
	yield pipe
	pipe.to("cpu")
	if hasattr(torch.cuda, "empty_cache"):
		torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def diffusion_config(pipeline: StableDiffusionPipeline, device: str) -> DiffusionConfig:
	latent_size = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
	num_steps = max(50, pipeline.scheduler.config.get("num_train_timesteps", 1000) // 20)
	return DiffusionConfig(
		scheduler=pipeline.scheduler,
		pipe=pipeline,
		device=device,
		guidance_scale=5.0,
		num_inference_steps=num_steps,
		num_images=1,
		image_size=(latent_size, latent_size),
		dtype=torch.float32,
		gen_seed=42,
		init_latents_seed=1234,
	)


@pytest.fixture(scope="session")
def video_pipeline(device: str) -> Generator[object, None, None]:
	if TextToVideoSDPipeline is None:
		pytest.skip("diffusers.TextToVideoSDPipeline is unavailable in this environment")
	if VIDEO_MODEL_ID is None:
		pytest.skip("Set MARKDIFFUSION_TEST_VIDEO_MODEL to enable VideoShield tests")
	if device != "cuda":
		pytest.skip("VideoShield tests require a CUDA-capable device")

	try:
		pipe = TextToVideoSDPipeline.from_pretrained(
			VIDEO_MODEL_ID,
			torch_dtype=torch.float16,
			local_files_only=os.getenv("MARKDIFFUSION_LOCAL_ONLY", "0") == "1",
		)
	except OSError:
		pytest.skip(
			f"Video model weights for '{VIDEO_MODEL_ID}' are unavailable. "
			"Download them or set MARKDIFFUSION_LOCAL_ONLY=0 to allow fetching."
		)

	pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
	pipe = pipe.to(device)
	pipe.set_progress_bar_config(disable=True)
	yield pipe
	pipe.to("cpu")
	if hasattr(torch.cuda, "empty_cache"):
		torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def video_diffusion_config(video_pipeline: object, device: str) -> DiffusionConfig:
	latent_size = video_pipeline.unet.config.sample_size * video_pipeline.vae_scale_factor
	num_steps = max(50, video_pipeline.scheduler.config.get("num_train_timesteps", 1000) // 20)
	return DiffusionConfig(
		scheduler=video_pipeline.scheduler,
		pipe=video_pipeline,
		device=device,
		guidance_scale=5.0,
		num_inference_steps=num_steps,
		num_images=1,
		image_size=(latent_size, latent_size),
		dtype=video_pipeline.unet.dtype if hasattr(video_pipeline.unet, "dtype") else torch.float16,
		gen_seed=42,
		init_latents_seed=5678,
		num_frames=VIDEO_NUM_FRAMES,
	)


@pytest.mark.parametrize("algorithm_name", [pytest.param(name) for name in AutoWatermark.list_supported_algorithms(PIPELINE_TYPE_IMAGE)])
def test_markdiffusion_watermark_generation_and_detection(
	algorithm_name: str,
	diffusion_config: DiffusionConfig,
) -> None:
	try:
		watermark = AutoWatermark.load(algorithm_name, diffusion_config=diffusion_config)
	except ImportError as exc:
		pytest.skip(f"{algorithm_name} requires optional dependency: {exc}")
		return

	watermarked_media = watermark.generate_watermarked_media(TEST_PROMPT)
	assert watermarked_media is not None

	unwatermarked_media = watermark.generate_unwatermarked_media(TEST_PROMPT)
	assert unwatermarked_media is not None

	marked_detection = watermark.detect_watermark_in_media(watermarked_media, prompt=TEST_PROMPT)
	assert isinstance(marked_detection, dict)
	assert marked_detection, "Detection dict for watermarked media should not be empty"

	unmarked_detection = watermark.detect_watermark_in_media(unwatermarked_media, prompt=TEST_PROMPT)
	assert isinstance(unmarked_detection, dict)
	assert unmarked_detection, "Detection dict for unwatermarked media should not be empty"


@pytest.mark.parametrize(
	"algorithm_name",
	[pytest.param(name) for name in AutoWatermark.list_supported_algorithms(PIPELINE_TYPE_TEXT_TO_VIDEO)],
)
def test_markdiffusion_video_watermark_generation_and_detection(
	algorithm_name: str,
	video_diffusion_config: DiffusionConfig,
) -> None:
	try:
		watermark = AutoWatermark.load(algorithm_name, diffusion_config=video_diffusion_config)
	except ImportError as exc:
		pytest.skip(f"{algorithm_name} requires optional dependency: {exc}")
		return

	watermarked_frames = watermark.generate_watermarked_media(
		VIDEO_PROMPT,
		num_frames=video_diffusion_config.num_frames,
	)
	assert isinstance(watermarked_frames, list)
	assert len(watermarked_frames) == video_diffusion_config.num_frames

	# Generate an unwatermarked baseline using the underlying pipeline
	pipe = watermark.config.pipe
	output = pipe(
		VIDEO_PROMPT,
		num_inference_steps=video_diffusion_config.num_inference_steps,
		guidance_scale=video_diffusion_config.guidance_scale,
		height=video_diffusion_config.image_size[0],
		width=video_diffusion_config.image_size[1],
		num_frames=video_diffusion_config.num_frames,
	)
	if hasattr(output, "frames"):
		unwatermarked_frames = convert_video_frames_to_images(output.frames[0])
	elif hasattr(output, "videos"):
		unwatermarked_frames = convert_video_frames_to_images(output.videos[0])
	else:
		unwatermarked_frames = convert_video_frames_to_images(output[0])
	assert len(unwatermarked_frames) == video_diffusion_config.num_frames

	marked_detection = watermark.detect_watermark_in_media(
		watermarked_frames,
		prompt=VIDEO_PROMPT,
		num_frames=video_diffusion_config.num_frames,
	)
	assert isinstance(marked_detection, dict)
	assert marked_detection, "Detection dict for watermarked video should not be empty"

	unmarked_detection = watermark.detect_watermark_in_media(
		unwatermarked_frames,
		prompt=VIDEO_PROMPT,
		num_frames=video_diffusion_config.num_frames,
	)
	assert isinstance(unmarked_detection, dict)
	assert unmarked_detection, "Detection dict for unwatermarked video should not be empty"
