"""
Pytest configuration and fixtures for MarkDiffusion watermark algorithm tests.

This file contains all pytest hooks, fixtures, and configuration that will be
automatically discovered and used by pytest.
"""

import pytest
import torch
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import gc
from markdiffusion.watermark.auto_watermark import AutoWatermark, PIPELINE_SUPPORTED_WATERMARKS
from markdiffusion.utils.diffusion_config import DiffusionConfig
from diffusers import (
    StableDiffusionPipeline,
    TextToVideoSDPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)
from markdiffusion.utils.pipeline_utils import (
    PIPELINE_TYPE_IMAGE,
    PIPELINE_TYPE_TEXT_TO_VIDEO,
    PIPELINE_TYPE_IMAGE_TO_VIDEO
)


# ============================================================================
# Test Configuration
# ============================================================================

# Default model paths (can be overridden via pytest options)
DEFAULT_IMAGE_MODEL_PATH = "huanzi05/stable-diffusion-2-1-base"
DEFAULT_VIDEO_MODEL_PATH = "ali-vilab/text-to-video-ms-1.7b"

# Test prompts
TEST_PROMPT_IMAGE = "A beautiful sunset over the ocean"
TEST_PROMPT_VIDEO = "A cinematic timelapse of city lights at night"

# Test parameters
IMAGE_SIZE = (64, 64)
NUM_INFERENCE_STEPS = 1
GUIDANCE_SCALE = 1.0
GEN_SEED = 42
NUM_FRAMES = 2

# Test dataset parameters
TEST_DATASET_MAX_SAMPLES = 1  # Small sample size for testing
TEST_DATASET_FOR_IMG = "MSCOCODataset"
TEST_DATASET_FOR_VIDEO = "VBenchDataset"


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--algorithm",
        action="store",
        default=None,
        help="Specific algorithm to test (e.g., TR, GS, VideoShield)"
    )
    parser.addoption(
        "--image-model-path",
        action="store",
        default=DEFAULT_IMAGE_MODEL_PATH,
        help="Path to image generation model"
    )
    parser.addoption(
        "--video-model-path",
        action="store",
        default=DEFAULT_VIDEO_MODEL_PATH,
        help="Path to video generation model"
    )
    parser.addoption(
        "--skip-generation",
        action="store_true",
        default=False,
        help="Skip generation tests (only test detection)"
    )
    parser.addoption(
        "--skip-detection",
        action="store_true",
        default=False,
        help="Skip detection tests (only test generation)"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "image: mark test as image watermark test")
    config.addinivalue_line("markers", "video: mark test as video watermark test")
    config.addinivalue_line("markers", "inversion: mark test as inversion module test")
    config.addinivalue_line("markers", "visualization: mark test as visualization test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "pipeline: mark test as pipeline test")
    config.addinivalue_line("markers", "detection: mark test as detection pipeline test")
    config.addinivalue_line("markers", "quality: mark test as quality analysis pipeline test")
    config.addinivalue_line("markers", "integration: mark test as integration test")

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information to pytest output."""
    terminalreporter.write_sep("=", "Watermark Algorithm Test Summary")

    # Count passed/failed tests by algorithm
    passed = terminalreporter.stats.get('passed', [])
    failed = terminalreporter.stats.get('failed', [])
    skipped = terminalreporter.stats.get('skipped', [])

    terminalreporter.write_line(f"Passed: {len(passed)}")
    terminalreporter.write_line(f"Failed: {len(failed)}")
    terminalreporter.write_line(f"Skipped: {len(skipped)}")

def pytest_collection_modifyitems(config, items):
    """
    Filter tests based on the --algorithm command line option.
    Handles case-insensitive matching against supported algorithms.
    """
    algo_str = config.getoption("--algorithm")
    if not algo_str:
        return

    user_whitelist = {a.strip().lower() for a in algo_str.split(",") if a.strip()}
    
    supported_algos = set()
    for algo_list in PIPELINE_SUPPORTED_WATERMARKS.values():
        supported_algos.update(algo_list)
    
    name_mapping = {name.lower(): name for name in supported_algos}

    final_whitelist = {name_mapping.get(u, u) for u in user_whitelist}
    
    print(f"\n[Filter] User input: {user_whitelist}")
    print(f"[Filter] Mapped to: {final_whitelist}")

    selected, deselected = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec and "algorithm_name" in callspec.params:
            current_algo = callspec.params["algorithm_name"]
            if current_algo in final_whitelist:
                selected.append(item)
            else:
                deselected.append(item)
        
        elif any(target.lower() in item.nodeid.lower() for target in final_whitelist):
             selected.append(item)
        
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get the device for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="session")
def image_model_path(request):
    """Get the image model path from command line or use default."""
    return request.config.getoption("--image-model-path")


@pytest.fixture(scope="session")
def video_model_path(request):
    """Get the video model path from command line or use default."""
    return request.config.getoption("--video-model-path")


@pytest.fixture(scope="session")
def skip_generation(request):
    """Check if generation tests should be skipped."""
    return request.config.getoption("--skip-generation")


@pytest.fixture(scope="session")
def skip_detection(request):
    """Check if detection tests should be skipped."""
    return request.config.getoption("--skip-detection")


@pytest.fixture(scope="session")
def image_pipeline(device, image_model_path):
    """Create and cache image generation pipeline."""
    try:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            image_model_path,
            subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            image_model_path,
            scheduler=scheduler
        ).to(device)
        return pipe, scheduler
    except Exception as e:
        pytest.skip(f"Failed to load image model: {e}")


@pytest.fixture(scope="session")
def video_pipeline(device, video_model_path):
    """Create and cache video generation pipeline."""
    try:
        scheduler = DDIMScheduler.from_pretrained(
            video_model_path,
            subfolder="scheduler"
        )
        pipe = TextToVideoSDPipeline.from_pretrained(
            video_model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
        return pipe, scheduler
    except Exception as e:
        pytest.skip(f"Failed to load video model: {e}")


@pytest.fixture
def image_diffusion_config(device, image_pipeline):
    """Create diffusion config for image generation."""
    pipe, scheduler = image_pipeline
    return DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=IMAGE_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        gen_seed=GEN_SEED,
        inversion_type="ddim"
    )


@pytest.fixture
def video_diffusion_config(device, video_pipeline):
    """Create diffusion config for video generation."""
    pipe, scheduler = video_pipeline
    return DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=IMAGE_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        gen_seed=GEN_SEED,
        inversion_type="ddim",
        num_frames=NUM_FRAMES
    )


# ============================================================================
# Fixtures for Pipeline Tests
# ============================================================================

@pytest.fixture
def test_image_dataset():
    """Create test dataset for image pipelines."""
    from markdiffusion.evaluation.dataset import MSCOCODataset
    return MSCOCODataset(
        max_samples=TEST_DATASET_MAX_SAMPLES,
        shuffle=False
    )


@pytest.fixture
def test_video_dataset():
    """Create test dataset for video pipelines."""
    from markdiffusion.evaluation.dataset import VBenchDataset
    return VBenchDataset(
        max_samples=TEST_DATASET_MAX_SAMPLES,
        dimension="subject_consistency",
        shuffle=False
    )


@pytest.fixture
def all_image_editors():
    """Get all image editor tools for saturation testing."""
    from markdiffusion.evaluation.tools.image_editor import (
        JPEGCompression,
        Rotation,
        CrSc,
        GaussianBlurring,
        GaussianNoise,
        Brightness,
        Mask,
        Overlay,
        AdaptiveNoiseInjection
    )

    return [
        JPEGCompression(),
        Rotation(),
        CrSc(),
        GaussianBlurring(),
        GaussianNoise(),
        Brightness(),
        Mask(),
        Overlay(),
        # AdaptiveNoiseInjection()
    ]


@pytest.fixture
def all_video_editors():
    """Get all video editor tools for saturation testing."""
    from markdiffusion.evaluation.tools.video_editor import (
        MPEG4Compression,
        FrameAverage,
        FrameRateAdapter,
        FrameSwap,
        FrameInterpolationAttack
    )

    return [
        MPEG4Compression(),
        FrameAverage(),
        FrameRateAdapter(),
        FrameSwap(),
        FrameInterpolationAttack()
    ]


@pytest.fixture
def all_image_quality_analyzers():
    """Get all image quality analyzers for testing."""
    from markdiffusion.evaluation.tools.image_quality_analyzer import (
        NIQECalculator,
        CLIPScoreCalculator,
        FIDCalculator,
        InceptionScoreCalculator,
        LPIPSAnalyzer,
        PSNRAnalyzer,
        SSIMAnalyzer,
        BRISQUEAnalyzer,
        VIFAnalyzer,
        FSIMAnalyzer
    )

    return {
        'direct': [NIQECalculator(patch_size=16), BRISQUEAnalyzer()],
        'referenced': [CLIPScoreCalculator()],
        'group': [FIDCalculator(), InceptionScoreCalculator()],
        'repeat': [LPIPSAnalyzer()],
        'compared': [PSNRAnalyzer(), SSIMAnalyzer(), VIFAnalyzer(), FSIMAnalyzer()]
    }


@pytest.fixture
def all_video_quality_analyzers():
    """Get all video quality analyzers for testing."""
    from markdiffusion.evaluation.tools.video_quality_analyzer import (
        SubjectConsistencyAnalyzer,
        MotionSmoothnessAnalyzer,
        DynamicDegreeAnalyzer,
        BackgroundConsistencyAnalyzer,
        ImagingQualityAnalyzer
    )

    return [
        SubjectConsistencyAnalyzer(),
        MotionSmoothnessAnalyzer(),
        DynamicDegreeAnalyzer(),
        BackgroundConsistencyAnalyzer(),
        ImagingQualityAnalyzer()
    ]


# Export constants for use in test files
__all__ = [
    'TEST_PROMPT_IMAGE',
    'TEST_PROMPT_VIDEO',
    'IMAGE_SIZE',
    'NUM_INFERENCE_STEPS',
    'GUIDANCE_SCALE',
    'GEN_SEED',
    'NUM_FRAMES',
    'TEST_DATASET_MAX_SAMPLES',
    'TEST_DATASET_FOR_IMG',
    'TEST_DATASET_FOR_VIDEO',
]


@pytest.fixture(autouse=True)
def cleanup_memory():
    """
    Automatic memory cleanup after EACH test case.
    This is critical for CI environments with limited RAM/VRAM.
    """
    yield
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@pytest.fixture
def test_image_dataset_group():
    """Create test dataset for image pipelines requiring multiple samples (e.g. FID)."""
    from markdiffusion.evaluation.dataset import MSCOCODataset
    return MSCOCODataset(
        max_samples=2,  # Minimum for FID
        shuffle=False
    )