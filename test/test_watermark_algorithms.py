"""
Parameterized pytest tests for all watermark algorithms in MarkDiffusion.

Usage:
    # Test all image watermark algorithms
    pytest test/test_watermark_algorithms.py -v

    # Test specific algorithm
    pytest test/test_watermark_algorithms.py -v -k "test_image_watermark[TR]"

    # Test specific algorithms using markers
    pytest test/test_watermark_algorithms.py -v -m "image"
    pytest test/test_watermark_algorithms.py -v -m "video"

    # Test with custom parameters
    pytest test/test_watermark_algorithms.py -v --algorithm TR --image-model-path /path/to/model
"""

import json
import types
import tempfile
import torch
import pytest
from PIL import Image
from typing import Dict, Any

from markdiffusion.watermark.auto_watermark import AutoWatermark, PIPELINE_SUPPORTED_WATERMARKS
from markdiffusion.utils.pipeline_utils import (
    get_pipeline_type,
    PIPELINE_TYPE_IMAGE,
    PIPELINE_TYPE_TEXT_TO_VIDEO,
)

# Import test constants from conftest
from .conftest import (
    TEST_PROMPT_IMAGE,
    TEST_PROMPT_VIDEO,
    IMAGE_SIZE,
    NUM_FRAMES,
)


# ============================================================================
# Test Cases - Image Watermarks
# ============================================================================

@pytest.mark.image
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_watermark_initialization(algorithm_name, image_diffusion_config):
    """Test that image watermark algorithms can be initialized correctly."""
    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )
        assert watermark is not None
        assert watermark.config is not None
        assert get_pipeline_type(watermark.config.pipe) == PIPELINE_TYPE_IMAGE
        print(f"✓ {algorithm_name} initialized successfully")
    except Exception as e:
        pytest.fail(f"Failed to initialize {algorithm_name}: {e}")


@pytest.mark.image
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_watermark_generation(algorithm_name, image_diffusion_config, skip_generation):
    """Test watermarked image generation for each algorithm."""
    if skip_generation:
        pytest.skip("Generation tests skipped by --skip-generation flag")

    try:
        # smaller test params for generation tests
        image_diffusion_config.num_inference_steps = 10
        image_diffusion_config.image_size = (512, 512)
        
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )
        
        

        # Generate watermarked image
        
        # more tests for specific algorithms if applicable
        if algorithm_name == "TR":
            tr_w_patterns = ["seed_ring", "seed_zeros", "seed_rand", "rand", "zeros", "const", "ring"]
            for w_pattern in tr_w_patterns:
                # Test TR with different w_patterns

                tr_config = {
                    "algorithm_name": "TR",
                    "w_seed": 999999,
                    "w_channel": 0,
                    "w_pattern": w_pattern,
                    "w_mask_shape": "circle",
                    "w_radius": 10,
                    "w_pattern_const": 0,
                    "threshold": 50
                }
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(tr_config, f)
                    temp_config_path = f.name
                
                try:
                    watermark_wm = AutoWatermark.load(
                        "TR",
                        algorithm_config=temp_config_path,
                        diffusion_config=image_diffusion_config
                    )
                    # Generate with wm_type="wm"
                    watermarked_image_wm = watermark_wm.generate_watermarked_media(TEST_PROMPT_IMAGE)
                    assert watermarked_image_wm is not None
                    assert isinstance(watermarked_image_wm, Image.Image)
                    assert watermarked_image_wm.size == (512, 512)
                    print(f"  ✓ TR generated watermarked image with {w_pattern} pattern successfully")
                finally:
                    import os
                    os.unlink(temp_config_path)
        elif algorithm_name == "GM":
            gm_w_patterns = ["seed_ring", "seed_zeros", "seed_rand", "rand", "zeros", "const", "signal_ring"]
            for w_pattern in gm_w_patterns:
                # Test GM with different w_patterns
                # Create a temporary config file with wm_type="wm"
                gm_config = {
                    "algorithm_name": "GM",
                    "channel_copy": 1,
                    "w_copy": 8,
                    "h_copy": 8,
                    "user_number": 1000000,
                    "fpr": 1e-6,
                    "chacha_key_seed": 123456,
                    "chacha_nonce_seed": 789012,
                    "watermark_seed": 0,
                    "w_seed": 999999,
                    "w_channel": -1,
                    "w_pattern": w_pattern,
                    "w_mask_shape": "circle",
                    "w_radius": 4,
                    "w_measurement": "l1_complex",
                    "w_injection": "complex",
                    "w_pattern_const": 0.0,
                    "w_length": None,
                    "huggingface_repo": "Generative-Watermark-Toolkits/MarkDiffusion-gm",
                    "gnr_checkpoint": "Generative-Watermark-Toolkits/MarkDiffusion-gm/model_final.pth",
                    "gnr_classifier_type": 0,
                    "gnr_model_nf": 128,
                    "gnr_binary_threshold": 0.5,
                    "gnr_use_for_decision": True,
                    "gnr_threshold": None,
                    "fuser_checkpoint": "Generative-Watermark-Toolkits/MarkDiffusion-gm/sd21_cls2.pkl",
                    "fuser_threshold": 0.5,
                    "fuser_frequency_scale": 0.01,
                    "hf_dir": "model_from_hf"
                }
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(gm_config, f)
                    temp_config_path = f.name
                
                try:
                    watermark_wm = AutoWatermark.load(
                        "GM",
                        algorithm_config=temp_config_path,
                        diffusion_config=image_diffusion_config
                    )
                    # Generate with wm_type="wm"
                    watermarked_image_wm = watermark_wm.generate_watermarked_media(TEST_PROMPT_IMAGE)
                    assert watermarked_image_wm is not None
                    assert isinstance(watermarked_image_wm, Image.Image)
                    assert watermarked_image_wm.size == (512, 512)
                    print(f"  ✓ GM generated watermarked image with {w_pattern} pattern successfully")
                finally:
                    import os
                    os.unlink(temp_config_path)
        elif algorithm_name == "SFW":
            # Test SFW with wm_type="wm" (non-HSQR mode uses Fourier treering pattern)
            # Create a temporary config file with wm_type="wm"
            sfw_config_wm = {
                "algorithm_name": "SFW",
                "w_seed": 42,
                "wm_type": "HSTR",  # Test with non-HSQR mode
                "delta": 1,
                "w_channel": 3,
                "threshold": 50
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sfw_config_wm, f)
                temp_config_path = f.name
            
            try:
                watermark_wm = AutoWatermark.load(
                    "SFW",
                    algorithm_config=temp_config_path,
                    diffusion_config=image_diffusion_config
                )
                # Generate with wm_type="wm"
                watermarked_image_wm = watermark_wm.generate_watermarked_media(TEST_PROMPT_IMAGE)
                assert watermarked_image_wm is not None
                assert isinstance(watermarked_image_wm, Image.Image)
                assert watermarked_image_wm.size == (512, 512)
                print(f"  ✓ SFW wm_type='wm' generation passed")
                # detect with wm_type="wm"
                detection_result = watermark_wm.detect_watermark_in_media(watermarked_image_wm)
                assert detection_result is not None
                detection_result = watermark_wm.detect_watermark_in_media(watermarked_image_wm, detector_type="p_value")
                assert detection_result is not None
                print(f"  ✓ SFW wm_type='wm' detection passed")

            finally:
                import os
                os.unlink(temp_config_path)

        
        watermarked_image = watermark.generate_watermarked_media(TEST_PROMPT_IMAGE)

        # Validate output
        assert watermarked_image is not None
        assert isinstance(watermarked_image, Image.Image)
        assert watermarked_image.size == (512, 512)

        print(f"✓ {algorithm_name} generated watermarked image successfully")

    except NotImplementedError:
        pytest.skip(f"{algorithm_name} does not implement watermarked image generation")
    except Exception as e:
        pytest.fail(f"Failed to generate watermarked image with {algorithm_name}: {e}")


@pytest.mark.image
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_unwatermarked_generation(algorithm_name, image_diffusion_config, skip_generation):
    """Test unwatermarked image generation for each algorithm."""
    if skip_generation:
        pytest.skip("Generation tests skipped by --skip-generation flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )

        # Generate unwatermarked image
        unwatermarked_image = watermark.generate_unwatermarked_media(TEST_PROMPT_IMAGE)

        # Validate output
        assert unwatermarked_image is not None
        assert isinstance(unwatermarked_image, Image.Image)
        assert unwatermarked_image.size == (IMAGE_SIZE[1], IMAGE_SIZE[0])

        print(f"✓ {algorithm_name} generated unwatermarked image successfully")

    except Exception as e:
        pytest.fail(f"Failed to generate unwatermarked image with {algorithm_name}: {e}")


@pytest.mark.image
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE])
def test_image_watermark_detection(algorithm_name, image_diffusion_config, skip_detection):
    """Test watermark detection in images for each algorithm."""
    if skip_detection:
        pytest.skip("Detection tests skipped by --skip-detection flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=image_diffusion_config
        )

        # Generate watermarked and unwatermarked images
        watermarked_image = watermark.generate_watermarked_media(TEST_PROMPT_IMAGE)
        unwatermarked_image = watermark.generate_unwatermarked_media(TEST_PROMPT_IMAGE)

        # Detect watermark in watermarked image
        detection_result_wm = watermark.detect_watermark_in_media(watermarked_image)
        assert detection_result_wm is not None
        assert isinstance(detection_result_wm, dict)
        assert detection_result_wm['is_watermarked'] is True
        
        # Test other detector_type for specific algorithms if applicable
        detector_types = []
        if algorithm_name == "RI":
            modes = ['real', 'imag']
            for mode in modes:
                detection_result_mode = watermark.detect_watermark_in_media(watermarked_image, mode=mode)
        elif algorithm_name == "TR":
            detection_result_mode = watermark.detect_watermark_in_media(watermarked_image, detector_type='p_value')
            assert detection_result_mode is not None
            assert isinstance(detection_result_mode, dict)
        elif algorithm_name == "ROBIN":
            detector_types = ['p_value', 'cosine_similarity']
            for detector_type in detector_types:
                detection_result_mode = watermark.detect_watermark_in_media(watermarked_image, detector_type=detector_type)
                assert detection_result_mode is not None
                assert isinstance(detection_result_mode, dict)
        elif algorithm_name == "GM":
            detector_types = ['message_acc', 'complex_l1', 'gnr_bit_acc', 'fused', 'all']
            for detector_type in detector_types:
                detection_result_mode = watermark.detect_watermark_in_media(watermarked_image, detector_type=detector_type)
                assert detection_result_mode is not None
                assert isinstance(detection_result_mode, dict)
            

        elif algorithm_name == "GS":
            # Test GS with chacha=False (non-ChaCha mode uses simple XOR key)
            # Create a temporary config file with chacha=False
            gs_config_no_chacha = {
                "algorithm_name": "GS",
                "channel_copy": 1,
                "wm_key": 42,
                "hw_copy": 8,
                "chacha": False,  # Test with chacha disabled
                "chacha_key_seed": 123456,
                "chacha_nonce_seed": 789012,
                "threshold": 0.7
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(gs_config_no_chacha, f)
                temp_config_path = f.name
            
            try:
                watermark_no_chacha = AutoWatermark.load(
                    "GS",
                    algorithm_config=temp_config_path,
                    diffusion_config=image_diffusion_config
                )
                # Generate and detect with chacha=False
                watermarked_image_no_chacha = watermark_no_chacha.generate_watermarked_media(TEST_PROMPT_IMAGE)
                detection_result_no_chacha = watermark_no_chacha.detect_watermark_in_media(watermarked_image_no_chacha)
                assert detection_result_no_chacha is not None
                assert isinstance(detection_result_no_chacha, dict)
                assert detection_result_no_chacha['is_watermarked'] is True
                print(f"  ✓ GS chacha=False detection passed: {detection_result_no_chacha}")
            finally:
                import os
                os.unlink(temp_config_path)

        # Detect watermark in unwatermarked image
        detection_result_unwm = watermark.detect_watermark_in_media(unwatermarked_image)
        assert detection_result_unwm is not None
        assert isinstance(detection_result_unwm, dict)
        assert detection_result_unwm['is_watermarked'] is False

        print(f"✓ {algorithm_name} detection results:")
        print(f"  Watermarked: {detection_result_wm}")
        print(f"  Unwatermarked: {detection_result_unwm}")

    except NotImplementedError:
        pytest.skip(f"{algorithm_name} does not implement watermark detection")
    except Exception as e:
        pytest.fail(f"Failed to detect watermark with {algorithm_name}: {e}")


# ============================================================================
# Test Cases - Video Watermarks
# ============================================================================

@pytest.mark.video
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_watermark_initialization(algorithm_name, video_diffusion_config):
    """Test that video watermark algorithms can be initialized correctly."""
    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )
        assert watermark is not None
        assert watermark.config is not None
        assert get_pipeline_type(watermark.config.pipe) == PIPELINE_TYPE_TEXT_TO_VIDEO
        print(f"✓ {algorithm_name} initialized successfully")
    except Exception as e:
        pytest.fail(f"Failed to initialize {algorithm_name}: {e}")


@pytest.mark.video
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_watermark_generation(algorithm_name, video_diffusion_config, skip_generation):
    """Test watermarked video generation for each algorithm."""
    if skip_generation:
        pytest.skip("Generation tests skipped by --skip-generation flag")

    try:
        # smaller test params for generation tests
        video_diffusion_config.num_inference_steps = 10
        video_diffusion_config.num_frames = 8
        video_diffusion_config.image_size = (128, 128)
        
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )

        # Generate watermarked video
        watermarked_frames = watermark.generate_watermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )

        # Validate output
        assert watermarked_frames is not None
        assert isinstance(watermarked_frames, list)
        assert len(watermarked_frames) > 0
        assert all(isinstance(frame, Image.Image) for frame in watermarked_frames)

        print(f"✓ {algorithm_name} generated {len(watermarked_frames)} watermarked frames")

    except NotImplementedError:
        pytest.skip(f"{algorithm_name} does not implement watermarked video generation")
    except Exception as e:
        pytest.fail(f"Failed to generate watermarked video with {algorithm_name}: {e}")


@pytest.mark.video
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_unwatermarked_generation(algorithm_name, video_diffusion_config, skip_generation):
    """Test unwatermarked video generation for each algorithm."""
    if skip_generation:
        pytest.skip("Generation tests skipped by --skip-generation flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )

        # Generate unwatermarked video
        unwatermarked_frames = watermark.generate_unwatermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )

        # Validate output
        assert unwatermarked_frames is not None
        assert isinstance(unwatermarked_frames, list)
        assert len(unwatermarked_frames) > 0
        assert all(isinstance(frame, Image.Image) for frame in unwatermarked_frames)

        print(f"✓ {algorithm_name} generated {len(unwatermarked_frames)} unwatermarked frames")

    except Exception as e:
        pytest.fail(f"Failed to generate unwatermarked video with {algorithm_name}: {e}")


@pytest.mark.video
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO])
def test_video_watermark_detection(algorithm_name, video_diffusion_config, skip_detection):
    """Test watermark detection in videos for each algorithm."""
    if skip_detection:
        pytest.skip("Detection tests skipped by --skip-detection flag")

    try:
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=video_diffusion_config
        )

        # Generate watermarked and unwatermarked videos
        watermarked_frames = watermark.generate_watermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )
        unwatermarked_frames = watermark.generate_unwatermarked_media(
            TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )

        # Detect watermark in watermarked video
        detection_result_wm = watermark.detect_watermark_in_media(
            watermarked_frames,
            prompt=TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )
        assert detection_result_wm is not None
        assert isinstance(detection_result_wm, dict)
        assert detection_result_wm['is_watermarked'] is True

        # Detect watermark in unwatermarked video
        detection_result_unwm = watermark.detect_watermark_in_media(
            unwatermarked_frames,
            prompt=TEST_PROMPT_VIDEO,
            num_frames=NUM_FRAMES
        )
        assert detection_result_unwm is not None
        assert isinstance(detection_result_unwm, dict)
        assert detection_result_unwm['is_watermarked'] is False

        print(f"✓ {algorithm_name} detection results:")
        print(f"  Watermarked: {detection_result_wm}")
        print(f"  Unwatermarked: {detection_result_unwm}")

    except NotImplementedError:
        pytest.skip(f"{algorithm_name} does not implement watermark detection")
    except Exception as e:
        pytest.fail(f"Failed to detect watermark with {algorithm_name}: {e}")


# ============================================================================
# Test Cases - Algorithm Compatibility
# ============================================================================

def test_algorithm_list():
    """Test that all algorithms are properly registered."""
    image_algorithms = AutoWatermark.list_supported_algorithms(PIPELINE_TYPE_IMAGE)
    video_algorithms = AutoWatermark.list_supported_algorithms(PIPELINE_TYPE_TEXT_TO_VIDEO)

    assert len(image_algorithms) > 0, "No image algorithms found"
    assert len(video_algorithms) > 0, "No video algorithms found"

    print(f"Image algorithms: {image_algorithms}")
    print(f"Video algorithms: {video_algorithms}")


def test_invalid_algorithm():
    """Test that invalid algorithm names raise appropriate errors."""
    with pytest.raises(ValueError, match="Invalid algorithm name"):
        AutoWatermark.load("InvalidAlgorithm", diffusion_config=None)


# ============================================================================
# Test Cases - Inversion Modules
# ============================================================================

@pytest.mark.inversion
@pytest.mark.parametrize("inversion_type", ["ddim", "exact"])
def test_inversion_4d_image_input(inversion_type, device, image_pipeline):
    """Test inversion modules with 4D image input (batch_size, channels, height, width)."""
    import torch
    from markdiffusion.inversions import DDIMInversion, ExactInversion

    pipe, scheduler = image_pipeline

    # Create inversion instance
    if inversion_type == "ddim":
        inversion = DDIMInversion(scheduler=scheduler, unet=pipe.unet, device=device)
    else:  # exact
        inversion = ExactInversion(scheduler=scheduler, unet=pipe.unet, device=device)

    # Create 4D test input: (batch_size, channels, height, width)
    batch_size = 1
    channels = 4  # latent space channels
    height = 64   # latent space height (512 / 8)
    width = 64    # latent space width (512 / 8)

    latents_input = torch.randn(batch_size, channels, height, width).to(device)

    # Get correct text embedding dimension from the model
    # Different SD versions use different text encoders (CLIP: 768, OpenCLIP: 1024)
    text_encoder = pipe.text_encoder
    with torch.no_grad():
        # Use a dummy prompt to get properly formatted embeddings
        text_inputs = pipe.tokenizer(
            "a test prompt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    try:
        # Test forward diffusion (image to noise)
        intermediate_latents = inversion.forward_diffusion(
            text_embeddings=text_embeddings,
            latents=latents_input,
            num_inference_steps=10,  # Use fewer steps for testing
            guidance_scale=1.0
        )

        # Validate output
        assert intermediate_latents is not None
        assert isinstance(intermediate_latents, list)
        assert len(intermediate_latents) > 0

        # Get final inverted latent (Z_T)
        z_t = intermediate_latents[-1]
        assert z_t.shape == latents_input.shape

        print(f"✓ {inversion_type} inversion for 4D image input successful")
        print(f"  Input shape: {latents_input.shape}")
        print(f"  Output Z_T shape: {z_t.shape}")
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        print(f"  Number of intermediate steps: {len(intermediate_latents)}")

    except Exception as e:
        pytest.fail(f"Failed to invert 4D image with {inversion_type}: {e}")


@pytest.mark.inversion
@pytest.mark.slow
@pytest.mark.parametrize("inversion_type", ["ddim", "exact"])
def test_inversion_5d_video_input(inversion_type, device, video_pipeline):
    """Test inversion modules with 5D video input (batch_size, num_frames, channels, height, width)."""
    import torch
    from markdiffusion.inversions import DDIMInversion, ExactInversion

    pipe, scheduler = video_pipeline

    # Create inversion instance
    if inversion_type == "ddim":
        inversion = DDIMInversion(scheduler=scheduler, unet=pipe.unet, device=device)
    else:  # exact
        inversion = ExactInversion(scheduler=scheduler, unet=pipe.unet, device=device)

    # Create 5D test input: (batch_size, num_frames, channels, height, width)
    batch_size = 1
    num_frames = 8   # number of video frames
    channels = 4     # latent space channels
    height = 64      # latent space height
    width = 64       # latent space width

    # Reshape to 5D for video: (batch_size, num_frames, channels, height, width)
    latents_input = torch.randn(batch_size, num_frames, channels, height, width).to(device)

    # Get correct text embeddings from the model
    text_encoder = pipe.text_encoder
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            "a test video prompt",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    try:
        # Test forward diffusion (video frames to noise)
        intermediate_latents = inversion.forward_diffusion(
            text_embeddings=text_embeddings,
            latents=latents_input.to(pipe.dtype),
            num_inference_steps=10,  # Use fewer steps for testing
            guidance_scale=1.0
        )

        # Validate output
        assert intermediate_latents is not None
        assert isinstance(intermediate_latents, list)
        assert len(intermediate_latents) > 0

        # Get final inverted latent (Z_T)
        z_t = intermediate_latents[-1]
        assert z_t.shape == latents_input.shape

        print(f"✓ {inversion_type} inversion for 5D video input successful")
        print(f"  Input shape: {latents_input.shape}")
        print(f"  Output Z_T shape: {z_t.shape}")
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        print(f"  Number of intermediate steps: {len(intermediate_latents)}")

    except Exception as e:
        pytest.fail(f"Failed to invert 5D video with {inversion_type}: {e}")


@pytest.mark.inversion
@pytest.mark.parametrize("inversion_type", ["ddim", "exact"])
def test_inversion_reconstruction_accuracy(device, image_pipeline, inversion_type):
    """Test that inversion can accurately reconstruct the latent vector."""
    import torch
    from markdiffusion.inversions import DDIMInversion, ExactInversion

    pipe, scheduler = image_pipeline
    if inversion_type == "ddim":
        inversion = DDIMInversion(scheduler=scheduler, unet=pipe.unet, device=device)
    else:  # exact
        inversion = ExactInversion(scheduler=scheduler, unet=pipe.unet, device=device)

    # Create test input
    latents_input = torch.randn(1, 4, 64, 64).to(device)

    # Get correct text embeddings from the model
    text_encoder = pipe.text_encoder
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            "a test prompt for reconstruction",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    try:
        # Forward diffusion: x_0 -> x_T
        forward_result = inversion.forward_diffusion(
            text_embeddings=text_embeddings,
            latents=latents_input,
            num_inference_steps=10,
            guidance_scale=1.0
        )

        z_t = forward_result[-1]

        # Backward diffusion: x_T -> x_0
        backward_result = inversion.backward_diffusion(
            text_embeddings=text_embeddings,
            latents=z_t,
            num_inference_steps=10,
            guidance_scale=1.0,
            reverse_process=False
        )

        reconstructed = backward_result[-1]

        # Calculate reconstruction error
        mse = torch.nn.functional.mse_loss(reconstructed, latents_input)

        print(f"✓ Inversion reconstruction test completed")
        print(f"  MSE between original and reconstructed: {mse.item():.6f}")
        print(f"  Original shape: {latents_input.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Text embeddings shape: {text_embeddings.shape}")

        # The reconstruction should be reasonably close
        # Note: DDIM is not perfectly reversible, so we expect some error
        assert mse.item() < 1.0, f"Reconstruction error too high: {mse.item()}"

    except Exception as e:
        pytest.fail(f"Failed reconstruction accuracy test: {e}")


# ============================================================================
# Test Cases - Visualization
# ============================================================================

def _get_visualizer_methods(visualizer, is_base_method=False):
    """Automatically discover methods from a visualizer instance.

    Args:
        visualizer: The visualizer instance to inspect
        is_base_method: If True, return base class methods; if False, return subclass-specific methods

    Returns:
        List of method names to test
    """
    import inspect

    cls = visualizer.__class__

    # Collect all parent class methods
    parent_methods = set()
    for base in cls.__mro__[1:]:  # Exclude cls itself
        for name, member in inspect.getmembers(base, inspect.isroutine):
            parent_methods.add(name)

    # Get all methods from the instance
    all_methods = [name for name, m in inspect.getmembers(visualizer, inspect.isroutine)]

    if is_base_method:
        # Return base class methods (excluding _ prefixed and visualize)
        filtered_methods = [
            m for m in all_methods
            if m in parent_methods
            and not m.startswith("_")
            and m != "visualize"
        ]
    else:
        # Return subclass-specific methods (excluding _ prefixed and visualize)
        filtered_methods = [
            m for m in all_methods
            if m not in parent_methods
            and not m.startswith("_")
            and m != "visualize"
        ]

    return filtered_methods


@pytest.mark.visualization
@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name",
                        list(PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_IMAGE]) +
                        list(PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO]))
def test_watermark_visualization(algorithm_name, image_diffusion_config, video_diffusion_config, tmp_path):
    """Unified test for watermark visualization of all algorithms.

    This test:
    1. Generates a watermarked image/video using the actual watermark algorithm
    2. Tests all base class visualization methods
    3. Tests all subclass-specific visualization methods
    4. Saves sample visualizations
    """
    from markdiffusion.visualize.auto_visualization import AutoVisualizer, VISUALIZATION_DATA_MAPPING
    from markdiffusion.visualize.data_for_visualization import DataForVisualization
    import matplotlib.pyplot as plt

    # Skip if visualization not supported for this algorithm
    if algorithm_name not in VISUALIZATION_DATA_MAPPING:
        pytest.skip(f"{algorithm_name} does not have visualization support")

    # Determine if this is a video or image algorithm
    is_video = algorithm_name in PIPELINE_SUPPORTED_WATERMARKS[PIPELINE_TYPE_TEXT_TO_VIDEO]
    diffusion_config = video_diffusion_config if is_video else image_diffusion_config
    test_prompt = TEST_PROMPT_VIDEO if is_video else TEST_PROMPT_IMAGE

    if is_video:
        diffusion_config.num_inference_steps = 10
        diffusion_config.num_frames = 8
        diffusion_config.image_size = (128, 128)
        diffusion_config.torch_dtype = torch.float32

    try:
        # Step 1: Load watermark algorithm
        watermark = AutoWatermark.load(
            algorithm_name,
            algorithm_config=f'markdiffusion/config/{algorithm_name}.json',
            diffusion_config=diffusion_config
        )

        if is_video:
            pipe = watermark.config.pipe
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe.to(device=device)
            for name in ["unet", "text_encoder", "vae"]:
                m = getattr(pipe, name, None)
                if m is not None:
                    m.to(device=device, dtype=torch.float32)

        # Step 2: Generate watermarked media
        watermarked_media = watermark.generate_watermarked_media(test_prompt)

        # Step 3: Get visualization data from the watermark instance
        if not hasattr(watermark, 'get_data_for_visualize'):
            pytest.skip(f"{algorithm_name} does not implement get_data_for_visualize()")

        vis_data = watermark.get_data_for_visualize(watermarked_media)

        # Validate visualization data
        assert vis_data is not None
        assert isinstance(vis_data, DataForVisualization)

        # Step 4: Load visualizer
        visualizer = AutoVisualizer.load(
            algorithm_name=algorithm_name,
            data_for_visualization=vis_data
        )

        assert visualizer is not None

        # Step 5: Test base class methods
        base_methods = _get_visualizer_methods(visualizer, is_base_method=True)
        base_tested = []
        base_failed = []

        print(f"\n{algorithm_name} - Testing base class methods:")
        print(f"  Found {len(base_methods)} base methods: {', '.join(base_methods[:5])}...")

        for method_name in base_methods:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            try:
                method = getattr(visualizer, method_name)

                # Determine appropriate parameters based on method signature
                import inspect
                sig = inspect.signature(method)
                params = {}

                # Add ax parameter if needed
                if 'ax' in sig.parameters:
                    params['ax'] = ax

                # For video methods, add frame parameter if available
                if is_video and 'frame' in sig.parameters:
                    params['frame'] = 0

                # Call the method
                method(**params)
                base_tested.append(method_name)
                plt.close(fig)
            except Exception as e:
                plt.close(fig)
                base_failed.append(f"{method_name}: {str(e)[:50]}")

        print(f"  ✓ Successfully tested {len(base_tested)}/{len(base_methods)} base methods")
        if base_failed:
            print(f"  ⚠ Failed methods: {base_failed[:3]}...")

        # Step 6: Test subclass-specific methods
        subclass_methods = _get_visualizer_methods(visualizer, is_base_method=False)
        subclass_tested = []
        subclass_failed = []

        print(f"\n{algorithm_name} - Testing subclass-specific methods:")
        print(f"  Found {len(subclass_methods)} subclass methods: {', '.join(subclass_methods[:5])}...")

        for method_name in subclass_methods:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            try:
                method = getattr(visualizer, method_name)

                # Determine appropriate parameters based on method signature
                import inspect
                sig = inspect.signature(method)
                params = {}

                # Add ax parameter if needed
                if 'ax' in sig.parameters:
                    params['ax'] = ax

                # For video methods, add frame parameter if available
                if is_video and 'frame' in sig.parameters:
                    params['frame'] = 0

                # Call the method
                method(**params)
                
                # add `channel` parameter if needed
                if 'channel' in sig.parameters:
                    params['channel'] = 0
                    method(**params)
                
                subclass_tested.append(method_name)
                plt.close(fig)
            except Exception as e:
                plt.close(fig)
                subclass_failed.append(f"{method_name}: {str(e)[:50]}")

        print(f"  ✓ Successfully tested {len(subclass_tested)}/{len(subclass_methods)} subclass methods")
        if subclass_tested:
            print(f"    Tested: {', '.join(subclass_tested[:5])}...")
        if subclass_failed:
            print(f"  ⚠ Failed methods: {subclass_failed[:3]}...")

        print(f"\n✓ {algorithm_name} visualization test completed successfully")
        print(f"  Base methods tested: {len(base_tested)}/{len(base_methods)}")
        print(f"  Subclass methods tested: {len(subclass_tested)}/{len(subclass_methods)}")

    except NotImplementedError as e:
        pytest.skip(f"{algorithm_name} visualization not fully implemented: {e}")
    except Exception as e:
        pytest.fail(f"Failed to test visualization for {algorithm_name}: {e}")


# ============================================================================
# Test Cases - BaseVisualizer Unit Tests
# ============================================================================

class TestBaseVisualizerMethods:
    """Unit tests for BaseVisualizer's visualize(), _draw_single_image, and _draw_video_frames methods."""

    @pytest.fixture
    def mock_data_for_image(self):
        """Create mock DataForVisualization for image tests."""
        import torch
        from unittest.mock import MagicMock
        
        mock_data = MagicMock()
        mock_data.algorithm_name = "TestAlgorithm"
        
        # Create a test image (PIL Image)
        test_image = Image.new("RGB", (64, 64), color=(128, 64, 192))
        mock_data.image = test_image
        
        # Create test latents for image: [B, C, H, W]
        mock_data.orig_watermarked_latents = torch.randn(1, 4, 8, 8)
        mock_data.reversed_latents = [torch.randn(1, 4, 8, 8) for _ in range(5)]
        
        return mock_data

    @pytest.fixture
    def mock_data_for_video(self):
        """Create mock DataForVisualization for video tests."""
        import torch
        from unittest.mock import MagicMock
        import numpy as np
        
        mock_data = MagicMock()
        mock_data.algorithm_name = "TestVideoAlgorithm"
        
        # Create test video frames (list of PIL Images)
        video_frames = [Image.new("RGB", (64, 64), color=(i * 30, 100, 200)) for i in range(8)]
        mock_data.video_frames = video_frames
        mock_data.image = None
        
        # Create test latents for video: [B, C, F, H, W]
        mock_data.orig_watermarked_latents = torch.randn(1, 4, 8, 8, 8)
        mock_data.reversed_latents = [torch.randn(1, 4, 8, 8, 8) for _ in range(5)]
        
        return mock_data

    @pytest.fixture
    def image_visualizer(self, mock_data_for_image):
        """Create a concrete visualizer for image tests."""
        from markdiffusion.visualize.base import BaseVisualizer
        
        class ConcreteImageVisualizer(BaseVisualizer):
            """Concrete implementation for testing."""
            pass
        
        return ConcreteImageVisualizer(
            data_for_visualization=mock_data_for_image,
            dpi=100,
            is_video=False
        )

    @pytest.fixture
    def video_visualizer(self, mock_data_for_video):
        """Create a concrete visualizer for video tests."""
        from markdiffusion.visualize.base import BaseVisualizer
        
        class ConcreteVideoVisualizer(BaseVisualizer):
            """Concrete implementation for testing."""
            pass
        
        return ConcreteVideoVisualizer(
            data_for_visualization=mock_data_for_video,
            dpi=100,
            is_video=True
        )

    # -------------------------------------------------------------------------
    # Tests for _draw_single_image
    # -------------------------------------------------------------------------

    @pytest.mark.visualization
    def test_draw_single_image_with_pil_image(self, image_visualizer):
        """Test _draw_single_image with PIL Image input."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        result = image_visualizer._draw_single_image(
            title="Test Single Image",
            ax=ax
        )
        
        assert result is not None
        assert result == ax
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_single_image_with_tensor(self, mock_data_for_image):
        """Test _draw_single_image with tensor input."""
        import torch
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer
        
        class ConcreteVisualizer(BaseVisualizer):
            pass
        
        # Replace image with tensor
        mock_data_for_image.image = torch.rand(1, 3, 64, 64)  # [B, C, H, W]
        
        visualizer = ConcreteVisualizer(
            data_for_visualization=mock_data_for_image,
            dpi=100,
            is_video=False
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        result = visualizer._draw_single_image(title="Tensor Image", ax=ax)
        
        assert result is not None
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_single_image_with_3d_tensor(self, mock_data_for_image):
        """Test _draw_single_image with 3D tensor input [C, H, W]."""
        import torch
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer
        
        class ConcreteVisualizer(BaseVisualizer):
            pass
        
        # Replace image with 3D tensor
        mock_data_for_image.image = torch.rand(3, 64, 64)  # [C, H, W]
        
        visualizer = ConcreteVisualizer(
            data_for_visualization=mock_data_for_image,
            dpi=100,
            is_video=False
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        result = visualizer._draw_single_image(title="3D Tensor Image", ax=ax)
        
        assert result is not None
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_single_image_with_normalized_tensor(self, mock_data_for_image):
        """Test _draw_single_image with tensor in [-1, 1] range."""
        import torch
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer
        
        class ConcreteVisualizer(BaseVisualizer):
            pass
        
        # Replace image with tensor in [-1, 1] range
        mock_data_for_image.image = torch.rand(1, 3, 64, 64) * 2 - 1  # [-1, 1]
        
        visualizer = ConcreteVisualizer(
            data_for_visualization=mock_data_for_image,
            dpi=100,
            is_video=False
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        result = visualizer._draw_single_image(title="Normalized Tensor", ax=ax)
        
        assert result is not None
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_single_image_empty_title(self, image_visualizer):
        """Test _draw_single_image with empty title."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        result = image_visualizer._draw_single_image(title="", ax=ax)
        
        assert result is not None
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Tests for _draw_video_frames
    # -------------------------------------------------------------------------

    @pytest.mark.visualization
    def test_draw_video_frames_basic(self, video_visualizer):
        """Test _draw_video_frames with basic parameters."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        result = video_visualizer._draw_video_frames(
            title="Test Video Frames",
            num_frames=4,
            ax=ax
        )
        
        assert result is not None
        assert result == ax
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_video_frames_single_frame(self, video_visualizer):
        """Test _draw_video_frames with single frame."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        result = video_visualizer._draw_video_frames(
            title="Single Frame",
            num_frames=1,
            ax=ax
        )
        
        assert result is not None
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_video_frames_all_frames(self, video_visualizer):
        """Test _draw_video_frames requesting more frames than available."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Request 16 frames but only 8 are available
        result = video_visualizer._draw_video_frames(
            title="All Frames",
            num_frames=16,
            ax=ax
        )
        
        assert result is not None
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_video_frames_with_numpy_frames(self, mock_data_for_video):
        """Test _draw_video_frames with numpy array frames."""
        import numpy as np
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer
        
        class ConcreteVisualizer(BaseVisualizer):
            pass
        
        # Replace frames with numpy arrays
        mock_data_for_video.video_frames = [
            np.random.rand(64, 64, 3).astype(np.float32) for _ in range(8)
        ]
        
        visualizer = ConcreteVisualizer(
            data_for_visualization=mock_data_for_video,
            dpi=100,
            is_video=True
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        result = visualizer._draw_video_frames(title="Numpy Frames", num_frames=4, ax=ax)
        
        assert result is not None
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_video_frames_with_tensor_frames(self, mock_data_for_video):
        """Test _draw_video_frames with tensor frames."""
        import torch
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer
        
        class ConcreteVisualizer(BaseVisualizer):
            pass
        
        # Replace frames with tensors [C, H, W]
        mock_data_for_video.video_frames = [
            torch.rand(3, 64, 64) for _ in range(8)
        ]
        
        visualizer = ConcreteVisualizer(
            data_for_visualization=mock_data_for_video,
            dpi=100,
            is_video=True
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        result = visualizer._draw_video_frames(title="Tensor Frames", num_frames=4, ax=ax)
        
        assert result is not None
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_video_frames_no_frames_raises_error(self, mock_data_for_video):
        """Test _draw_video_frames raises error when no frames available."""
        import matplotlib.pyplot as plt
        from markdiffusion.visualize.base import BaseVisualizer
        
        class ConcreteVisualizer(BaseVisualizer):
            pass
        
        # Remove video_frames
        mock_data_for_video.video_frames = None
        
        visualizer = ConcreteVisualizer(
            data_for_visualization=mock_data_for_video,
            dpi=100,
            is_video=True
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        with pytest.raises(ValueError, match="No video frames available"):
            visualizer._draw_video_frames(title="No Frames", num_frames=4, ax=ax)
        
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_video_frames_empty_title(self, video_visualizer):
        """Test _draw_video_frames with empty title."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        result = video_visualizer._draw_video_frames(title="", num_frames=4, ax=ax)
        
        assert result is not None
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Tests for draw_watermarked_image (dispatches to _draw_single_image or _draw_video_frames)
    # -------------------------------------------------------------------------

    @pytest.mark.visualization
    def test_draw_watermarked_image_dispatches_to_single_image(self, image_visualizer):
        """Test draw_watermarked_image dispatches to _draw_single_image for images."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        result = image_visualizer.draw_watermarked_image(
            title="Watermarked Image",
            ax=ax
        )
        
        assert result is not None
        plt.close(fig)

    @pytest.mark.visualization
    def test_draw_watermarked_image_dispatches_to_video_frames(self, video_visualizer):
        """Test draw_watermarked_image dispatches to _draw_video_frames for videos."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        result = video_visualizer.draw_watermarked_image(
            title="Watermarked Video",
            num_frames=4,
            ax=ax
        )
        
        assert result is not None
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Tests for visualize() method
    # -------------------------------------------------------------------------

    @pytest.mark.visualization
    def test_visualize_single_method(self, image_visualizer, tmp_path):
        """Test visualize() with single method."""
        result = image_visualizer.visualize(
            rows=1,
            cols=1,
            methods=["draw_watermarked_image"],
            figsize=(5, 5)
        )
        
        assert result is not None
        import matplotlib.pyplot as plt
        plt.close(result)

    @pytest.mark.visualization
    def test_visualize_multiple_methods(self, image_visualizer, tmp_path):
        """Test visualize() with multiple methods in grid layout."""
        result = image_visualizer.visualize(
            rows=2,
            cols=2,
            methods=[
                "draw_watermarked_image",
                "draw_orig_latents",
                "draw_orig_latents_fft",
                "draw_inverted_latents"
            ],
            figsize=(10, 10),
            method_kwargs=[
                {"title": "Image"},
                {"channel": 0, "title": "Original Latents Ch0"},
                {"channel": 0, "title": "FFT Ch0"},
                {"channel": 0, "title": "Inverted Latents Ch0"}
            ]
        )
        
        assert result is not None
        import matplotlib.pyplot as plt
        plt.close(result)

    @pytest.mark.visualization
    def test_visualize_saves_to_file(self, image_visualizer, tmp_path):
        """Test visualize() saves figure to file."""
        save_path = str(tmp_path / "test_visualization.png")
        
        result = image_visualizer.visualize(
            rows=1,
            cols=1,
            methods=["draw_watermarked_image"],
            save_path=save_path
        )
        
        assert result is not None
        assert (tmp_path / "test_visualization.png").exists()
        
        import matplotlib.pyplot as plt
        plt.close(result)

    @pytest.mark.visualization
    def test_visualize_with_default_figsize(self, image_visualizer):
        """Test visualize() with default figsize calculation."""
        result = image_visualizer.visualize(
            rows=2,
            cols=2,
            methods=[
                "draw_watermarked_image",
                "draw_orig_latents",
                "draw_orig_latents_fft",
                "draw_inverted_latents"
            ],
            method_kwargs=[
                {},
                {"channel": 0},
                {"channel": 0},
                {"channel": 0}
            ]
        )
        
        assert result is not None
        import matplotlib.pyplot as plt
        plt.close(result)

    @pytest.mark.visualization
    def test_visualize_mismatched_layout_raises_error(self, image_visualizer):
        """Test visualize() raises error when methods don't match layout."""
        with pytest.raises(ValueError, match="not compatible with the layout"):
            image_visualizer.visualize(
                rows=2,
                cols=2,
                methods=["draw_watermarked_image"]  # Only 1 method for 2x2 layout
            )

    @pytest.mark.visualization
    def test_visualize_invalid_method_raises_error(self, image_visualizer):
        """Test visualize() raises error for invalid method name."""
        with pytest.raises(ValueError, match="Method .* not found"):
            image_visualizer.visualize(
                rows=1,
                cols=1,
                methods=["nonexistent_method"]
            )

    @pytest.mark.visualization
    def test_visualize_video_with_frame_selection(self, video_visualizer, tmp_path):
        """Test visualize() for video with frame parameter."""
        result = video_visualizer.visualize(
            rows=1,
            cols=2,
            methods=[
                "draw_watermarked_image",
                "draw_orig_latents"
            ],
            method_kwargs=[
                {"num_frames": 4},
                {"channel": 0, "frame": 0}
            ]
        )
        
        assert result is not None
        import matplotlib.pyplot as plt
        plt.close(result)

    @pytest.mark.visualization
    def test_visualize_single_row(self, image_visualizer):
        """Test visualize() with single row layout."""
        result = image_visualizer.visualize(
            rows=1,
            cols=3,
            methods=[
                "draw_watermarked_image",
                "draw_orig_latents",
                "draw_orig_latents_fft"
            ],
            method_kwargs=[
                {},
                {"channel": 0},
                {"channel": 0}
            ]
        )
        
        assert result is not None
        import matplotlib.pyplot as plt
        plt.close(result)

    @pytest.mark.visualization
    def test_visualize_single_column(self, image_visualizer):
        """Test visualize() with single column layout."""
        result = image_visualizer.visualize(
            rows=3,
            cols=1,
            methods=[
                "draw_watermarked_image",
                "draw_orig_latents",
                "draw_orig_latents_fft"
            ],
            method_kwargs=[
                {},
                {"channel": 0},
                {"channel": 0}
            ]
        )
        
        assert result is not None
        import matplotlib.pyplot as plt
        plt.close(result)
