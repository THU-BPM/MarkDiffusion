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

"""
Unit tests for image editor classes in MarkDiffusion.

Tests cover:
- ImageEditor: Base class
- JPEGCompression: JPEG quality compression
- Rotation: Image rotation
- CrSc: Crop and scale
- GaussianBlurring: Gaussian blur filter
- GaussianNoise: Add Gaussian noise
- Brightness: Brightness adjustment
- Mask: Random rectangular masks
- Overlay: Random stroke overlays
- AdaptiveNoiseInjection: Adaptive noise based on image features
"""

import pytest
import numpy as np
from PIL import Image
import os
import tempfile

from markdiffusion.evaluation.tools.image_editor import (
    ImageEditor,
    JPEGCompression,
    Rotation,
    CrSc,
    GaussianBlurring,
    GaussianNoise,
    Brightness,
    Mask,
    Overlay,
    AdaptiveNoiseInjection,
    DiffusionPurification,
    NeuralCodecCompression,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB test image."""
    return Image.new("RGB", (256, 256), color="red")


@pytest.fixture
def sample_gradient_image():
    """Create a gradient image for testing edge detection."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        arr[:, i, :] = i  # Horizontal gradient
    return Image.fromarray(arr)


@pytest.fixture
def sample_complex_image():
    """Create a more complex image with varied content."""
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_dark_image():
    """Create a dark image for testing adaptive noise."""
    arr = np.full((256, 256, 3), 30, dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_bright_image():
    """Create a bright image for testing."""
    arr = np.full((256, 256, 3), 220, dtype=np.uint8)
    return Image.fromarray(arr)


# ============================================================================
# Tests for ImageEditor Base Class
# ============================================================================

class TestImageEditor:
    """Tests for ImageEditor base class."""

    def test_initialization(self):
        """Test base class can be instantiated."""
        editor = ImageEditor()
        assert editor is not None

    def test_edit_method_exists(self):
        """Test edit method exists."""
        editor = ImageEditor()
        assert hasattr(editor, 'edit')

    def test_edit_returns_none_by_default(self, sample_rgb_image):
        """Test base edit method returns None."""
        editor = ImageEditor()
        result = editor.edit(sample_rgb_image)
        assert result is None


# ============================================================================
# Tests for JPEGCompression
# ============================================================================

class TestJPEGCompression:
    """Tests for JPEGCompression editor."""

    def test_default_quality(self):
        """Test default quality is 95."""
        editor = JPEGCompression()
        assert editor.quality == 95

    def test_custom_quality(self):
        """Test custom quality setting."""
        editor = JPEGCompression(quality=50)
        assert editor.quality == 50

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = JPEGCompression(quality=75)
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_size(self, sample_rgb_image):
        """Test edit preserves image size."""
        editor = JPEGCompression(quality=75)
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_low_quality_changes_image(self, sample_complex_image):
        """Test low quality compression changes the image."""
        editor = JPEGCompression(quality=10)
        result = editor.edit(sample_complex_image)

        # Convert to arrays and compare
        original_arr = np.array(sample_complex_image)
        result_arr = np.array(result)

        # Should be different due to compression artifacts
        assert not np.array_equal(original_arr, result_arr)

    def test_high_quality_preserves_more(self, sample_complex_image):
        """Test high quality compression preserves more detail."""
        low_editor = JPEGCompression(quality=10)
        high_editor = JPEGCompression(quality=95)

        original_arr = np.array(sample_complex_image)
        low_result = np.array(low_editor.edit(sample_complex_image))
        high_result = np.array(high_editor.edit(sample_complex_image))

        low_diff = np.mean(np.abs(original_arr.astype(float) - low_result.astype(float)))
        high_diff = np.mean(np.abs(original_arr.astype(float) - high_result.astype(float)))

        # High quality should have less difference
        assert high_diff < low_diff

    def test_temp_file_cleanup(self, sample_rgb_image):
        """Test temporary file is cleaned up."""
        editor = JPEGCompression(quality=75)
        editor.edit(sample_rgb_image)
        assert not os.path.exists("temp.jpg")

    def test_various_quality_levels(self, sample_rgb_image):
        """Test various quality levels."""
        for quality in [1, 25, 50, 75, 100]:
            editor = JPEGCompression(quality=quality)
            result = editor.edit(sample_rgb_image)
            assert isinstance(result, Image.Image)


# ============================================================================
# Tests for Rotation
# ============================================================================

class TestRotation:
    """Tests for Rotation editor."""

    def test_default_parameters(self):
        """Test default rotation parameters."""
        editor = Rotation()
        assert editor.angle == 30
        assert editor.expand is False

    def test_custom_angle(self):
        """Test custom rotation angle."""
        editor = Rotation(angle=45)
        assert editor.angle == 45

    def test_custom_expand(self):
        """Test custom expand parameter."""
        editor = Rotation(angle=30, expand=True)
        assert editor.expand is True

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = Rotation(angle=45)
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_no_expand_preserves_size(self, sample_rgb_image):
        """Test rotation without expand preserves size."""
        editor = Rotation(angle=45, expand=False)
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_expand_changes_size(self, sample_rgb_image):
        """Test rotation with expand may change size."""
        editor = Rotation(angle=45, expand=True)
        result = editor.edit(sample_rgb_image)
        # Rotated image with expand should be larger
        orig_w, orig_h = sample_rgb_image.size
        new_w, new_h = result.size
        assert new_w >= orig_w or new_h >= orig_h

    def test_zero_rotation(self, sample_rgb_image):
        """Test zero rotation doesn't change image significantly."""
        editor = Rotation(angle=0)
        result = editor.edit(sample_rgb_image)

        original_arr = np.array(sample_rgb_image)
        result_arr = np.array(result)

        np.testing.assert_array_equal(original_arr, result_arr)

    def test_360_rotation(self, sample_rgb_image):
        """Test 360 degree rotation returns similar image."""
        editor = Rotation(angle=360)
        result = editor.edit(sample_rgb_image)

        original_arr = np.array(sample_rgb_image)
        result_arr = np.array(result)

        np.testing.assert_array_equal(original_arr, result_arr)

    def test_negative_angle(self, sample_rgb_image):
        """Test negative rotation angle."""
        editor = Rotation(angle=-45)
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)


# ============================================================================
# Tests for CrSc (Crop and Scale)
# ============================================================================

class TestCrSc:
    """Tests for CrSc (Crop and Scale) editor."""

    def test_default_crop_ratio(self):
        """Test default crop ratio is 0.8."""
        editor = CrSc()
        assert editor.crop_ratio == 0.8

    def test_custom_crop_ratio(self):
        """Test custom crop ratio."""
        editor = CrSc(crop_ratio=0.5)
        assert editor.crop_ratio == 0.5

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = CrSc(crop_ratio=0.8)
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_size(self, sample_rgb_image):
        """Test edit preserves original size after scaling back."""
        editor = CrSc(crop_ratio=0.8)
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_center_crop(self, sample_gradient_image):
        """Test that crop is centered."""
        editor = CrSc(crop_ratio=0.5)
        result = editor.edit(sample_gradient_image)

        # Result should be scaled back, but content should be from center
        assert result.size == sample_gradient_image.size

    def test_various_crop_ratios(self, sample_rgb_image):
        """Test various crop ratios."""
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            editor = CrSc(crop_ratio=ratio)
            result = editor.edit(sample_rgb_image)
            assert result.size == sample_rgb_image.size

    def test_crop_ratio_one(self, sample_rgb_image):
        """Test crop ratio of 1.0 (no crop)."""
        editor = CrSc(crop_ratio=1.0)
        result = editor.edit(sample_rgb_image)

        # Should be very similar to original
        original_arr = np.array(sample_rgb_image)
        result_arr = np.array(result)

        # Allow small differences due to resize
        assert np.allclose(original_arr, result_arr, atol=1)

    def test_default_position_is_center(self):
        """Default position is 'center' (backward compatible)."""
        editor = CrSc()
        assert editor.position == "center"

    def test_position_random_produces_valid_image(self, sample_rgb_image):
        """position='random' returns an image of original size."""
        editor = CrSc(crop_ratio=0.5, position="random")
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)
        assert result.size == sample_rgb_image.size

    def test_position_explicit_topleft_differs_from_center(self, sample_gradient_image):
        """Explicit (0.0, 0.0) crop should differ from center crop on a non-uniform image."""
        center_result = np.array(CrSc(crop_ratio=0.5, position="center").edit(sample_gradient_image))
        topleft_result = np.array(CrSc(crop_ratio=0.5, position=(0.0, 0.0)).edit(sample_gradient_image))
        assert not np.array_equal(center_result, topleft_result)

    def test_position_invalid_string_raises(self):
        with pytest.raises(ValueError):
            CrSc(position="bottomleft")

    def test_position_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CrSc(position=(1.5, 0.5))


# ============================================================================
# Tests for GaussianBlurring
# ============================================================================

class TestGaussianBlurring:
    """Tests for GaussianBlurring editor."""

    def test_default_radius(self):
        """Test default blur radius is 2."""
        editor = GaussianBlurring()
        assert editor.radius == 2

    def test_custom_radius(self):
        """Test custom blur radius."""
        editor = GaussianBlurring(radius=5)
        assert editor.radius == 5

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = GaussianBlurring(radius=2)
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_size(self, sample_rgb_image):
        """Test edit preserves image size."""
        editor = GaussianBlurring(radius=2)
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_blur_reduces_variance(self, sample_complex_image):
        """Test blur reduces local variance in complex image."""
        editor = GaussianBlurring(radius=5)
        result = editor.edit(sample_complex_image)

        original_arr = np.array(sample_complex_image).astype(float)
        result_arr = np.array(result).astype(float)

        # Blurred image should have lower local variance
        # Compare variance of small patches
        orig_var = np.var(original_arr)
        result_var = np.var(result_arr)

        assert result_var < orig_var

    def test_larger_radius_more_blur(self, sample_complex_image):
        """Test larger radius produces more blur."""
        small_blur = GaussianBlurring(radius=1)
        large_blur = GaussianBlurring(radius=10)

        small_result = np.array(small_blur.edit(sample_complex_image))
        large_result = np.array(large_blur.edit(sample_complex_image))

        # Larger blur should have lower variance
        assert np.var(large_result) < np.var(small_result)


# ============================================================================
# Tests for GaussianNoise
# ============================================================================

class TestGaussianNoise:
    """Tests for GaussianNoise editor."""

    def test_default_sigma(self):
        """Test default sigma is 25.0."""
        editor = GaussianNoise()
        assert editor.sigma == 25.0

    def test_custom_sigma(self):
        """Test custom sigma."""
        editor = GaussianNoise(sigma=50.0)
        assert editor.sigma == 50.0

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = GaussianNoise(sigma=25.0)
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_size(self, sample_rgb_image):
        """Test edit preserves image size."""
        editor = GaussianNoise(sigma=25.0)
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_noise_changes_image(self, sample_rgb_image):
        """Test noise changes the image."""
        editor = GaussianNoise(sigma=25.0)
        result = editor.edit(sample_rgb_image)

        original_arr = np.array(sample_rgb_image)
        result_arr = np.array(result)

        assert not np.array_equal(original_arr, result_arr)

    def test_higher_sigma_more_noise(self, sample_rgb_image):
        """Test higher sigma produces more noise."""
        low_noise = GaussianNoise(sigma=10.0)
        high_noise = GaussianNoise(sigma=100.0)

        original_arr = np.array(sample_rgb_image).astype(float)
        low_result = np.array(low_noise.edit(sample_rgb_image)).astype(float)
        high_result = np.array(high_noise.edit(sample_rgb_image)).astype(float)

        low_diff = np.mean(np.abs(original_arr - low_result))
        high_diff = np.mean(np.abs(original_arr - high_result))

        assert high_diff > low_diff

    def test_output_clipped_to_valid_range(self, sample_rgb_image):
        """Test output values are in valid [0, 255] range."""
        editor = GaussianNoise(sigma=100.0)
        result = editor.edit(sample_rgb_image)
        result_arr = np.array(result)

        assert result_arr.min() >= 0
        assert result_arr.max() <= 255

    def test_zero_sigma_preserves_image(self, sample_rgb_image):
        """Test zero sigma doesn't add noise."""
        editor = GaussianNoise(sigma=0.0)
        result = editor.edit(sample_rgb_image)

        original_arr = np.array(sample_rgb_image)
        result_arr = np.array(result)

        np.testing.assert_array_equal(original_arr, result_arr)


# ============================================================================
# Tests for Brightness
# ============================================================================

class TestBrightness:
    """Tests for Brightness editor."""

    def test_default_factor(self):
        """Test default brightness factor is 1.2."""
        editor = Brightness()
        assert editor.factor == 1.2

    def test_custom_factor(self):
        """Test custom brightness factor."""
        editor = Brightness(factor=0.5)
        assert editor.factor == 0.5

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = Brightness(factor=1.5)
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_size(self, sample_rgb_image):
        """Test edit preserves image size."""
        editor = Brightness(factor=1.5)
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_factor_one_preserves_image(self, sample_rgb_image):
        """Test factor of 1.0 preserves image."""
        editor = Brightness(factor=1.0)
        result = editor.edit(sample_rgb_image)

        original_arr = np.array(sample_rgb_image)
        result_arr = np.array(result)

        np.testing.assert_array_equal(original_arr, result_arr)

    def test_higher_factor_increases_brightness(self, sample_dark_image):
        """Test higher factor increases brightness."""
        editor = Brightness(factor=2.0)
        result = editor.edit(sample_dark_image)

        original_mean = np.mean(np.array(sample_dark_image))
        result_mean = np.mean(np.array(result))

        assert result_mean > original_mean

    def test_lower_factor_decreases_brightness(self, sample_bright_image):
        """Test lower factor decreases brightness."""
        editor = Brightness(factor=0.5)
        result = editor.edit(sample_bright_image)

        original_mean = np.mean(np.array(sample_bright_image))
        result_mean = np.mean(np.array(result))

        assert result_mean < original_mean


# ============================================================================
# Tests for Mask
# ============================================================================

class TestMask:
    """Tests for Mask editor."""

    def test_default_parameters(self):
        """Test default mask parameters."""
        editor = Mask()
        assert editor.mask_ratio == 0.1
        assert editor.num_masks == 5

    def test_custom_parameters(self):
        """Test custom mask parameters."""
        editor = Mask(mask_ratio=0.2, num_masks=10)
        assert editor.mask_ratio == 0.2
        assert editor.num_masks == 10

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = Mask()
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_size(self, sample_rgb_image):
        """Test edit preserves image size."""
        editor = Mask()
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_masks_add_black_regions(self, sample_bright_image):
        """Test masks add black regions."""
        editor = Mask(num_masks=10)
        result = editor.edit(sample_bright_image)

        result_arr = np.array(result)

        # Should have some black pixels (all zeros)
        black_pixels = np.all(result_arr == 0, axis=2)
        assert np.any(black_pixels)

    def test_original_not_modified(self, sample_rgb_image):
        """Test original image is not modified."""
        original_arr = np.array(sample_rgb_image).copy()

        editor = Mask()
        editor.edit(sample_rgb_image)

        current_arr = np.array(sample_rgb_image)
        np.testing.assert_array_equal(original_arr, current_arr)


# ============================================================================
# Tests for Overlay
# ============================================================================

class TestOverlay:
    """Tests for Overlay editor."""

    def test_default_parameters(self):
        """Test default overlay parameters."""
        editor = Overlay()
        assert editor.num_strokes == 10
        assert editor.stroke_width == 5
        assert editor.stroke_type == 'random'

    def test_custom_parameters(self):
        """Test custom overlay parameters."""
        editor = Overlay(num_strokes=20, stroke_width=10, stroke_type='black')
        assert editor.num_strokes == 20
        assert editor.stroke_width == 10
        assert editor.stroke_type == 'black'

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = Overlay()
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_size(self, sample_rgb_image):
        """Test edit preserves image size."""
        editor = Overlay()
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_overlay_changes_image(self, sample_rgb_image):
        """Test overlay changes the image."""
        editor = Overlay(num_strokes=20)
        result = editor.edit(sample_rgb_image)

        original_arr = np.array(sample_rgb_image)
        result_arr = np.array(result)

        assert not np.array_equal(original_arr, result_arr)

    def test_black_stroke_type(self, sample_bright_image):
        """Test black stroke type adds black pixels."""
        editor = Overlay(num_strokes=20, stroke_width=10, stroke_type='black')
        result = editor.edit(sample_bright_image)

        result_arr = np.array(result)
        black_pixels = np.all(result_arr == 0, axis=2)
        assert np.any(black_pixels)

    def test_white_stroke_type(self, sample_dark_image):
        """Test white stroke type adds white pixels."""
        editor = Overlay(num_strokes=20, stroke_width=10, stroke_type='white')
        result = editor.edit(sample_dark_image)

        result_arr = np.array(result)
        white_pixels = np.all(result_arr == 255, axis=2)
        assert np.any(white_pixels)

    def test_original_not_modified(self, sample_rgb_image):
        """Test original image is not modified."""
        original_arr = np.array(sample_rgb_image).copy()

        editor = Overlay()
        editor.edit(sample_rgb_image)

        current_arr = np.array(sample_rgb_image)
        np.testing.assert_array_equal(original_arr, current_arr)


# ============================================================================
# Tests for AdaptiveNoiseInjection
# ============================================================================

class TestAdaptiveNoiseInjection:
    """Tests for AdaptiveNoiseInjection editor."""

    def test_default_parameters(self):
        """Test default parameters."""
        editor = AdaptiveNoiseInjection()
        assert editor.intensity == 0.5
        assert editor.auto_select is True

    def test_custom_parameters(self):
        """Test custom parameters."""
        editor = AdaptiveNoiseInjection(intensity=0.8, auto_select=False)
        assert editor.intensity == 0.8
        assert editor.auto_select is False

    def test_edit_returns_image(self, sample_rgb_image):
        """Test edit returns a PIL Image."""
        editor = AdaptiveNoiseInjection()
        result = editor.edit(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_edit_preserves_size(self, sample_rgb_image):
        """Test edit preserves image size."""
        editor = AdaptiveNoiseInjection()
        result = editor.edit(sample_rgb_image)
        assert result.size == sample_rgb_image.size

    def test_noise_changes_image(self, sample_rgb_image):
        """Test noise injection changes the image."""
        editor = AdaptiveNoiseInjection(intensity=0.5)
        result = editor.edit(sample_rgb_image)

        original_arr = np.array(sample_rgb_image)
        result_arr = np.array(result)

        assert not np.array_equal(original_arr, result_arr)

    def test_analyze_image_features(self, sample_complex_image):
        """Test _analyze_image_features returns expected keys."""
        editor = AdaptiveNoiseInjection()
        img_arr = np.array(sample_complex_image).astype(np.float32)

        features = editor._analyze_image_features(img_arr)

        assert 'brightness_mean' in features
        assert 'brightness_std' in features
        assert 'edge_density' in features
        assert 'texture_complexity' in features

    def test_select_noise_type_dark_image(self, sample_dark_image):
        """Test noise type selection for dark image."""
        editor = AdaptiveNoiseInjection()
        img_arr = np.array(sample_dark_image).astype(np.float32)

        features = editor._analyze_image_features(img_arr)
        noise_type = editor._select_noise_type(features)

        # Dark images should use gaussian noise
        assert noise_type == 'gaussian'

    def test_auto_select_false_uses_mixed_noise(self, sample_rgb_image):
        """Test auto_select=False uses mixed noise."""
        editor = AdaptiveNoiseInjection(auto_select=False)
        result = editor.edit(sample_rgb_image)

        assert isinstance(result, Image.Image)

    def test_add_gaussian_noise(self, sample_rgb_image):
        """Test _add_gaussian_noise method."""
        editor = AdaptiveNoiseInjection()
        img_arr = np.array(sample_rgb_image).astype(np.float32)

        noisy = editor._add_gaussian_noise(img_arr, sigma=25)

        assert noisy.shape == img_arr.shape
        assert noisy.dtype == np.uint8
        assert not np.array_equal(noisy, img_arr.astype(np.uint8))

    def test_add_salt_pepper_noise(self, sample_rgb_image):
        """Test _add_salt_pepper_noise method."""
        editor = AdaptiveNoiseInjection()
        img_arr = np.array(sample_rgb_image).astype(np.float32)

        noisy = editor._add_salt_pepper_noise(img_arr, amount=0.1)

        assert noisy.shape == img_arr.shape
        # Should have some extreme values (0 or 255)
        assert np.any(noisy == 0) or np.any(noisy == 255)

    def test_add_poisson_noise(self, sample_rgb_image):
        """Test _add_poisson_noise method."""
        editor = AdaptiveNoiseInjection()
        img_arr = np.array(sample_rgb_image).astype(np.float32)

        noisy = editor._add_poisson_noise(img_arr)

        assert noisy.shape == img_arr.shape
        assert noisy.dtype == np.uint8

    def test_add_speckle_noise(self, sample_rgb_image):
        """Test _add_speckle_noise method."""
        editor = AdaptiveNoiseInjection()
        img_arr = np.array(sample_rgb_image).astype(np.float32)

        noisy = editor._add_speckle_noise(img_arr, variance=0.5)

        assert noisy.shape == img_arr.shape
        assert noisy.dtype == np.uint8

    def test_output_clipped_to_valid_range(self, sample_rgb_image):
        """Test output values are in valid [0, 255] range."""
        editor = AdaptiveNoiseInjection(intensity=1.0)
        result = editor.edit(sample_rgb_image)
        result_arr = np.array(result)

        assert result_arr.min() >= 0
        assert result_arr.max() <= 255

    def test_grayscale_feature_analysis(self):
        """Test feature analysis works with grayscale-like input."""
        editor = AdaptiveNoiseInjection()

        # 2D array (grayscale)
        gray_arr = np.random.randint(0, 256, (256, 256)).astype(np.float32)
        features = editor._analyze_image_features(gray_arr)

        assert 'brightness_mean' in features


# ============================================================================
# Tests for DiffusionPurification
# ============================================================================

class TestDiffusionPurification:
    """Tests for DiffusionPurification editor.

    These tests cover constructor argument handling. The end-to-end purification
    path requires loading a Stable Diffusion pipeline (heavy / network-bound) and
    is exercised by the @pytest.mark.slow integration test below.
    """

    class _DummyConfig:
        device = "cpu"
        image_size = (512, 512)
        num_inference_steps = 50
        pipe = object()

    def test_default_strength(self):
        editor = DiffusionPurification(self._DummyConfig())
        assert editor.purification_strength == 0.3
        assert editor.default_prompt == ""
        assert editor.pipe is self._DummyConfig.pipe

    def test_custom_strength(self):
        editor = DiffusionPurification(self._DummyConfig(), purification_strength=0.5)
        assert editor.purification_strength == 0.5

    def test_strength_out_of_range_raises(self):
        with pytest.raises(ValueError):
            DiffusionPurification(self._DummyConfig(), purification_strength=0.0)
        with pytest.raises(ValueError):
            DiffusionPurification(self._DummyConfig(), purification_strength=1.5)

    def test_purifier_pipe_override(self):
        sentinel = object()
        editor = DiffusionPurification(self._DummyConfig(), purifier_pipe=sentinel)
        assert editor.pipe is sentinel

    @pytest.mark.slow
    def test_edit_smoke_with_real_pipe(self, sample_rgb_image):
        """End-to-end smoke test. Requires network + GPU/CPU diffusion model load."""
        pytest.importorskip("diffusers")
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        from markdiffusion.utils.diffusion_config import DiffusionConfig
        import torch

        model_path = "huanzi05/stable-diffusion-2-1-base"
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, scheduler=scheduler, safety_checker=None,
            torch_dtype=torch.float32,
        ).to("cpu")
        cfg = DiffusionConfig(
            scheduler=scheduler, pipe=pipe, device="cpu",
            image_size=(256, 256), num_inference_steps=10, guidance_scale=1.0,
            gen_seed=0, inversion_type="ddim",
        )
        editor = DiffusionPurification(cfg, purification_strength=0.2)
        out = editor.edit(sample_rgb_image)
        assert isinstance(out, Image.Image)
        assert out.size == sample_rgb_image.size


# ============================================================================
# Tests for NeuralCodecCompression
# ============================================================================

class TestNeuralCodecCompression:
    """Tests for NeuralCodecCompression editor.

    Constructor and validation are tested without loading a model. The model
    download / encode-decode path is exercised by the slow integration test.
    """

    def test_default_parameters(self):
        editor = NeuralCodecCompression()
        assert editor.quality == 5
        assert editor.model_name == "cheng2020-anchor"

    def test_custom_quality(self):
        editor = NeuralCodecCompression(quality=8)
        assert editor.quality == 8

    def test_invalid_model_name_raises_on_load(self):
        editor = NeuralCodecCompression(model_name="nonexistent-codec")
        # Validation is lazy: error occurs only when we try to materialize the model.
        try:
            import compressai  # noqa: F401
        except ImportError:
            pytest.skip("compressai not installed; cannot exercise lazy load path.")
        with pytest.raises(ValueError):
            editor._get_model()

    @pytest.mark.slow
    def test_edit_smoke_with_real_codec(self, sample_rgb_image):
        """End-to-end: cheng2020-anchor encode/decode round-trip."""
        pytest.importorskip("compressai")
        editor = NeuralCodecCompression(quality=1, device="cpu")
        out = editor.edit(sample_rgb_image)
        assert isinstance(out, Image.Image)
        assert out.size == sample_rgb_image.size


# ============================================================================
# Integration Tests
# ============================================================================

class TestEditorChaining:
    """Test chaining multiple editors."""

    def test_chain_multiple_editors(self, sample_rgb_image):
        """Test applying multiple editors in sequence."""
        editors = [
            JPEGCompression(quality=75),
            Rotation(angle=15),
            GaussianBlurring(radius=2),
            Brightness(factor=1.1),
        ]

        result = sample_rgb_image
        for editor in editors:
            result = editor.edit(result)

        assert isinstance(result, Image.Image)
        assert result.size[0] > 0 and result.size[1] > 0

    def test_all_editors_work_together(self, sample_complex_image):
        """Test all editors can process the same image."""
        editors = [
            JPEGCompression(quality=90),
            Rotation(angle=10),
            CrSc(crop_ratio=0.9),
            GaussianBlurring(radius=1),
            GaussianNoise(sigma=10),
            Brightness(factor=1.05),
            Mask(num_masks=2),
            Overlay(num_strokes=5),
            AdaptiveNoiseInjection(intensity=0.3),
        ]

        for editor in editors:
            result = editor.edit(sample_complex_image)
            assert isinstance(result, Image.Image)
            assert result.size[0] > 0 and result.size[1] > 0


class TestEditorConsistency:
    """Test editor consistency and reproducibility."""

    def test_same_params_same_result_deterministic(self, sample_rgb_image):
        """Test deterministic editors produce same result."""
        # JPEGCompression, Rotation, CrSc, Brightness are deterministic
        editor = Rotation(angle=45)

        result1 = editor.edit(sample_rgb_image)
        result2 = editor.edit(sample_rgb_image)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_random_editors_produce_different_results(self, sample_rgb_image):
        """Test random editors may produce different results."""
        editor = GaussianNoise(sigma=50)

        result1 = editor.edit(sample_rgb_image)
        result2 = editor.edit(sample_rgb_image)

        # Very unlikely to be identical
        assert not np.array_equal(np.array(result1), np.array(result2))
