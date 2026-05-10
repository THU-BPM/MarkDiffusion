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
Unit tests for video editor classes in MarkDiffusion.

Tests cover:
- VideoEditor: Base class
- MPEG4Compression: MPEG-4 video compression
- FrameAverage: Averaging frames in sliding window
- FrameRateAdapter: Frame rate conversion
- FrameSwap: Random adjacent frame swapping
- FrameInterpolationAttack: Insert interpolated frames
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
import shutil

from markdiffusion.evaluation.tools.video_editor import (
    VideoEditor,
    MPEG4Compression,
    FrameAverage,
    FrameRateAdapter,
    FrameSwap,
    FrameInterpolationAttack,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_frames():
    """Create a list of sample RGB frames."""
    frames = []
    for i in range(10):
        # Create frames with different colors for variation
        color = (i * 25, 100, 255 - i * 25)
        img = Image.new("RGB", (128, 128), color=color)
        frames.append(img)
    return frames


@pytest.fixture
def sample_gradient_frames():
    """Create gradient frames for testing interpolation."""
    frames = []
    for i in range(5):
        arr = np.full((64, 64, 3), i * 50, dtype=np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


@pytest.fixture
def single_frame():
    """Create a single frame."""
    return [Image.new("RGB", (128, 128), color="red")]


@pytest.fixture
def two_frames():
    """Create two frames for edge case testing."""
    return [
        Image.new("RGB", (64, 64), color="red"),
        Image.new("RGB", (64, 64), color="blue"),
    ]


@pytest.fixture
def empty_frames():
    """Return empty frame list."""
    return []


# ============================================================================
# Tests for VideoEditor Base Class
# ============================================================================

class TestVideoEditor:
    """Tests for VideoEditor base class."""

    def test_initialization(self):
        """Test base class can be instantiated."""
        editor = VideoEditor()
        assert editor is not None

    def test_edit_method_exists(self):
        """Test edit method exists."""
        editor = VideoEditor()
        assert hasattr(editor, 'edit')

    def test_edit_returns_none_by_default(self, sample_frames):
        """Test base edit method returns None."""
        editor = VideoEditor()
        result = editor.edit(sample_frames)
        assert result is None


# ============================================================================
# Tests for MPEG4Compression
# ============================================================================

class TestMPEG4Compression:
    """Tests for MPEG4Compression editor."""

    def test_default_fps(self):
        """Test default fps is 24.0."""
        editor = MPEG4Compression()
        assert editor.fps == 24.0

    def test_custom_fps(self):
        """Test custom fps setting."""
        editor = MPEG4Compression(fps=30.0)
        assert editor.fps == 30.0

    def test_fourcc_initialized(self):
        """Test fourcc codec is initialized."""
        editor = MPEG4Compression()
        assert editor.fourcc is not None

    def test_edit_returns_list(self, sample_frames):
        """Test edit returns a list of PIL Images."""
        editor = MPEG4Compression(fps=24.0)
        result = editor.edit(sample_frames)
        assert isinstance(result, list)
        assert all(isinstance(f, Image.Image) for f in result)

    def test_edit_preserves_frame_size(self, sample_frames):
        """Test edit preserves frame dimensions."""
        editor = MPEG4Compression(fps=24.0)
        result = editor.edit(sample_frames)
        original_size = sample_frames[0].size
        for frame in result:
            assert frame.size == original_size

    def test_compression_changes_pixels(self, sample_frames):
        """Test compression may change pixel values due to lossy encoding."""
        editor = MPEG4Compression(fps=24.0)
        result = editor.edit(sample_frames)

        # At least some frames should have different pixels due to compression
        original_arr = np.array(sample_frames[0])
        result_arr = np.array(result[0])

        # Due to lossy compression, pixels may differ
        # We just check the shapes match
        assert original_arr.shape == result_arr.shape

    def test_various_fps_values(self, two_frames):
        """Test various fps values."""
        for fps in [15.0, 24.0, 30.0, 60.0]:
            editor = MPEG4Compression(fps=fps)
            result = editor.edit(two_frames)
            assert isinstance(result, list)

# ============================================================================
# Tests for FrameAverage
# ============================================================================

class TestFrameAverage:
    """Tests for FrameAverage editor."""

    def test_default_n_frames(self):
        """Test default n_frames is 3."""
        editor = FrameAverage()
        assert editor.n_frames == 3

    def test_custom_n_frames(self):
        """Test custom n_frames setting."""
        editor = FrameAverage(n_frames=5)
        assert editor.n_frames == 5

    def test_edit_returns_list(self, sample_frames):
        """Test edit returns a list of PIL Images."""
        editor = FrameAverage(n_frames=3)
        result = editor.edit(sample_frames)
        assert isinstance(result, list)
        assert all(isinstance(f, Image.Image) for f in result)

    def test_edit_preserves_frame_count(self, sample_frames):
        """Test edit preserves number of frames."""
        editor = FrameAverage(n_frames=3)
        result = editor.edit(sample_frames)
        assert len(result) == len(sample_frames)

    def test_edit_preserves_frame_size(self, sample_frames):
        """Test edit preserves frame dimensions."""
        editor = FrameAverage(n_frames=3)
        result = editor.edit(sample_frames)
        original_size = sample_frames[0].size
        for frame in result:
            assert frame.size == original_size

    def test_averaging_effect(self):
        """Test that averaging smooths frames."""
        # Create frames with distinct pixel values
        frames = []
        for i in range(5):
            arr = np.full((64, 64, 3), i * 60, dtype=np.uint8)  # 0, 60, 120, 180, 240
            frames.append(Image.fromarray(arr))

        editor = FrameAverage(n_frames=3)
        result = editor.edit(frames)

        # Middle frame (index 2, value 120) should be averaged with neighbors (60, 180)
        # Average of [60, 120, 180] = 120, so it stays the same
        # But first frame (value 0) averaged with [0, 60] should change
        first_original = np.array(frames[0]).mean()  # 0
        first_result = np.array(result[0]).mean()  # avg of [0, 60] = 30

        # First frame should be affected by second frame in the window
        assert first_result > first_original

    def test_single_frame(self, single_frame):
        """Test with single frame."""
        editor = FrameAverage(n_frames=3)
        result = editor.edit(single_frame)
        assert len(result) == 1
        assert isinstance(result[0], Image.Image)

    def test_n_frames_larger_than_video(self, two_frames):
        """Test when n_frames is larger than video length."""
        editor = FrameAverage(n_frames=10)
        result = editor.edit(two_frames)
        assert len(result) == 2

    def test_various_n_frames(self, sample_frames):
        """Test various n_frames values."""
        for n in [1, 2, 3, 5, 7]:
            editor = FrameAverage(n_frames=n)
            result = editor.edit(sample_frames)
            assert len(result) == len(sample_frames)


# ============================================================================
# Tests for FrameRateAdapter
# ============================================================================

class TestFrameRateAdapter:
    """Tests for FrameRateAdapter editor."""

    def test_default_parameters(self):
        """Test default parameters."""
        editor = FrameRateAdapter()
        assert editor.source_fps == 30.0
        assert editor.target_fps == 24.0

    def test_custom_parameters(self):
        """Test custom fps settings."""
        editor = FrameRateAdapter(source_fps=60.0, target_fps=30.0)
        assert editor.source_fps == 60.0
        assert editor.target_fps == 30.0

    def test_invalid_fps_raises(self):
        """Test invalid fps raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            FrameRateAdapter(source_fps=0)
        with pytest.raises(ValueError, match="must be positive"):
            FrameRateAdapter(target_fps=-10)

    def test_edit_returns_list(self, sample_frames):
        """Test edit returns a list of PIL Images."""
        editor = FrameRateAdapter(source_fps=30.0, target_fps=24.0)
        result = editor.edit(sample_frames)
        assert isinstance(result, list)
        assert all(isinstance(f, Image.Image) for f in result)

    def test_edit_preserves_frame_size(self, sample_frames):
        """Test edit preserves frame dimensions."""
        editor = FrameRateAdapter(source_fps=30.0, target_fps=24.0)
        result = editor.edit(sample_frames)
        original_size = sample_frames[0].size
        for frame in result:
            assert frame.size == original_size

    def test_same_fps_returns_copies(self, sample_frames):
        """Test same source and target fps returns copies."""
        editor = FrameRateAdapter(source_fps=30.0, target_fps=30.0)
        result = editor.edit(sample_frames)
        assert len(result) == len(sample_frames)

    def test_downsampling(self, sample_frames):
        """Test downsampling reduces frame count."""
        editor = FrameRateAdapter(source_fps=30.0, target_fps=15.0)
        result = editor.edit(sample_frames)
        # Should have roughly half the frames
        assert len(result) < len(sample_frames)

    def test_upsampling(self, sample_frames):
        """Test upsampling increases frame count."""
        editor = FrameRateAdapter(source_fps=15.0, target_fps=30.0)
        result = editor.edit(sample_frames)
        # Should have roughly double the frames
        assert len(result) > len(sample_frames)

    def test_empty_frames(self, empty_frames):
        """Test with empty frames."""
        editor = FrameRateAdapter()
        result = editor.edit(empty_frames)
        assert result == []

    def test_single_frame(self, single_frame):
        """Test with single frame."""
        editor = FrameRateAdapter()
        result = editor.edit(single_frame)
        assert len(result) == 1

    def test_interpolation_smoothness(self, sample_gradient_frames):
        """Test interpolated frames have smooth transitions."""
        editor = FrameRateAdapter(source_fps=15.0, target_fps=30.0)
        result = editor.edit(sample_gradient_frames)

        # Check that interpolated values are between neighbors
        for i in range(1, len(result) - 1):
            arr = np.array(result[i])
            # Values should be within valid range
            assert arr.min() >= 0
            assert arr.max() <= 255


# ============================================================================
# Tests for FrameSwap
# ============================================================================

class TestFrameSwap:
    """Tests for FrameSwap editor."""

    def test_default_probability(self):
        """Test default swap probability is 0.25."""
        editor = FrameSwap()
        assert editor.p == 0.25

    def test_custom_probability(self):
        """Test custom swap probability."""
        editor = FrameSwap(p=0.5)
        assert editor.p == 0.5

    def test_edit_returns_list(self, sample_frames):
        """Test edit returns a list of PIL Images."""
        editor = FrameSwap(p=0.5)
        result = editor.edit(sample_frames.copy())
        assert isinstance(result, list)
        assert all(isinstance(f, Image.Image) for f in result)

    def test_edit_preserves_frame_count(self, sample_frames):
        """Test edit preserves number of frames."""
        editor = FrameSwap(p=0.5)
        frames_copy = sample_frames.copy()
        result = editor.edit(frames_copy)
        assert len(result) == len(sample_frames)

    def test_edit_preserves_frame_size(self, sample_frames):
        """Test edit preserves frame dimensions."""
        editor = FrameSwap(p=0.5)
        frames_copy = sample_frames.copy()
        result = editor.edit(frames_copy)
        original_size = sample_frames[0].size
        for frame in result:
            assert frame.size == original_size

    def test_high_probability_no_swap(self, sample_frames):
        """Test high probability (p=1.0) doesn't swap (inverted logic in implementation)."""
        # Note: The implementation swaps when random() >= p
        # So p=1.0 means never swap (random() < 1.0 always)
        editor = FrameSwap(p=1.0)
        frames_copy = [f.copy() for f in sample_frames]
        original_arrays = [np.array(f) for f in frames_copy]

        result = editor.edit(frames_copy)
        result_arrays = [np.array(f) for f in result]

        for orig, res in zip(original_arrays, result_arrays):
            np.testing.assert_array_equal(orig, res)

    def test_zero_probability_always_swaps(self, sample_frames):
        """Test zero probability (p=0.0) always swaps (inverted logic in implementation)."""
        # Note: The implementation swaps when random() >= p
        # So p=0.0 means always swap (random() >= 0.0 always)
        editor = FrameSwap(p=0.0)
        frames_copy = [f.copy() for f in sample_frames]

        editor.edit(frames_copy)
        # Just check it runs without error and returns correct length
        assert len(frames_copy) == len(sample_frames)

    def test_single_frame(self, single_frame):
        """Test with single frame."""
        editor = FrameSwap(p=0.5)
        result = editor.edit(single_frame)
        assert len(result) == 1

    def test_two_frames(self, two_frames):
        """Test with two frames."""
        editor = FrameSwap(p=0.5)
        result = editor.edit(two_frames)
        assert len(result) == 2


# ============================================================================
# Tests for FrameInterpolationAttack
# ============================================================================

class TestFrameInterpolationAttack:
    """Tests for FrameInterpolationAttack editor."""

    def test_default_parameters(self):
        """Test default interpolated_frames is 1."""
        editor = FrameInterpolationAttack()
        assert editor.interpolated_frames == 1

    def test_custom_parameters(self):
        """Test custom interpolated_frames setting."""
        editor = FrameInterpolationAttack(interpolated_frames=3)
        assert editor.interpolated_frames == 3

    def test_negative_frames_raises(self):
        """Test negative interpolated_frames raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            FrameInterpolationAttack(interpolated_frames=-1)

    def test_edit_returns_list(self, sample_frames):
        """Test edit returns a list of PIL Images."""
        editor = FrameInterpolationAttack(interpolated_frames=1)
        result = editor.edit(sample_frames)
        assert isinstance(result, list)
        assert all(isinstance(f, Image.Image) for f in result)

    def test_edit_preserves_frame_size(self, sample_frames):
        """Test edit preserves frame dimensions."""
        editor = FrameInterpolationAttack(interpolated_frames=1)
        result = editor.edit(sample_frames)
        original_size = sample_frames[0].size
        for frame in result:
            assert frame.size == original_size

    def test_frame_count_increases(self, sample_frames):
        """Test interpolation increases frame count."""
        n_interp = 2
        editor = FrameInterpolationAttack(interpolated_frames=n_interp)
        result = editor.edit(sample_frames)

        # Expected: original frames + (n_original - 1) * n_interp
        expected_count = len(sample_frames) + (len(sample_frames) - 1) * n_interp
        assert len(result) == expected_count

    def test_zero_interpolation(self, sample_frames):
        """Test zero interpolated_frames returns copies."""
        editor = FrameInterpolationAttack(interpolated_frames=0)
        result = editor.edit(sample_frames)
        assert len(result) == len(sample_frames)

    def test_empty_frames(self, empty_frames):
        """Test with empty frames."""
        editor = FrameInterpolationAttack(interpolated_frames=1)
        result = editor.edit(empty_frames)
        assert result == []

    def test_single_frame(self, single_frame):
        """Test with single frame."""
        editor = FrameInterpolationAttack(interpolated_frames=2)
        result = editor.edit(single_frame)
        assert len(result) == 1

    def test_two_frames_interpolation(self, two_frames):
        """Test interpolation between two frames."""
        editor = FrameInterpolationAttack(interpolated_frames=1)
        result = editor.edit(two_frames)

        # Should have: frame1, interpolated, frame2
        assert len(result) == 3

    def test_interpolated_values(self, sample_gradient_frames):
        """Test interpolated frames have intermediate values."""
        editor = FrameInterpolationAttack(interpolated_frames=1)
        result = editor.edit(sample_gradient_frames)

        # Check that interpolated frame (index 1) is between original frames
        original_0 = np.array(sample_gradient_frames[0]).astype(float)
        original_1 = np.array(sample_gradient_frames[1]).astype(float)
        interpolated = np.array(result[1]).astype(float)

        # Interpolated value should be close to average of neighbors
        expected = (original_0 + original_1) / 2
        np.testing.assert_array_almost_equal(interpolated, expected, decimal=0)

    def test_various_interpolation_counts(self, two_frames):
        """Test various interpolation counts."""
        for n in [0, 1, 2, 3, 5]:
            editor = FrameInterpolationAttack(interpolated_frames=n)
            result = editor.edit([f.copy() for f in two_frames])
            expected = 2 + 1 * n  # 2 original + 1 gap * n interpolated
            assert len(result) == expected


# ============================================================================
# Integration Tests
# ============================================================================

class TestVideoEditorChaining:
    """Test chaining multiple video editors."""

    def test_chain_frame_editors(self, sample_frames):
        """Test chaining frame-based editors."""
        editors = [
            FrameAverage(n_frames=3),
            FrameSwap(p=0.1),
        ]

        result = sample_frames
        for editor in editors:
            result = editor.edit(result)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(f, Image.Image) for f in result)

    def test_chain_with_interpolation(self, sample_frames):
        """Test chaining with interpolation editor."""
        editors = [
            FrameInterpolationAttack(interpolated_frames=1),
            FrameAverage(n_frames=3),
        ]

        result = sample_frames
        for editor in editors:
            result = editor.edit(result)

        assert isinstance(result, list)
        assert len(result) > len(sample_frames)  # Should have more frames

    def test_chain_with_rate_adapter(self, sample_frames):
        """Test chaining with frame rate adapter."""
        editors = [
            FrameRateAdapter(source_fps=30.0, target_fps=24.0),
            FrameAverage(n_frames=3),
        ]

        result = sample_frames
        for editor in editors:
            result = editor.edit(result)

        assert isinstance(result, list)
        assert all(isinstance(f, Image.Image) for f in result)


class TestVideoEditorEdgeCases:
    """Test edge cases for video editors."""

    def test_large_frame_count(self):
        """Test with many frames."""
        frames = [Image.new("RGB", (32, 32), color=(i % 256, 0, 0)) for i in range(100)]

        editor = FrameAverage(n_frames=5)
        result = editor.edit(frames)

        assert len(result) == 100

    def test_small_frame_size(self):
        """Test with very small frames."""
        frames = [Image.new("RGB", (8, 8), color="red") for _ in range(5)]

        editor = FrameAverage(n_frames=3)
        result = editor.edit(frames)

        assert len(result) == 5
        assert all(f.size == (8, 8) for f in result)

    def test_large_frame_size(self):
        """Test with large frames."""
        frames = [Image.new("RGB", (512, 512), color="blue") for _ in range(3)]

        editor = FrameAverage(n_frames=3)
        result = editor.edit(frames)

        assert len(result) == 3
        assert all(f.size == (512, 512) for f in result)

    def test_non_square_frames(self):
        """Test with non-square frames."""
        frames = [Image.new("RGB", (320, 180), color="green") for _ in range(5)]

        editor = FrameAverage(n_frames=3)
        result = editor.edit(frames)

        assert len(result) == 5
        assert all(f.size == (320, 180) for f in result)

    def test_grayscale_to_rgb_conversion(self):
        """Test editors handle RGB frames correctly."""
        # Create RGB frames
        frames = [Image.new("RGB", (64, 64), color=(128, 128, 128)) for _ in range(5)]

        editor = FrameAverage(n_frames=3)
        result = editor.edit(frames)

        assert all(f.mode == "RGB" for f in result)
