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
Unit tests for the utils module.

This file contains tests for:
- utils/utils.py: General utility functions
- utils/callbacks.py: Callback classes for diffusion models
- utils/pipeline_utils.py: Pipeline type detection utilities
- utils/media_utils.py: Media conversion utilities
- utils/diffusion_config.py: Diffusion configuration
"""

import pytest
import torch
import numpy as np
import json
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, MagicMock, patch
from diffusers import (
    StableDiffusionPipeline, 
    TextToVideoSDPipeline, 
    StableVideoDiffusionPipeline,
    DPMSolverMultistepScheduler
)
from markdiffusion.inversions.exact_inversion import ExactInversion


# ============================================================================
# Tests for utils/utils.py
# ============================================================================

class TestInheritDocstring:
    """Tests for inherit_docstring decorator."""

    def test_inherit_docstring_from_base_class(self):
        """Test that docstrings are inherited from base classes."""
        from markdiffusion.utils.utils import inherit_docstring

        class Base:
            def method(self):
                """Base method docstring."""
                pass

        @inherit_docstring
        class Derived(Base):
            def method(self):
                pass

        assert Derived.method.__doc__ == "Base method docstring."

    def test_no_override_existing_docstring(self):
        """Test that existing docstrings are not overridden."""
        from markdiffusion.utils.utils import inherit_docstring

        class Base:
            def method(self):
                """Base method docstring."""
                pass

        @inherit_docstring
        class Derived(Base):
            def method(self):
                """Derived method docstring."""
                pass

        assert Derived.method.__doc__ == "Derived method docstring."

    def test_no_docstring_in_base(self):
        """Test behavior when base class has no docstring."""
        from markdiffusion.utils.utils import inherit_docstring

        class Base:
            def method(self):
                pass

        @inherit_docstring
        class Derived(Base):
            def method(self):
                pass

        assert Derived.method.__doc__ is None

    def test_multiple_inheritance(self):
        """Test docstring inheritance with multiple base classes."""
        from markdiffusion.utils.utils import inherit_docstring

        class Base1:
            def method(self):
                """Base1 method docstring."""
                pass

        class Base2:
            def method(self):
                """Base2 method docstring."""
                pass

        @inherit_docstring
        class Derived(Base1, Base2):
            def method(self):
                pass

        # Should inherit from first base class
        assert Derived.method.__doc__ == "Base1 method docstring."


class TestLoadConfigFile:
    """Tests for load_config_file function."""

    def test_load_valid_json(self, tmp_path):
        """Test loading a valid JSON configuration file."""
        from markdiffusion.utils.utils import load_config_file

        config_data = {"key": "value", "number": 42, "nested": {"a": 1}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_config_file(str(config_file))
        assert result == config_data

    def test_load_nonexistent_file(self, capsys):
        """Test loading a nonexistent file returns None."""
        from markdiffusion.utils.utils import load_config_file

        result = load_config_file("/nonexistent/path/config.json")
        assert result is None

        captured = capsys.readouterr()
        assert "does not exist" in captured.out

    def test_load_invalid_json(self, tmp_path, capsys):
        """Test loading an invalid JSON file returns None."""
        from markdiffusion.utils.utils import load_config_file

        config_file = tmp_path / "invalid.json"
        config_file.write_text("{invalid json content")

        result = load_config_file(str(config_file))
        assert result is None

        captured = capsys.readouterr()
        assert "Error decoding JSON" in captured.out

    def test_load_empty_json(self, tmp_path):
        """Test loading an empty JSON object."""
        from markdiffusion.utils.utils import load_config_file

        config_file = tmp_path / "empty.json"
        config_file.write_text("{}")

        result = load_config_file(str(config_file))
        assert result == {}


class TestLoadJsonAsList:
    """Tests for load_json_as_list function."""

    def test_load_jsonl_file(self, tmp_path):
        """Test loading a JSONL file (one JSON object per line)."""
        from markdiffusion.utils.utils import load_json_as_list

        data = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
            {"id": 3, "name": "third"},
        ]
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        result = load_json_as_list(str(jsonl_file))
        assert result == data

    def test_load_empty_jsonl(self, tmp_path):
        """Test loading an empty JSONL file."""
        from markdiffusion.utils.utils import load_json_as_list

        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        result = load_json_as_list(str(jsonl_file))
        assert result == []


class TestCreateDirectoryForFile:
    """Tests for create_directory_for_file function."""

    def test_create_directory(self, tmp_path):
        """Test creating a directory for a file path."""
        from markdiffusion.utils.utils import create_directory_for_file

        file_path = tmp_path / "new_dir" / "subdir" / "file.txt"
        create_directory_for_file(str(file_path))

        assert (tmp_path / "new_dir" / "subdir").exists()
        assert (tmp_path / "new_dir" / "subdir").is_dir()

    def test_existing_directory(self, tmp_path):
        """Test that existing directories don't cause errors."""
        from markdiffusion.utils.utils import create_directory_for_file

        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        file_path = existing_dir / "file.txt"

        # Should not raise an error
        create_directory_for_file(str(file_path))
        assert existing_dir.exists()


class TestSetRandomSeed:
    """Tests for set_random_seed function."""

    def test_reproducibility_torch(self):
        """Test that torch random is reproducible with same seed."""
        from markdiffusion.utils.utils import set_random_seed

        set_random_seed(42)
        tensor1 = torch.randn(10)

        set_random_seed(42)
        tensor2 = torch.randn(10)

        assert torch.allclose(tensor1, tensor2)

    def test_reproducibility_numpy(self):
        """Test that numpy random is reproducible with same seed."""
        from markdiffusion.utils.utils import set_random_seed

        set_random_seed(42)
        arr1 = np.random.randn(10)

        set_random_seed(42)
        arr2 = np.random.randn(10)

        np.testing.assert_array_almost_equal(arr1, arr2)

    def test_reproducibility_python_random(self):
        """Test that Python random is reproducible with same seed."""
        from markdiffusion.utils.utils import set_random_seed
        import random

        set_random_seed(42)
        val1 = [random.random() for _ in range(10)]

        set_random_seed(42)
        val2 = [random.random() for _ in range(10)]

        assert val1 == val2

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        from markdiffusion.utils.utils import set_random_seed

        set_random_seed(42)
        tensor1 = torch.randn(10)

        set_random_seed(123)
        tensor2 = torch.randn(10)

        assert not torch.allclose(tensor1, tensor2)


# ============================================================================
# Tests for utils/callbacks.py
# ============================================================================

class TestDenoisingLatentsCollector:
    """Tests for DenoisingLatentsCollector class."""

    def test_init_default_parameters(self):
        """Test default initialization parameters."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector()
        assert collector.save_every_n_steps == 1
        assert collector.to_cpu is True
        assert collector.data == []
        assert collector._call_count == 0

    def test_init_custom_parameters(self):
        """Test custom initialization parameters."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector(save_every_n_steps=5, to_cpu=False)
        assert collector.save_every_n_steps == 5
        assert collector.to_cpu is False

    def test_call_saves_latents(self):
        """Test that __call__ saves latents correctly."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector()
        latents = torch.randn(1, 4, 64, 64)

        collector(step=0, timestep=1000, latents=latents)

        assert len(collector.data) == 1
        assert collector.data[0]["step"] == 0
        assert collector.data[0]["timestep"] == 1000
        assert collector.data[0]["call_count"] == 1
        assert collector.data[0]["latents"].shape == latents.shape

    def test_call_respects_save_every_n_steps(self):
        """Test that latents are saved every n steps."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector(save_every_n_steps=2)
        latents = torch.randn(1, 4, 64, 64)

        for i in range(5):
            collector(step=i, timestep=1000 - i * 100, latents=latents)

        # Should save at call 2 and 4 (1-indexed)
        assert len(collector.data) == 2
        assert collector.data[0]["call_count"] == 2
        assert collector.data[1]["call_count"] == 4

    def test_call_moves_to_cpu(self):
        """Test that latents are moved to CPU when to_cpu=True."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector(to_cpu=True)
        latents = torch.randn(1, 4, 64, 64)

        collector(step=0, timestep=1000, latents=latents)

        assert collector.data[0]["latents"].device == torch.device("cpu")

    def test_latents_list_property(self):
        """Test latents_list property returns list of latents."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector()

        for i in range(3):
            collector(step=i, timestep=1000 - i * 100, latents=torch.randn(1, 4, 64, 64))

        latents_list = collector.latents_list
        assert len(latents_list) == 3
        assert all(isinstance(l, torch.Tensor) for l in latents_list)

    def test_timesteps_list_property(self):
        """Test timesteps_list property returns list of timesteps."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector()
        timesteps = [1000, 800, 600]

        for i, ts in enumerate(timesteps):
            collector(step=i, timestep=ts, latents=torch.randn(1, 4, 64, 64))

        assert collector.timesteps_list == timesteps

    def test_get_latents_at_step(self):
        """Test get_latents_at_step returns correct latents."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector()
        latents_0 = torch.randn(1, 4, 64, 64)
        latents_1 = torch.randn(1, 4, 64, 64)

        collector(step=0, timestep=1000, latents=latents_0)
        collector(step=1, timestep=800, latents=latents_1)

        result = collector.get_latents_at_step(0)
        assert torch.allclose(result, latents_0.cpu())

    def test_get_latents_at_step_not_found(self):
        """Test get_latents_at_step raises ValueError for missing step."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector()
        collector(step=0, timestep=1000, latents=torch.randn(1, 4, 64, 64))

        with pytest.raises(ValueError, match="No latents found for step"):
            collector.get_latents_at_step(999)

    def test_clear(self):
        """Test clear method resets collector state."""
        from markdiffusion.utils.callbacks import DenoisingLatentsCollector

        collector = DenoisingLatentsCollector()

        for i in range(3):
            collector(step=i, timestep=1000, latents=torch.randn(1, 4, 64, 64))

        collector.clear()

        assert collector.data == []
        assert collector._call_count == 0


# ============================================================================
# Tests for utils/pipeline_utils.py
# ============================================================================

class TestPipelineUtils:
    """Tests for pipeline utility functions."""

    def test_pipeline_type_constants(self):
        """Test pipeline type constants are defined correctly."""
        from markdiffusion.utils.pipeline_utils import (
            PIPELINE_TYPE_IMAGE,
            PIPELINE_TYPE_TEXT_TO_VIDEO,
            PIPELINE_TYPE_IMAGE_TO_VIDEO,
        )

        assert PIPELINE_TYPE_IMAGE == "image"
        assert PIPELINE_TYPE_TEXT_TO_VIDEO == "t2v"
        assert PIPELINE_TYPE_IMAGE_TO_VIDEO == "i2v"

    def test_get_pipeline_type_unknown(self):
        """Test get_pipeline_type returns None for unknown pipeline."""
        from markdiffusion.utils.pipeline_utils import get_pipeline_type

        mock_pipeline = Mock()
        result = get_pipeline_type(mock_pipeline)
        assert result is None

    def test_is_video_pipeline_with_mock(self):
        """Test is_video_pipeline with mocked pipelines."""
        from markdiffusion.utils.pipeline_utils import is_video_pipeline, get_pipeline_type

        # Mock image pipeline
        mock_image_pipe = Mock()
        with patch("markdiffusion.utils.pipeline_utils.get_pipeline_type", return_value="image"):
            assert is_video_pipeline(mock_image_pipe) is False

        # Mock video pipeline
        mock_video_pipe = Mock()
        with patch("markdiffusion.utils.pipeline_utils.get_pipeline_type", return_value="t2v"):
            assert is_video_pipeline(mock_video_pipe) is True

    def test_is_image_pipeline_with_mock(self):
        """Test is_image_pipeline with mocked pipelines."""
        from markdiffusion.utils.pipeline_utils import is_image_pipeline

        mock_pipe = Mock()
        with patch("markdiffusion.utils.pipeline_utils.get_pipeline_type", return_value="image"):
            assert is_image_pipeline(mock_pipe) is True

        with patch("markdiffusion.utils.pipeline_utils.get_pipeline_type", return_value="t2v"):
            assert is_image_pipeline(mock_pipe) is False

    def test_is_t2v_pipeline_with_mock(self):
        """Test is_t2v_pipeline with mocked pipelines."""
        from markdiffusion.utils.pipeline_utils import is_t2v_pipeline

        mock_pipe = Mock()
        with patch("markdiffusion.utils.pipeline_utils.get_pipeline_type", return_value="t2v"):
            assert is_t2v_pipeline(mock_pipe) is True

        with patch("markdiffusion.utils.pipeline_utils.get_pipeline_type", return_value="i2v"):
            assert is_t2v_pipeline(mock_pipe) is False

    def test_is_i2v_pipeline_with_mock(self):
        """Test is_i2v_pipeline with mocked pipelines."""
        from markdiffusion.utils.pipeline_utils import is_i2v_pipeline

        mock_pipe = Mock()
        with patch("markdiffusion.utils.pipeline_utils.get_pipeline_type", return_value="i2v"):
            assert is_i2v_pipeline(mock_pipe) is True

        with patch("markdiffusion.utils.pipeline_utils.get_pipeline_type", return_value="t2v"):
            assert is_i2v_pipeline(mock_pipe) is False

    def test_get_pipeline_requirements_image(self):
        """Test get_pipeline_requirements for image pipeline."""
        from markdiffusion.utils.pipeline_utils import get_pipeline_requirements, PIPELINE_TYPE_IMAGE

        result = get_pipeline_requirements(PIPELINE_TYPE_IMAGE)
        assert result["required_params"] == []
        assert "height" in result["optional_params"]
        assert "width" in result["optional_params"]

    def test_get_pipeline_requirements_t2v(self):
        """Test get_pipeline_requirements for text-to-video pipeline."""
        from markdiffusion.utils.pipeline_utils import get_pipeline_requirements, PIPELINE_TYPE_TEXT_TO_VIDEO

        result = get_pipeline_requirements(PIPELINE_TYPE_TEXT_TO_VIDEO)
        assert "num_frames" in result["required_params"]
        assert "fps" in result["optional_params"]

    def test_get_pipeline_requirements_i2v(self):
        """Test get_pipeline_requirements for image-to-video pipeline."""
        from markdiffusion.utils.pipeline_utils import get_pipeline_requirements, PIPELINE_TYPE_IMAGE_TO_VIDEO

        result = get_pipeline_requirements(PIPELINE_TYPE_IMAGE_TO_VIDEO)
        assert "input_image" in result["required_params"]
        assert "num_frames" in result["required_params"]

    def test_get_pipeline_requirements_unknown(self):
        """Test get_pipeline_requirements for unknown pipeline type."""
        from markdiffusion.utils.pipeline_utils import get_pipeline_requirements

        result = get_pipeline_requirements("unknown")
        assert result["required_params"] == []
        assert result["optional_params"] == []


# ============================================================================
# Tests for utils/media_utils.py
# ============================================================================

class TestTorchToNumpy:
    """Tests for torch_to_numpy function."""

    def test_image_tensor_conversion(self):
        """Test conversion of 4D image tensor."""
        from markdiffusion.utils.media_utils import torch_to_numpy

        # Create tensor in range [-1, 1]
        tensor = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        result = torch_to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 64, 64, 3)  # B, H, W, C
        assert result.min() >= 0 and result.max() <= 1

    def test_video_tensor_conversion(self):
        """Test conversion of 5D video tensor."""
        from markdiffusion.utils.media_utils import torch_to_numpy

        # Create tensor in range [-1, 1]
        tensor = torch.randn(1, 3, 8, 64, 64).clamp(-1, 1)
        result = torch_to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 8, 64, 64, 3)  # B, F, H, W, C

    def test_unsupported_dimension(self):
        """Test that unsupported dimensions raise ValueError."""
        from markdiffusion.utils.media_utils import torch_to_numpy

        tensor = torch.randn(3, 64, 64)  # 3D tensor
        with pytest.raises(ValueError, match="Unsupported tensor dimension"):
            torch_to_numpy(tensor)


class TestPilToTorch:
    """Tests for pil_to_torch function."""

    def test_basic_conversion(self):
        """Test basic PIL to torch conversion."""
        from markdiffusion.utils.media_utils import pil_to_torch

        img = Image.new("RGB", (64, 64), color="red")
        tensor = pil_to_torch(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_normalized_range(self):
        """Test that normalized output is in [-1, 1] range."""
        from markdiffusion.utils.media_utils import pil_to_torch

        img = Image.new("RGB", (64, 64), color="white")
        tensor = pil_to_torch(img, normalize=True)

        # White pixels should be close to 1.0 after normalization
        assert tensor.max() <= 1.0
        assert tensor.min() >= -1.0

    def test_unnormalized_range(self):
        """Test that unnormalized output is in [0, 1] range."""
        from markdiffusion.utils.media_utils import pil_to_torch

        img = Image.new("RGB", (64, 64), color="white")
        tensor = pil_to_torch(img, normalize=False)

        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0


class TestNumpyToPil:
    """Tests for numpy_to_pil function."""

    def test_uint8_array(self):
        """Test conversion of uint8 numpy array."""
        from markdiffusion.utils.media_utils import numpy_to_pil

        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = numpy_to_pil(arr)

        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)

    def test_float_array(self):
        """Test conversion of float numpy array in [0, 1] range."""
        from markdiffusion.utils.media_utils import numpy_to_pil

        arr = np.random.rand(64, 64, 3).astype(np.float32)
        img = numpy_to_pil(arr)

        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)


class TestCv2ToPil:
    """Tests for cv2_to_pil function."""

    def test_uint8_array(self):
        """Test conversion of uint8 cv2 array."""
        from markdiffusion.utils.media_utils import cv2_to_pil

        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = cv2_to_pil(arr)

        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)

    def test_float_array(self):
        """Test conversion of float cv2 array."""
        from markdiffusion.utils.media_utils import cv2_to_pil

        arr = np.random.rand(64, 64, 3).astype(np.float32)
        img = cv2_to_pil(arr)

        assert isinstance(img, Image.Image)


class TestPilToCv2:
    """Tests for pil_to_cv2 function."""

    def test_basic_conversion(self):
        """Test basic PIL to cv2 conversion."""
        from markdiffusion.utils.media_utils import pil_to_cv2

        img = Image.new("RGB", (64, 64), color="red")
        arr = pil_to_cv2(img)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.float64
        assert arr.max() <= 1.0 and arr.min() >= 0.0


class TestTransformToModelFormat:
    """Tests for transform_to_model_format function."""

    def test_single_pil_image(self):
        """Test transformation of single PIL image."""
        from markdiffusion.utils.media_utils import transform_to_model_format

        img = Image.new("RGB", (64, 64), color="blue")
        tensor = transform_to_model_format(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)
        # Check normalization to [-1, 1]
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0

    def test_single_pil_image_with_resize(self):
        """Test transformation with resize."""
        from markdiffusion.utils.media_utils import transform_to_model_format

        img = Image.new("RGB", (128, 128), color="blue")
        tensor = transform_to_model_format(img, target_size=64)

        assert tensor.shape == (3, 64, 64)

    def test_list_of_pil_images(self):
        """Test transformation of list of PIL images."""
        from markdiffusion.utils.media_utils import transform_to_model_format

        frames = [Image.new("RGB", (64, 64), color="red") for _ in range(4)]
        tensor = transform_to_model_format(frames)

        assert tensor.shape == (4, 3, 64, 64)

    def test_list_of_numpy_arrays(self):
        """Test transformation of list of numpy arrays."""
        from markdiffusion.utils.media_utils import transform_to_model_format

        frames = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(4)]
        tensor = transform_to_model_format(frames)

        assert tensor.shape == (4, 3, 64, 64)

    def test_single_numpy_frame(self):
        """Test transformation of single numpy frame (3D array)."""
        from markdiffusion.utils.media_utils import transform_to_model_format

        frame = np.random.rand(64, 64, 3).astype(np.float32)
        tensor = transform_to_model_format(frame)

        assert tensor.shape == (3, 64, 64)

    def test_numpy_video_array(self):
        """Test transformation of numpy video array (4D)."""
        from markdiffusion.utils.media_utils import transform_to_model_format

        video = np.random.rand(8, 64, 64, 3).astype(np.float32)
        tensor = transform_to_model_format(video)

        assert tensor.shape == (8, 3, 64, 64)

    def test_unsupported_type(self):
        """Test that unsupported types raise ValueError."""
        from markdiffusion.utils.media_utils import transform_to_model_format

        with pytest.raises(ValueError, match="Unsupported media type"):
            transform_to_model_format("not_valid_input")

    def test_mixed_frame_types_raises_error(self):
        """Test that mixed frame types raise ValueError."""
        from markdiffusion.utils.media_utils import transform_to_model_format

        frames = [
            Image.new("RGB", (64, 64)),
            np.random.rand(64, 64, 3),
        ]
        with pytest.raises(ValueError, match="All frames must be either"):
            transform_to_model_format(frames)


class TestConvertVideoFramesToImages:
    """Tests for convert_video_frames_to_images function."""

    def test_numpy_frames(self):
        """Test conversion of numpy frames to PIL images."""
        from markdiffusion.utils.media_utils import convert_video_frames_to_images

        frames = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(4)]
        result = convert_video_frames_to_images(frames)

        assert len(result) == 4
        assert all(isinstance(img, Image.Image) for img in result)

    def test_pil_frames(self):
        """Test that PIL frames pass through unchanged."""
        from markdiffusion.utils.media_utils import convert_video_frames_to_images

        frames = [Image.new("RGB", (64, 64)) for _ in range(4)]
        result = convert_video_frames_to_images(frames)

        assert len(result) == 4
        assert all(isinstance(img, Image.Image) for img in result)

    def test_unsupported_frame_type(self):
        """Test that unsupported frame types raise ValueError."""
        from markdiffusion.utils.media_utils import convert_video_frames_to_images

        frames = ["not_a_frame"]
        with pytest.raises(ValueError, match="Unsupported frame type"):
            convert_video_frames_to_images(frames)


class TestSaveVideoFrames:
    """Tests for save_video_frames function."""

    def test_save_numpy_frames(self, tmp_path):
        """Test saving numpy frames to disk."""
        from markdiffusion.utils.media_utils import save_video_frames

        frames = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(4)]
        save_dir = str(tmp_path)
        save_video_frames(frames, save_dir)

        saved_files = list(tmp_path.glob("*.png"))
        assert len(saved_files) == 4

    def test_save_pil_frames(self, tmp_path):
        """Test saving PIL frames to disk."""
        from markdiffusion.utils.media_utils import save_video_frames

        frames = [Image.new("RGB", (64, 64), color="red") for _ in range(4)]
        save_dir = str(tmp_path)
        save_video_frames(frames, save_dir)

        saved_files = list(tmp_path.glob("*.png"))
        assert len(saved_files) == 4

    def test_frame_naming(self, tmp_path):
        """Test that frames are named with zero-padded indices."""
        from markdiffusion.utils.media_utils import save_video_frames

        frames = [Image.new("RGB", (64, 64)) for _ in range(3)]
        save_dir = str(tmp_path)
        save_video_frames(frames, save_dir)

        assert (tmp_path / "00.png").exists()
        assert (tmp_path / "01.png").exists()
        assert (tmp_path / "02.png").exists()


# ============================================================================
# Tests for utils/diffusion_config.py
# ============================================================================

class TestDiffusionConfig:
    """Tests for DiffusionConfig class."""

    @pytest.fixture
    def mock_image_pipeline(self):
        """Create a mock image pipeline."""
        from diffusers import StableDiffusionPipeline

        mock_pipe = MagicMock(spec=StableDiffusionPipeline)
        return mock_pipe

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        from diffusers import DPMSolverMultistepScheduler

        mock_scheduler = MagicMock(spec=DPMSolverMultistepScheduler)
        return mock_scheduler

    def test_default_parameters(self, mock_image_pipeline, mock_scheduler):
        """Test DiffusionConfig with default parameters."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        config = DiffusionConfig(
            scheduler=mock_scheduler,
            pipe=mock_image_pipeline,
            device="cpu",
        )

        assert config.guidance_scale == 7.5
        assert config.num_images == 1
        assert config.num_inference_steps == 50
        assert config.image_size == (512, 512)
        assert config.dtype == torch.float16
        assert config.gen_seed == 0
        assert config.inversion_type == "ddim"

    def test_custom_parameters(self, mock_image_pipeline, mock_scheduler):
        """Test DiffusionConfig with custom parameters."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        config = DiffusionConfig(
            scheduler=mock_scheduler,
            pipe=mock_image_pipeline,
            device="cuda",
            guidance_scale=10.0,
            num_inference_steps=30,
            image_size=(256, 256),
            gen_seed=42,
        )

        assert config.guidance_scale == 10.0
        assert config.num_inference_steps == 30
        assert config.image_size == (256, 256)
        assert config.gen_seed == 42

    def test_invalid_inversion_type(self, mock_image_pipeline, mock_scheduler):
        """Test that invalid inversion type raises AssertionError."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with pytest.raises(AssertionError, match="Invalid inversion type"):
            DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipeline,
                device="cpu",
                inversion_type="invalid",
            )

    def test_num_inversion_steps_defaults_to_inference_steps(
        self, mock_image_pipeline, mock_scheduler
    ):
        """Test num_inversion_steps defaults to num_inference_steps."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        config = DiffusionConfig(
            scheduler=mock_scheduler,
            pipe=mock_image_pipeline,
            device="cpu",
            num_inference_steps=30,
        )

        assert config.num_inversion_steps == 30

    def test_explicit_num_inversion_steps(self, mock_image_pipeline, mock_scheduler):
        """Test explicit num_inversion_steps."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        config = DiffusionConfig(
            scheduler=mock_scheduler,
            pipe=mock_image_pipeline,
            device="cpu",
            num_inference_steps=30,
            num_inversion_steps=20,
        )

        assert config.num_inversion_steps == 20

    def test_pipeline_type_property(self, mock_image_pipeline, mock_scheduler):
        """Test pipeline_type property."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        config = DiffusionConfig(
            scheduler=mock_scheduler,
            pipe=mock_image_pipeline,
            device="cpu",
        )

        # Should return "image" for StableDiffusionPipeline
        assert config.pipeline_type == "image"

    def test_is_image_pipeline_property(self, mock_image_pipeline, mock_scheduler):
        """Test is_image_pipeline property."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        config = DiffusionConfig(
            scheduler=mock_scheduler,
            pipe=mock_image_pipeline,
            device="cpu",
        )

        assert config.is_image_pipeline is True
        assert config.is_video_pipeline is False

    def test_gen_kwargs_stored(self, mock_image_pipeline, mock_scheduler):
        """Test that extra kwargs are stored in gen_kwargs."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        config = DiffusionConfig(
            scheduler=mock_scheduler,
            pipe=mock_image_pipeline,
            device="cpu",
            custom_param="value",
            another_param=42,
        )

        assert config.gen_kwargs["custom_param"] == "value"
        assert config.gen_kwargs["another_param"] == 42


# ============================================================================
# Integration Tests
# ============================================================================

class TestMediaConversionRoundTrip:
    """Integration tests for media conversion round trips."""

    def test_pil_torch_pil_roundtrip(self):
        """Test PIL -> Torch -> numpy -> PIL roundtrip."""
        from markdiffusion.utils.media_utils import pil_to_torch, torch_to_numpy, numpy_to_pil

        original = Image.new("RGB", (64, 64), color=(128, 64, 192))
        tensor = pil_to_torch(original, normalize=True)
        tensor = tensor.unsqueeze(0)  # Add batch dim
        numpy_arr = torch_to_numpy(tensor)
        result = numpy_to_pil(numpy_arr[0])

        # Check sizes match
        assert result.size == original.size

    def test_numpy_pil_numpy_roundtrip(self):
        """Test numpy -> PIL -> numpy roundtrip."""
        from markdiffusion.utils.media_utils import numpy_to_pil, pil_to_cv2

        original = np.random.rand(64, 64, 3).astype(np.float32)
        pil_img = numpy_to_pil(original)
        result = pil_to_cv2(pil_img)

        # Shape should be preserved
        assert result.shape == original.shape


# ============================================================================
# Additional Tests for media_utils.py - Coverage Improvement
# ============================================================================

class TestSetInversion:
    """Tests for set_inversion function."""

    def test_set_inversion_ddim(self):
        """Test set_inversion with ddim type."""
        from markdiffusion.utils.media_utils import set_inversion

        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.unet = MagicMock()
        mock_pipe.device = "cpu"

        with patch("markdiffusion.inversions.DDIMInversion") as mock_ddim:
            mock_ddim.return_value = MagicMock()
            result = set_inversion(mock_pipe, "ddim")
            mock_ddim.assert_called_once_with(mock_pipe.scheduler, mock_pipe.unet, mock_pipe.device)
            assert result is not None

    def test_set_inversion_exact(self):
        """Test set_inversion with exact type."""
        from markdiffusion.utils.media_utils import set_inversion

        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.unet = MagicMock()
        mock_pipe.device = "cpu"

        with patch("markdiffusion.inversions.ExactInversion") as mock_exact:
            mock_exact.return_value = MagicMock()
            result = set_inversion(mock_pipe, "exact")
            mock_exact.assert_called_once_with(mock_pipe.scheduler, mock_pipe.unet, mock_pipe.device)
            assert result is not None

    def test_set_inversion_invalid_type(self):
        """Test set_inversion with invalid type raises ValueError."""
        from markdiffusion.utils.media_utils import set_inversion

        mock_pipe = MagicMock()
        with pytest.raises(ValueError, match="Invalid inversion type"):
            set_inversion(mock_pipe, "invalid")


class TestTensor2Vid:
    """Tests for tensor2vid function."""

    def test_tensor2vid_np_output(self):
        """Test tensor2vid with numpy output."""
        from markdiffusion.utils.media_utils import tensor2vid

        # Create mock video tensor [B, C, F, H, W]
        video = torch.rand(1, 3, 4, 64, 64)
        
        mock_processor = MagicMock()
        mock_processor.postprocess.return_value = np.random.rand(4, 64, 64, 3)
        
        result = tensor2vid(video, mock_processor, output_type="np")
        
        assert isinstance(result, np.ndarray)
        mock_processor.postprocess.assert_called()

    def test_tensor2vid_pt_output(self):
        """Test tensor2vid with pytorch output."""
        from markdiffusion.utils.media_utils import tensor2vid

        video = torch.rand(1, 3, 4, 64, 64)
        
        mock_processor = MagicMock()
        mock_processor.postprocess.return_value = torch.rand(4, 64, 64, 3)
        
        result = tensor2vid(video, mock_processor, output_type="pt")
        
        assert isinstance(result, torch.Tensor)

    def test_tensor2vid_pil_output(self):
        """Test tensor2vid with PIL output."""
        from markdiffusion.utils.media_utils import tensor2vid

        video = torch.rand(1, 3, 4, 64, 64)
        
        mock_processor = MagicMock()
        mock_processor.postprocess.return_value = [Image.new("RGB", (64, 64)) for _ in range(4)]
        
        result = tensor2vid(video, mock_processor, output_type="pil")
        
        assert isinstance(result, list)

    def test_tensor2vid_invalid_output_type(self):
        """Test tensor2vid with invalid output type raises ValueError."""
        from markdiffusion.utils.media_utils import tensor2vid

        video = torch.rand(1, 3, 4, 64, 64)
        mock_processor = MagicMock()
        mock_processor.postprocess.return_value = "invalid"
        
        with pytest.raises(ValueError, match="does not exist"):
            tensor2vid(video, mock_processor, output_type="invalid")


class TestGetMediaLatents:
    """Tests for get_media_latents function."""

    def test_get_media_latents_image(self):
        """Test get_media_latents for image pipeline."""
        from markdiffusion.utils.media_utils import get_media_latents

        mock_pipe = MagicMock()
        
        image = torch.rand(1, 3, 512, 512)
        
        with patch("markdiffusion.utils.media_utils.get_pipeline_type", return_value="image"):
            with patch("markdiffusion.utils.media_utils._get_image_latents") as mock_get:
                mock_get.return_value = torch.rand(1, 4, 64, 64)
                result = get_media_latents(mock_pipe, image)
                mock_get.assert_called_once()
                assert result is not None

    def test_get_media_latents_t2v(self):
        """Test get_media_latents for text-to-video pipeline."""
        from markdiffusion.utils.media_utils import get_media_latents
        from diffusers import TextToVideoSDPipeline

        mock_pipe = MagicMock(spec=TextToVideoSDPipeline)
        
        video = torch.rand(8, 3, 512, 512)
        
        with patch("markdiffusion.utils.media_utils.get_pipeline_type", return_value="t2v"):
            with patch("markdiffusion.utils.media_utils._get_video_latents") as mock_get:
                mock_get.return_value = torch.rand(1, 4, 8, 64, 64)
                result = get_media_latents(mock_pipe, video)
                mock_get.assert_called_once()
                assert result is not None

    def test_get_media_latents_unsupported_pipeline(self):
        """Test get_media_latents with unsupported pipeline type."""
        from markdiffusion.utils.media_utils import get_media_latents

        mock_pipe = MagicMock()
        image = torch.rand(1, 3, 512, 512)
        
        with patch("markdiffusion.utils.media_utils.get_pipeline_type", return_value=None):
            with pytest.raises(ValueError, match="Unsupported pipeline type"):
                get_media_latents(mock_pipe, image)


class TestDecodeMediaLatents:
    """Tests for decode_media_latents function."""

    def test_decode_media_latents_image(self):
        """Test decode_media_latents for image pipeline."""
        from markdiffusion.utils.media_utils import decode_media_latents
        from diffusers import StableDiffusionPipeline

        mock_pipe = MagicMock(spec=StableDiffusionPipeline)
        
        latents = torch.rand(1, 4, 64, 64)
        
        with patch("markdiffusion.utils.media_utils.get_pipeline_type", return_value="image"):
            with patch("markdiffusion.utils.media_utils._decode_image_latents") as mock_decode:
                mock_decode.return_value = torch.rand(1, 3, 512, 512)
                result = decode_media_latents(mock_pipe, latents)
                mock_decode.assert_called_once()
                assert result is not None

    def test_decode_media_latents_video(self):
        """Test decode_media_latents for video pipeline."""
        from markdiffusion.utils.media_utils import decode_media_latents
        from diffusers import TextToVideoSDPipeline

        mock_pipe = MagicMock(spec=TextToVideoSDPipeline)
        
        latents = torch.rand(1, 4, 8, 64, 64)
        
        with patch("markdiffusion.utils.media_utils.get_pipeline_type", return_value="t2v"):
            with patch("markdiffusion.utils.media_utils._decode_video_latents") as mock_decode:
                mock_decode.return_value = np.random.rand(1, 8, 512, 512, 3)
                result = decode_media_latents(mock_pipe, latents)
                mock_decode.assert_called_once()
                assert result is not None

    def test_decode_media_latents_unsupported_pipeline(self):
        """Test decode_media_latents with unsupported pipeline type."""
        from markdiffusion.utils.media_utils import decode_media_latents

        mock_pipe = MagicMock()
        latents = torch.rand(1, 4, 64, 64)
        
        with patch("markdiffusion.utils.media_utils.get_pipeline_type", return_value=None):
            with pytest.raises(ValueError, match="Unsupported pipeline type"):
                decode_media_latents(mock_pipe, latents)


class TestGetVideoLatents:
    """Tests for _get_video_latents function."""

    def test_get_video_latents_sample(self):
        """Test _get_video_latents with sampling."""
        from markdiffusion.utils.media_utils import _get_video_latents

        mock_pipe = MagicMock()
        mock_dist = MagicMock()
        mock_dist.sample.return_value = torch.rand(8, 4, 64, 64)
        mock_pipe.vae.encode.return_value.latent_dist = mock_dist
        
        video_frames = torch.rand(8, 3, 512, 512)
        
        result = _get_video_latents(mock_pipe, video_frames, sample=True, permute=True)
        
        assert result is not None
        mock_dist.sample.assert_called_once()

    def test_get_video_latents_mode(self):
        """Test _get_video_latents with mode (no sampling)."""
        from markdiffusion.utils.media_utils import _get_video_latents

        mock_pipe = MagicMock()
        mock_dist = MagicMock()
        mock_dist.mode.return_value = torch.rand(8, 4, 64, 64)
        mock_pipe.vae.encode.return_value.latent_dist = mock_dist
        
        video_frames = torch.rand(8, 3, 512, 512)
        
        result = _get_video_latents(mock_pipe, video_frames, sample=False, permute=False)
        
        assert result is not None
        mock_dist.mode.assert_called_once()

    def test_get_video_latents_decoder_inv_not_implemented(self):
        """Test _get_video_latents raises NotImplementedError for decoder_inv."""
        from markdiffusion.utils.media_utils import _get_video_latents

        mock_pipe = MagicMock()
        mock_dist = MagicMock()
        mock_dist.sample.return_value = torch.rand(8, 4, 64, 64)
        mock_pipe.vae.encode.return_value.latent_dist = mock_dist
        
        video_frames = torch.rand(8, 3, 512, 512)
        
        with pytest.raises(NotImplementedError, match="Decoder inversion is not implemented"):
            _get_video_latents(mock_pipe, video_frames, decoder_inv=True)


class TestDecodeVideoLatents:
    """Tests for _decode_video_latents function."""

    def test_decode_video_latents_with_num_frames(self):
        """Test _decode_video_latents with num_frames specified."""
        from markdiffusion.utils.media_utils import _decode_video_latents

        mock_pipe = MagicMock()
        mock_pipe.decode_latents.return_value = torch.rand(1, 3, 8, 64, 64)
        mock_pipe.video_processor.postprocess.return_value = np.random.rand(8, 64, 64, 3)
        
        latents = torch.rand(1, 4, 8, 64, 64)
        
        with patch("markdiffusion.utils.media_utils.tensor2vid") as mock_tensor2vid:
            mock_tensor2vid.return_value = np.random.rand(1, 8, 64, 64, 3)
            result = _decode_video_latents(mock_pipe, latents, num_frames=8)
            mock_pipe.decode_latents.assert_called_once_with(latents, 8)

    def test_decode_video_latents_without_num_frames(self):
        """Test _decode_video_latents without num_frames."""
        from markdiffusion.utils.media_utils import _decode_video_latents

        mock_pipe = MagicMock()
        mock_pipe.decode_latents.return_value = torch.rand(1, 3, 8, 64, 64)
        
        latents = torch.rand(1, 4, 8, 64, 64)
        
        with patch("markdiffusion.utils.media_utils.tensor2vid") as mock_tensor2vid:
            mock_tensor2vid.return_value = np.random.rand(1, 8, 64, 64, 3)
            result = _decode_video_latents(mock_pipe, latents)
            mock_pipe.decode_latents.assert_called_once_with(latents)


class TestGetImageLatents:
    """Tests for _get_image_latents function."""

    def test_get_image_latents_sample(self):
        """Test _get_image_latents with sampling."""
        from markdiffusion.utils.media_utils import _get_image_latents

        mock_pipe = MagicMock()
        mock_dist = MagicMock()
        mock_dist.sample.return_value = torch.rand(1, 4, 64, 64)
        mock_pipe.vae.encode.return_value.latent_dist = mock_dist
        
        image = torch.rand(1, 3, 512, 512)
        
        result = _get_image_latents(mock_pipe, image, sample=True)
        
        assert result is not None
        mock_dist.sample.assert_called_once()

    def test_get_image_latents_mode(self):
        """Test _get_image_latents with mode (no sampling)."""
        from markdiffusion.utils.media_utils import _get_image_latents

        mock_pipe = MagicMock()
        mock_dist = MagicMock()
        mock_dist.mode.return_value = torch.rand(1, 4, 64, 64)
        mock_pipe.vae.encode.return_value.latent_dist = mock_dist
        
        image = torch.rand(1, 3, 512, 512)
        
        result = _get_image_latents(mock_pipe, image, sample=False)
        
        assert result is not None
        mock_dist.mode.assert_called_once()


class TestDecodeImageLatents:
    """Tests for _decode_image_latents function."""

    def test_decode_image_latents(self):
        """Test _decode_image_latents basic functionality."""
        from markdiffusion.utils.media_utils import _decode_image_latents

        mock_pipe = MagicMock()
        mock_pipe.vae.decode.return_value = [torch.rand(1, 3, 512, 512)]
        
        latents = torch.rand(1, 4, 64, 64)
        
        result = _decode_image_latents(mock_pipe, latents)
        
        assert result is not None
        mock_pipe.vae.decode.assert_called_once()


# ============================================================================
# Additional Tests for diffusion_config.py - Coverage Improvement
# ============================================================================

class TestDiffusionConfig:
    """Comprehensive tests for DiffusionConfig class."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        return MagicMock(spec=DPMSolverMultistepScheduler)

    @pytest.fixture
    def mock_image_pipe(self):
        """Create a mock StableDiffusionPipeline."""
        return MagicMock(spec=StableDiffusionPipeline)

    @pytest.fixture
    def mock_t2v_pipe(self):
        """Create a mock TextToVideoSDPipeline."""
        return MagicMock(spec=TextToVideoSDPipeline)

    @pytest.fixture
    def mock_i2v_pipe(self):
        """Create a mock StableVideoDiffusionPipeline."""
        return MagicMock(spec=StableVideoDiffusionPipeline)

    def test_basic_initialization_with_defaults(self, mock_scheduler, mock_image_pipe):
        """Test basic initialization with default parameters."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda"
            )

            assert config.device == "cuda"
            assert config.scheduler == mock_scheduler
            assert config.pipe == mock_image_pipe
            assert config.guidance_scale == 7.5
            assert config.num_images == 1
            assert config.num_inference_steps == 50
            assert config.num_inversion_steps == 50
            assert config.image_size == (512, 512)
            assert config.dtype == torch.float16
            assert config.gen_seed == 0
            assert config.init_latents_seed == 0
            assert config.inversion_type == "ddim"
            assert config.num_frames == -1

    def test_initialization_with_custom_parameters(self, mock_scheduler, mock_image_pipe):
        """Test initialization with custom parameters."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cpu",
                guidance_scale=10.0,
                num_images=4,
                num_inference_steps=100,
                num_inversion_steps=80,
                image_size=(768, 768),
                dtype=torch.float32,
                gen_seed=42,
                init_latents_seed=123,
                inversion_type="exact",
                num_frames=-1
            )

            assert config.device == "cpu"
            assert config.guidance_scale == 10.0
            assert config.num_images == 4
            assert config.num_inference_steps == 100
            assert config.num_inversion_steps == 80
            assert config.image_size == (768, 768)
            assert config.dtype == torch.float32
            assert config.gen_seed == 42
            assert config.init_latents_seed == 123
            assert config.inversion_type == "exact"
            assert config.num_frames == -1

    def test_initialization_with_kwargs(self, mock_scheduler, mock_image_pipe):
        """Test initialization with additional kwargs."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda",
                custom_param1="value1",
                custom_param2=42
            )

            assert config.gen_kwargs == {"custom_param1": "value1", "custom_param2": 42}

    def test_num_inversion_steps_defaults_to_num_inference_steps(self, mock_scheduler, mock_image_pipe):
        """Test that num_inversion_steps defaults to num_inference_steps when not provided."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda",
                num_inference_steps=75
            )

            assert config.num_inversion_steps == 75

    def test_invalid_inversion_type_raises_error(self, mock_scheduler, mock_image_pipe):
        """Test that invalid inversion type raises AssertionError."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            with pytest.raises(AssertionError, match="Invalid inversion type"):
                DiffusionConfig(
                    scheduler=mock_scheduler,
                    pipe=mock_image_pipe,
                    device="cuda",
                    inversion_type="invalid_type"
                )

    def test_image_pipeline_properties(self, mock_scheduler, mock_image_pipe):
        """Test properties for image pipeline."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda"
            )

            assert config.pipeline_type == 'image'
            assert config.is_image_pipeline == True
            assert config.is_video_pipeline == False

    def test_t2v_pipeline_properties(self, mock_scheduler, mock_t2v_pipe):
        """Test properties for text-to-video pipeline."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='t2v'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_t2v_pipe,
                device="cuda",
                num_frames=16
            )

            assert config.pipeline_type == 't2v'
            assert config.is_video_pipeline == True
            assert config.is_image_pipeline == False
            assert config.num_frames == 16

    def test_i2v_pipeline_properties(self, mock_scheduler, mock_i2v_pipe):
        """Test properties for image-to-video pipeline."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='i2v'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_i2v_pipe,
                device="cuda",
                num_frames=24
            )

            assert config.pipeline_type == 'i2v'
            assert config.is_video_pipeline == True
            assert config.is_image_pipeline == False
            assert config.num_frames == 24

    def test_image_pipeline_auto_corrects_num_frames(self, mock_scheduler, mock_image_pipe):
        """Test that image pipeline auto-corrects num_frames to -1."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda",
                num_frames=16  # Invalid for image pipeline
            )

            # Should be auto-corrected to -1
            assert config.num_frames == -1

    def test_video_pipeline_requires_positive_num_frames(self, mock_scheduler, mock_t2v_pipe):
        """Test that video pipeline requires num_frames >= 1."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='t2v'):
            with pytest.raises(ValueError, match="num_frames must be >= 1"):
                DiffusionConfig(
                    scheduler=mock_scheduler,
                    pipe=mock_t2v_pipe,
                    device="cuda",
                    num_frames=-1  # Invalid for video pipeline
                )

    def test_video_pipeline_with_zero_frames_raises_error(self, mock_scheduler, mock_t2v_pipe):
        """Test that video pipeline with num_frames=0 raises error."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='t2v'):
            with pytest.raises(ValueError, match="num_frames must be >= 1"):
                DiffusionConfig(
                    scheduler=mock_scheduler,
                    pipe=mock_t2v_pipe,
                    device="cuda",
                    num_frames=0
                )

    def test_unsupported_pipeline_type_raises_error(self, mock_scheduler):
        """Test that unsupported pipeline type raises ValueError."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        unsupported_pipe = MagicMock()

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value=None):
            with pytest.raises(ValueError, match="Unsupported pipeline type"):
                DiffusionConfig(
                    scheduler=mock_scheduler,
                    pipe=unsupported_pipe,
                    device="cuda"
                )

    def test_ddim_inversion_type(self, mock_scheduler, mock_image_pipe):
        """Test DDIM inversion type."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda",
                inversion_type="ddim"
            )

            assert config.inversion_type == "ddim"

    def test_exact_inversion_type(self, mock_scheduler, mock_image_pipe):
        """Test exact inversion type."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda",
                inversion_type="exact"
            )

            assert config.inversion_type == "exact"

    def test_different_devices(self, mock_scheduler, mock_image_pipe):
        """Test initialization with different device types."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            # Test CPU
            config_cpu = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cpu"
            )
            assert config_cpu.device == "cpu"

            # Test CUDA
            config_cuda = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda"
            )
            assert config_cuda.device == "cuda"

            # Test specific CUDA device
            config_cuda_0 = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda:0"
            )
            assert config_cuda_0.device == "cuda:0"

    def test_different_dtypes(self, mock_scheduler, mock_image_pipe):
        """Test initialization with different dtypes."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            # Test float16
            config_fp16 = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda",
                dtype=torch.float16
            )
            assert config_fp16.dtype == torch.float16

            # Test float32
            config_fp32 = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda",
                dtype=torch.float32
            )
            assert config_fp32.dtype == torch.float32

            # Test bfloat16
            config_bf16 = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_image_pipe,
                device="cuda",
                dtype=torch.bfloat16
            )
            assert config_bf16.dtype == torch.bfloat16

    def test_various_image_sizes(self, mock_scheduler, mock_image_pipe):
        """Test initialization with various image sizes."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        test_sizes = [
            (256, 256),
            (512, 512),
            (768, 768),
            (1024, 1024),
            (512, 768),  # Non-square
        ]

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            for size in test_sizes:
                config = DiffusionConfig(
                    scheduler=mock_scheduler,
                    pipe=mock_image_pipe,
                    device="cuda",
                    image_size=size
                )
                assert config.image_size == size

    def test_various_guidance_scales(self, mock_scheduler, mock_image_pipe):
        """Test initialization with various guidance scales."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        test_scales = [1.0, 5.0, 7.5, 10.0, 15.0, 20.0]

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            for scale in test_scales:
                config = DiffusionConfig(
                    scheduler=mock_scheduler,
                    pipe=mock_image_pipe,
                    device="cuda",
                    guidance_scale=scale
                )
                assert config.guidance_scale == scale

    def test_various_num_inference_steps(self, mock_scheduler, mock_image_pipe):
        """Test initialization with various num_inference_steps."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        test_steps = [10, 25, 50, 100, 150]

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='image'):
            for steps in test_steps:
                config = DiffusionConfig(
                    scheduler=mock_scheduler,
                    pipe=mock_image_pipe,
                    device="cuda",
                    num_inference_steps=steps
                )
                assert config.num_inference_steps == steps
                assert config.num_inversion_steps == steps  # Should default to same value

    def test_i2v_pipeline_validation(self, mock_scheduler, mock_i2v_pipe):
        """Test validation for image-to-video pipeline."""
        from markdiffusion.utils.diffusion_config import DiffusionConfig

        with patch('markdiffusion.utils.diffusion_config.get_pipeline_type', return_value='i2v'):
            # Valid configuration
            config = DiffusionConfig(
                scheduler=mock_scheduler,
                pipe=mock_i2v_pipe,
                device="cuda",
                num_frames=14
            )
            assert config.num_frames == 14


# ============================================================================
# Additional Tests for utils.py - Coverage Improvement  
# ============================================================================

class TestLoadConfigFileAdditional:
    """Additional tests for load_config_file function."""

    def test_load_config_file_unexpected_error(self, tmp_path, capsys):
        """Test load_config_file handles unexpected errors."""
        from markdiffusion.utils.utils import load_config_file

        # Create a file that will cause an unexpected error by making it a directory
        config_path = tmp_path / "config"
        config_path.mkdir()
        
        result = load_config_file(str(config_path))
        assert result is None
        
        captured = capsys.readouterr()
        assert "unexpected error" in captured.out.lower() or "error" in captured.out.lower()


class TestCreateDirectoryForFileAdditional:
    """Additional tests for create_directory_for_file function."""

    def test_create_directory_for_file_with_relative_path(self, tmp_path):
        """Test create_directory_for_file with relative path in tmp directory."""
        from markdiffusion.utils.utils import create_directory_for_file
        import os
        
        # Create a file path with a subdirectory in tmp_path
        file_path = str(tmp_path / "subdir" / "file.txt")
        create_directory_for_file(file_path)
        
        # Verify the directory was created
        assert (tmp_path / "subdir").exists()
        assert (tmp_path / "subdir").is_dir()