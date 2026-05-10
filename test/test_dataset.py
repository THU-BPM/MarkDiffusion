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
Unit tests for dataset classes in MarkDiffusion.

Tests cover:
- BaseDataset: Base class functionality
- StableDiffusionPromptsDataset: Prompt-only dataset
- MSCOCODataset: Image-caption dataset
- VBenchDataset: Video benchmark dataset
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image
import pandas as pd

from markdiffusion.evaluation.dataset import (
    BaseDataset,
    StableDiffusionPromptsDataset,
    MSCOCODataset,
    VBenchDataset,
)


# ============================================================================
# Tests for BaseDataset
# ============================================================================

class TestBaseDataset:
    """Tests for BaseDataset class."""

    def test_initialization_with_default_max_samples(self):
        """Test BaseDataset initializes with default max_samples."""
        dataset = BaseDataset()
        assert dataset.max_samples == 200
        assert dataset.prompts == []
        assert dataset.references == []

    def test_initialization_with_custom_max_samples(self):
        """Test BaseDataset initializes with custom max_samples."""
        dataset = BaseDataset(max_samples=50)
        assert dataset.max_samples == 50

    def test_num_samples_property(self):
        """Test num_samples returns length of prompts."""
        dataset = BaseDataset()
        assert dataset.num_samples == 0

        dataset.prompts = ["prompt1", "prompt2", "prompt3"]
        assert dataset.num_samples == 3

    def test_num_references_property(self):
        """Test num_references returns length of references."""
        dataset = BaseDataset()
        assert dataset.num_references == 0

        dataset.references = [Mock(), Mock()]
        assert dataset.num_references == 2

    def test_len_method(self):
        """Test __len__ returns num_samples."""
        dataset = BaseDataset()
        dataset.prompts = ["a", "b", "c", "d"]
        assert len(dataset) == 4

    def test_get_prompt(self):
        """Test get_prompt returns correct prompt at index."""
        dataset = BaseDataset()
        dataset.prompts = ["first", "second", "third"]

        assert dataset.get_prompt(0) == "first"
        assert dataset.get_prompt(1) == "second"
        assert dataset.get_prompt(2) == "third"

    def test_get_prompt_index_error(self):
        """Test get_prompt raises IndexError for invalid index."""
        dataset = BaseDataset()
        dataset.prompts = ["only_one"]

        with pytest.raises(IndexError):
            dataset.get_prompt(5)

    def test_get_reference(self):
        """Test get_reference returns correct reference at index."""
        dataset = BaseDataset()
        mock_images = [Mock(spec=Image.Image), Mock(spec=Image.Image)]
        dataset.references = mock_images

        assert dataset.get_reference(0) == mock_images[0]
        assert dataset.get_reference(1) == mock_images[1]

    def test_get_reference_index_error(self):
        """Test get_reference raises IndexError for invalid index."""
        dataset = BaseDataset()
        dataset.references = []

        with pytest.raises(IndexError):
            dataset.get_reference(0)

    def test_getitem_without_references(self):
        """Test __getitem__ returns only prompt when no references."""
        dataset = BaseDataset()
        dataset.prompts = ["prompt1", "prompt2"]

        assert dataset[0] == "prompt1"
        assert dataset[1] == "prompt2"

    def test_getitem_with_references(self):
        """Test __getitem__ returns (prompt, reference) tuple when references exist."""
        dataset = BaseDataset()
        dataset.prompts = ["prompt1", "prompt2"]
        mock_images = [Mock(spec=Image.Image), Mock(spec=Image.Image)]
        dataset.references = mock_images

        result = dataset[0]
        assert isinstance(result, tuple)
        assert result[0] == "prompt1"
        assert result[1] == mock_images[0]

    def test_load_data_is_noop(self):
        """Test _load_data does nothing in base class."""
        dataset = BaseDataset()
        dataset._load_data()  # Should not raise
        assert dataset.prompts == []
        assert dataset.references == []


# ============================================================================
# Tests for StableDiffusionPromptsDataset
# ============================================================================

class TestStableDiffusionPromptsDataset:
    """Tests for StableDiffusionPromptsDataset class."""

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_initialization(self, mock_load_dataset):
        """Test dataset initializes correctly."""
        # Setup mock
        mock_data = {"Prompt": ["prompt1", "prompt2", "prompt3"]}
        mock_load_dataset.return_value = mock_data

        dataset = StableDiffusionPromptsDataset(max_samples=2)

        assert dataset.max_samples == 2
        assert dataset.split == "test"
        assert dataset.shuffle is False
        mock_load_dataset.assert_called_once()

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_name_property(self, mock_load_dataset):
        """Test name property returns correct name."""
        mock_load_dataset.return_value = {"Prompt": []}
        dataset = StableDiffusionPromptsDataset(max_samples=1)
        assert dataset.name == "Stable Diffusion Prompts"

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_prompts_loaded(self, mock_load_dataset):
        """Test prompts are loaded from dataset."""
        test_prompts = ["A cat sitting on a mat", "A dog running in park", "A bird flying"]
        mock_load_dataset.return_value = {"Prompt": test_prompts}

        dataset = StableDiffusionPromptsDataset(max_samples=3)

        assert len(dataset.prompts) == 3
        assert dataset.prompts == test_prompts

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_max_samples_limit(self, mock_load_dataset):
        """Test max_samples limits the number of prompts loaded."""
        test_prompts = ["p1", "p2", "p3", "p4", "p5"]
        mock_load_dataset.return_value = {"Prompt": test_prompts}

        dataset = StableDiffusionPromptsDataset(max_samples=2)

        assert len(dataset.prompts) == 2
        assert dataset.prompts == ["p1", "p2"]

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_shuffle_option(self, mock_load_dataset):
        """Test shuffle option is passed to dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = lambda self, key: ["p1", "p2"]
        mock_dataset.shuffle.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        dataset = StableDiffusionPromptsDataset(max_samples=2, shuffle=True)

        assert dataset.shuffle is True
        mock_dataset.shuffle.assert_called_once()

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_custom_split(self, mock_load_dataset):
        """Test custom split option."""
        mock_load_dataset.return_value = {"Prompt": ["p1"]}

        dataset = StableDiffusionPromptsDataset(max_samples=1, split="train")

        assert dataset.split == "train"
        mock_load_dataset.assert_called_with(
            "dataset/stable_diffusion_prompts", split="train"
        )

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_no_references(self, mock_load_dataset):
        """Test that StableDiffusionPromptsDataset has no references."""
        mock_load_dataset.return_value = {"Prompt": ["p1", "p2"]}

        dataset = StableDiffusionPromptsDataset(max_samples=2)

        assert dataset.num_references == 0
        assert dataset.references == []


# ============================================================================
# Tests for MSCOCODataset
# ============================================================================

class TestMSCOCODataset:
    """Tests for MSCOCODataset class."""

    @patch('markdiffusion.evaluation.dataset.pd.read_parquet')
    @patch('markdiffusion.evaluation.dataset.tqdm')
    def test_initialization(self, mock_tqdm, mock_read_parquet):
        """Test dataset initializes correctly."""
        # Setup mock DataFrame
        mock_df = pd.DataFrame({
            'TEXT': ['caption1', 'caption2'],
            'URL': ['http://example.com/1.jpg', 'http://example.com/2.jpg']
        })
        mock_read_parquet.return_value = mock_df
        mock_tqdm.return_value = range(2)

        with patch.object(MSCOCODataset, '_load_image_from_url', return_value=Mock(spec=Image.Image)):
            dataset = MSCOCODataset(max_samples=2)

        assert dataset.max_samples == 2
        assert dataset.shuffle is False

    @patch('markdiffusion.evaluation.dataset.pd.read_parquet')
    def test_name_property(self, mock_read_parquet):
        """Test name property returns correct name."""
        mock_read_parquet.return_value = pd.DataFrame({'TEXT': [], 'URL': []})

        with patch.object(MSCOCODataset, '_load_data'):
            dataset = MSCOCODataset.__new__(MSCOCODataset)
            dataset.max_samples = 0
            dataset.prompts = []
            dataset.references = []
            dataset.shuffle = False

        assert dataset.name == "MS-COCO 2017"

    @patch('markdiffusion.evaluation.dataset.requests.get')
    def test_load_image_from_url_success(self, mock_get):
        """Test _load_image_from_url successfully loads an image."""
        # Create a mock response with image data
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        # Create a simple PNG image bytes
        from io import BytesIO
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        mock_response.content = img_bytes.getvalue()
        mock_get.return_value = mock_response

        # Create dataset instance without loading data
        dataset = BaseDataset.__new__(MSCOCODataset)
        result = dataset._load_image_from_url("http://example.com/image.jpg")

        assert isinstance(result, Image.Image)
        mock_get.assert_called_once_with("http://example.com/image.jpg")

    @patch('markdiffusion.evaluation.dataset.requests.get')
    def test_load_image_from_url_failure(self, mock_get, capsys):
        """Test _load_image_from_url returns None on failure."""
        mock_get.side_effect = Exception("Connection error")

        dataset = BaseDataset.__new__(MSCOCODataset)
        result = dataset._load_image_from_url("http://example.com/bad.jpg")

        assert result is None
        captured = capsys.readouterr()
        assert "Load image from url failed" in captured.out

    @patch('markdiffusion.evaluation.dataset.pd.read_parquet')
    @patch('markdiffusion.evaluation.dataset.tqdm')
    def test_shuffle_option(self, mock_tqdm, mock_read_parquet):
        """Test shuffle option shuffles the DataFrame."""
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.iloc = MagicMock()
        mock_df.sample.return_value.reset_index.return_value = mock_df
        mock_read_parquet.return_value = mock_df
        mock_tqdm.return_value = []

        with patch.object(MSCOCODataset, '_load_image_from_url'):
            dataset = MSCOCODataset(max_samples=0, shuffle=True)

        assert dataset.shuffle is True
        mock_df.sample.assert_called_once_with(frac=1)


# ============================================================================
# Tests for VBenchDataset
# ============================================================================

class TestVBenchDataset:
    """Tests for VBenchDataset class."""

    @patch('builtins.open')
    def test_initialization(self, mock_open):
        """Test dataset initializes correctly."""
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_file.readlines.return_value = ["prompt1\n", "prompt2\n", "prompt3\n"]
        mock_open.return_value = mock_file

        dataset = VBenchDataset(max_samples=2, dimension="subject_consistency")

        assert dataset.max_samples == 2
        assert dataset.dimension == "subject_consistency"
        assert dataset.shuffle is False

    @patch('builtins.open')
    def test_name_property(self, mock_open):
        """Test name property returns correct name."""
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_file.readlines.return_value = []
        mock_open.return_value = mock_file

        dataset = VBenchDataset(max_samples=0, dimension="test")

        assert dataset.name == "VBench"

    @patch('builtins.open')
    def test_prompts_loaded(self, mock_open):
        """Test prompts are loaded from file."""
        test_prompts = ["A man walking\n", "A car driving\n", "A plane flying\n"]
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_file.readlines.return_value = test_prompts
        mock_open.return_value = mock_file

        dataset = VBenchDataset(max_samples=3, dimension="test")

        assert len(dataset.prompts) == 3
        assert dataset.prompts == ["A man walking", "A car driving", "A plane flying"]

    @patch('builtins.open')
    def test_max_samples_limit(self, mock_open):
        """Test max_samples limits the number of prompts."""
        test_prompts = ["p1\n", "p2\n", "p3\n", "p4\n", "p5\n"]
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_file.readlines.return_value = test_prompts
        mock_open.return_value = mock_file

        dataset = VBenchDataset(max_samples=2, dimension="test")

        assert len(dataset.prompts) == 2

    @patch('builtins.open')
    @patch('markdiffusion.evaluation.dataset.random.shuffle')
    def test_shuffle_option(self, mock_shuffle, mock_open):
        """Test shuffle option shuffles the prompts."""
        test_prompts = ["p1\n", "p2\n", "p3\n"]
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_file.readlines.return_value = test_prompts
        mock_open.return_value = mock_file

        dataset = VBenchDataset(max_samples=3, dimension="test", shuffle=True)

        assert dataset.shuffle is True
        mock_shuffle.assert_called_once()

    @patch('builtins.open')
    def test_file_path_format(self, mock_open):
        """Test correct file path is used based on dimension."""
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_file.readlines.return_value = []
        mock_open.return_value = mock_file

        dataset = VBenchDataset(max_samples=0, dimension="motion_smoothness")

        mock_open.assert_called_with(
            "dataset/vbench/prompts_per_dimension/motion_smoothness.txt", "r"
        )

    @patch('builtins.open')
    def test_no_references(self, mock_open):
        """Test that VBenchDataset has no references."""
        mock_file = MagicMock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        mock_file.readlines.return_value = ["p1\n"]
        mock_open.return_value = mock_file

        dataset = VBenchDataset(max_samples=1, dimension="test")

        assert dataset.num_references == 0

    def test_file_not_found(self):
        """Test FileNotFoundError when dimension file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            VBenchDataset(max_samples=1, dimension="nonexistent_dimension")


# ============================================================================
# Integration Tests
# ============================================================================

class TestDatasetIntegration:
    """Integration tests for dataset functionality."""

    def test_base_dataset_iteration(self):
        """Test iterating over BaseDataset."""
        dataset = BaseDataset()
        dataset.prompts = ["p1", "p2", "p3"]

        collected = []
        for i in range(len(dataset)):
            collected.append(dataset[i])

        assert collected == ["p1", "p2", "p3"]

    def test_dataset_with_references_iteration(self):
        """Test iterating over dataset with references."""
        dataset = BaseDataset()
        dataset.prompts = ["p1", "p2"]
        mock_images = [Mock(spec=Image.Image), Mock(spec=Image.Image)]
        dataset.references = mock_images

        for i in range(len(dataset)):
            prompt, ref = dataset[i]
            assert prompt == dataset.prompts[i]
            assert ref == mock_images[i]

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_stable_diffusion_dataset_as_base_dataset(self, mock_load_dataset):
        """Test StableDiffusionPromptsDataset works as BaseDataset."""
        mock_load_dataset.return_value = {"Prompt": ["test_prompt"]}

        dataset = StableDiffusionPromptsDataset(max_samples=1)

        # Should have all BaseDataset functionality
        assert isinstance(dataset, BaseDataset)
        assert len(dataset) == dataset.num_samples
        assert dataset.get_prompt(0) == "test_prompt"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestDatasetEdgeCases:
    """Test edge cases for dataset classes."""

    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        dataset = BaseDataset(max_samples=0)
        assert len(dataset) == 0
        assert dataset.num_samples == 0
        assert dataset.num_references == 0

    def test_single_item_dataset(self):
        """Test dataset with single item."""
        dataset = BaseDataset()
        dataset.prompts = ["only_prompt"]

        assert len(dataset) == 1
        assert dataset[0] == "only_prompt"
        assert dataset.get_prompt(0) == "only_prompt"

    def test_large_max_samples(self):
        """Test with very large max_samples."""
        dataset = BaseDataset(max_samples=1000000)
        assert dataset.max_samples == 1000000

    def test_negative_index(self):
        """Test negative indexing behavior."""
        dataset = BaseDataset()
        dataset.prompts = ["first", "second", "third"]

        # Python lists support negative indexing
        assert dataset[-1] == "third"
        assert dataset[-2] == "second"

    @patch('markdiffusion.evaluation.dataset.load_dataset')
    def test_unicode_prompts(self, mock_load_dataset):
        """Test handling of unicode prompts."""
        unicode_prompts = ["日本語プロンプト", "中文提示", "🎨 emoji art"]
        mock_load_dataset.return_value = {"Prompt": unicode_prompts}

        dataset = StableDiffusionPromptsDataset(max_samples=3)

        assert dataset.prompts == unicode_prompts
        assert dataset.get_prompt(0) == "日本語プロンプト"
