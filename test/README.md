# MarkDiffusion Watermark Algorithm Unit Tests

This directory contains all parameterized unit tests for the watermark algorithms and inversion modules in the MarkDiffusion project.

## 📋 Directory Structure

```text
test/
├── test_watermark_algorithms.py  # Main test file for initialization, generation, detection, inversion, visualization of watermark algorithms
├── test_pipelines.py             # Evaluation pipelines tests
├── test_dataset.py               # Dataset classes tests
├── test_utils.py                 # Utility functions tests
├── test_exceptions.py            # Exception classes tests
├── test_image_editor.py          # Image editor module tests
├── test_video_editor.py          # Video editor module tests
├── test_edge_cases.py            # Edge cases tests
├── conftest.py                   # Pytest configuration and fixtures
├── pytest.ini                    # Pytest config file
├── requirements-test.txt         # Test dependencies
└── README.md                     # This document
```

## 🎯 What Is Covered by the Tests

### Watermark Algorithms

#### Image watermark algorithms (9)

- **TR**
- **GS**
- **PRC**
- **RI**
- **SEAL**
- **ROBIN**
- **WIND**
- **GM**
- **SFW**

#### Video watermark algorithms (2)

- **VideoShield**
- **VideoMark**

### Inversion Modules

- **DDIM Inversion** – supports image / video latents 
- **Exact Inversion** – supports image / video latents 

### Visualization Modules

- Visualization support for all image and video watermark algorithms
- Visualization content includes: watermarked images, original latent vectors, inverted latent vectors, frequency-domain analysis, etc.
- Each algorithm has its own dedicated visualizer

### Evaluation Pipelines and Datasets

- **Dataset classes** (3): StableDiffusionPromptsDataset, MSCOCODataset, VBenchDataset
- **Detection pipelines** (2): WatermarkedMediaDetectionPipeline, UnWatermarkedMediaDetectionPipeline
- **Image quality analysis pipelines** (5): DirectImageQualityAnalysisPipeline, ReferencedImageQualityAnalysisPipeline, GroupImageQualityAnalysisPipeline, RepeatImageQualityAnalysisPipeline, ComparedImageQualityAnalysisPipeline
- **Video quality analysis pipeline** (1): DirectVideoQualityAnalysisPipeline
- **Total**: 8 evaluation pipelines + 3 dataset classes

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
pip install -r test/requirements-test.txt
```

Test dependencies include:

- pytest
- pytest-timeout
- pytest-html (optional, to generate HTML reports)
- pytest-cov (optional, for coverage reports)
- pytest-xdist (optional, for parallel testing)

Note: 
- The testing process may require downloading models from Hugging Face. If access is restricted, please set up a proper mirror, such as `export HF_ENDPOINT=https://hf-mirror.com`.
- This test includes **672 test cases, with 94.73% code coverage**. The entire testing process is relatively time-consuming and is expected to take around 30 minutes on a single modern GPU (e.g. H20/A100).


### 2. Run Tests

#### Run directly with pytest

```bash
# Test the whole project and report coverage & HTML report
pytest test -v \
  --cov=. \
  --cov-report=html \
  --cov-report=term-missing \
  --html=report.html

# Test all pipelines
pytest test/test_pipelines.py -v

# Test all algorithms and modules
pytest test/test_watermark_algorithms.py -v

# Test a specific algorithm
pytest test/test_watermark_algorithms.py -v --algorithm TR

# Quick tests (initialization only)
pytest test/test_watermark_algorithms.py -v -k initialization

# Test all utils
pytest test/test_utils.py -v

# Test all exceptions
pytest test/test_exceptions.py -v

# Test all dataset classes
pytest test/test_dataset.py -v

# Test all image editor modules
pytest test/test_image_editor.py -v

# Test all video editor modules
pytest test/test_video_editor.py -v

# Test all edge cases
pytest test/test_edge_cases.py -v
```

## 🏷️ Test Markers

| Marker | Description | How to use |
|--------|-------------|------------|
| `@pytest.mark.image` | Image watermark tests | `-m image` |
| `@pytest.mark.video` | Video watermark tests | `-m video` |
| `@pytest.mark.inversion` | Inversion module tests | `-m inversion` |
| `@pytest.mark.visualization` | Visualization tests | `-m visualization` |
| `@pytest.mark.pipeline` | Pipeline tests | `-m pipeline` |
| `@pytest.mark.detection` | Detection pipeline tests | `-m detection` |
| `@pytest.mark.quality` | Quality analysis pipeline tests | `-m quality` |
| `@pytest.mark.integration` | Integration tests | `-m integration` |
| `@pytest.mark.summary` | Summary tests | `-m summary` |
| `@pytest.mark.slow` | Slow tests (generation and detection) | `-m "not slow"` |

## 📄 License

These test codes follow the MarkDiffusion project’s Apache 2.0 license.
