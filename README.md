<div align="center">

<img src="img/markdiffusion-color-1.jpg" style="width: 65%;"/>

# An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models

[![Home](https://img.shields.io/badge/Home-5F259F?style=for-the-badge&logo=homepage&logoColor=white)](https://generative-watermark.github.io/)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.10569)
[![Models](https://img.shields.io/badge/Models-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Generative-Watermark-Toolkits) 
[![Colab](https://img.shields.io/badge/Google--Colab-%23D97700?style=for-the-badge&logo=Google-colab&logoColor=white)](https://colab.research.google.com/drive/1N1C9elDAB5zwF4FxKKYMCqR3eSpCSqAW?usp=sharing) 
[![DOC](https://img.shields.io/badge/Readthedocs-%2300A89C?style=for-the-badge&logo=readthedocs&logoColor=#8CA1AF)](https://markdiffusion.readthedocs.io) 
[![PYPI](https://img.shields.io/badge/PYPI-%23193440?style=for-the-badge&logo=pypi&logoColor=#3775A9)](https://pypi.org/project/markdiffusion) 
[![CONDA-FORGE](https://img.shields.io/badge/Conda--Forge-%23000000?style=for-the-badge&logo=condaforge&logoColor=#FFFFFF)](https://github.com/conda-forge/markdiffusion-feedstock)




**Language Versions:** [English](README.md) | [中文](README_zh.md) | [Français](README_fr.md) | [Español](README_es.md)
</div>

> 🔥 **As a new released project, We welcome PRs!** If you have implemented a LDM watermarking algorithm or are interested in contributing one, we'd love to include it in MarkDiffusion. Join our community and help make generative watermarking more accessible to everyone!

## Contents
- [Updates](#-updates)
- [Introduction to MarkDiffusion](#-introduction-to-markdiffusion)
  - [Overview](#-overview)
  - [Key Features](#-key-features)
  - [Implemented Algorithms](#-implemented-algorithms)
  - [Evaluation Module](#-evaluation-module)
- [Quick Start](#-quick-start)
    - [Google Colab Demo](#google-colab-demo)
    - [Installation](#installation)
    - [How to Use the Toolkit](#how-to-use-the-toolkit)
- [Test Modules](#-test-modules)
- [Citation](#citation)


## 🔥 Updates
🛠 **(2026.05.15)** Expanded the test suite to 672 unit tests with 94.73% code coverage (full GPU + CPU regression on the new `markdiffusion-test` env).

🏗️ **(2026.05.10)** Restructured the repo into a proper `markdiffusion/` Python package so `pip install -e .` and PyPI installs share the same import paths (`from markdiffusion.watermark import AutoWatermark`). Editable installs and CI now run from a single source layout.

🎯 **(2026.05.10)** Add *DiffusionPurification* and *NeuralCodecCompression* regeneration attacks; *CrSc* (Crop & Scale) gains `position="random"` and explicit offset support — thanks contributors!

🛠 **(2025.12.19)** Add a complete test suite for all functionality with 658 test cases.

🛠 **(2025.12.10)** Add a continuous integration testing system using github actions.

🎯 **(2025.10.10)** Add *Mask, Overlay, AdaptiveNoiseInjection* image attack tools, thanks Zheyu Fu for his PR!

🎯 **(2025.10.09)** Add *FrameRateAdapter, FrameInterpolationAttack* video attack tools, thanks Luyang Si for his PR!

🎯 **(2025.10.08)** Add *SSIM, BRISQUE, VIF, FSIM* image quality analyzer, thanks Huan Wang for her PR!

✨ **(2025.10.07)** Add [SFW](https://arxiv.org/pdf/2509.07647) watermarking method, thanks Huan Wang for her PR!

✨ **(2025.10.07)** Add [VideoMark](https://arxiv.org/abs/2504.16359) watermarking method, thanks Hanqian Li for his PR!

✨ **(2025.9.29)** Add [GaussMarker](https://arxiv.org/abs/2506.11444) watermarking method, thanks Luyang Si for his PR!

## 🔓 Introduction to MarkDiffusion

### 👀 Overview

MarkDiffusion is an open-source Python toolkit for generative watermarking of latent diffusion models. As the use of diffusion-based generative models expands, ensuring the authenticity and origin of generated media becomes critical. MarkDiffusion simplifies the access, understanding, and assessment of watermarking technologies, making it accessible to both researchers and the broader community. *Note: if you are interested in LLM watermarking (text watermark), please refer to the [MarkLLM](https://github.com/THU-BPM/MarkLLM) toolkit from our group.*

The toolkit comprises three key components: a unified implementation framework for streamlined watermarking algorithm integrations and user-friendly interfaces; a mechanism visualization suite that intuitively showcases added and extracted watermark patterns to aid public understanding; and a comprehensive evaluation module offering standard implementations of 33 tools across three essential aspects—detectability, robustness, and output quality, plus 8 automated evaluation pipelines.

<img src="img/fig1_overview.png" alt="MarkDiffusion Overview" style="zoom:50%;" />

### 💍 Key Features

- **Unified Implementation Framework:** MarkDiffusion provides a modular architecture supporting eleven state-of-the-art generative image/video watermarking algorithms of LDMs.

- **Comprehensive Algorithm Support:** Currently implements 11 watermarking algorithms from two major categories: Pattern-based methods (Tree-Ring, Ring-ID, ROBIN, WIND, SFW) and Key-based methods (Gaussian-Shading, PRC, SEAL, VideoShield, GaussMarker, VideoMark).

- **Visualization Solutions:** The toolkit includes custom visualization tools that enable clear and insightful views into how different watermarking algorithms operate under various scenarios. These visualizations help demystify the algorithms' mechanisms, making them more understandable for users.

- **Evaluation Module:** With 33 evaluation tools covering detectability, robustness, and impact on output quality, MarkDiffusion provides comprehensive assessment capabilities. It features 8 automated evaluation pipelines: Watermark Detection Pipeline, Image Quality Analysis Pipeline, Video Quality Analysis Pipeline, and specialized robustness assessment tools.

### ✨ Implemented Algorithms

| **Algorithm** | **Category** | **Target** | **Reference** |
|---------------|-------------|------------|---------------|
| Tree-Ring | Pattern | Image | [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/abs/2305.20030) |
| Ring-ID | Pattern | Image | [RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification](https://arxiv.org/abs/2404.14055) |
| ROBIN | Pattern | Image | [ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization](https://arxiv.org/abs/2411.03862) |
| WIND | Pattern | Image | [Hidden in the Noise: Two-Stage Robust Watermarking for Images](https://arxiv.org/abs/2412.04653) |
| SFW | Pattern | Image | [Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](https://arxiv.org/abs/2509.07647) |
| Gaussian-Shading | Key | Image | [Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models](https://arxiv.org/abs/2404.04956) |
| GaussMarker | Key | Image | [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](https://arxiv.org/abs/2506.11444) |
| PRC | Key | Image | [An undetectable watermark for generative image models](https://arxiv.org/abs/2410.07369) |
| SEAL | Key | Image | [SEAL: Semantic Aware Image Watermarking](https://arxiv.org/abs/2503.12172) |
| VideoShield | Key | Video | [VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking](https://arxiv.org/abs/2501.14195) |
| VideoMark | Key | Video | [VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models](https://arxiv.org/abs/2504.16359) |

### 🎯 Evaluation Module
#### Evaluation Pipelines

MarkDiffusion supports eight pipelines, two for detection (WatermarkedMediaDetectionPipeline and UnWatermarkedMediaDetectionPipeline), and six for quality analysis. The table below details the quality analysis pipelines.

| **Quality Analysis Pipeline** | **Input Type** | **Required Data** | **Applicable Metrics** |  
| --- | --- | --- | --- |
| DirectImageQualityAnalysisPipeline | Single image | Generated watermarked/unwatermarked image | Metrics for single image evaluation | 
| ReferencedImageQualityAnalysisPipeline | Image + reference content | Generated watermarked/unwatermarked image + reference image/text | Metrics requiring computation between single image and reference content (text/image) | 
| GroupImageQualityAnalysisPipeline | Image set (+ reference image set) | Generated watermarked/unwatermarked image set (+reference image set) | Metrics requiring computation on image sets | 
| RepeatImageQualityAnalysisPipeline | Image set | Repeatedly generated watermarked/unwatermarked image set | Metrics for evaluating repeatedly generated image sets | 
| ComparedImageQualityAnalysisPipeline | Two images for comparison | Generated watermarked and unwatermarked images | Metrics measuring differences between two images | 
| DirectVideoQualityAnalysisPipeline | Single video | Generated video frame set | Metrics for overall video evaluation |

#### Evaluation Tools

| **Tool Name** | **Evaluation Category** | **Function Description** | **Output Metrics** |
| --- | --- | --- | --- |
| FundamentalSuccessRateCalculator | Detectability | Calculate classification metrics for fixed-threshold watermark detection | Various classification metrics |
| DynamicThresholdSuccessRateCalculator | Detectability | Calculate classification metrics for dynamic-threshold watermark detection | Various classification metrics |
| **Image Attack Tools** | | | |
| Rotation | Robustness (Image) | Image rotation attack, testing watermark resistance to rotation transforms | Rotated images/frames |
| CrSc (Crop & Scale) | Robustness (Image) | Cropping and scaling attack, evaluating watermark robustness to size changes (supports `position="center"`, `"random"`, or an explicit `(x_ratio, y_ratio)` offset) | Cropped/scaled images/frames |
| GaussianNoise | Robustness (Image) | Gaussian noise attack, testing watermark resistance to noise interference | Noise-corrupted images/frames |
| GaussianBlurring | Robustness (Image) | Gaussian blur attack, evaluating watermark resistance to blur processing | Blurred images/frames |
| JPEGCompression | Robustness (Image) | JPEG compression attack, testing watermark robustness to lossy compression | Compressed images/frames |
| Brightness | Robustness (Image) | Brightness adjustment attack, evaluating watermark resistance to brightness changes | Brightness-modified images/frames |
| Mask | Robustness (Image) | Image masking attack, testing watermark resistance to partial occlusion by random black rectangles | Masked images/frames |
| Overlay | Robustness (Image) | Image overlay attack, testing watermark resistance to graffiti-style strokes and annotations | Overlaid images/frames |
| AdaptiveNoiseInjection | Robustness (Image) | Adaptive noise injection attack, testing watermark resistance to content-aware noise (Gaussian/Salt-pepper/Poisson/Speckle) | Noisy images/frames with adaptive noise |
| DiffusionPurification | Robustness (Image) | Regeneration attack: encodes the image to latent space, injects noise at a configurable schedule fraction, and reverse-denoises through the diffusion pipe. Reference: Nie et al., DiffPure (ICML 2022). | Purified (regenerated) images |
| NeuralCodecCompression | Robustness (Image) | Regeneration attack: round-trips the image through a pretrained learned image codec (compressai, default `cheng2020-anchor`) at a target quality level. Reference: Cheng et al., CVPR 2020. Requires the `[optional]` extras. | Codec-compressed images |
| **Video Attack Tools** | | | |
| MPEG4Compression | Robustness (Video) | MPEG-4 video compression attack, testing video watermark compression robustness | Compressed video frames |
| FrameAverage | Robustness (Video) | Frame averaging attack, destroying watermarks through inter-frame averaging | Averaged video frames |
| FrameSwap | Robustness (Video) | Frame swapping attack, testing robustness by changing frame sequences | Swapped video frames |
| FrameRateAdapter | Robustness (Video) | Frame rate conversion attack that resamples frames while preserving duration | Resampled frame sequence |
| FrameInterpolationAttack | Robustness (Video) | Frame interpolation attack inserting blended frames to alter temporal density | Interpolated video frames |
| **Image Quality Analyzers** | | | |
| InceptionScoreCalculator | Quality (Image) | Evaluate generated image quality and diversity | IS score |
| FIDCalculator | Quality (Image) | Fréchet Inception Distance, measuring distribution difference between generated and real images | FID value |
| LPIPSAnalyzer | Quality (Image) | Learned Perceptual Image Patch Similarity, evaluating perceptual quality | LPIPS distance |
| CLIPScoreCalculator | Quality (Image) | CLIP-based text-image consistency evaluation | CLIP similarity score |
| PSNRAnalyzer | Quality (Image) | Peak Signal-to-Noise Ratio, measuring image distortion | PSNR value (dB) |
| NIQECalculator | Quality (Image) | Natural Image Quality Evaluator, reference-free quality assessment | NIQE score |
| SSIMAnalyzer | Quality (Image) | Structural Similarity Index between two images | SSIM value |
| BRISQUEAnalyzer | Quality (Image) | Blind/Referenceless Image Spatial Quality Evaluator, evaluating perceptual quality of an image without requiring a reference | BRISQUE score |
| VIFAnalyzer | Quality (Image) | Visual Information Fidelity analyzer, comparing a distorted image with a reference image to quantify the amount of visual information preserved | VIF value |
| FSIMAnalyzer | Quality (Image) | Feature Similarity Index analyzer, comparing structural similarity between two images based on phase congruency and gradient magnitude | FSIM value |
| **Video Quality Analyzers** | | | |
| SubjectConsistencyAnalyzer | Quality (Video) | Evaluate consistency of subject objects in video | Subject consistency score |
| BackgroundConsistencyAnalyzer | Quality (Video) | Evaluate background coherence and stability in video | Background consistency score |
| MotionSmoothnessAnalyzer | Quality (Video) | Evaluate smoothness of video motion | Motion smoothness metric |
| DynamicDegreeAnalyzer | Quality (Video) | Measure dynamic level and change magnitude in video | Dynamic degree value |
| ImagingQualityAnalyzer | Quality (Video) | Comprehensive evaluation of video imaging quality | Imaging quality score |

## 🧩 Quick Start
### Google Colab Demo
If you're interested in trying out MarkDiffusion without installing anything, you can use [Google Colab](https://colab.research.google.com/drive/1N1C9elDAB5zwF4FxKKYMCqR3eSpCSqAW?usp=sharing#scrollTo=-kWt7m9Y3o-G) to see how it works.

### Installation

MarkDiffusion can be installed in three ways. Pick the one that matches what you plan to do:

| Mode | Command | Use when... |
|---|---|---|
| **A. PyPI (recommended for users)** | `pip install markdiffusion[optional]` | You just want to *call* watermarking algorithms from your own script/notebook. No need to read or modify the library source, run the bundled tests, or generate visualizations from the demo notebooks. |
| **B. Editable install from source (recommended for contributors/researchers)** | `git clone … && cd MarkDiffusion && pip install -e ".[optional]"` | You want to (a) run the `MarkDiffusion_demo.ipynb` / `test/` suite, (b) modify or add watermarking algorithms, (c) reproduce paper results, or (d) submit PRs. Source edits in `markdiffusion/` take effect immediately. |
| **C. conda-forge (conda-only environments)** | `conda install -c conda-forge markdiffusion` | You are restricted to conda channels. Some advanced features (those depending on PyPI-only packages such as `pyiqa` / `lpips`) are not bundled; install them separately if needed. |

**Mode A — PyPI install:**

```bash
conda create -n markdiffusion python=3.11
conda activate markdiffusion
pip install markdiffusion[optional]
```

**Mode B — Editable install from source:**

```bash
git clone https://github.com/THU-BPM/MarkDiffusion.git
cd MarkDiffusion
conda create -n markdiffusion python=3.11
conda activate markdiffusion
pip install -e ".[optional]"
# (optional) install test extras to run pytest, coverage, parallel tests
pip install -r test/requirements-test.txt
```

`pyproject.toml` pins `torch>=2.4,<2.11` and `setuptools<81` so the resolver picks a CUDA-12.x wheel that works on driver ≥ 525 and avoids dropping `pkg_resources` (still required by `openai-clip`).

**Mode C — conda-forge:**

```bash
conda create -n markdiffusion python=3.11
conda activate markdiffusion
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install markdiffusion
```

### How to Use the Toolkit

The same `markdiffusion.*` import paths work in both PyPI and editable installs. The two demo notebooks differ only in scope:

- **`MarkDiffusion_pypi_demo.ipynb`** — minimal end-to-end usage; safe starting point for PyPI users.
- **`MarkDiffusion_demo.ipynb`** — exhaustive walkthrough of all 11 algorithms plus visualization and evaluation pipelines; ships in the source repo (Mode B).

Here is the minimal end-to-end example for both modes:

```python
import torch
from markdiffusion.watermark import AutoWatermark
from markdiffusion.utils import DiffusionConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "huanzi05/stable-diffusion-2-1-base"
scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    scheduler=scheduler,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
).to(device)

diffusion_config = DiffusionConfig(
    scheduler=scheduler,
    pipe=pipe,
    device=device,
    image_size=(512, 512),
    num_inference_steps=50,
    guidance_scale=7.5,
    gen_seed=42,
    inversion_type="ddim",
)

# AutoWatermark picks up the bundled default config (markdiffusion/config/TR.json)
# automatically. Pass `algorithm_config=` only if you want to override it.
tr_watermark = AutoWatermark.load("TR", diffusion_config=diffusion_config)

prompt = "A beautiful landscape with mountains and a river at sunset"
watermarked_image = tr_watermark.generate_watermarked_media(input_data=prompt)
watermarked_image.save("watermarked_image.png")

detection_result = tr_watermark.detect_watermark_in_media(watermarked_image)
print(detection_result)
```

If you installed in **Mode B**, you can additionally point `algorithm_config=` at a JSON in your working copy (e.g. `markdiffusion/config/TR.json`) to experiment with parameters without reinstalling. From the source repo you can also run the demo notebook end-to-end:

```bash
jupyter nbconvert --to notebook --execute MarkDiffusion_demo.ipynb \
    --ExecutePreprocessor.kernel_name=markdiffusion \
    --ExecutePreprocessor.timeout=1800
```

## 🛠 Test Modules
We provide a comprehensive set of test modules to ensure the quality of the code. The module includes 672 unit tests of 94.73% code coverage. Please refer to the `test/` directory for more details. Here are the [full coverage report](https://thu-bpm.github.io/MarkDiffusion/ToReviewers/htmlcov/index.html) and the [result report](https://thu-bpm.github.io/MarkDiffusion/ToReviewers/report.html?sort=result) directly exported via pytest.

## Citation
```
@article{pan2025markdiffusion,
  title={MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models},
  author={Pan, Leyi and Guan, Sheng and Fu, Zheyu and Si, Luyang and Wang, Zian and Hu, Xuming and King, Irwin and Yu, Philip S and Liu, Aiwei and Wen, Lijie},
  journal={arXiv preprint arXiv:2509.10569},
  year={2025}
}
```
