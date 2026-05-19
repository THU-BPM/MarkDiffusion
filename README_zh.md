<div align="center">

<img src="img/markdiffusion-color-1.jpg" style="width: 65%;"/>

# 潜在扩散模型生成式水印的开源工具包

[![Home](https://img.shields.io/badge/Home-5F259F?style=for-the-badge&logo=homepage&logoColor=white)](https://generative-watermark.github.io/)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.10569)
[![Models](https://img.shields.io/badge/Models-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Generative-Watermark-Toolkits) 
[![Colab](https://img.shields.io/badge/Google--Colab-%23D97700?style=for-the-badge&logo=Google-colab&logoColor=white)](https://colab.research.google.com/drive/1N1C9elDAB5zwF4FxKKYMCqR3eSpCSqAW?usp=sharing) 
[![DOC](https://img.shields.io/badge/Readthedocs-%2300A89C?style=for-the-badge&logo=readthedocs&logoColor=#8CA1AF)](https://markdiffusion.readthedocs.io) 
[![PYPI](https://img.shields.io/badge/PYPI-%23193440?style=for-the-badge&logo=pypi&logoColor=#3775A9)](https://pypi.org/project/markdiffusion) 
[![CONDA-FORGE](https://img.shields.io/badge/Conda--Forge-%23000000?style=for-the-badge&logo=condaforge&logoColor=#FFFFFF)](https://github.com/conda-forge/markdiffusion-feedstock)



**语言版本:** [English](README.md) | [中文](README_zh.md) | [Français](README_fr.md) | [Español](README_es.md)
</div>

> 🔥 **作为一个新发布的项目，我们欢迎 PR！** 如果您已经实现了 LDM 水印算法或有兴趣贡献一个算法，我们很乐意将其包含在 MarkDiffusion 中。加入我们的社区，帮助让生成式水印技术对每个人都更易用！

## 目录
- [更新日志](#-更新日志)
- [MarkDiffusion 简介](#-markdiffusion-简介)
  - [概述](#-概述)
  - [核心特性](#-核心特性)
  - [已实现算法](#-已实现算法)
  - [评估模块](#-评估模块)
- [快速开始](#-快速开始)
    - [Google Colab 演示](#google-colab-演示)
    - [安装](#安装)
    - [如何使用工具包](#如何使用工具包)
- [测试模块](#-测试模块)
- [引用](#引用)


## 🔥 更新日志
🛠 **(2026.05.15)** 将测试套件扩展到 672 个单元测试，代码覆盖率达 94.73%（在新的 `markdiffusion-test` 环境上完成 GPU + CPU 全量回归）。

🏗️ **(2026.05.10)** 将仓库重构为标准的 `markdiffusion/` Python 包结构，使 `pip install -e .` 与 PyPI 安装共享同一套 import 路径（`from markdiffusion.watermark import AutoWatermark`）。editable 安装与 CI 现在基于统一的源码布局。

🎯 **(2026.05.10)** 新增 *DiffusionPurification* 与 *NeuralCodecCompression* 再生成攻击；*CrSc*（裁剪与缩放）支持 `position="random"` 以及显式偏移参数 —— 感谢贡献者!

🛠 **(2025.12.19)** 为所有功能添加了包含658个测试用例的完整测试套件。

🛠 **(2025.12.10)** 使用 GitHub Actions 添加了持续集成测试系统。

🎯 **(2025.10.10)** 添加 *Mask、Overlay、AdaptiveNoiseInjection* 图像攻击工具，感谢付哲语的 PR！

🎯 **(2025.10.09)** 添加 *FrameRateAdapter、FrameInterpolationAttack* 视频攻击工具，感谢司路阳的 PR！

🎯 **(2025.10.08)** 添加 *SSIM、BRISQUE、VIF、FSIM* 图像质量分析器，感谢王欢的 PR！

✨ **(2025.10.07)** 添加 [SFW](https://arxiv.org/pdf/2509.07647) 水印方法，感谢王欢的 PR！

✨ **(2025.10.07)** 添加 [VideoMark](https://arxiv.org/abs/2504.16359) 水印方法，感谢李翰乾的 PR！

✨ **(2025.9.29)** 添加 [GaussMarker](https://arxiv.org/abs/2506.11444) 水印方法，感谢司路阳的 PR！

## 🔓 MarkDiffusion 简介

### 👀 概述

MarkDiffusion 是一个用于潜在扩散模型生成式水印的开源 Python 工具包。随着基于扩散的生成模型应用范围的扩大，确保生成媒体的真实性和来源变得至关重要。MarkDiffusion 简化了水印技术的访问、理解和评估，使研究人员和更广泛的社区都能轻松使用。*注意：如果您对 LLM 水印（文本水印）感兴趣，请参考我们团队的 [MarkLLM](https://github.com/THU-BPM/MarkLLM) 工具包。*

该工具包包含三个关键组件：统一的实现框架，用于简化水印算法集成和用户友好的界面；机制可视化套件，直观地展示添加和提取的水印模式，帮助公众理解；以及全面的评估模块，提供 33 个工具的标准实现，涵盖三个关键方面——可检测性、鲁棒性和输出质量，以及 8 个自动化评估流水线。

<img src="img/fig1_overview.png" alt="MarkDiffusion Overview" style="zoom:50%;" />

### 💍 核心特性

- **统一实现框架：** MarkDiffusion 提供了一个模块化架构，支持十一种最先进的 LDM 生成式图像/视频水印算法。

- **全面的算法支持：** 目前实现了来自两大类别的 11 种水印算法：基于模式的方法（Tree-Ring、Ring-ID、ROBIN、WIND、SFW）和基于密钥的方法（Gaussian-Shading、PRC、SEAL、VideoShield、GaussMarker、VideoMark）。

- **可视化解决方案：** 该工具包包含定制的可视化工具，能够清晰而深入地展示不同水印算法在各种场景下的运行方式。这些可视化有助于揭示算法机制，使其对用户更易理解。

- **评估模块：** 拥有 33 个评估工具，涵盖可检测性、鲁棒性和对输出质量的影响，MarkDiffusion 提供全面的评估能力。它具有 8 个自动化评估流水线：水印检测流水线、图像质量分析流水线、视频质量分析流水线以及专门的鲁棒性评估工具。

### ✨ 已实现算法

| **算法** | **类别** | **目标** | **参考文献** |
|---------------|-------------|------------|---------------|
| Tree-Ring | 模式 | 图像 | [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/abs/2305.20030) |
| Ring-ID | 模式 | 图像 | [RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification](https://arxiv.org/abs/2404.14055) |
| ROBIN | 模式 | 图像 | [ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization](https://arxiv.org/abs/2411.03862) |
| WIND | 模式 | 图像 | [Hidden in the Noise: Two-Stage Robust Watermarking for Images](https://arxiv.org/abs/2412.04653) |
| SFW | 模式 | 图像 | [Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](https://arxiv.org/abs/2509.07647) |
| Gaussian-Shading | 密钥 | 图像 | [Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models](https://arxiv.org/abs/2404.04956) |
| GaussMarker | 密钥 | 图像 | [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](https://arxiv.org/abs/2506.11444) |
| PRC | 密钥 | 图像 | [An undetectable watermark for generative image models](https://arxiv.org/abs/2410.07369) |
| SEAL | 密钥 | 图像 | [SEAL: Semantic Aware Image Watermarking](https://arxiv.org/abs/2503.12172) |
| VideoShield | 密钥 | 视频 | [VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking](https://arxiv.org/abs/2501.14195) |
| VideoMark | 密钥 | 视频 | [VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models](https://arxiv.org/abs/2504.16359) |

### 🎯 评估模块
#### 评估流水线

MarkDiffusion 支持八个流水线，两个用于检测（WatermarkedMediaDetectionPipeline 和 UnWatermarkedMediaDetectionPipeline），六个用于质量分析。下表详细说明了质量分析流水线。

| **质量分析流水线** | **输入类型** | **所需数据** | **适用指标** |  
| --- | --- | --- | --- |
| DirectImageQualityAnalysisPipeline | 单张图像 | 生成的有/无水印图像 | 单张图像评估指标 | 
| ReferencedImageQualityAnalysisPipeline | 图像 + 参考内容 | 生成的有/无水印图像 + 参考图像/文本 | 需要在单张图像和参考内容（文本/图像）之间计算的指标 | 
| GroupImageQualityAnalysisPipeline | 图像集（+ 参考图像集） | 生成的有/无水印图像集（+ 参考图像集） | 需要在图像集上计算的指标 | 
| RepeatImageQualityAnalysisPipeline | 图像集 | 重复生成的有/无水印图像集 | 用于评估重复生成图像集的指标 | 
| ComparedImageQualityAnalysisPipeline | 两张对比图像 | 生成的有水印和无水印图像 | 测量两张图像之间差异的指标 | 
| DirectVideoQualityAnalysisPipeline | 单个视频 | 生成的视频帧集 | 整体视频评估指标 |

#### 评估工具

| **工具名称** | **评估类别** | **功能描述** | **输出指标** |
| --- | --- | --- | --- |
| FundamentalSuccessRateCalculator | 可检测性 | 计算固定阈值水印检测的分类指标 | 各种分类指标 |
| DynamicThresholdSuccessRateCalculator | 可检测性 | 计算动态阈值水印检测的分类指标 | 各种分类指标 |
| **图像攻击工具** | | | |
| Rotation | 鲁棒性（图像） | 图像旋转攻击，测试水印对旋转变换的抗性 | 旋转后的图像/帧 |
| CrSc（裁剪与缩放） | 鲁棒性（图像） | 裁剪和缩放攻击，评估水印对尺寸变化的鲁棒性 | 裁剪/缩放后的图像/帧 |
| GaussianNoise | 鲁棒性（图像） | 高斯噪声攻击，测试水印对噪声干扰的抗性 | 噪声损坏的图像/帧 |
| GaussianBlurring | 鲁棒性（图像） | 高斯模糊攻击，评估水印对模糊处理的抗性 | 模糊后的图像/帧 |
| JPEGCompression | 鲁棒性（图像） | JPEG 压缩攻击，测试水印对有损压缩的鲁棒性 | 压缩后的图像/帧 |
| Brightness | 鲁棒性（图像） | 亮度调整攻击，评估水印对亮度变化的抗性 | 亮度修改后的图像/帧 |
| Mask | 鲁棒性（图像） | 图像遮罩攻击，测试水印对随机黑色矩形部分遮挡的抗性 | 遮罩后的图像/帧 |
| Overlay | 鲁棒性（图像） | 图像覆盖攻击，测试水印对涂鸦式笔触和注释的抗性 | 覆盖后的图像/帧 |
| AdaptiveNoiseInjection | 鲁棒性（图像） | 自适应噪声注入攻击，测试水印对内容感知噪声的抗性（高斯/椒盐/泊松/斑点） | 自适应噪声处理后的图像/帧 |
| **视频攻击工具** | | | |
| MPEG4Compression | 鲁棒性（视频） | MPEG-4 视频压缩攻击，测试视频水印的压缩鲁棒性 | 压缩后的视频帧 |
| FrameAverage | 鲁棒性（视频） | 帧平均攻击，通过帧间平均破坏水印 | 平均后的视频帧 |
| FrameSwap | 鲁棒性（视频） | 帧交换攻击，通过改变帧序列测试鲁棒性 | 交换后的视频帧 |
| FrameRateAdapter | 鲁棒性（视频） | 帧率转换攻击，在保持时长的同时重采样帧 | 重采样后的帧序列 |
| FrameInterpolationAttack | 鲁棒性（视频） | 帧插值攻击，插入混合帧以改变时间密度 | 插值后的视频帧 |
| **图像质量分析器** | | | |
| InceptionScoreCalculator | 质量（图像） | 评估生成图像的质量和多样性 | IS 分数 |
| FIDCalculator | 质量（图像） | Fréchet Inception Distance，测量生成图像和真实图像之间的分布差异 | FID 值 |
| LPIPSAnalyzer | 质量（图像） | 学习感知图像块相似度，评估感知质量 | LPIPS 距离 |
| CLIPScoreCalculator | 质量（图像） | 基于 CLIP 的文本-图像一致性评估 | CLIP 相似度分数 |
| PSNRAnalyzer | 质量（图像） | 峰值信噪比，测量图像失真 | PSNR 值（dB） |
| NIQECalculator | 质量（图像） | 自然图像质量评估器，无参考质量评估 | NIQE 分数 |
| SSIMAnalyzer | 质量（图像） | 两张图像之间的结构相似性指数 | SSIM 值 |
| BRISQUEAnalyzer | 质量（图像） | 盲/无参考图像空间质量评估器，无需参考即可评估图像的感知质量 | BRISQUE 分数 |
| VIFAnalyzer | 质量（图像） | 视觉信息保真度分析器，比较失真图像与参考图像以量化保留的视觉信息量 | VIF 值 |
| FSIMAnalyzer | 质量（图像） | 特征相似性指数分析器，基于相位一致性和梯度幅度比较两张图像的结构相似性 | FSIM 值 |
| **视频质量分析器** | | | |
| SubjectConsistencyAnalyzer | 质量（视频） | 评估视频中主体对象的一致性 | 主体一致性分数 |
| BackgroundConsistencyAnalyzer | 质量（视频） | 评估视频中背景的连贯性和稳定性 | 背景一致性分数 |
| MotionSmoothnessAnalyzer | 质量（视频） | 评估视频运动的平滑度 | 运动平滑度指标 |
| DynamicDegreeAnalyzer | 质量（视频） | 测量视频中的动态水平和变化幅度 | 动态度值 |
| ImagingQualityAnalyzer | 质量（视频） | 综合评估视频成像质量 | 成像质量分数 |

## 🧩 快速开始
### Google Colab 演示
如果您想在不安装任何内容的情况下试用 MarkDiffusion，可以使用 [Google Colab](https://colab.research.google.com/drive/1N1C9elDAB5zwF4FxKKYMCqR3eSpCSqAW?usp=sharing#scrollTo=-kWt7m9Y3o-G) 查看其工作方式。

### 安装

MarkDiffusion 提供三种安装方式，请根据使用场景选择：

| 方式 | 命令 | 适用场景 |
|---|---|---|
| **A. PyPI（推荐普通用户）** | `pip install markdiffusion[optional]` | 你只想在自己的脚本/notebook 里**调用**水印算法，不需要阅读或修改库源码，也不需要跑仓库内的测试套件 / demo notebook。 |
| **B. 源码 editable 安装（推荐贡献者 / 研究者）** | `git clone … && cd MarkDiffusion && pip install -e ".[optional]"` | 你想（a）跑 `MarkDiffusion_demo.ipynb` / `test/` 测试套件，（b）修改或新增水印算法，（c）复现论文结果，或（d）提 PR。`markdiffusion/` 下的源码改动立即生效。 |
| **C. conda-forge（仅 conda 环境）** | `conda install -c conda-forge markdiffusion` | 你只能使用 conda 渠道。部分依赖 PyPI-only 包（如 `pyiqa` / `lpips`）的高级功能未打包进 conda 发布，如需使用请单独安装。 |

**方式 A —— PyPI 安装：**

```bash
conda create -n markdiffusion python=3.11
conda activate markdiffusion
pip install markdiffusion[optional]
```

**方式 B —— 源码 editable 安装：**

```bash
git clone https://github.com/THU-BPM/MarkDiffusion.git
cd MarkDiffusion
conda create -n markdiffusion python=3.11
conda activate markdiffusion
pip install -e ".[optional]"
# （可选）安装测试附加依赖，以运行 pytest / 覆盖率 / 并行测试
pip install -r test/requirements-test.txt
```

`pyproject.toml` 已将 `torch>=2.4,<2.11` 与 `setuptools<81` pin 住，依赖解析器会选 CUDA-12.x wheel（需要驱动 ≥ 525），同时保留 `pkg_resources`（`openai-clip` 仍依赖）。

**方式 C —— conda-forge：**

```bash
conda create -n markdiffusion python=3.11
conda activate markdiffusion
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install markdiffusion
```

### 如何使用工具包

PyPI 安装与 editable 安装使用同一套 `markdiffusion.*` import 路径。两本 demo notebook 仅作用范围不同：

- **`MarkDiffusion_pypi_demo.ipynb`** —— 端到端最小示例；PyPI 用户的安全起点。
- **`MarkDiffusion_demo.ipynb`** —— 覆盖全部 11 个算法、可视化和评估流水线的完整演练；仅随源码仓库提供（方式 B）。

两种安装方式都可直接使用的最小端到端示例：

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

# AutoWatermark 默认从包内置 config（markdiffusion/config/TR.json）加载；
# 如需覆盖，传 `algorithm_config=`。
tr_watermark = AutoWatermark.load("TR", diffusion_config=diffusion_config)

prompt = "A beautiful landscape with mountains and a river at sunset"
watermarked_image = tr_watermark.generate_watermarked_media(input_data=prompt)
watermarked_image.save("watermarked_image.png")

detection_result = tr_watermark.detect_watermark_in_media(watermarked_image)
print(detection_result)
```

如果使用**方式 B**，还可以让 `algorithm_config=` 指向工作目录中的 JSON（例如 `markdiffusion/config/TR.json`），无需重新安装即可调参。在源码仓库下也可以直接执行完整 demo notebook：

```bash
jupyter nbconvert --to notebook --execute MarkDiffusion_demo.ipynb \
    --ExecutePreprocessor.kernel_name=markdiffusion \
    --ExecutePreprocessor.timeout=1800
```

## 🛠 测试模块
我们提供了一套全面的测试模块来确保代码质量。该模块包含 672 个单元测试，代码覆盖率 94.73%。详情请参考 `test/` 目录。这里是直接由 pytest 导出的[完整覆盖率报告](https://thu-bpm.github.io/MarkDiffusion/ToReviewers/htmlcov/index.html)与[测试结果报告](https://thu-bpm.github.io/MarkDiffusion/ToReviewers/report.html?sort=result)。

## 引用
```
@article{pan2025markdiffusion,
  title={MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models},
  author={Pan, Leyi and Guan, Sheng and Fu, Zheyu and Si, Luyang and Wang, Zian and Hu, Xuming and King, Irwin and Yu, Philip S and Liu, Aiwei and Wen, Lijie},
  journal={arXiv preprint arXiv:2509.10569},
  year={2025}
}
```

