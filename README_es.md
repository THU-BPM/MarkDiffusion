<div align="center">

<img src="img/markdiffusion-color-1.jpg" style="width: 65%;"/>

# Un Kit de Herramientas de Código Abierto para Marcas de Agua Generativas de Modelos de Difusión Latente

[![Home](https://img.shields.io/badge/Home-5F259F?style=for-the-badge&logo=homepage&logoColor=white)](https://generative-watermark.github.io/)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.10569)
[![Models](https://img.shields.io/badge/Models-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Generative-Watermark-Toolkits) 
[![Colab](https://img.shields.io/badge/Google--Colab-%23D97700?style=for-the-badge&logo=Google-colab&logoColor=white)](https://colab.research.google.com/drive/1N1C9elDAB5zwF4FxKKYMCqR3eSpCSqAW?usp=sharing) 
[![DOC](https://img.shields.io/badge/Readthedocs-%2300A89C?style=for-the-badge&logo=readthedocs&logoColor=#8CA1AF)](https://markdiffusion.readthedocs.io) 
[![PYPI](https://img.shields.io/badge/PYPI-%23193440?style=for-the-badge&logo=pypi&logoColor=#3775A9)](https://pypi.org/project/markdiffusion) 
[![CONDA-FORGE](https://img.shields.io/badge/Conda--Forge-%23000000?style=for-the-badge&logo=condaforge&logoColor=#FFFFFF)](https://github.com/conda-forge/markdiffusion-feedstock)



**Versiones de idioma:** [English](README.md) | [中文](README_zh.md) | [Français](README_fr.md) | [Español](README_es.md)
</div>

> 🔥 **¡Como un proyecto recién lanzado, damos la bienvenida a PRs!** Si has implementado un algoritmo de marcas de agua LDM o estás interesado en contribuir con uno, nos encantaría incluirlo en MarkDiffusion. ¡Únete a nuestra comunidad y ayuda a hacer las marcas de agua generativas más accesibles para todos!

## Contenidos
- [Actualizaciones](#-actualizaciones)
- [Introducción a MarkDiffusion](#-introducción-a-markdiffusion)
  - [Descripción general](#-descripción-general)
  - [Características clave](#-características-clave)
  - [Algoritmos implementados](#-algoritmos-implementados)
  - [Módulo de evaluación](#-módulo-de-evaluación)
- [Inicio rápido](#-inicio-rápido)
    - [Demo de Google Colab](#demo-de-google-colab)
    - [Instalación](#instalación)
    - [Cómo usar el kit de herramientas](#cómo-usar-el-kit-de-herramientas)
- [Módulos de prueba](#-módulos-de-prueba)
- [Citación](#citación)


## 🔥 Actualizaciones
🛠 **(2026.05.15)** Suite de pruebas ampliada a 672 pruebas unitarias con un 94,73 % de cobertura de código (regresión completa GPU + CPU en el nuevo entorno `markdiffusion-test`).

🏗️ **(2026.05.10)** Repositorio reestructurado como un auténtico paquete Python `markdiffusion/`, de modo que `pip install -e .` y la instalación desde PyPI comparten las mismas rutas de import (`from markdiffusion.watermark import AutoWatermark`). Las instalaciones editable y la CI ahora se basan en una disposición única.

🎯 **(2026.05.10)** Añadidos los ataques de regeneración *DiffusionPurification* y *NeuralCodecCompression*; *CrSc* (Crop & Scale) admite ahora `position="random"` y un desplazamiento explícito — ¡gracias a los contribuyentes!

🛠 **(2025.12.19)** Agregada una suite de pruebas completa para todas las funcionalidades con 658 casos de prueba.

🛠 **(2025.12.10)** Agregado un sistema de pruebas de integración continua usando GitHub Actions.

🎯 **(2025.10.10)** Agregadas herramientas de ataque de imagen *Mask, Overlay, AdaptiveNoiseInjection*, ¡gracias a Zheyu Fu por su PR!

🎯 **(2025.10.09)** Agregadas herramientas de ataque de video *FrameRateAdapter, FrameInterpolationAttack*, ¡gracias a Luyang Si por su PR!

🎯 **(2025.10.08)** Agregados analizadores de calidad de imagen *SSIM, BRISQUE, VIF, FSIM*, ¡gracias a Huan Wang por su PR!

✨ **(2025.10.07)** Agregado el método de marca de agua [SFW](https://arxiv.org/pdf/2509.07647), ¡gracias a Huan Wang por su PR!

✨ **(2025.10.07)** Agregado el método de marca de agua [VideoMark](https://arxiv.org/abs/2504.16359), ¡gracias a Hanqian Li por su PR!

✨ **(2025.9.29)** Agregado el método de marca de agua [GaussMarker](https://arxiv.org/abs/2506.11444), ¡gracias a Luyang Si por su PR!

## 🔓 Introducción a MarkDiffusion

### 👀 Descripción general

MarkDiffusion es un kit de herramientas de Python de código abierto para marcas de agua generativas de modelos de difusión latente. A medida que se expande el uso de modelos generativos basados en difusión, garantizar la autenticidad y el origen de los medios generados se vuelve crítico. MarkDiffusion simplifica el acceso, la comprensión y la evaluación de tecnologías de marcas de agua, haciéndolo accesible tanto para investigadores como para la comunidad en general. *Nota: si estás interesado en marcas de agua LLM (marca de agua de texto), consulta el kit de herramientas [MarkLLM](https://github.com/THU-BPM/MarkLLM) de nuestro grupo.*

El kit de herramientas comprende tres componentes clave: un marco de implementación unificado para integraciones simplificadas de algoritmos de marcas de agua e interfaces fáciles de usar; un conjunto de visualización de mecanismos que muestra intuitivamente los patrones de marcas de agua agregados y extraídos para ayudar a la comprensión pública; y un módulo de evaluación integral que ofrece implementaciones estándar de 33 herramientas en tres aspectos esenciales: detectabilidad, robustez y calidad de salida, además de 8 pipelines de evaluación automatizados.

<img src="img/fig1_overview.png" alt="MarkDiffusion Overview" style="zoom:50%;" />

### 💍 Características clave

- **Marco de implementación unificado:** MarkDiffusion proporciona una arquitectura modular que admite once algoritmos de marcas de agua generativas de imagen/video de última generación para LDMs.

- **Soporte integral de algoritmos:** Actualmente implementa 11 algoritmos de marcas de agua de dos categorías principales: métodos basados en patrones (Tree-Ring, Ring-ID, ROBIN, WIND, SFW) y métodos basados en claves (Gaussian-Shading, PRC, SEAL, VideoShield, GaussMarker, VideoMark).

- **Soluciones de visualización:** El kit de herramientas incluye herramientas de visualización personalizadas que permiten vistas claras y perspicaces sobre cómo operan los diferentes algoritmos de marcas de agua en varios escenarios. Estas visualizaciones ayudan a desmitificar los mecanismos de los algoritmos, haciéndolos más comprensibles para los usuarios.

- **Módulo de evaluación:** Con 33 herramientas de evaluación que cubren detectabilidad, robustez e impacto en la calidad de salida, MarkDiffusion proporciona capacidades de evaluación integral. Cuenta con 8 pipelines de evaluación automatizados: Pipeline de detección de marcas de agua, Pipeline de análisis de calidad de imagen, Pipeline de análisis de calidad de video y herramientas especializadas de evaluación de robustez.

### ✨ Algoritmos implementados

| **Algoritmo** | **Categoría** | **Objetivo** | **Referencia** |
|---------------|-------------|------------|---------------|
| Tree-Ring | Patrón | Imagen | [Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/abs/2305.20030) |
| Ring-ID | Patrón | Imagen | [RingID: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification](https://arxiv.org/abs/2404.14055) |
| ROBIN | Patrón | Imagen | [ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization](https://arxiv.org/abs/2411.03862) |
| WIND | Patrón | Imagen | [Hidden in the Noise: Two-Stage Robust Watermarking for Images](https://arxiv.org/abs/2412.04653) |
| SFW | Patrón | Imagen | [Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](https://arxiv.org/abs/2509.07647) |
| Gaussian-Shading | Clave | Imagen | [Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models](https://arxiv.org/abs/2404.04956) |
| GaussMarker | Clave | Imagen | [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](https://arxiv.org/abs/2506.11444) |
| PRC | Clave | Imagen | [An undetectable watermark for generative image models](https://arxiv.org/abs/2410.07369) |
| SEAL | Clave | Imagen | [SEAL: Semantic Aware Image Watermarking](https://arxiv.org/abs/2503.12172) |
| VideoShield | Clave | Video | [VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking](https://arxiv.org/abs/2501.14195) |
| VideoMark | Clave | Video | [VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models](https://arxiv.org/abs/2504.16359) |

### 🎯 Módulo de evaluación
#### Pipelines de evaluación

MarkDiffusion admite ocho pipelines, dos para detección (WatermarkedMediaDetectionPipeline y UnWatermarkedMediaDetectionPipeline), y seis para análisis de calidad. La tabla a continuación detalla los pipelines de análisis de calidad.

| **Pipeline de análisis de calidad** | **Tipo de entrada** | **Datos requeridos** | **Métricas aplicables** |  
| --- | --- | --- | --- |
| DirectImageQualityAnalysisPipeline | Imagen única | Imagen generada con/sin marca de agua | Métricas para evaluación de imagen única | 
| ReferencedImageQualityAnalysisPipeline | Imagen + contenido de referencia | Imagen generada con/sin marca de agua + imagen/texto de referencia | Métricas que requieren cálculo entre imagen única y contenido de referencia (texto/imagen) | 
| GroupImageQualityAnalysisPipeline | Conjunto de imágenes (+ conjunto de imágenes de referencia) | Conjunto de imágenes generadas con/sin marca de agua (+ conjunto de imágenes de referencia) | Métricas que requieren cálculo en conjuntos de imágenes | 
| RepeatImageQualityAnalysisPipeline | Conjunto de imágenes | Conjunto de imágenes generadas repetidamente con/sin marca de agua | Métricas para evaluar conjuntos de imágenes generadas repetidamente | 
| ComparedImageQualityAnalysisPipeline | Dos imágenes para comparación | Imágenes generadas con y sin marca de agua | Métricas que miden diferencias entre dos imágenes | 
| DirectVideoQualityAnalysisPipeline | Video único | Conjunto de fotogramas de video generados | Métricas para evaluación general de video |

#### Herramientas de evaluación

| **Nombre de la herramienta** | **Categoría de evaluación** | **Descripción de la función** | **Métricas de salida** |
| --- | --- | --- | --- |
| FundamentalSuccessRateCalculator | Detectabilidad | Calcular métricas de clasificación para detección de marca de agua con umbral fijo | Varias métricas de clasificación |
| DynamicThresholdSuccessRateCalculator | Detectabilidad | Calcular métricas de clasificación para detección de marca de agua con umbral dinámico | Varias métricas de clasificación |
| **Herramientas de ataque de imagen** | | | |
| Rotation | Robustez (Imagen) | Ataque de rotación de imagen, probando la resistencia de la marca de agua a transformaciones de rotación | Imágenes/fotogramas rotados |
| CrSc (Crop & Scale) | Robustez (Imagen) | Ataque de recorte y escalado, evaluando la robustez de la marca de agua a cambios de tamaño | Imágenes/fotogramas recortados/escalados |
| GaussianNoise | Robustez (Imagen) | Ataque de ruido gaussiano, probando la resistencia de la marca de agua a interferencias de ruido | Imágenes/fotogramas corrompidos por ruido |
| GaussianBlurring | Robustez (Imagen) | Ataque de desenfoque gaussiano, evaluando la resistencia de la marca de agua al procesamiento de desenfoque | Imágenes/fotogramas desenfocados |
| JPEGCompression | Robustez (Imagen) | Ataque de compresión JPEG, probando la robustez de la marca de agua a la compresión con pérdida | Imágenes/fotogramas comprimidos |
| Brightness | Robustez (Imagen) | Ataque de ajuste de brillo, evaluando la resistencia de la marca de agua a cambios de brillo | Imágenes/fotogramas modificados en brillo |
| Mask | Robustez (Imagen) | Ataque de enmascaramiento de imagen, probando la resistencia de la marca de agua a la oclusión parcial por rectángulos negros aleatorios | Imágenes/fotogramas enmascarados |
| Overlay | Robustez (Imagen) | Ataque de superposición de imagen, probando la resistencia de la marca de agua a trazos y anotaciones estilo grafiti | Imágenes/fotogramas superpuestos |
| AdaptiveNoiseInjection | Robustez (Imagen) | Ataque de inyección de ruido adaptativo, probando la resistencia de la marca de agua al ruido consciente del contenido (Gaussiano/Sal-pimienta/Poisson/Moteado) | Imágenes/fotogramas ruidosos con ruido adaptativo |
| **Herramientas de ataque de video** | | | |
| MPEG4Compression | Robustez (Video) | Ataque de compresión de video MPEG-4, probando la robustez de compresión de marca de agua de video | Fotogramas de video comprimidos |
| FrameAverage | Robustez (Video) | Ataque de promedio de fotogramas, destruyendo marcas de agua a través del promedio entre fotogramas | Fotogramas de video promediados |
| FrameSwap | Robustez (Video) | Ataque de intercambio de fotogramas, probando la robustez cambiando secuencias de fotogramas | Fotogramas de video intercambiados |
| FrameRateAdapter | Robustez (Video) | Ataque de conversión de velocidad de fotogramas que remuestrea fotogramas preservando la duración | Secuencia de fotogramas remuestreada |
| FrameInterpolationAttack | Robustez (Video) | Ataque de interpolación de fotogramas insertando fotogramas mezclados para alterar la densidad temporal | Fotogramas de video interpolados |
| **Analizadores de calidad de imagen** | | | |
| InceptionScoreCalculator | Calidad (Imagen) | Evaluar calidad y diversidad de imagen generada | Puntuación IS |
| FIDCalculator | Calidad (Imagen) | Distancia de Inception de Fréchet, midiendo la diferencia de distribución entre imágenes generadas y reales | Valor FID |
| LPIPSAnalyzer | Calidad (Imagen) | Similitud de parche de imagen perceptual aprendida, evaluando calidad perceptual | Distancia LPIPS |
| CLIPScoreCalculator | Calidad (Imagen) | Evaluación de consistencia texto-imagen basada en CLIP | Puntuación de similitud CLIP |
| PSNRAnalyzer | Calidad (Imagen) | Relación señal-ruido de pico, midiendo la distorsión de imagen | Valor PSNR (dB) |
| NIQECalculator | Calidad (Imagen) | Evaluador de calidad de imagen natural, evaluación de calidad sin referencia | Puntuación NIQE |
| SSIMAnalyzer | Calidad (Imagen) | Índice de similitud estructural entre dos imágenes | Valor SSIM |
| BRISQUEAnalyzer | Calidad (Imagen) | Evaluador de calidad espacial de imagen ciega/sin referencia, evaluando la calidad perceptual de una imagen sin requerir una referencia | Puntuación BRISQUE |
| VIFAnalyzer | Calidad (Imagen) | Analizador de fidelidad de información visual, comparando una imagen distorsionada con una imagen de referencia para cuantificar la cantidad de información visual preservada | Valor VIF |
| FSIMAnalyzer | Calidad (Imagen) | Analizador de índice de similitud de características, comparando similitud estructural entre dos imágenes basada en congruencia de fase y magnitud de gradiente | Valor FSIM |
| **Analizadores de calidad de video** | | | |
| SubjectConsistencyAnalyzer | Calidad (Video) | Evaluar consistencia de objetos sujeto en video | Puntuación de consistencia de sujeto |
| BackgroundConsistencyAnalyzer | Calidad (Video) | Evaluar coherencia y estabilidad del fondo en video | Puntuación de consistencia de fondo |
| MotionSmoothnessAnalyzer | Calidad (Video) | Evaluar suavidad del movimiento del video | Métrica de suavidad de movimiento |
| DynamicDegreeAnalyzer | Calidad (Video) | Medir nivel dinámico y magnitud de cambio en video | Valor de grado dinámico |
| ImagingQualityAnalyzer | Calidad (Video) | Evaluación integral de calidad de imagen de video | Puntuación de calidad de imagen |

## 🧩 Inicio rápido
### Demo de Google Colab
Si deseas probar MarkDiffusion sin instalar nada, puedes usar [Google Colab](https://colab.research.google.com/drive/1N1C9elDAB5zwF4FxKKYMCqR3eSpCSqAW?usp=sharing#scrollTo=-kWt7m9Y3o-G) para ver cómo funciona.

### Instalación

MarkDiffusion se puede instalar de tres formas. Elige la que se ajuste a tu uso:

| Modo | Comando | Cuándo usarlo... |
|---|---|---|
| **A. PyPI (recomendado para usuarios)** | `pip install markdiffusion[optional]` | Solo quieres *invocar* algoritmos de marca de agua desde tu propio script/notebook. No necesitas leer o modificar el código fuente de la biblioteca, ni ejecutar las pruebas o los notebooks de demostración. |
| **B. Instalación editable desde la fuente (recomendado para contribuyentes/investigadores)** | `git clone … && cd MarkDiffusion && pip install -e ".[optional]"` | Quieres (a) ejecutar `MarkDiffusion_demo.ipynb` / la suite `test/`, (b) modificar o añadir algoritmos, (c) reproducir resultados del paper, o (d) enviar PR. Las ediciones en `markdiffusion/` surten efecto de inmediato. |
| **C. conda-forge (entornos solo conda)** | `conda install -c conda-forge markdiffusion` | Estás restringido a canales conda. Algunas características avanzadas (que dependen de paquetes solo en PyPI como `pyiqa` / `lpips`) no están incluidas; instálalas por separado si las necesitas. |

**Modo A — Instalación PyPI:**

```bash
conda create -n markdiffusion python=3.11
conda activate markdiffusion
pip install markdiffusion[optional]
```

**Modo B — Instalación editable desde la fuente:**

```bash
git clone https://github.com/THU-BPM/MarkDiffusion.git
cd MarkDiffusion
conda create -n markdiffusion python=3.11
conda activate markdiffusion
pip install -e ".[optional]"
# (opcional) instalar extras de prueba para ejecutar pytest, cobertura, pruebas paralelas
pip install -r test/requirements-test.txt
```

`pyproject.toml` fija `torch>=2.4,<2.11` y `setuptools<81` para que el resolvedor elija una wheel de CUDA-12.x compatible con controladores ≥ 525 y conserve `pkg_resources` (aún requerido por `openai-clip`).

**Modo C — conda-forge:**

```bash
conda create -n markdiffusion python=3.11
conda activate markdiffusion
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install markdiffusion
```

### Cómo usar el kit de herramientas

Las mismas rutas de import `markdiffusion.*` funcionan tanto para la instalación PyPI como para la editable. Los dos notebooks de demostración solo difieren en su alcance:

- **`MarkDiffusion_pypi_demo.ipynb`** — uso mínimo de extremo a extremo; punto de partida seguro para usuarios PyPI.
- **`MarkDiffusion_demo.ipynb`** — recorrido exhaustivo por los 11 algoritmos junto con herramientas de visualización y pipelines de evaluación; se entrega con el repositorio fuente (Modo B).

Ejemplo mínimo de extremo a extremo válido para ambos modos:

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

# AutoWatermark toma automáticamente la configuración por defecto integrada
# (markdiffusion/config/TR.json). Pasa `algorithm_config=` para sobrescribirla.
tr_watermark = AutoWatermark.load("TR", diffusion_config=diffusion_config)

prompt = "A beautiful landscape with mountains and a river at sunset"
watermarked_image = tr_watermark.generate_watermarked_media(input_data=prompt)
watermarked_image.save("watermarked_image.png")

detection_result = tr_watermark.detect_watermark_in_media(watermarked_image)
print(detection_result)
```

En el **Modo B**, además puedes apuntar `algorithm_config=` a un JSON de tu copia de trabajo (por ejemplo `markdiffusion/config/TR.json`) para experimentar con parámetros sin reinstalar. Desde el repositorio fuente también puedes ejecutar el notebook de demostración de extremo a extremo:

```bash
jupyter nbconvert --to notebook --execute MarkDiffusion_demo.ipynb \
    --ExecutePreprocessor.kernel_name=markdiffusion \
    --ExecutePreprocessor.timeout=1800
```

## 🛠 Módulos de prueba
Proporcionamos un conjunto completo de módulos de prueba para garantizar la calidad del código. El módulo incluye 672 pruebas unitarias con un 94,73 % de cobertura de código. Consulta el directorio `test/` para más detalles. Aquí están el [informe completo de cobertura](https://thu-bpm.github.io/MarkDiffusion/ToReviewers/htmlcov/index.html) y el [informe de resultados](https://thu-bpm.github.io/MarkDiffusion/ToReviewers/report.html?sort=result) exportados directamente vía pytest.

## Citación
```
@article{pan2025markdiffusion,
  title={MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models},
  author={Pan, Leyi and Guan, Sheng and Fu, Zheyu and Si, Luyang and Wang, Zian and Hu, Xuming and King, Irwin and Yu, Philip S and Liu, Aiwei and Wen, Lijie},
  journal={arXiv preprint arXiv:2509.10569},
  year={2025}
}
```

