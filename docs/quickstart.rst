.. _quickstart:

Google Colab Demo
=================

If you're interested in trying out MarkDiffusion without installing anything, you can use 
`Google Colab <https://colab.research.google.com/drive/1N1C9elDAB5zwF4FxKKYMCqR3eSpCSqAW?usp=sharing>`_ 
to see how it works.

Installation
============

MarkDiffusion can be installed in three ways. Pick the one that matches what you plan to do:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Mode
     - Command
     - Use when...
   * - **A. PyPI** (recommended for users)
     - ``pip install markdiffusion[optional]``
     - You just want to *call* watermarking algorithms from your own script or notebook. No need to read or modify the library source, run the bundled tests, or generate visualizations from the demo notebooks.
   * - **B. Editable install from source** (recommended for contributors/researchers)
     - ``git clone … && cd MarkDiffusion && pip install -e ".[optional]"``
     - You want to (a) run ``MarkDiffusion_demo.ipynb`` and the ``test/`` suite, (b) modify or add watermarking algorithms, (c) reproduce paper results, or (d) submit PRs. Source edits in ``markdiffusion/`` take effect immediately.
   * - **C. conda-forge** (conda-only environments)
     - ``conda install -c conda-forge markdiffusion``
     - You are restricted to conda channels. Some advanced features that depend on PyPI-only packages (``pyiqa`` / ``lpips`` …) are not bundled; install them separately if needed.

**Mode A — PyPI install:**

.. code-block:: bash

   conda create -n markdiffusion python=3.11
   conda activate markdiffusion
   pip install markdiffusion[optional]

**Mode B — Editable install from source:**

.. code-block:: bash

   git clone https://github.com/THU-BPM/MarkDiffusion.git
   cd MarkDiffusion
   conda create -n markdiffusion python=3.11
   conda activate markdiffusion
   pip install -e ".[optional]"
   # (optional) install test extras to run pytest, coverage, parallel tests
   pip install -r test/requirements-test.txt

.. note::
   ``pyproject.toml`` pins ``torch>=2.4,<2.11`` and ``setuptools<81`` so the resolver picks
   a CUDA-12.x wheel that runs on NVIDIA driver ≥ 525 and keeps ``pkg_resources`` available
   (still required by ``openai-clip``).

**Mode C — conda-forge:**

.. code-block:: bash

   conda create -n markdiffusion python=3.11
   conda activate markdiffusion
   conda config --add channels conda-forge
   conda config --set channel_priority strict
   conda install markdiffusion

How to Use the Toolkit
======================

The same ``markdiffusion.*`` import paths work in both PyPI and editable installs. The two demo
notebooks differ only in scope:

- ``MarkDiffusion_pypi_demo.ipynb`` — minimal end-to-end usage; safe starting point for PyPI users.
- ``MarkDiffusion_demo.ipynb`` — exhaustive walkthrough of all 11 algorithms plus visualization and
  evaluation pipelines; ships in the source repo (Mode B).

Here is the minimal end-to-end example for both modes:

.. code-block:: python

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

If you installed in **Mode B**, you can additionally point ``algorithm_config=`` at a JSON in your
working copy (e.g. ``markdiffusion/config/TR.json``) to experiment with parameters without
reinstalling. From the source repo you can also run the demo notebook end-to-end:

.. code-block:: bash

   jupyter nbconvert --to notebook --execute MarkDiffusion_demo.ipynb \
       --ExecutePreprocessor.kernel_name=markdiffusion \
       --ExecutePreprocessor.timeout=1800

Next Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you're familiar with the basics, explore more:

- :doc:`user_guide/algorithms` - Learn about specific algorithms
- :doc:`user_guide/visualization` - Visualize watermarking mechanisms
- :doc:`user_guide/evaluation` - Evaluate watermark quality and robustness
