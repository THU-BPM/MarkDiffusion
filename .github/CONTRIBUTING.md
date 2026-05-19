# Contributing to MarkDiffusion

Thanks for your interest in contributing! This page covers two flows:

- [Submit a pull request](#create-a-pull-request) — the generic GitHub workflow.
- [Add a new watermarking algorithm](#add-a-new-watermarking-algorithm) — the
  domain-specific recipe for plugging a new method into MarkDiffusion.

For deeper references see:
- [User Guide → Algorithms](https://markdiffusion.readthedocs.io/en/latest/user_guide/algorithms.html)
- [Advanced → Custom Algorithms](https://markdiffusion.readthedocs.io/en/latest/advanced/custom_algorithms.html)
- [Advanced → Configuration](https://markdiffusion.readthedocs.io/en/latest/advanced/configuration.html)

---

## Create a Pull Request

1. Fork the [repository](https://github.com/THU-BPM/MarkDiffusion) by clicking on the [Fork](https://github.com/THU-BPM/MarkDiffusion/fork) button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

```bash
# Replace [username] with your GitHub username
git clone git@github.com:[username]/MarkDiffusion.git
cd MarkDiffusion

# Add the original repository as "upstream" to keep your fork synced
git remote add upstream https://github.com/THU-BPM/MarkDiffusion.git
```

3. Create a new branch to hold your development changes:

```bash
# It is good practice to sync with upstream before creating a branch
git fetch upstream
git checkout -b dev_your_branch upstream/main
```

4. Set up a development environment (editable install — Mode B in the README):

```bash
conda create -n markdiffusion python=3.11
conda activate markdiffusion
pip install -e ".[optional]"
pip install -r test/requirements-test.txt
```

5. Run the test suite before committing:

```bash
pytest test/ -m "not slow"      # quick checks; 615 of 672 tests run in ~10 min
pytest test/                    # full run (requires GPU + cached HF models)
```

6. Submit changes:

```bash
git add .
git commit -m "feat: add awesome feature"

# Sync with upstream again to avoid conflicts
git fetch upstream
git rebase upstream/main

# Push to your own fork (origin)
git push -u origin dev_your_branch
```

7. Open a Pull Request from your branch `dev_your_branch` at your fork page (GitHub will prompt you to merge into THU-BPM/MarkDiffusion).

---

## Add a New Watermarking Algorithm

This is the most common contribution. The codebase is organized so that a new
algorithm only needs to drop files into four places and register two strings.
The pattern below is followed by every existing algorithm in
`markdiffusion/watermark/`.

### 1. File layout

For an algorithm called `MyMark`, create:

```text
markdiffusion/
├── config/
│   └── MyMark.json                    # default hyper-parameters
├── watermark/
│   └── mymark/
│       ├── __init__.py                # `from .mymark import MyMark, MyMarkConfig`
│       └── mymark.py                  # MyMarkConfig + MyMark
├── detection/
│   └── mymark/
│       ├── __init__.py
│       └── mymark_detection.py        # MyMarkDetector (optional but recommended)
└── visualize/                         # optional, only if you want a visualizer
    └── mymark/
        ├── __init__.py
        └── mymark_visualizer.py
```

### 2. Base classes & required methods

All abstract bases live in `markdiffusion/watermark/base.py`,
`markdiffusion/detection/base.py`, and `markdiffusion/visualize/base.py`.

| You inherit from | You **must** implement | You **may** override |
| --- | --- | --- |
| `BaseConfig` | `initialize_parameters(self)`, `algorithm_name` property | — |
| `BaseWatermark` | `_generate_watermarked_image` *or* `_generate_watermarked_video`,<br>`_detect_watermark_in_image` *or* `_detect_watermark_in_video`,<br>`get_data_for_visualize` | `_generate_unwatermarked_image/video` (defaults are usually fine) |
| `BaseDetector` *(optional helper)* | `eval_watermark(self, reversed_latents, …)` | — |
| `BaseVisualizer` *(optional)* | `visualize(self, …)` plus any `draw_*` helpers you want | — |

Minimal skeleton (image algorithm):

```python
# markdiffusion/watermark/mymark/mymark.py
from typing import Dict
from PIL import Image

from markdiffusion.watermark.base import BaseConfig, BaseWatermark
from markdiffusion.visualize.data_for_visualization import DataForVisualization
from markdiffusion.utils.utils import set_random_seed


class MyMarkConfig(BaseConfig):
    @property
    def algorithm_name(self) -> str:
        return "MyMark"

    def initialize_parameters(self) -> None:
        # Read your hyper-parameters from self.config_dict (loaded from the JSON).
        self.strength = self.config_dict["strength"]
        self.threshold = self.config_dict["threshold"]


class MyMark(BaseWatermark):
    def __init__(self, watermark_config: MyMarkConfig, *args, **kwargs):
        self.config = watermark_config
        # Build any precomputed patterns / keys here so they live on self.config.device.

    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        set_random_seed(self.config.gen_seed)
        # 1. Take self.config.init_latents and inject your watermark.
        watermarked_latents = ...
        # 2. Call self.config.pipe(prompt, latents=watermarked_latents, ...).
        return self.config.pipe(prompt, latents=watermarked_latents, ...).images[0]

    def _detect_watermark_in_image(self, image: Image.Image, prompt: str = "",
                                   *args, **kwargs) -> Dict[str, float]:
        # 1. Encode image to latents, run DDIM inversion via self.config.inversion.
        # 2. Compare reversed_latents to your watermark pattern.
        return {"is_watermarked": ..., "score": ...}

    def get_data_for_visualize(self, image: Image.Image, prompt: str = "",
                               *args, **kwargs) -> DataForVisualization:
        return DataForVisualization(
            config=self.config,
            utils=None,
            image=image,
            # Add algorithm-specific tensors the visualizer needs as kwargs.
        )
```

The `BaseWatermark.generate_watermarked_media` and `detect_watermark_in_media`
methods on the public API dispatch automatically to your `_*_image` or
`_*_video` overrides based on the pipeline type — you do not implement the
public methods yourself.

### 3. Device-agnostic code

Every tensor you create must follow `self.config.device` (which can be `cpu`,
`cuda`, or `mps`). Concretely:

- Use `torch.Generator(device=self.config.device)` for reproducible seeds, or
  `torch.Generator(device="cpu")` if you need cross-device-identical patterns
  (see `markdiffusion/watermark/gm/gm.py` for the latter pattern).
- Move tensors with `.to(self.config.device)`. **Do not** hardcode `.cuda()`
  or `.half()`; that is what caused the CPU regression fixed in v1.0.2.
- Derive latent shapes from `self.config.image_size` /
  `self.pipe.vae_scale_factor` — never hardcode `64x64`.

### 4. Config JSON

Drop the default hyper-parameters into `markdiffusion/config/MyMark.json`:

```json
{
    "algorithm_name": "MyMark",
    "strength": 1.0,
    "threshold": 0.5
}
```

`algorithm_name` must match what `MyMarkConfig.algorithm_name` returns.
Keys read inside `initialize_parameters` should appear here.

### 5. Register with `AutoWatermark`

Two mappings, both in `markdiffusion/watermark/`:

```python
# auto_watermark.py
WATERMARK_MAPPING_NAMES = {
    ...
    "MyMark": "markdiffusion.watermark.mymark.MyMark",
}

PIPELINE_SUPPORTED_WATERMARKS = {
    PIPELINE_TYPE_IMAGE: [..., "MyMark"],        # add to image / t2v / i2v as appropriate
}
```

```python
# auto_config.py
CONFIG_MAPPING_NAMES = {
    ...
    "MyMark": "markdiffusion.watermark.mymark.MyMarkConfig",
}
```

After this, users can do `AutoWatermark.load("MyMark", diffusion_config=cfg)`
and the bundled `markdiffusion/config/MyMark.json` is picked up automatically.

### 6. (Optional) Visualizer

If you want a visualizer in `MarkDiffusion_demo.ipynb`, inherit from
`BaseVisualizer` and register in `markdiffusion/visualize/auto_visualization.py`:

```python
# markdiffusion/visualize/auto_visualization.py
VISUALIZER_MAPPING_NAMES = {
    ...
    "MyMark": "markdiffusion.visualize.mymark.MyMarkVisualizer",
}
```

Provide at least `visualize(rows, cols, methods, …)` plus any `draw_*` helpers
the notebook calls.

### 7. Tests

Most of the test suite is parameterised over `PIPELINE_SUPPORTED_WATERMARKS`,
so once you register your algorithm it gets picked up automatically. Confirm:

```bash
# Initialization-only smoke test (fast, no GPU required)
pytest test/test_watermark_algorithms.py \
    -k "initialization and MyMark" -v

# Full image-watermark slow test (requires GPU + cached SD pipeline)
pytest test/test_watermark_algorithms.py \
    -k "MyMark" -m slow -v
```

If your algorithm needs custom assertions beyond what the parameterised tests
cover, add a `test_mymark.py` next to the existing `test_*.py` files. Aim to
keep the coverage badge at ≥ 94 % (see
[`test/README.md`](../test/README.md) for the current target and how to
generate coverage locally).

### 8. Documentation

- Add a row to the algorithm tables in `README.md`, `README_zh.md`,
  `README_fr.md`, `README_es.md`, and `docs/user_guide/algorithms.rst`.
- If your algorithm shipped via a paper, link the arXiv ID under
  `## 🔥 Updates` in the four READMEs.
- For deep architectural notes, extend
  `docs/advanced/custom_algorithms.rst`.

That's it — open a PR, the CI selective-test workflow will run a subset of
tests scoped to your changes, and a maintainer will review.
