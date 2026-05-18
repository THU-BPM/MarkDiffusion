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

"""MarkDiffusion - An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models."""

__version__ = "1.0.2"
__author__ = "THU-BPM MarkDiffusion Team"
__license__ = "Apache-2.0"

from markdiffusion.watermark.base import BaseWatermark, BaseConfig
from markdiffusion.watermark.auto_watermark import AutoWatermark
from markdiffusion.watermark.auto_config import AutoConfig
from markdiffusion.utils.diffusion_config import DiffusionConfig

__all__ = [
    "__version__",
    "BaseWatermark",
    "BaseConfig",
    "AutoWatermark",
    "AutoConfig",
    "DiffusionConfig",
]
