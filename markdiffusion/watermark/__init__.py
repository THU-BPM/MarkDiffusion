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
MarkDiffusion - An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models.

This package provides watermarking algorithms for diffusion models including:
- Tree-Ring (TR)
- Gaussian Shading (GS)
- RingID (RI)
- PRC
- ROBIN
- Gaussian Marking (GM)
- SFW (Stable Few Watermarks)
- SEAL
- WIND
- VideoMark
- VideoShield
"""

__version__ = "0.1.6.post1"
__author__ = "THU-BPM MarkDiffusion Team"
__license__ = "Apache-2.0"

from .base import BaseWatermark, BaseConfig
from .auto_watermark import AutoWatermark
from .auto_config import AutoConfig

__all__ = [
    "__version__",
    "BaseWatermark",
    "BaseConfig",
    "AutoWatermark",
    "AutoConfig",
]
