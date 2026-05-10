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


import torch
from markdiffusion.detection.base import BaseDetector
from scipy.stats import ncx2

class TRDetector(BaseDetector):
    
    def __init__(self,
                 watermarking_mask: torch.Tensor,
                 gt_patch: torch.Tensor,
                 threshold: float,
                 device: torch.device):
        super().__init__(threshold, device)
        self.watermarking_mask = watermarking_mask
        self.gt_patch = gt_patch
        
    def eval_watermark(self,
                        reversed_latents: torch.Tensor,
                        reference_latents: torch.Tensor = None,
                        detector_type: str = "l1_distance") -> float:
        if detector_type == 'l1_distance':
            reversed_latents_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2)) #[self.watermarking_mask].flatten()
            target_patch = self.gt_patch #[self.watermarking_mask].flatten()
            l1_distance = torch.abs(reversed_latents_fft[self.watermarking_mask] - target_patch[self.watermarking_mask]).mean().item()
            return {
                'is_watermarked': l1_distance < self.threshold, 
                'l1_distance': l1_distance
            }
        elif detector_type == 'p_value':
            reversed_latents_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))[self.watermarking_mask].flatten()
            target_patch = self.gt_patch[self.watermarking_mask].flatten()
            target_patch = torch.concatenate([target_patch.real, target_patch.imag])
            reversed_latents_fft = torch.concatenate([reversed_latents_fft.real, reversed_latents_fft.imag])
            sigma_ = reversed_latents_fft.std()
            lambda_ = (target_patch ** 2 / sigma_ ** 2).sum().item()
            x = (((reversed_latents_fft - target_patch) / sigma_) ** 2).sum().item()
            p = ncx2.cdf(x=x, df=len(target_patch), nc=lambda_)
            return {
                'is_watermarked': bool(p < self.threshold), 
                'p_value': p
            }
        else:
            raise ValueError(f"Tree Ring's watermark detector type {self.detector_type} not supported")

