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
from detection.base import BaseDetector
from scipy.stats import ncx2

class SFWDetector(BaseDetector):
    
    def __init__(self,
                 watermarking_mask: torch.Tensor,
                 gt_patch: torch.Tensor,
                 w_channel: int,
                 threshold: float,
                 device: torch.device,
                 wm_type: str = "HSTR"):
        super().__init__(threshold, device)
        self.watermarking_mask = watermarking_mask
        self.gt_patch = gt_patch
        self.w_channel = w_channel
        self.wm_type = wm_type

    @torch.no_grad()
    def get_distance_hsqr(self,qr_gt_bool, target_fft,p=1):
        """
        qr_gt_bool : (c_wm,42,42) boolean
        target_fft : (1,4,64,64) complex64
        """
        qr_gt_bool = qr_gt_bool.squeeze(0)
        center_row = target_fft.shape[-2] // 2 # 32
        qr_pix_len = qr_gt_bool.shape[-1]    # 42
        qr_pix_half = (qr_pix_len + 1) // 2 # 21
        qr_gt_f32 = torch.where(qr_gt_bool, torch.tensor(45.0), torch.tensor(-45.0)).to(torch.float32) # (c_wm,42,42) boolean -> float32
        qr_left = qr_gt_f32[0,:, :qr_pix_half]   # (42,21) float32
        qr_right = qr_gt_f32[0,:, qr_pix_half:]  # (42,21) float32
        qr_complex = torch.complex(qr_left, qr_right).to(target_fft.device) # (42,21) complex64
        row_start = 10 + 1 # 11
        row_end = row_start + qr_pix_len # 53 = 11+42
        col_start = center_row + 1 # 33 = 32+1
        col_end = col_start + qr_pix_half # 54 = 33+21
        qr_slice = (0, self.w_channel, slice(row_start, row_end), slice(col_start, col_end)) # (42,21)
        diff = torch.abs(qr_complex - target_fft[qr_slice]) # (42,21)
        return torch.mean(diff).item()
        
    def eval_watermark(self,
                        reversed_latents: torch.Tensor,
                        reference_latents: torch.Tensor = None,
                        detector_type: str = "l1_distance") -> float:
        start, end = 10, 54
        center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))
        reversed_latents_fft = torch.zeros_like(reversed_latents, dtype=torch.complex64)
        reversed_latents_fft[center_slice] = torch.fft.fftshift(torch.fft.fft2(reversed_latents[center_slice]), dim=(-1, -2))
        if self.wm_type == "HSQR":
            if detector_type == 'l1_distance':
                hsqr_distance = self.get_distance_hsqr(qr_gt_bool=self.gt_patch, target_fft=reversed_latents_fft)
                return {
                    'is_watermark': hsqr_distance < self.threshold, 
                    'l1_distance': hsqr_distance
                }
            else:
                raise ValueError(f"SFW(HSQR)'s watermark detector type {self.detector_type} not supported")
        else:
            if detector_type == 'l1_distance':
                target_patch = self.gt_patch #[self.watermarking_mask].flatten()
                l1_distance = torch.abs(reversed_latents_fft[self.watermarking_mask] - target_patch[self.watermarking_mask]).mean().item()
                return {
                    'is_watermark': l1_distance < self.threshold, 
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
                    'is_watermark': bool(p < self.threshold), 
                    'p_value': p
                }
            else:
                raise ValueError(f"SFW(HSTR)'s watermark detector type {self.detector_type} not supported")

