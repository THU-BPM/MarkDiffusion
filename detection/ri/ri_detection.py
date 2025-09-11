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
import numpy as np

class RIDetector(BaseDetector):
    
    def __init__(self,
                 watermarking_mask: torch.Tensor,
                 ring_watermark_channel: list,
                 heter_watermark_channel: list,
                 pattern_list: list,
                 threshold: float,
                 device: torch.device):
        super().__init__(threshold, device)
        self.watermarking_mask = watermarking_mask
        self.ring_watermark_channel = ring_watermark_channel
        self.heter_watermark_channel = heter_watermark_channel
        self.watermark_channel = sorted(self.heter_watermark_channel + self.ring_watermark_channel)
        self.pattern_list = pattern_list
        
    def _get_distance(self, tensor1, tensor2, mask, p, mode, channel_min=False):
        channel = self.watermark_channel
        if tensor1.shape != tensor2.shape:
            raise ValueError(f'Shape mismatch during eval: {tensor1.shape} vs {tensor2.shape}')
        if mode not in ['complex', 'real', 'imag']:
            raise NotImplemented(f'Eval mode not implemented: {mode}')

        if not channel_min:
            if p == 1:
                # a faster implementation for l1 distance
                if mode == 'complex':
                    return torch.mean(torch.abs(tensor1[0][channel] - tensor2[0][channel])[mask]).item()
                if mode == 'real':
                    return torch.mean(torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)[mask]).item()
                if mode == 'imag':
                    return torch.mean(torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)[mask]).item()
            else:
                if mode == 'complex':
                    return torch.norm(torch.abs(tensor1[0][channel][mask] - tensor2[0][channel][mask]),
                                      p=p).item() / torch.sum(mask)
                if mode == 'real':
                    return torch.norm(torch.abs(tensor1[0][channel][mask].real - tensor2[0][channel][mask].real),
                                      p=p).item() / torch.sum(mask)
                if mode == 'imag':
                    return torch.norm(torch.abs(tensor1[0][channel][mask].imag - tensor2[0][channel][mask].imag),
                                      p=p).item() / torch.sum(mask)
        else:
            # argmin TODO: normalize
            if len(self.ring_watermark_channel) > 1 and len(self.heter_watermark_channel) > 0:
                ring_channel_idx_list = [idx for idx, c_id in enumerate(self.watermark_channel) if
                                         c_id in self.ring_watermark_channel]
                heter_channel_idx_list = [idx for idx, c_id in enumerate(self.watermark_channel) if
                                          c_id in self.heter_watermark_channel]
                if mode == 'complex':
                    diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])  # [c, h, w]
                elif mode == 'real':
                    diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)  # [c, h, w]
                elif mode == 'imag':
                    diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)  # [c, h, w]
                l1_list = []
                num_items = []
                for c_idx in range(len(mask)):
                    mask_c = torch.zeros_like(mask)
                    mask_c[c_idx] = mask[c_idx]
                    l1_list.append(torch.mean(diff[mask_c]).item())
                    num_items.append(torch.sum(mask_c).item())
                total = 0
                num = 0
                for ring_channel_idx in ring_channel_idx_list:
                    total += l1_list[ring_channel_idx] * num_items[ring_channel_idx]
                    num += num_items[ring_channel_idx]
                ring_channels_mean = total / num
                return min(ring_channels_mean, min([l1_list[idx] for idx in heter_channel_idx_list]))
            elif len(self.ring_watermark_channel) == 1 and len(self.heter_watermark_channel) > 0:
                if mode == 'complex':
                    diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])  # [c, h, w]
                elif mode == 'real':
                    diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)  # [c, h, w]
                elif mode == 'imag':
                    diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)  # [c, h, w]
                l1_list = []
                for c_idx in range(len(mask)):
                    mask_c = torch.zeros_like(mask)
                    mask_c[c_idx] = mask[c_idx]
                    l1_list.append(torch.mean(diff[mask_c]).item())
                return min(l1_list)
            else:
                raise NotImplementedError
        
        
    def eval_watermark(self,
                        reversed_latents: torch.Tensor,
                        reference_latents: torch.Tensor = None,
                        detector_type: str = "l1_distance") -> float:
        
        if detector_type != 'l1_distance':
            raise ValueError(f"Detector type {detector_type} not supported")
        
        reversed_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
        all_distances = []
        
        for idx, pattern in enumerate(self.pattern_list):
            dist = self._get_distance(
                pattern,
                reversed_fft,
                self.watermarking_mask,
                p=1,
                mode="complex",
                channel_min=False
            )
            all_distances.append(dist)
        
        min_idx = int(np.argmin(all_distances))
        min_dist = all_distances[min_idx]
        
        return {
            'is_watermark': bool(min_dist < self.threshold),
            'l1_distance': min_dist
        }
        
        