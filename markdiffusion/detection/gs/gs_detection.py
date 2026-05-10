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
import numpy as np
from Crypto.Random import get_random_bytes
from Crypto.Cipher import ChaCha20
from scipy.stats import truncnorm, norm
from functools import reduce
from detection.base import BaseDetector
from typing import Union

class GSDetector(BaseDetector):
    
    def __init__(self,
                 watermarking_mask: torch.Tensor,
                 chacha: bool,
                 wm_key: Union[int, tuple[int, int]],
                 channel_copy: int,
                 hw_copy: int,
                 vote_threshold: int,
                 threshold: float,
                 device: torch.device):
        super().__init__(threshold, device)
        self.chacha = chacha
        self.channel_copy = channel_copy
        self.hw_copy = hw_copy
        self.watermark = watermarking_mask
        self.vote_threshold = vote_threshold
        if self.chacha:
            self.chacha_key, self.chacha_nonce = wm_key
        else:
            self.key = wm_key
    
    def _stream_key_decrypt(self, reversed_m):
        """Decrypt the watermark using ChaCha20 cipher."""
        cipher = ChaCha20.new(key=self.chacha_key, nonce=self.chacha_nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        
        # Calculate dimensions dynamically
        total_elements = sd_bit.size
        spatial_size = int(np.sqrt(total_elements / 4))
        
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, spatial_size, spatial_size).to(torch.uint8)
        return sd_tensor.to(self.device)
    
    def _diffusion_inverse(self, reversed_sd):
        """Inverse the diffusion process to extract the watermark."""
        _, _, H, W = reversed_sd.shape
        
        ch_stride = 4 // self.channel_copy
        hw_stride_h = H // self.hw_copy
        hw_stride_w = W // self.hw_copy
        
        ch_list = [ch_stride] * self.channel_copy
        hw_list_h = [hw_stride_h] * self.hw_copy
        hw_list_w = [hw_stride_w] * self.hw_copy
        
        split_dim1 = torch.cat(torch.split(reversed_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list_h), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list_w), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.vote_threshold] = 0
        vote[vote > self.vote_threshold] = 1
        return vote
    
    def eval_watermark(self, reversed_latents: torch.Tensor, detector_type: str = "bit_acc") -> float:
        """Evaluate watermark in reversed latents."""
        if detector_type != 'bit_acc':
            raise ValueError(f'Detector type {detector_type} is not supported for Gaussian Shading.')
        reversed_m = (reversed_latents > 0).int()
        if self.chacha:
            reversed_sd = self._stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        else:
            reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self._diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        
        return {
            'is_watermarked': bool(correct > self.threshold),
            'bit_acc': correct
        }
        