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
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any

class WINDetector(BaseDetector):
    """WIND Watermark Detector (Two-Stage Robust Detection)"""
    
    def __init__(self,
                 noise_groups: Dict[int, torch.Tensor],
                 group_patterns: Dict[int, torch.Tensor],
                 threshold: float,
                 device: torch.device,
                 group_radius: int = 10):  
        super().__init__(threshold, device)
        
        self.noise_groups = {
            int(g): [noise.to(device) for noise in noise_list]
            for g, noise_list in noise_groups.items()
        }
        
        self.group_patterns = {
            int(g): pattern.to(device)
            for g, pattern in group_patterns.items()
        }
        
        self.group_radius = group_radius 
        self.device = device
        self.group_threshold = 0.7
        self.logger = logging.getLogger(__name__)
        

    def _circle_mask(self, size: int, r: int) -> torch.Tensor:
        """Using circle mask(same with Utils)"""
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device),
            torch.arange(size, device=self.device),
            indexing='ij'
        )
        center = size // 2
        dist = (x - center)**2 + (y - center)**2
        return ((dist >= (r-2)**2) & (dist <= r**2)).float()

    def _fft_transform(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert to Fourier space with shift"""
        return torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))

   
    def _retrieve_group(self, z_fft: torch.Tensor) -> int:
        similarities = []
        mask = self._circle_mask(z_fft.shape[-1], self.group_radius)  
    
        for group_id, pattern in self.group_patterns.items():
            try:
                z_masked = torch.abs(z_fft) * mask
                pattern_masked = torch.abs(pattern) * mask
            
                sim = F.cosine_similarity(
                    z_masked.flatten().unsqueeze(0),
                    pattern_masked.flatten().unsqueeze(0)
                ).item()
            
                similarities.append((group_id, sim))
            except Exception as e:
                   self.logger.warning(f"Error processing group {group_id}: {str(e)}")
                   continue
            return max(similarities, key=lambda x: x[1])[0] if similarities else -1

    def _match_noise(self, z: torch.Tensor, group_id: int) -> Dict[str, Any]:
        if group_id not in self.noise_groups or group_id == -1:
            return {'cosine_similarity': 0.0, 'best_match': None}

        mask = self._circle_mask(z.shape[-1], self.group_radius)
        z_fft = torch.fft.fftshift(torch.fft.fft2(z), dim=(-1, -2))
        z_fft = z_fft - self.group_patterns[group_id] * mask  
        z_cleaned = torch.fft.ifft2(torch.fft.ifftshift(z_fft)).real

        max_sim = -1.0
        best_noise = None

        for candidate in self.noise_groups[group_id]:
            sim = F.cosine_similarity(
                z_cleaned.flatten().unsqueeze(0),
                candidate.flatten().unsqueeze(0)
            ).item()
            if sim > max_sim:
                max_sim = sim
                best_noise = candidate

        return {
            'cosine_similarity': max(max_sim, 0.0),
            'best_match': best_noise
        }
    
    def eval_watermark(self,
                      reversed_latents: torch.Tensor,
                      reference_latents: torch.Tensor = None,
                      detector_type: str = "cosine_similarity") -> Dict[str, Any]:
        """
        Two-stage watermark detection
        
        Args:
            reversed_latents: Latents obtained through reverse diffusion [C,H,W]
            reference_latents: Not used (for API compatibility)
            detector_type: Detection method ('cosine_similarity' only supported)
            
        Returns:
            Dictionary containing detection results:
            - group_id: Identified group ID
            - similarity: Highest similarity score
            - is_watermarked: Detection result
            - best_match: Best matching noise tensor
        """
        if detector_type != "cosine_similarity":
            raise ValueError(f"WIND detector only supports 'cosine' method, got {detector_type}")
        
        try:
            # Input validation
            if not isinstance(reversed_latents, torch.Tensor):
                reversed_latents = torch.tensor(reversed_latents, device=self.device)
            reversed_latents = reversed_latents.to(self.device)
            
            # Stage 1: Group identification
            z_fft = self._fft_transform(reversed_latents)
            group_id = self._retrieve_group(z_fft)
            
            # Stage 2: Noise matching
            match_result = self._match_noise(reversed_latents, group_id)
            
            return {
                'group_id': group_id,
                'cosine_similarity': match_result['cosine_similarity'],
                'is_watermarked': bool(match_result['cosine_similarity'] > self.threshold),
                #'best_match': match_result['best_match']
            }
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            return {
                'group_id': -1,
                'cosine_similarity': 0.0,
                'is_watermarked': False,
                # 'best_match': None
            }

