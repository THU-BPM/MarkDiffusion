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
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import math

class SEALDetector(BaseDetector):
    
    def __init__(self,
                 k: int,
                 b: int,
                 theta_mid: int,
                 cap_processor: Blip2Processor,
                 cap_model: Blip2ForConditionalGeneration,
                 sentence_transformer: SentenceTransformer,
                 patch_distance_threshold: float,
                 device: torch.device):
        super().__init__(patch_distance_threshold, device)
        self.k = k
        self.b = b
        self.theta_mid = theta_mid
        self.cap_processor = cap_processor
        self.cap_model = cap_model
        self.sentence_transformer = sentence_transformer
        self.patch_distance_threshold = patch_distance_threshold
        
        # Calculate the match threshold
        # m^{\text{match}} = \left\lfloor n \rho(\theta^{\text{mid}}) \right\rfloor
        # \rho(\theta) = \left( 1 - \frac{\theta}{180^\circ} \right)^b
        # n = k
        self.match_threshold = math.floor(self.k * ((1 - self.theta_mid / 180) ** self.b))
        
    def _calculate_patch_l2(self, noise1: torch.Tensor, noise2: torch.Tensor, k: int) -> torch.Tensor:
        """
            Calculate L2 distances patch by patch. Returns a list of L2 values for the first k patches.
        """
        l2_list = []
        patch_per_side_h = int(math.ceil(math.sqrt(k)))
        patch_per_side_w = int(math.ceil(k / patch_per_side_h))
        patch_height = 64 // patch_per_side_h
        patch_width = 64 // patch_per_side_w
        patch_count = 0
        for i in range(patch_per_side_h):
            for j in range(patch_per_side_w):
                if patch_count >= k:
                    break
                y_start = i * patch_height
                x_start = j * patch_width
                y_end = min(y_start + patch_height, 64)
                x_end = min(x_start + patch_width, 64)
                patch1 = noise1[:, :, y_start:y_end, x_start:x_end]
                patch2 = noise2[:, :, y_start:y_end, x_start:x_end]
                l2_val = torch.norm(patch1 - patch2).item()
                l2_list.append(l2_val)
                patch_count += 1
        return l2_list

    def eval_watermark(self,
                        reversed_latents: torch.Tensor,
                        reference_latents: torch.Tensor,
                        detector_type: str = "patch_accuracy") -> float:
        
        if detector_type != "patch_accuracy":
            raise ValueError(f"Detector type {detector_type} is not supported for SEAL detector")
        
        l2_patch_list = self._calculate_patch_l2(reversed_latents, reference_latents, self.k)
        
        # Count the number of patches that are less than the threshold
        num_patches_below_threshold = sum(1 for l2 in l2_patch_list if l2 < self.patch_distance_threshold)
        
        return {
            "is_watermarked": bool(num_patches_below_threshold >= self.match_threshold),
            "patch_accuracy": num_patches_below_threshold / self.k,
        }
        
    