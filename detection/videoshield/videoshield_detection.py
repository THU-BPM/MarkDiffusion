import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from detection.base import BaseDetector
from Crypto.Cipher import ChaCha20
from scipy.stats import truncnorm, norm
from functools import reduce
import logging

logger = logging.getLogger(__name__)


class VideoShieldDetector(BaseDetector):
    """VideoShield watermark detector class."""

    def __init__(self, 
                 watermark: torch.Tensor, 
                 threshold: float,
                 device: torch.device,
                 chacha_key: Optional[bytes] = None,
                 chacha_nonce: Optional[bytes] = None,
                 height: int = 64,
                 width: int = 64,
                 num_frames: int = 0,
                 k_f: int = 8,
                 k_c: int = 1, 
                 k_h: int = 4,
                 k_w: int = 4) -> None:
        """Initialize the VideoShield detector.
        
        Args:
            watermark: The watermarking bits
            threshold: Threshold for watermark detection
            device: The device to use for computation
            chacha_key: ChaCha20 encryption key (optional)
            chacha_nonce: ChaCha20 nonce (optional)
            height: Height of the video
            width: Width of the video
            num_frames: Number of frames in the video
            k_f: Frame repetition factor
            k_c: Channel repetition factor
            k_h: Height repetition factor
            k_w: Width repetition factor
        """
        super().__init__(threshold, device)
        self.watermark = watermark.to(device)
        self.chacha_key = chacha_key
        self.chacha_nonce = chacha_nonce
        self.num_frames = num_frames
        self.height = height
        self.width = width
        # Repetition factors
        self.k_f = k_f
        self.k_c = k_c
        self.k_h = k_h
        self.k_w = k_w
        
        # Calculate voting threshold
        if k_f == 1 and k_c == 1 and k_h == 1 and k_w == 1:
            self.vote_threshold = 1
        else:
            self.vote_threshold = (k_f * k_c * k_h * k_w) // 2
        
    def _stream_key_decrypt(self, reversed_m: np.ndarray) -> np.ndarray:
        """Decrypt the watermark using ChaCha20 cipher."""
        if self.chacha_key is None or self.chacha_nonce is None:
            # If no encryption keys provided, return as-is
            return reversed_m
            
        cipher = ChaCha20.new(key=self.chacha_key, nonce=self.chacha_nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        
        return sd_bit
    
    def _diffusion_inverse(self, watermark_r: torch.Tensor, is_video: bool = False) -> torch.Tensor:
        """Inverse the diffusion process to extract the watermark through voting.
        
        Args:
            watermark_r: The reversed watermark tensor
            is_video: Whether this is video (5D) or image (4D) data
            
        Returns:
            Extracted watermark through voting
        """
        if is_video and watermark_r.dim() == 5:
            return self._video_diffusion_inverse(watermark_r)
        else:
            return self._image_diffusion_inverse(watermark_r)
    
    def _video_diffusion_inverse(self, watermark_r: torch.Tensor) -> torch.Tensor:
        """Video-specific diffusion inverse with frame dimension handling."""
        # Get tensor dimensions
        batch, channels, frames, height, width = watermark_r.shape
        
        ch_stride = channels // self.k_c
        frame_stride = frames // self.k_f
        h_stride = height // self.k_h
        w_stride = width // self.k_w
        
        ch_list = [ch_stride] * self.k_c
        frame_list = [frame_stride] * self.k_f
        h_list = [h_stride] * self.k_h
        w_list = [w_stride] * self.k_w
        
        # Split and reorganize dimensions for voting
        try:
            split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
            split_dim2 = torch.cat(torch.split(split_dim1, tuple(frame_list), dim=2), dim=0)
            split_dim3 = torch.cat(torch.split(split_dim2, tuple(h_list), dim=3), dim=0)
            split_dim4 = torch.cat(torch.split(split_dim3, tuple(w_list), dim=4), dim=0)
            
            # Voting
            vote = torch.sum(split_dim4, dim=0).clone()
            vote[vote <= self.vote_threshold] = 0
            vote[vote > self.vote_threshold] = 1
            
            return vote
        except Exception as e:
            logger.error(f"Video diffusion inverse failed: {e}")
            # Return a fallback result
            return torch.zeros_like(self.gt_patch)
    
    def _image_diffusion_inverse(self, watermark_r: torch.Tensor) -> torch.Tensor:
        """Image-specific diffusion inverse."""
        # Handle both 4D and 5D tensors by squeezing if needed
        if watermark_r.dim() == 5:
            watermark_r = watermark_r.squeeze(2)  # Remove frame dimension
            
        batch, channels, height, width = watermark_r.shape
        
        ch_stride = channels // self.k_c
        h_stride = height // self.k_h
        w_stride = width // self.k_w
        
        ch_list = [ch_stride] * self.k_c
        h_list = [h_stride] * self.k_h
        w_list = [w_stride] * self.k_w
        
        try:
            split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
            split_dim2 = torch.cat(torch.split(split_dim1, tuple(h_list), dim=2), dim=0)
            split_dim3 = torch.cat(torch.split(split_dim2, tuple(w_list), dim=3), dim=0)
            
            vote = torch.sum(split_dim3, dim=0).clone()
            vote[vote <= self.vote_threshold] = 0
            vote[vote > self.vote_threshold] = 1
            
            return vote
        except Exception as e:
            logger.error(f"Image diffusion inverse failed: {e}")
            # Return a fallback result
            return torch.zeros_like(self.gt_patch)
        
    def eval_watermark(self,
                       reversed_latents: torch.Tensor,
                       detector_type: str = "bit_acc") -> Dict[str, Union[bool, float]]:
        """Evaluate the watermark in the reversed latents.
        
        Args:
            reversed_latents: The reversed latents from forward diffusion
            detector_type: The type of detector to use ('bit_acc', 'standard', etc.)
                
        Returns:
            Dict containing detection results and confidence scores
        """
        if detector_type not in ['bit_acc']:
            raise ValueError(f'Detector type {detector_type} is not supported for VideoShield.')
            
        # Basic validation
        if reversed_latents.numel() == 0:
            return {'is_watermarked': False, 'bit_acc': 0.0, 'confidence': 0.0}
        
        # Convert latents to binary bits
        reversed_m = (reversed_latents > 0).int()
        
        # Decrypt if encryption keys are available
        if self.chacha_key is not None and self.chacha_nonce is not None:
            reversed_sd = self._stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        else:
            # No decryption, use reversed bits directly
            reversed_sd = reversed_m
            
        # Reshape back to tensor format
        if reversed_latents.dim() == 5:
            # Video case
            # [B, C, F, H, W] for T2V model
            # [B, F, C, H, W] for I2V model
            batch, channels_or_frames, frames_or_channels, height, width = reversed_latents.shape
            reversed_sd_tensor = torch.from_numpy(reversed_sd).reshape(
                batch, channels_or_frames, frames_or_channels, height, width
            ).to(torch.uint8).to(self.device)
        else:
            # Image case
            batch, channels, height, width = reversed_latents.shape
            reversed_sd_tensor = torch.from_numpy(reversed_sd).reshape(
                batch, channels, height, width
            ).to(torch.uint8).to(self.device)
        
        # Extract watermark through voting
        is_video = reversed_latents.dim() == 5
        reversed_watermark = self._diffusion_inverse(reversed_sd_tensor, is_video)
        
        correct = (reversed_watermark == self.watermark).float().mean().item()
        
        return {
            'is_watermarked': bool(correct > self.threshold),
            'bit_acc': correct,
        }