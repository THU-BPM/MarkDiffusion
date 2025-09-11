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
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization
from Crypto.Cipher import ChaCha20


class VideoShieldVisualizer(BaseVisualizer):
    """VideoShield watermark visualization class.
    
    This visualizer handles watermark visualization for VideoShield algorithm,
    which extends Gaussian Shading to the video domain by adding frame dimensions.
    
    Key Members for VideoShieldVisualizer:
        - self.data.orig_watermarked_latents: [B, C, F, H, W]
        - self.data.reversed_latents: List[[B, C, F, H, W]]
    """
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1, is_video: bool = True):
        super().__init__(data_for_visualization, dpi, watermarking_step, is_video)
    
    def _stream_key_decrypt(self, reversed_m: np.ndarray) -> torch.Tensor:
        """Decrypt the watermark using ChaCha20 cipher.
        
        Args:
            reversed_m: Encrypted binary message array
            
        Returns:
            Decrypted watermark tensor
        """
        cipher = ChaCha20.new(key=self.data.chacha_key, nonce=self.data.chacha_nonce)
        
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        
        return sd_bit
    
    def _diffusion_inverse(self, reversed_sd: torch.Tensor) -> torch.Tensor:
        """Video-specific diffusion inverse with frame dimension handling.
        
        Args:
            reversed_sd: Video watermark tensor with shape (B, C, F, H, W)
            
        Returns:
            Extracted watermark pattern
        """
        ch_stride = 4 // self.data.k_c
        frame_stride = self.data.num_frames // self.data.k_f
        h_stride = self.data.latents_height // self.data.k_h
        w_stride = self.data.latents_width // self.data.k_w
        
        ch_list = [ch_stride] * self.data.k_c
        frame_list = [frame_stride] * self.data.k_f
        h_list = [h_stride] * self.data.k_h
        w_list = [w_stride] * self.data.k_w
        
        # Split and reorganize dimensions for voting
        split_dim1 = torch.cat(torch.split(reversed_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(frame_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(h_list), dim=3), dim=0)
        split_dim4 = torch.cat(torch.split(split_dim3, tuple(w_list), dim=4), dim=0)
        
        # Voting
        vote = torch.sum(split_dim4, dim=0).clone()
        vote[vote <= self.data.vote_threshold] = 0
        vote[vote > self.data.vote_threshold] = 1
        
        return vote
    
    def draw_watermark_bits(self,
                           channel: int | None = None,
                           frame: int | None = None,
                           title: str = "Original Watermark Bits",
                           cmap: str = "binary",
                           ax: Axes | None = None) -> Axes:
        """Draw the original watermark bits for VideoShield.
        
        For video watermarks, this method can visualize specific frames or average
        across frames to create a 2D visualization.
        
        Args:
            channel: The channel to visualize. If None, all channels are shown.
            frame: The frame to visualize. If None, uses middle frame for videos.
            title: The title of the plot.
            cmap: The colormap to use.
            ax: The axes to plot on.
            
        Returns:
            The plotted axes.
        """
        # Reshape watermark to video dimensions based on repetition factors
        # VideoShield watermark shape: [1, C//k_c, F//k_f, H//k_h, W//k_w]
        ch_stride = 4 // self.data.k_c
        frame_stride = self.data.num_frames // self.data.k_f
        h_stride = self.data.latents_height // self.data.k_h
        w_stride = self.data.latents_width // self.data.k_w
        
        watermark = self.data.watermark.reshape(1, ch_stride, frame_stride, h_stride, w_stride)
        
        if channel is not None:
            # Single channel visualization
            if channel >= ch_stride:
                raise ValueError(f"Channel {channel} is out of range. Max channel: {ch_stride - 1}")
            
            # Select specific frame or use middle frame
            if frame is not None:
                if frame >= frame_stride:
                    raise ValueError(f"Frame {frame} is out of range. Max frame: {frame_stride - 1}")
                watermark_data = watermark[0, channel, frame].cpu().numpy()
                frame_info = f" - Frame {frame}"
            else:
                # Use middle frame
                mid_frame = frame_stride // 2
                watermark_data = watermark[0, channel, mid_frame].cpu().numpy()
                frame_info = f" - Frame {mid_frame} (middle)"
            
            ax.imshow(watermark_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            if title != "":
                ax.set_title(f"{title} - Channel {channel}{frame_info}", fontsize=10)
            ax.axis('off')
        else:
            # Multi-channel visualization
            num_channels = ch_stride
            
            # Calculate grid layout
            rows = int(np.ceil(np.sqrt(num_channels)))
            cols = int(np.ceil(num_channels / rows))
            
            # Clear the axis and set title
            ax.clear()
            if title != "":
                if frame is not None:
                    ax.set_title(f"{title} - Frame {frame}", pad=20, fontsize=10)
                else:
                    mid_frame = frame_stride // 2
                    ax.set_title(f"{title} - Frame {mid_frame} (middle)", pad=20, fontsize=10)
            ax.axis('off')
            
            # Use gridspec for better control
            gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=ax.get_subplotspec(), 
                                         wspace=0.3, hspace=0.4)
            
            # Create subplots
            for i in range(num_channels):
                row_idx = i // cols
                col_idx = i % cols
                
                # Create subplot using gridspec
                sub_ax = ax.figure.add_subplot(gs[row_idx, col_idx])
                
                # Select specific frame or use middle frame
                if frame is not None:
                    if frame >= frame_stride:
                        raise ValueError(f"Frame {frame} is out of range. Max frame: {frame_stride - 1}")
                    watermark_data = watermark[0, i, frame].cpu().numpy()
                else:
                    mid_frame = frame_stride // 2
                    watermark_data = watermark[0, i, mid_frame].cpu().numpy()
                
                # Draw the watermark channel
                sub_ax.imshow(watermark_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
        
        return ax
        
    def draw_reconstructed_watermark_bits(self,
                                          channel: int | None = None,
                                          frame: int | None = None,
                                          title: str = "Reconstructed Watermark Bits",
                                          cmap: str = "binary",
                                          ax: Axes | None = None) -> Axes:
        """Draw the reconstructed watermark bits for VideoShield.
        
        Args:
            channel: The channel to visualize. If None, all channels are shown.
            frame: The frame to visualize. If None, uses middle frame for videos.
            title: The title of the plot.
            cmap: The colormap to use.
            ax: The axes to plot on.
            
        Returns:
            The plotted axes.
        """
        # Step 1: Get reversed latents and reconstruct the watermark bits
        reversed_latent = self.data.reversed_latents[self.watermarking_step]
        
        # Convert to binary bits
        reversed_m = (reversed_latent > 0).int()
        
        # Decrypt 
        reversed_sd_flat = self._stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        # Reshape back to video tensor
        reversed_sd = torch.from_numpy(reversed_sd_flat).reshape(reversed_latent.shape).to(torch.uint8)
        
        # Extract watermark through voting mechanism
        reversed_watermark = self._diffusion_inverse(reversed_sd.cuda())
        
        # Calculate bit accuracy
        bit_acc = (reversed_watermark == self.data.watermark).float().mean().item()
        
        # Reshape to video dimensions for visualization
        ch_stride = 4 // self.data.k_c
        frame_stride = self.data.num_frames // self.data.k_f
        h_stride = self.data.latents_height // self.data.k_h
        w_stride = self.data.latents_width // self.data.k_w
        
        reconstructed_watermark = reversed_watermark.reshape(1, ch_stride, frame_stride, h_stride, w_stride)
        
        if channel is not None:
            # Single channel visualization
            if channel >= ch_stride:
                raise ValueError(f"Channel {channel} is out of range. Max channel: {ch_stride - 1}")
            
            # Select specific frame or use middle frame
            if frame is not None:
                if frame >= frame_stride:
                    raise ValueError(f"Frame {frame} is out of range. Max frame: {frame_stride - 1}")
                reconstructed_watermark_data = reconstructed_watermark[0, channel, frame].cpu().numpy()
                frame_info = f" - Frame {frame}"
            else:
                # Use middle frame
                mid_frame = frame_stride // 2
                reconstructed_watermark_data = reconstructed_watermark[0, channel, mid_frame].cpu().numpy()
                frame_info = f" - Frame {mid_frame} (middle)"
            
            ax.imshow(reconstructed_watermark_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            if title != "":
                ax.set_title(f"{title} - Channel {channel}{frame_info} (Bit Acc: {bit_acc:.3f})", fontsize=10)
            else:
                ax.set_title(f"Channel {channel}{frame_info} (Bit Acc: {bit_acc:.3f})", fontsize=10)
            ax.axis('off')
        else:
            # Multi-channel visualization
            num_channels = ch_stride
            
            # Calculate grid layout
            rows = int(np.ceil(np.sqrt(num_channels)))
            cols = int(np.ceil(num_channels / rows))
            
            # Clear the axis and set title with bit accuracy
            ax.clear()
            if title != "":
                if frame is not None:
                    ax.set_title(f'{title} - Frame {frame} (Bit Acc: {bit_acc:.3f})', pad=20, fontsize=10)
                else:
                    mid_frame = frame_stride // 2
                    ax.set_title(f'{title} - Frame {mid_frame} (middle) (Bit Acc: {bit_acc:.3f})', pad=20, fontsize=10)
            else:
                if frame is not None:
                    ax.set_title(f'Frame {frame} (Bit Acc: {bit_acc:.3f})', pad=20, fontsize=10)
                else:
                    mid_frame = frame_stride // 2
                    ax.set_title(f'Frame {mid_frame} (middle) (Bit Acc: {bit_acc:.3f})', pad=20, fontsize=10)
            ax.axis('off')
            
            # Use gridspec for better control
            gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=ax.get_subplotspec(), 
                                         wspace=0.3, hspace=0.4)
            
            # Create subplots
            for i in range(num_channels):
                row_idx = i // cols
                col_idx = i % cols
                
                # Create subplot using gridspec
                sub_ax = ax.figure.add_subplot(gs[row_idx, col_idx])
                
                # Select specific frame or use middle frame
                if frame is not None:
                    if frame >= frame_stride:
                        raise ValueError(f"Frame {frame} is out of range. Max frame: {frame_stride - 1}")
                    reconstructed_watermark_data = reconstructed_watermark[0, i, frame].cpu().numpy()
                else:
                    mid_frame = frame_stride // 2
                    reconstructed_watermark_data = reconstructed_watermark[0, i, mid_frame].cpu().numpy()
                
                # Draw the reconstructed watermark channel
                sub_ax.imshow(reconstructed_watermark_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
        
        return ax