import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization
from Crypto.Cipher import ChaCha20

class GaussianShadingVisualizer(BaseVisualizer):
    """Gaussian Shading watermark visualization class"""
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1):
        super().__init__(data_for_visualization, dpi, watermarking_step)
    
    def _stream_key_decrypt(self, reversed_m):
        """Decrypt the watermark using ChaCha20 cipher."""
        cipher = ChaCha20.new(key=self.data.chacha_key, nonce=self.data.chacha_nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor.cuda()
    
    def _diffusion_inverse(self, reversed_sd):
        """Inverse the diffusion process to extract the watermark."""
        ch_stride = 4 // self.data.channel_copy
        hw_stride = 64 // self.data.hw_copy
        ch_list = [ch_stride] * self.data.channel_copy
        hw_list = [hw_stride] * self.data.hw_copy
        split_dim1 = torch.cat(torch.split(reversed_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.data.vote_threshold] = 0
        vote[vote > self.data.vote_threshold] = 1
        return vote
    
    def draw_watermark_bits(self,
                               channel: int | None = None,
                               title: str = "Original Watermark Bits",
                               cmap: str = "binary",
                               ax: Axes | None = None) -> Axes:
        """
        Draw the original watermark bits.(sd in GS class). draw ch // channel_copy images in one ax.
        
        Parameters:
            channel (int | None): The channel to visualize. If None, all channels are shown.
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        # Step 1: reshape self.data.watermark to [1, 4 // self.data.channel_copy, 64 // self.data.hw_copy, 64 // self.data.hw_copy]
        watermark = self.data.watermark.reshape(1, 4 // self.data.channel_copy, 64 // self.data.hw_copy, 64 // self.data.hw_copy)
        
        if channel is not None:
            # Single channel visualization
            if channel >= 4 // self.data.channel_copy:
                raise ValueError(f"Channel {channel} is out of range. Max channel: {4 // self.data.channel_copy - 1}")
            
            watermark_data = watermark[0, channel].cpu().numpy()
            im = ax.imshow(watermark_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            if title != "":
                ax.set_title(f"{title} - Channel {channel}", fontsize=10)
            ax.axis('off')
            
            cbar = ax.figure.colorbar(im, ax=ax, alpha=0.0)
            cbar.ax.set_visible(False)
        else:
            # Multi-channel visualization
            # Step 2: draw watermark for (4 // self.data.channel_copy) images in this ax
            num_channels = 4 // self.data.channel_copy
            
            # Calculate grid layout
            rows = int(np.ceil(np.sqrt(num_channels)))
            cols = int(np.ceil(num_channels / rows))
            
            # Clear the axis and set title
            ax.clear()
            if title != "":
                ax.set_title(title, pad=20)
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
                
                # Draw the watermark channel
                watermark_data = watermark[0, i].cpu().numpy()
                sub_ax.imshow(watermark_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
        
        return ax
        
    def draw_reconstructed_watermark_bits(self,
                                          channel: int | None = None,
                                          title: str = "Reconstructed Watermark Bits",
                                          cmap: str = "binary",
                                          ax: Axes | None = None) -> Axes:
        """
        Draw the reconstructed watermark bits.(reversed_latents in GS class). draw ch // channel_copy images in one ax.
        
        Parameters:
            channel (int | None): The channel to visualize. If None, all channels are shown.
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        # Step 1: reconstruct the watermark bits
        reversed_latent = self.data.reversed_latents[self.watermarking_step]
        
        reversed_m = (reversed_latent > 0).int()
        if self.data.chacha:
            reversed_sd = self._stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        else:
            reversed_sd = (reversed_m + self.data.key) % 2
        
        reversed_watermark = self._diffusion_inverse(reversed_sd)
        bit_acc = (reversed_watermark == self.data.watermark).float().mean().item()
        reconstructed_watermark = reversed_watermark.reshape(1, 4 // self.data.channel_copy, 64 // self.data.hw_copy, 64 // self.data.hw_copy)
        
        if channel is not None:
            # Single channel visualization
            if channel >= 4 // self.data.channel_copy:
                raise ValueError(f"Channel {channel} is out of range. Max channel: {4 // self.data.channel_copy - 1}")
            
            reconstructed_watermark_data = reconstructed_watermark[0, channel].cpu().numpy()
            im = ax.imshow(reconstructed_watermark_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            if title != "":
                ax.set_title(f"{title} - Channel {channel} (Bit Acc: {bit_acc:.3f})", fontsize=10)
            else:
                ax.set_title(f"Channel {channel} (Bit Acc: {bit_acc:.3f})", fontsize=10)
            ax.axis('off')
            
            cbar = ax.figure.colorbar(im, ax=ax, alpha=0.0)
            cbar.ax.set_visible(False)
        else:
            # Multi-channel visualization
            # Step 2: draw reconstructed_watermark for (4 // self.data.channel_copy) images in this ax(add Bit_acc to the title)
            num_channels = 4 // self.data.channel_copy
            
            # Calculate grid layout
            rows = int(np.ceil(np.sqrt(num_channels)))
            cols = int(np.ceil(num_channels / rows))
            
            # Clear the axis and set title with bit accuracy
            ax.clear()
            if title != "":
                ax.set_title(f'{title} (Bit Acc: {bit_acc:.3f})', pad=20)
            else:
                ax.set_title(f'(Bit Acc: {bit_acc:.3f})', pad=20)
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
                
                # Draw the reconstructed watermark channel
                reconstructed_watermark_data = reconstructed_watermark[0, i].cpu().numpy()
                sub_ax.imshow(reconstructed_watermark_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
        
        return ax
