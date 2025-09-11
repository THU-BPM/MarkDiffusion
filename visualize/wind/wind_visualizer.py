import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization
from matplotlib.gridspec import GridSpecFromSubplotSpec

class WINDVisualizer(BaseVisualizer):
    """WIND watermark visualization class"""
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1):
        super().__init__(data_for_visualization, dpi, watermarking_step)
        index = self.data.current_index % self.data.M
        self.group_pattern = self.data.group_patterns[index] # shape: [4, 64, 64]
    
    def draw_group_pattern_fft(self,
                           channel: int | None = None,
                           title: str = "Group Pattern in Fourier Domain",
                           cmap: str = "viridis",
                           use_color_bar: bool = True,
                           vmin: float | None = None,
                           vmax: float | None = None,
                           ax: Axes | None = None,
                           **kwargs) -> Axes:
        """
        Draw the group pattern in Fourier Domain.
        
        Parameters:
            channel (int | None): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        if channel is not None:
            im = ax.imshow(np.abs(self.group_pattern[channel].cpu().numpy()), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
            if title != "":
                ax.set_title(title)
            if use_color_bar:
                ax.figure.colorbar(im, ax=ax)
            ax.axis('off')
        else:
            # Multi-channel visualization
            num_channels = 4
            rows = 2
            cols = 2
            
            # Clear the axis and set title
            ax.clear()
            if title != "":
                ax.set_title(title, pad=20)
            ax.axis('off')
            
            # Use gridspec for better control
            gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=ax.get_subplotspec(), 
                                         wspace=0.3, hspace=0.4)
            
            # Create subplots for each channel
            for i in range(num_channels):
                row_idx = i // cols
                col_idx = i % cols
                
                # Create subplot using gridspec
                sub_ax = ax.figure.add_subplot(gs[row_idx, col_idx])
                
                # Draw the latent channel
                latent_data = self.group_pattern[i].cpu().numpy()
                im = sub_ax.imshow(np.abs(latent_data), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
        
        return ax
        
    
    def draw_orig_noise_wo_group_pattern(self,
                                         channel: int | None = None,
                         title: str = "Original Noise without Group Pattern",
                         cmap: str = "viridis",
                         use_color_bar: bool = True,
                         vmin: float | None = None,
                         vmax: float | None = None,
                         ax: Axes | None = None,
                         **kwargs) -> Axes:
        """
        Draw the original noise without group pattern.
        
        Parameters:
            channel (int | None): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (plt.Axes): The axes to plot on.
            
        Returns:
            plt.axes.Axes: The plotted axes.
        """
        if channel is not None:
            # Single channel visualization
            orig_noise_fft = self._fft_transform(self.data.orig_watermarked_latents[0, channel])
            z_fft = orig_noise_fft - self.group_pattern[channel].cpu().numpy()
            z_cleaned = self._ifft_transform(z_fft).real
            im = ax.imshow(z_cleaned, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
            if title != "":
                ax.set_title(title)
            if use_color_bar:
                ax.figure.colorbar(im, ax=ax)
            ax.axis('off')
        else:
            # Multi-channel visualization
            num_channels = 4
            rows = 2
            cols = 2
            
            # Clear the axis and set title
            ax.clear()
            if title != "":
                ax.set_title(title, pad=20)
            ax.axis('off')
            
            # Use gridspec for better control
            gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=ax.get_subplotspec(), 
                                         wspace=0.3, hspace=0.4)
            
            # Create subplots for each channel
            for i in range(num_channels):
                row_idx = i // cols
                col_idx = i % cols
                
                # Create subplot using gridspec
                sub_ax = ax.figure.add_subplot(gs[row_idx, col_idx])
                
                # Compute original noise without group pattern for this channel
                orig_noise_fft = self._fft_transform(self.data.orig_watermarked_latents[0, i])
                z_fft = orig_noise_fft - self.group_pattern[i].cpu().numpy()
                z_cleaned = self._ifft_transform(z_fft).real
                
                # Draw the cleaned noise channel
                im = sub_ax.imshow(z_cleaned, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
        
        return ax
    
    def draw_inverted_noise_wo_group_pattern(self,
                                             channel: int | None = None,
                          title: str = "Inverted Noise without Group Pattern",
                          cmap: str = "viridis",
                          use_color_bar: bool = True,
                          vmin: float | None = None,
                          vmax: float | None = None,
                          ax: Axes | None = None,
                          **kwargs) -> Axes:
        """
        Draw the inverted noise without group pattern.
        
        Parameters:
            channel (int | None): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (plt.Axes): The axes to plot on.
            
        Returns:
            plt.axes.Axes: The plotted axes.
        """
        if channel is not None:
            # Single channel visualization
            reversed_latent = self.data.reversed_latents[self.watermarking_step]
            reversed_latent_fft = self._fft_transform(reversed_latent[0, channel])
            z_fft = reversed_latent_fft - self.group_pattern[channel].cpu().numpy()
            z_cleaned = self._ifft_transform(z_fft).real
            im = ax.imshow(z_cleaned, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
            if title != "":
                ax.set_title(title)
            if use_color_bar:
                ax.figure.colorbar(im, ax=ax)
            ax.axis('off')
        else:
            # Multi-channel visualization
            num_channels = 4
            rows = 2
            cols = 2
            
            # Clear the axis and set title
            ax.clear()
            if title != "":
                ax.set_title(title, pad=20)
            ax.axis('off')
            
            # Use gridspec for better control
            gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=ax.get_subplotspec(), 
                                         wspace=0.3, hspace=0.4)
            
            # Create subplots for each channel
            for i in range(num_channels):
                row_idx = i // cols
                col_idx = i % cols
                
                # Create subplot using gridspec
                sub_ax = ax.figure.add_subplot(gs[row_idx, col_idx])
                
                # Compute inverted noise without group pattern for this channel
                reversed_latent = self.data.reversed_latents[self.watermarking_step]
                reversed_latent_fft = self._fft_transform(reversed_latent[0, i])
                z_fft = reversed_latent_fft - self.group_pattern[i].cpu().numpy()
                z_cleaned = self._ifft_transform(z_fft).real
                
                # Draw the cleaned inverted noise channel
                im = sub_ax.imshow(z_cleaned, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
        
        return ax
    
    def draw_diff_noise_wo_group_pattern(self,
                                         channel: int | None = None,
                          title: str = "Difference map without Group Pattern",
                          cmap: str = "coolwarm",
                          use_color_bar: bool = True,
                          vmin: float | None = None,
                          vmax: float | None = None,
                          ax: Axes | None = None,
                          **kwargs) -> Axes:
        """
        Draw the difference between original and inverted noise after removing group pattern.
        
        Parameters:
            channel (int | None): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (plt.Axes): The axes to plot on.
            
        Returns:
            plt.axes.Axes: The plotted axes.
        """
        if channel is not None:
            # Single channel visualization
            # Process original latents
            orig_latent_channel = self.data.orig_watermarked_latents[0, channel]
            orig_noise_fft = self._fft_transform(orig_latent_channel)
            orig_z_fft = orig_noise_fft - self.group_pattern[channel].cpu().numpy()
            orig_z_cleaned = self._ifft_transform(orig_z_fft).real
            
            # Process inverted latents
            reversed_latent = self.data.reversed_latents[self.watermarking_step]
            reversed_latent_fft = self._fft_transform(reversed_latent[0, channel])
            inv_z_fft = reversed_latent_fft - self.group_pattern[channel].cpu().numpy()
            inv_z_cleaned = self._ifft_transform(inv_z_fft).real
            
            # Compute difference
            diff_cleaned = orig_z_cleaned - inv_z_cleaned
            
            im = ax.imshow(diff_cleaned, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
            if title != "":
                ax.set_title(title)
            if use_color_bar:
                ax.figure.colorbar(im, ax=ax)
            ax.axis('off')
        else:
            # Multi-channel visualization
            num_channels = 4
            rows = 2
            cols = 2
            
            # Clear the axis and set title
            ax.clear()
            if title != "":
                ax.set_title(title, pad=20)
            ax.axis('off')
            
            # Use gridspec for better control
            gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=ax.get_subplotspec(), 
                                         wspace=0.3, hspace=0.4)
            
            # Create subplots for each channel
            for i in range(num_channels):
                row_idx = i // cols
                col_idx = i % cols
                
                # Create subplot using gridspec
                sub_ax = ax.figure.add_subplot(gs[row_idx, col_idx])
                
                # Process original latents for this channel
                orig_latent_channel = self.data.orig_watermarked_latents[0, i]
                orig_noise_fft = self._fft_transform(orig_latent_channel)
                orig_z_fft = orig_noise_fft - self.group_pattern[i].cpu().numpy()
                orig_z_cleaned = self._ifft_transform(orig_z_fft).real
                
                # Process inverted latents for this channel
                reversed_latent = self.data.reversed_latents[self.watermarking_step]
                reversed_latent_fft = self._fft_transform(reversed_latent[0, i])
                inv_z_fft = reversed_latent_fft - self.group_pattern[i].cpu().numpy()
                inv_z_cleaned = self._ifft_transform(inv_z_fft).real
                
                # Compute difference
                diff_cleaned = orig_z_cleaned - inv_z_cleaned
                
                # Draw the difference channel
                im = sub_ax.imshow(diff_cleaned, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
        
        return ax