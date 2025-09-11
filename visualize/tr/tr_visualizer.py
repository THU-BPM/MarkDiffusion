import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization

class TreeRingVisualizer(BaseVisualizer):
    """Tree-Ring watermark visualization class"""
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1):
        super().__init__(data_for_visualization, dpi, watermarking_step)
    
    def draw_pattern_fft(self, 
                      title: str = "Tree-Ring FFT with Watermark Area",
                      cmap: str = "viridis",
                      use_color_bar: bool = True,
                      vmin: float | None = None,
                      vmax: float | None = None,
                      ax: Axes | None = None,
                      **kwargs) -> Axes:
        """
        Draw FFT visualization with original watermark pattern, with all 0 background.
        
        Parameters:
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        orig_latent = self.data.orig_watermarked_latents[0, self.data.w_channel].cpu()
        watermarking_mask = self.data.watermarking_mask[0, self.data.w_channel].cpu()
        
        fft_data = torch.from_numpy(self._fft_transform(orig_latent))
        fft_vis = torch.zeros_like(fft_data)
        fft_vis[watermarking_mask] = fft_data[watermarking_mask]
        
        im = ax.imshow(np.abs(fft_vis.cpu().numpy()), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if title != "":
            ax.set_title(title)
        if use_color_bar:
            ax.figure.colorbar(im, ax=ax)
        ax.axis('off')
        
        return ax
        
    def draw_inverted_pattern_fft(self, 
                      step: int | None = None,
                      title: str = "Tree-Ring FFT with Inverted Watermark Area",
                      cmap: str = "viridis",
                      use_color_bar: bool = True,
                      vmin: float | None = None,
                      vmax: float | None = None,
                      ax: Axes | None = None,
                      **kwargs) -> Axes:
        """
        Draw FFT visualization with inverted pattern, with all 0 background.
        
        Parameters:
            step (int | None): The timestep of the inverted latents. If None, the last timestep is used.
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        if step is None:
            inverted_latent = self.data.reversed_latents[self.watermarking_step][0, self.data.w_channel]
        else:
            inverted_latent = self.data.reversed_latents[step][0, self.data.w_channel]
        
        watermarking_mask = self.data.watermarking_mask[0, self.data.w_channel].cpu()
        
        fft_data = torch.from_numpy(self._fft_transform(inverted_latent))
        fft_vis = torch.zeros_like(fft_data).to(fft_data.device)
        fft_vis[watermarking_mask] = fft_data[watermarking_mask]
        
        im = ax.imshow(np.abs(fft_vis.cpu().numpy()), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if title != "":
            ax.set_title(title)
        if use_color_bar:
            ax.figure.colorbar(im, ax=ax)
        ax.axis('off')
        
        return ax
        
