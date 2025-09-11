import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization

class ROBINVisualizer(BaseVisualizer):
    """ROBIN watermark visualization class"""
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1):
        super().__init__(data_for_visualization, dpi, watermarking_step)
        # ROBIN uses a specific watermarking step
        if hasattr(self.data, 'watermarking_step'):
            self.watermarking_step = self.data.watermarking_step
        else:
            raise ValueError("watermarking_step is required for ROBIN visualization")
    
    def draw_pattern_fft(self, 
                      title: str = None,
                      cmap: str = "viridis",
                      use_color_bar: bool = True,
                      vmin: float | None = None,
                      vmax: float | None = None,
                      ax: Axes | None = None,
                      **kwargs) -> Axes:
        """
        Draw FFT visualization with original watermark pattern, with all 0 background.
        
        Parameters:
            title (str): The title of the plot. If None, includes watermarking step info.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        # Use custom title with watermarking step if not provided
        if title is None:
            title = f"ROBIN FFT with Watermark Area (Step {self.watermarking_step})"
            
        orig_latent = self.data.optimized_watermark[0, self.data.w_channel].cpu()
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
                      title: str = None,
                      cmap: str = "viridis",
                      use_color_bar: bool = True,
                      vmin: float | None = None,
                      vmax: float | None = None,
                      ax: Axes | None = None,
                      **kwargs) -> Axes:
        """
        Draw FFT visualization with inverted pattern, with all 0 background.
        
        Parameters:
            step (int | None): The timestep of the inverted latents. If None, uses ROBIN's specific step.
            title (str): The title of the plot. If None, includes watermarking step info.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        # For ROBIN, we need to use the specific watermarking step
        if step is None:
            # Calculate the actual step index for ROBIN
            # ROBIN uses: num_steps_to_use - 1 - self.config.watermarking_step
            num_steps = len(self.data.reversed_latents)
            actual_step = num_steps - 1 - self.watermarking_step
            inverted_latent = self.data.reversed_latents[actual_step][0, self.data.w_channel]
        else:
            inverted_latent = self.data.reversed_latents[step][0, self.data.w_channel]
        
        # Use custom title with watermarking step if not provided
        if title is None:
            title = f"ROBIN FFT with Inverted Watermark Area (Step {self.watermarking_step})"
        
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
        
    def draw_optimized_watermark(self,
                                title: str = None,
                                cmap: str = "viridis",
                                use_color_bar: bool = True,
                                vmin: float | None = None,
                                vmax: float | None = None,
                                ax: Axes | None = None,
                                **kwargs) -> Axes:
        """
        Draw the optimized watermark pattern (ROBIN-specific).
        
        Parameters:
            title (str): The title of the plot. If None, includes watermarking step info.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        # Use custom title with watermarking step if not provided
        if title is None:
            title = f"ROBIN Optimized Watermark (Step {self.watermarking_step})"
            
        optimized_watermark = self.data.optimized_watermark[0, self.data.w_channel].cpu()
        
        im = ax.imshow(np.abs(optimized_watermark.numpy()), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if title != "":
            ax.set_title(title)
        if use_color_bar:
            ax.figure.colorbar(im, ax=ax)
        ax.axis('off')
        
        return ax