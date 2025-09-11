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
import numpy as np
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization

class RingIDVisualizer(BaseVisualizer):
    """RingID watermark visualization class"""
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1):
        super().__init__(data_for_visualization, dpi, watermarking_step)
    
    def draw_ring_pattern_fft(self,
                          title: str = "Ring Watermark Pattern (FFT)",
                          cmap: str = "viridis",
                          use_color_bar: bool = True,
                          vmin: float | None = None,
                          vmax: float | None = None,
                          ax: Axes | None = None,
                          **kwargs) -> Axes:
        """
        Draw the ring watermark pattern in the Fourier Domain.(background all zeros)
        
        Parameters:
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        watermarked_latents_fft = torch.from_numpy(self._fft_transform(self.data.latents))
        background = torch.zeros_like(watermarked_latents_fft)
        watermark_channel = self.data.ring_watermark_channel
        pattern = self.data.pattern.cpu()
        mask = self.data.mask.cpu()
        for channel, channel_mask in zip(watermark_channel, mask):
            watermarked_latents_fft[:, channel] = background[:, channel] + pattern[:,
                                                                             channel] * channel_mask
        
        im = ax.imshow(np.abs(watermarked_latents_fft[0, watermark_channel][0].cpu().numpy()), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if title != "":
            ax.set_title(title)
        if use_color_bar:
            ax.figure.colorbar(im, ax=ax)
        ax.axis('off')
        
        return ax
    
    def draw_heter_pattern_fft(self,
                          title: str = "Heter Watermark Pattern (FFT)",
                          cmap: str = "viridis",
                          use_color_bar: bool = True,
                          vmin: float | None = None,
                          vmax: float | None = None,
                          ax: Axes | None = None,
                          **kwargs) -> Axes:
        """
        Draw the heter watermark pattern in the Fourier Domain.(background all zeros)
        
        Parameters:
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        watermarked_latents_fft = torch.from_numpy(self._fft_transform(self.data.latents))
        background = torch.zeros_like(watermarked_latents_fft)
        watermark_channel = self.data.heter_watermark_channel
        pattern = self.data.pattern.cpu()
        mask = self.data.mask.cpu()
        for channel, channel_mask in zip(watermark_channel, mask):
            watermarked_latents_fft[:, channel] = background[:, channel] + pattern[:,
                                                                             channel] * channel_mask
        
        im = ax.imshow(np.abs(watermarked_latents_fft[0, self.data.heter_watermark_channel][0].cpu().numpy()), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if title != "":
            ax.set_title(title)
        if use_color_bar:
            ax.figure.colorbar(im, ax=ax)
        ax.axis('off')
        
        return ax
    
    def draw_inverted_ring_pattern_fft(self,
                          title: str = "Inverted Ring Watermark Pattern (FFT)",
                          cmap: str = "viridis",
                          use_color_bar: bool = True,
                          vmin: float | None = None,
                          vmax: float | None = None,
                          ax: Axes | None = None,
                          **kwargs) -> Axes:
        """
        Extract and visualize watermark pattern from reversed_latents[-1] using FFT.
        
        Parameters:
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            ax (Axes): The axes to plot on.
            
        Returns:
            Axes: The plotted axes.
        """
        # Extract the last reversed latent
        reversed_latent = self.data.reversed_latents[-1]
        
        # Apply FFT transform
        reversed_latent_fft = torch.from_numpy(self._fft_transform(reversed_latent))
        
        # Extract watermark pattern from ring watermark channel
        watermark_channel = self.data.ring_watermark_channel
        pattern = self.data.pattern.cpu()
        mask = self.data.mask.cpu()
        
        # Create extracted pattern
        extracted_pattern = torch.zeros_like(reversed_latent_fft)
        for channel, channel_mask in zip(watermark_channel, mask):
            extracted_pattern[:, channel] = reversed_latent_fft[:, channel] * channel_mask
        
        # Visualize the extracted pattern
        im = ax.imshow(np.abs(extracted_pattern[0, watermark_channel][0].cpu().numpy()), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if title != "":
            ax.set_title(title)
        if use_color_bar:
            ax.figure.colorbar(im, ax=ax)
        ax.axis('off')
        
        return ax
