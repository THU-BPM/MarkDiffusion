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
from matplotlib.gridspec import GridSpec
import numpy as np
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization

class PRCVisualizer(BaseVisualizer):
    """PRC (Pseudorandom Codes) watermark visualizer"""
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1):
        """
        Initialize PRC visualizer
        
        Args:
            data_for_visualization: DataForVisualization object containing visualization data
            dpi: DPI for visualization (default: 300)
            watermarking_step: The step for inserting the watermark (default: -1)
        """
        super().__init__(data_for_visualization, dpi, watermarking_step)
        
        # Pre-detach all tensors while maintaining device compatibility
        if hasattr(self.data, 'watermarked_latents') and self.data.watermarked_latents is not None:
            self.data.watermarked_latents = self.data.watermarked_latents.detach()
        if hasattr(self.data, 'orig_latents') and self.data.orig_latents is not None:
            self.data.orig_latents = self.data.orig_latents.detach()
        if hasattr(self.data, 'inverted_latents') and self.data.inverted_latents is not None:
            self.data.inverted_latents = self.data.inverted_latents.detach()
        if hasattr(self.data, 'prc_codeword') and self.data.prc_codeword is not None:
            self.data.prc_codeword = self.data.prc_codeword.detach()
        if hasattr(self.data, 'generator_matrix') and self.data.generator_matrix is not None:
            self.data.generator_matrix = self.data.generator_matrix.detach()
    
    def draw_generator_matrix(self,
                             title: str = "Generator Matrix G",
                             cmap: str = "Blues",
                             use_color_bar: bool = True,
                             max_display_size: int = 50,
                             ax: Axes | None = None,
                             **kwargs) -> Axes:
        """
        Draw the generator matrix visualization
        
        Parameters:
            title (str): The title of the plot
            cmap (str): The colormap to use
            use_color_bar (bool): Whether to display the colorbar
            max_display_size (int): Maximum size to display (for large matrices)
            ax (Axes): The axes to plot on
            
        Returns:
            Axes: The plotted axes
        """
        if hasattr(self.data, 'generator_matrix') and self.data.generator_matrix is not None:
            gen_matrix = self.data.generator_matrix.cpu().numpy()
            
            # Show a sample of the matrix if it's too large
            if gen_matrix.shape[0] > max_display_size or gen_matrix.shape[1] > max_display_size:
                sample_size = min(max_display_size, min(gen_matrix.shape))
                matrix_sample = gen_matrix[:sample_size, :sample_size]
                title += f" (Sample {sample_size}x{sample_size})"
            else:
                matrix_sample = gen_matrix
            
            im = ax.imshow(matrix_sample, cmap=cmap, aspect='auto', **kwargs)
            
            if use_color_bar:
                plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Generator Matrix\nNot Available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        return ax
    
    def draw_codeword(self,
                     title: str = "PRC Codeword",
                     cmap: str = "viridis",
                     use_color_bar: bool = True,
                     ax: Axes | None = None,
                     **kwargs) -> Axes:
        """
        Draw the PRC codeword visualization
        
        Parameters:
            title (str): The title of the plot
            cmap (str): The colormap to use
            use_color_bar (bool): Whether to display the colorbar
            ax (Axes): The axes to plot on
            
        Returns:
            Axes: The plotted axes
        """
        if hasattr(self.data, 'prc_codeword') and self.data.prc_codeword is not None:
            codeword = self.data.prc_codeword.cpu().numpy()
            
            # If 1D, reshape for visualization
            if len(codeword.shape) == 1:
                # Create a reasonable 2D shape
                length = len(codeword)
                height = int(np.sqrt(length))
                width = length // height
                if height * width < length:
                    width += 1
                # Pad if necessary
                padded_codeword = np.zeros(height * width)
                padded_codeword[:length] = codeword
                codeword = padded_codeword.reshape(height, width)
            
            im = ax.imshow(codeword, cmap=cmap, aspect='equal', **kwargs)
            
            if use_color_bar:
                plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'PRC Codeword\nNot Available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        ax.set_title(title, fontsize=12)
        return ax
    
    def draw_recovered_codeword(self,
                              title: str = "Recovered Codeword (c̃)",
                              cmap: str = "viridis",
                              use_color_bar: bool = True,
                              vmin: float = -1.0,
                              vmax: float = 1.0,
                              ax: Axes | None = None,
                              **kwargs) -> Axes:
        """
        Draw the recovered codeword (c̃) from PRC detection
        
        This visualizes the recovered codeword from prc_detection.recovered_prc
        
        Parameters:
            title (str): The title of the plot
            cmap (str): The colormap to use
            use_color_bar (bool): Whether to display the colorbar
            vmin (float): Minimum value for colormap (-1.0)
            vmax (float): Maximum value for colormap (1.0)
            ax (Axes): The axes to plot on
            
        Returns:
            Axes: The plotted axes
        """
        if hasattr(self.data, 'recovered_prc') and self.data.recovered_prc is not None:
            recovered_codeword = self.data.recovered_prc.cpu().numpy().flatten()
            
            # Ensure it's the expected length
            if len(recovered_codeword) == 16384:
                # Reshape to 2D for visualization (128x128 = 16384)
                codeword_2d = recovered_codeword.reshape((128, 128))
                
                im = ax.imshow(codeword_2d, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', **kwargs)
                
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Codeword Value', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'Recovered Codeword\nUnexpected Length: {len(recovered_codeword)}\n(Expected: 16384)', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Recovered Codeword (c̃)\nNot Available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        return ax
                                  
    def draw_difference_map(self,
                           title: str = "Difference Map",
                           cmap: str = "hot",
                           use_color_bar: bool = True,
                           channel: int = 0,
                           ax: Axes | None = None,
                           **kwargs) -> Axes:
        """
        Draw difference map between watermarked and inverted latents
        
        Parameters:
            title (str): The title of the plot
            cmap (str): The colormap to use
            use_color_bar (bool): Whether to display the colorbar
            channel (int): The channel to visualize
            ax (Axes): The axes to plot on
            
        Returns:
            Axes: The plotted axes
        """
        if (hasattr(self.data, 'watermarked_latents') and self.data.watermarked_latents is not None and
            hasattr(self.data, 'inverted_latents') and self.data.inverted_latents is not None):
            
            wm_latents = self._get_latent_data(self.data.watermarked_latents, channel=channel).cpu().numpy()
            inv_latents = self._get_latent_data(self.data.inverted_latents, channel=channel).cpu().numpy()
            
            diff_map = np.abs(wm_latents - inv_latents)
            im = ax.imshow(diff_map, cmap=cmap, aspect='equal', **kwargs)
            
            if use_color_bar:
                plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Difference Map\nNot Available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        return ax
