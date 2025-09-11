import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import math
from visualize.base import BaseVisualizer
from visualize.data_for_visualization import DataForVisualization

class SEALVisualizer(BaseVisualizer):
    """SEAL watermark visualization class"""
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1):
        super().__init__(data_for_visualization, dpi, watermarking_step)
        
    def draw_embedding_distributions(self,
                                    title: str = "Embedding Distributions",
                                    ax: Axes | None = None,
                                    show_legend: bool = True,
                                    show_label: bool = True,
                                    show_axis: bool = True) -> Axes:
        """
        Draw histogram of embedding distributions comparison(original_embedding vs inspected_embedding).
        
        Parameters:
            title (str): The title of the plot.
            ax (plt.Axes): The axes to plot on.
            show_legend (bool): Whether to show the legend. Default: True.
            show_label (bool): Whether to show axis labels. Default: True.
            show_axis (bool): Whether to show axis ticks and labels. Default: True.
            
        Returns:
            Axes: The plotted axes.
        """
        original_embedding = self.data.original_embedding 
        inspected_embedding = self.data.inspected_embedding 
        
        # Convert to numpy arrays and flatten if necessary
        original_data = original_embedding.cpu().numpy().flatten()
        inspected_data = inspected_embedding.cpu().numpy().flatten()
        
        # Create overlapping histograms with transparency
        ax.hist(original_data, bins=50, alpha=0.7, color='blue', 
                label='Original Embedding', density=True, edgecolor='darkblue', linewidth=0.5)
        ax.hist(inspected_data, bins=50, alpha=0.7, color='red', 
                label='Inspected Embedding', density=True, edgecolor='darkred', linewidth=0.5)
        
        # Set labels and title
        if title != "":
            ax.set_title(title, fontsize=12)
        if show_label:
            ax.set_xlabel('Embedding Values')
            ax.set_ylabel('Density')
        if show_legend:
            ax.legend()
        if not show_axis:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        
        # Create a hidden colorbar for nice visualization
        im = ax.scatter([], [], c=[], cmap='viridis')  
        cbar = ax.figure.colorbar(im, ax=ax, alpha=0.0)  
        cbar.ax.set_visible(False)  
        
        return ax
        
        
    def draw_patch_diff(self,
                        title: str = "Patch Difference",
                        cmap: str = 'RdBu',
                        use_color_bar: bool = True,
                        vmin: float | None = None,
                        vmax: float | None = None,
                        show_number: bool = False,
                        ax: Axes | None = None,
                        **kwargs) -> Axes:
        """
        Draw the difference between the reference_noise and reversed_latents in patch.
        
        Parameters:
            title (str): The title of the plot.
            cmap (str): The colormap to use.
            use_color_bar (bool): Whether to display the colorbar.
            vmin (float | None): Minimum value for colormap normalization.
            vmax (float | None): Maximum value for colormap normalization.
            show_number (bool): Whether to display numerical values on each patch. Default: False.
            ax (plt.Axes): The axes to plot on.
            
        Returns:
            plt.axes.Axes: The plotted axes.
        """
        reversed_latent = self.data.reversed_latents[self.watermarking_step] # shape: [1, 4, 64, 64]
        reference_noise = self.data.reference_noise # shape: [1, 4, 64, 64]
        k = self.data.k_value
        
        patch_per_side_h = int(math.ceil(math.sqrt(k)))
        patch_per_side_w = int(math.ceil(k / patch_per_side_h))
        patch_height = 64 // patch_per_side_h
        patch_width = 64 // patch_per_side_w
        diff_map = torch.zeros((patch_per_side_h, patch_per_side_w))

        patch_count = 0  # Initialize patch counter
        for i in range(patch_per_side_h):
            for j in range(patch_per_side_w):
                if patch_count >= k:
                    break
                y_start = i * patch_height
                x_start = j * patch_width
                y_end = min(y_start + patch_height, 64)
                x_end = min(x_start + patch_width, 64)
                patch1 = reversed_latent[:, :, y_start:y_end, x_start:x_end]
                patch2 = reference_noise[:, :, y_start:y_end, x_start:x_end]
                l2_val = torch.norm(patch1 - patch2).item()
                diff_map[i, j] = l2_val
                patch_count += 1  # Increment patch counter
        
        im = ax.imshow(diff_map.cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if title != "":
            ax.set_title(title)
        if use_color_bar:
            ax.figure.colorbar(im, ax=ax)
        if show_number:
            # Calculate appropriate font size based on patch size
            patch_size = min(patch_per_side_h, patch_per_side_w)
            if patch_size >= 8:
                fontsize = 8
                format_str = '{:.2f}'
            elif patch_size >= 4:
                fontsize = 6
                format_str = '{:.1f}'
            else:
                fontsize = 4
                format_str = '{:.0f}'
            fontsize = 4
            format_str = '{:.0f}'
            for i in range(patch_per_side_h):
                for j in range(patch_per_side_w):
                    if i * patch_per_side_w + j < k:  # Only show numbers for valid patches
                        value = diff_map[i, j].item()
                        ax.text(j, i, format_str.format(value), 
                               ha='center', va='center', color='white', 
                               fontsize=fontsize, fontweight='bold')
        ax.axis('off')
        
        return ax
