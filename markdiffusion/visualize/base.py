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


from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import torch
from PIL import Image
from visualize.data_for_visualization import DataForVisualization
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import Tuple
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from PIL import Image
from matplotlib.gridspec import GridSpecFromSubplotSpec

class BaseVisualizer(ABC):
    """Base class for watermark visualization data"""
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1, is_video: bool = False):
        """Initialize with common attributes"""
        self.data = data_for_visualization
        self.dpi = dpi
        self.watermarking_step = -1 # The step for inserting the watermark, defaults to -1 for the last step
        self.is_video = is_video  # Whether this is for T2V (video) or T2I (image) model
    
    def _fft_transform(self, latent: torch.Tensor) -> np.ndarray:
        """
        Apply FFT transform to the latent tensor of the watermarked image.
        """
        return fftshift(fft2(latent.cpu().numpy()))
    
    def _ifft_transform(self, fft_data: np.ndarray) -> np.ndarray:
        """
        Apply inverse FFT transform to the fft data.
        """
        return ifft2(ifftshift(fft_data))
    
    def _get_latent_data(self, latents: torch.Tensor, channel: Optional[int] = None, frame: Optional[int] = None) -> torch.Tensor:
        """
        Extract latent data with proper indexing for both T2I and T2V models.
        
        Parameters:
            latents: The latent tensor [B, C, H, W] for T2I or [B, C, F, H, W] for T2V
            channel: The channel index to extract
            frame: The frame index to extract (only for T2V models)
            
        Returns:
            The extracted latent tensor
        """
        if self.is_video:
            # T2V model: [B, C, F, H, W]
            if frame is not None:
                if channel is not None:
                    return latents[0, channel, frame]  # [H, W]
                else:
                    return latents[0, :, frame]  # [C, H, W]
            else:
                # If no frame specified, use the middle frame
                mid_frame = latents.shape[2] // 2
                if channel is not None:
                    return latents[0, channel, mid_frame]  # [H, W]
                else:
                    return latents[0, :, mid_frame]  # [C, H, W]
        else:
            # T2I model: [B, C, H, W]
            if channel is not None:
                return latents[0, channel]  # [H, W]
            else:
                return latents[0]  # [C, H, W]
    
    def draw_orig_latents(self, 
                          channel: Optional[int] = None,
                          frame: Optional[int] = None,
                          title: str = "Original Latents", 
                          cmap: str = "viridis", 
                          use_color_bar: bool = True,
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          ax: Optional[Axes] = None,
                          **kwargs) -> Axes:
        """
            Draw the original latents of the watermarked image.

            Parameters:
                channel (Optional[int]): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
                frame (Optional[int]): The frame index for T2V models. If None, uses middle frame for videos.
                title (str): The title of the plot.
                cmap (str): The colormap to use.
                use_color_bar (bool): Whether to display the colorbar.
                ax (Axes): The axes to plot on.
                
            Returns:
                Axes: The plotted axes.
        """
        if channel is not None:
            # Single channel visualization
            latent_data = self._get_latent_data(self.data.orig_watermarked_latents, channel, frame).cpu().numpy()
            im = ax.imshow(latent_data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
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
                latent_data = self._get_latent_data(self.data.orig_watermarked_latents, i, frame).cpu().numpy()
                im = sub_ax.imshow(latent_data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
                
        return ax
        
    def draw_orig_latents_fft(self, 
                          channel: Optional[int] = None,
                          frame: Optional[int] = None,
                          title: str = "Original Latents in Fourier Domain", 
                          cmap: str = "viridis", 
                          use_color_bar: bool = True,
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          ax: Optional[Axes] = None,
                          **kwargs) -> Axes:
        """
            Draw the original latents of the watermarked image in the Fourier domain.
            
            Parameters:
                channel (Optional[int]): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
                frame (Optional[int]): The frame index for T2V models. If None, uses middle frame for videos.
                title (str): The title of the plot.
                cmap (str): The colormap to use.
                use_color_bar (bool): Whether to display the colorbar.
                ax (Axes): The axes to plot on.

            Returns:
                Axes: The plotted axes.
        """
        if channel is not None:
            # Single channel visualization
            latent_data = self._get_latent_data(self.data.orig_watermarked_latents, channel, frame)
            fft_data = self._fft_transform(latent_data)
            
            im = ax.imshow(np.abs(fft_data), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
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
                
                # Draw the FFT of latent channel
                latent_data = self._get_latent_data(self.data.orig_watermarked_latents, i, frame)
                fft_data = self._fft_transform(latent_data)
                im = sub_ax.imshow(np.abs(fft_data), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
            
        return ax
    
    def draw_inverted_latents(self, 
                              channel: Optional[int] = None,
                              frame: Optional[int] = None,
                              step: Optional[int] = None,
                              title: str = "Inverted Latents", 
                              cmap: str = "viridis", 
                              use_color_bar: bool = True,
                              vmin: Optional[float] = None,
                              vmax: Optional[float] = None,
                              ax: Optional[Axes] = None,
                              **kwargs) -> Axes:
        """
            Draw the inverted latents of the watermarked image.
            
            Parameters:
                channel (Optional[int]): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
                frame (Optional[int]): The frame index for T2V models. If None, uses middle frame for videos.
                step (Optional[int]): The timestep of the inverted latents. If None, the last timestep is used.
                title (str): The title of the plot.
                cmap (str): The colormap to use.
                use_color_bar (bool): Whether to display the colorbar.
                ax (Axes): The axes to plot on.
                
            Returns:
                Axes: The plotted axes.
        """
        if channel is not None:
            # Single channel visualization
            # Get inverted latents data
            if step is None:
                reversed_latents = self.data.reversed_latents[self.watermarking_step]
            else:
                reversed_latents = self.data.reversed_latents[step]
            
            latent_data = self._get_latent_data(reversed_latents, channel, frame).cpu().numpy()
            im = ax.imshow(latent_data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
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
                
                # Get inverted latents data
                if step is None:
                    reversed_latents = self.data.reversed_latents[self.watermarking_step]
                else:
                    reversed_latents = self.data.reversed_latents[step]
                
                latent_data = self._get_latent_data(reversed_latents, i, frame).cpu().numpy()
                
                # Draw the latent channel
                im = sub_ax.imshow(latent_data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
            
        return ax
    
    def draw_inverted_latents_fft(self, 
                                 channel: Optional[int] = None,
                                 frame: Optional[int] = None,
                                 step: int = -1, 
                                 title: str = "Inverted Latents in Fourier Domain", 
                                 cmap: str = "viridis", 
                                 use_color_bar: bool = True,
                                 vmin: Optional[float] = None,
                                 vmax: Optional[float] = None,
                                 ax: Optional[Axes] = None,
                                 **kwargs) -> Axes:
        """
            Draw the inverted latents of the watermarked image in the Fourier domain.
            
            Parameters:
                channel (Optional[int]): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
                frame (Optional[int]): The frame index for T2V models. If None, uses middle frame for videos.
                step (Optional[int]): The timestep of the inverted latents. If None, the last timestep is used.
                title (str): The title of the plot.
                cmap (str): The colormap to use.
                use_color_bar (bool): Whether to display the colorbar.
                ax (Axes): The axes to plot on.
                
            Returns:
                Axes: The plotted axes.
        """
        if channel is not None:
            # Single channel visualization
            reversed_latents = self.data.reversed_latents[step]
            latent_data = self._get_latent_data(reversed_latents, channel, frame)
            fft_data = self._fft_transform(latent_data)
            
            im = ax.imshow(np.abs(fft_data), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
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
                
                # Draw the FFT of inverted latent channel
                reversed_latents = self.data.reversed_latents[step]
                latent_data = self._get_latent_data(reversed_latents, i, frame)
                fft_data = self._fft_transform(latent_data)
                im = sub_ax.imshow(np.abs(fft_data), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
            
        return ax
    
    def draw_diff_latents_fft(self, 
                              channel: Optional[int] = None,
                              frame: Optional[int] = None,
                              title: str = "Difference between Original and Inverted Latents in Fourier Domain", 
                              cmap: str = "coolwarm", 
                              use_color_bar: bool = True,
                              vmin: Optional[float] = None,
                              vmax: Optional[float] = None,
                              ax: Optional[Axes] = None,
                              **kwargs) -> Axes:
        """
            Draw the difference between the original and inverted initial latents of the watermarked image in the Fourier domain.
            
            Parameters:
                channel (Optional[int]): The channel of the latent tensor to visualize. If None, all 4 channels are shown.
                frame (Optional[int]): The frame index for T2V models. If None, uses middle frame for videos.
                title (str): The title of the plot.
                cmap (str): The colormap to use.
                use_color_bar (bool): Whether to display the colorbar.
                ax (Axes): The axes to plot on.
                
            Returns:
                Axes: The plotted axes.
        """
        if channel is not None:
            # Single channel visualization
            # Get original and inverted latents
            orig_data = self._get_latent_data(self.data.orig_watermarked_latents, channel, frame).cpu().numpy()
            
            reversed_latents = self.data.reversed_latents[self.watermarking_step]
            inv_data = self._get_latent_data(reversed_latents, channel, frame).cpu().numpy()
                
            # Compute difference
            diff_data = orig_data - inv_data
            
            # Convert to tensor for FFT transform
            diff_tensor = torch.tensor(diff_data)
            fft_data = self._fft_transform(diff_tensor)
            
            im = ax.imshow(np.abs(fft_data), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
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
                
                # Get original and inverted latents
                orig_data = self._get_latent_data(self.data.orig_watermarked_latents, i, frame).cpu().numpy()
                
                reversed_latents = self.data.reversed_latents[self.watermarking_step]
                inv_data = self._get_latent_data(reversed_latents, i, frame).cpu().numpy()
                
                # Compute difference and FFT
                diff_data = orig_data - inv_data
                diff_tensor = torch.tensor(diff_data)
                fft_data = self._fft_transform(diff_tensor)
                
                # Draw the FFT of difference
                im = sub_ax.imshow(np.abs(fft_data), cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                sub_ax.set_title(f'Channel {i}', fontsize=8, pad=3)
                sub_ax.axis('off')
                # Add small colorbar for each subplot
                if use_color_bar:
                    cbar = ax.figure.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
            
        return ax

    def draw_watermarked_image(self,
                               title: str = "Watermarked Image",
                               num_frames: int = 4,
                               vmin: Optional[float] = None,
                               vmax: Optional[float] = None,
                               ax: Optional[Axes] = None,
                               **kwargs) -> Axes:
        """
        Draw the watermarked image or video frames.

        For images (is_video=False), displays a single image.
        For videos (is_video=True), displays a grid of video frames.

        Parameters:
            title (str): The title of the plot.
            num_frames (int): Number of frames to display for videos (default: 4).
            vmin (Optional[float]): Minimum value for colormap.
            vmax (Optional[float]): Maximum value for colormap.
            ax (Axes): The axes to plot on.

        Returns:
            Axes: The plotted axes.
        """
        if self.is_video:
            # Video visualization: display multiple frames
            return self._draw_video_frames(title=title, num_frames=num_frames, ax=ax, **kwargs)
        else:
            # Image visualization: display single image
            return self._draw_single_image(title=title, vmin=vmin, vmax=vmax, ax=ax, **kwargs)

    def _draw_single_image(self,
                          title: str = "Watermarked Image",
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          ax: Optional[Axes] = None,
                          **kwargs) -> Axes:
        """
        Draw a single watermarked image.

        Parameters:
            title (str): The title of the plot.
            vmin (Optional[float]): Minimum value for colormap.
            vmax (Optional[float]): Maximum value for colormap.
            ax (Axes): The axes to plot on.

        Returns:
            Axes: The plotted axes.
        """
        # Convert image data to numpy array
        if torch.is_tensor(self.data.image):
            # Handle tensor format (like in RI watermark)
            if self.data.image.dim() == 4:  # [B, C, H, W]
                image_array = self.data.image[0].permute(1, 2, 0).cpu().numpy()
            elif self.data.image.dim() == 3:  # [C, H, W]
                image_array = self.data.image.permute(1, 2, 0).cpu().numpy()
            else:
                image_array = self.data.image.cpu().numpy()

            # Normalize to 0-1 if needed
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
                
            # Normalize [-1, 1] range to [0, 1] for imshow
            if image_array.min() < 0:
                image_array = (image_array + 1.0) / 2.0

            # Normalize [-1, 1] range to [0, 1] for imshow
            if image_array.min() < 0:
                image_array = (image_array + 1.0) / 2.0

            # Clip to valid range
            image_array = np.clip(image_array, 0, 1)
        else:
            # Handle PIL Image format
            image_array = np.array(self.data.image)

        im = ax.imshow(image_array, vmin=vmin, vmax=vmax, **kwargs)
        if title != "":
            ax.set_title(title, fontsize=12)
        ax.axis('off')

        # Hidden colorbar for nice visualization
        cbar = ax.figure.colorbar(im, ax=ax, alpha=0.0)
        cbar.ax.set_visible(False)

        return ax

    def _draw_video_frames(self,
                          title: str = "Watermarked Video Frames",
                          num_frames: int = 4,
                          ax: Optional[Axes] = None,
                          **kwargs) -> Axes:
        """
        Draw multiple frames from the watermarked video.

        This method displays a grid of video frames to show the temporal
        consistency of the watermarked video.

        Parameters:
            title (str): The title of the plot.
            num_frames (int): Number of frames to display (default: 4).
            ax (Axes): The axes to plot on.

        Returns:
            Axes: The plotted axes.
        """
        if not hasattr(self.data, 'video_frames') or self.data.video_frames is None:
            raise ValueError("No video frames available for visualization. Please ensure video_frames is provided in data_for_visualization.")

        video_frames = self.data.video_frames
        total_frames = len(video_frames)

        # Limit num_frames to available frames
        num_frames = min(num_frames, total_frames)

        # Calculate which frames to show (evenly distributed)
        if num_frames == 1:
            frame_indices = [total_frames // 2]  # Middle frame
        else:
            frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

        # Calculate grid layout
        rows = int(np.ceil(np.sqrt(num_frames)))
        cols = int(np.ceil(num_frames / rows))

        # Clear the axis and set title
        ax.clear()
        if title != "":
            ax.set_title(title, pad=20, fontsize=12)
        ax.axis('off')

        # Use gridspec for better control
        gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=ax.get_subplotspec(),
                                     wspace=0.1, hspace=0.4)

        # Create subplots for each frame
        for i, frame_idx in enumerate(frame_indices):
            row_idx = i // cols
            col_idx = i % cols

            # Create subplot using gridspec
            sub_ax = ax.figure.add_subplot(gs[row_idx, col_idx])

            # Get the frame
            frame = video_frames[frame_idx]

            # Convert frame to displayable format
            try:
                # First, convert tensor to numpy if needed
                if hasattr(frame, 'cpu'):  # PyTorch tensor
                    frame = frame.cpu().numpy()
                elif hasattr(frame, 'numpy'):  # Other tensor types
                    frame = frame.numpy()
                elif hasattr(frame, 'convert'):  # PIL Image
                    frame = np.array(frame)

                # Handle channels-first format (C, H, W) -> (H, W, C) for numpy arrays
                if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                    if frame.shape[0] in [1, 3, 4]:  # Channels first
                        frame = np.transpose(frame, (1, 2, 0))

                # Ensure proper data type for matplotlib
                if isinstance(frame, np.ndarray):
                    if frame.dtype == np.float64:
                        frame = frame.astype(np.float32)
                    elif frame.dtype not in [np.uint8, np.float32]:
                        # Convert to float32 and normalize if needed
                        frame = frame.astype(np.float32)
                        if frame.max() > 1.0:
                            frame = frame / 255.0
                            
                    # Normalize [-1, 1] range to [0, 1] for imshow
                    if frame.min() < 0:
                        frame = (frame + 1.0) / 2.0

                    # Clip to valid range [0, 1]
                    frame = np.clip(frame, 0, 1)

                    # Normalize [-1, 1] range to [0, 1] for imshow
                    if frame.min() < 0:
                        frame = (frame + 1.0) / 2.0

                    # Clip to valid range [0, 1]
                    frame = np.clip(frame, 0, 1)

                im = sub_ax.imshow(frame)

            except Exception as e:
                print(f"Error displaying frame {frame_idx}: {e}")

            sub_ax.set_title(f'Frame {frame_idx}', fontsize=10, pad=5)
            sub_ax.axis('off')

        # Hide unused subplots
        for i in range(num_frames, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            if row_idx < rows and col_idx < cols:
                empty_ax = ax.figure.add_subplot(gs[row_idx, col_idx])
                empty_ax.axis('off')

        return ax

    def visualize(self, 
                  rows: int,
                  cols: int,
                  methods: List[str],
                  figsize: Optional[Tuple[int, int]] = None,
                  method_kwargs: Optional[List[Dict[str, Any]]] = None,
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Comprehensive visualization of watermark analysis.
        
        Parameters:
            rows (int): The number of rows of the subplots.
            cols (int): The number of columns of the subplots.
            methods (List[str]): List of methods to call.
            method_kwargs (Optional[List[Dict[str, Any]]]): List of keyword arguments for each method.
            figsize (Tuple[int, int]): The size of the figure.
            save_path (Optional[str]): The path to save the figure.
            
        Returns:
            plt.Figure: The matplotlib figure object.
        """
        # Check if the rows and cols are compatible with the number of methods
        if len(methods) != rows * cols:
            raise ValueError(f"The number of methods ({len(methods)}) is not compatible with the layout ({rows}x{cols})")
        
        # Initialize the figure size if not provided
        if figsize is None:
            figsize = (cols * 5, rows * 5) 
        
        # Create figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Ensure axes is always a 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
            
        if method_kwargs is None:
            method_kwargs = [{} for _ in methods]
        
        # Plot each method
        for i, method_name in enumerate(methods):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            try:
                method = getattr(self, method_name)
            except AttributeError:
                raise ValueError(f"Method '{method_name}' not found in {self.__class__.__name__}")
            
            try:
                # print(method_kwargs[i])
                method(ax=ax, **method_kwargs[i])
            except TypeError:
                raise ValueError(f"Method '{method_name}' does not accept the given arguments: {method_kwargs[i]}")
        
        # if the number of methods is less than the number of axes, hide the unused axes
        for i in range(len(methods), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.0) 
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            
        return fig