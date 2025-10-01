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


class VideoMarkVisualizer(BaseVisualizer):
    """VideoMark watermark visualization class.
    
    This visualizer handles watermark visualization for VideoShield algorithm,
    which extends Gaussian Shading to the video domain by adding frame dimensions.
    
    Key Members for VideoMarkVisualizer:
        - self.data.orig_watermarked_latents: [B, C, F, H, W]
        - self.data.reversed_latents: List[[B, C, F, H, W]]
    """
    
    def __init__(self, data_for_visualization: DataForVisualization, dpi: int = 300, watermarking_step: int = -1, is_video: bool = True):
        super().__init__(data_for_visualization, dpi, watermarking_step, is_video)
    
    def draw_watermarked_video_frames(self,
                                    num_frames: int = 4,
                                    title: str = "Watermarked Video Frames",
                                    ax: Axes | None = None) -> Axes:
        """Draw multiple frames from the watermarked video.
        
        This method displays a grid of video frames to show the temporal
        consistency of the watermarked video.
        
        Args:
            num_frames: Number of frames to display (default: 4)
            title: The title of the plot
            ax: The axes to plot on
            
        Returns:
            The plotted axes
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
            
            # Get the frame - keep it simple like the demo
            frame = video_frames[frame_idx]
            
            # Convert frame to displayable format - keep it simple and robust
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
