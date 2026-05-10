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


from PIL import Image
from typing import List
import cv2
import numpy as np
import tempfile
import os
import random
import subprocess
import shutil

class VideoEditor:
    """Base class for video editors."""
    
    def __init__(self):
        pass
        
    def edit(self, frames: List[Image.Image], prompt: str = None) -> List[Image.Image]:
        pass
    
class MPEG4Compression(VideoEditor):
    """MPEG-4 compression video editor."""
    
    def __init__(self, fps: float = 24.0):
        """Initialize the MPEG-4 compression video editor.

        Args:
            fps (float, optional): The frames per second of the compressed video. Defaults to 24.0.
        """
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = fps
        
    def edit(self, frames: List[Image.Image], prompt: str = None) -> List[Image.Image]:
        """Compress the video using MPEG-4 compression.

        Args:
            frames (List[Image.Image]): The frames to compress.
            prompt (str, optional): The prompt for video editing. Defaults to None.

        Returns:
            List[Image.Image]: The compressed frames.
        """
        # Transform PIL images to numpy arrays and convert to BGR format
        frame_arrays = [cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR) for f in frames]

        # Get frame size
        height, width, _ = frame_arrays[0].shape

        # Use a temporary file to save the mp4 video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name

        # Write mp4 video (MPEG-4 encoding)
        out = cv2.VideoWriter(video_path, self.fourcc, self.fps, (width, height))

        for frame in frame_arrays:
            out.write(frame)
        out.release()

        # Read mp4 video and decode back to frames
        cap = cv2.VideoCapture(video_path)
        compressed_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Transform back to PIL.Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            compressed_frames.append(pil_img)
        cap.release()

        # Clean up temporary file
        os.remove(video_path)

        return compressed_frames
        
    
class FrameAverage(VideoEditor):
    """Frame average video editor."""
    
    def __init__(self, n_frames: int = 3):
        """Initialize the frame average video editor.

        Args:
            n_frames (int, optional): The number of frames to average. Defaults to 3.
        """
        self.n_frames = n_frames
        
    def edit(self, frames: List[Image.Image], prompt: str = None) -> List[Image.Image]:
        """Average frames in a window of size n_frames.

        Args:
            frames (List[Image.Image]): The frames to average.
            prompt (str, optional): The prompt for video editing. Defaults to None.

        Returns:
            List[Image.Image]: The averaged frames.
        """
        n = self.n_frames
        num_frames = len(frames)
        # Transform all PIL images to numpy arrays and convert to float32 for averaging
        arrays = [np.asarray(img).astype(np.float32) for img in frames]
        result = []
        for i in range(num_frames):
            # Determine current window
            start = max(0, i - n // 2)
            end = min(num_frames, start + n)
            # If the end exceeds, move the window to the left
            start = max(0, end - n)
            window = arrays[start:end]
            avg = np.mean(window, axis=0).astype(np.uint8)
            result.append(Image.fromarray(avg))
        return result


class FrameRateAdapter(VideoEditor):
    """Resample videos to a target frame rate using linear interpolation."""

    def __init__(self, source_fps: float = 30.0, target_fps: float = 24.0):
        """Initialize the frame rate adapter.

        Args:
            source_fps (float, optional): Original frames per second. Defaults to 30.0.
            target_fps (float, optional): Desired frames per second. Defaults to 24.0.
        """
        if source_fps <= 0 or target_fps <= 0:
            raise ValueError("source_fps and target_fps must be positive numbers")
        self.source_fps = source_fps
        self.target_fps = target_fps

    def edit(self, frames: List[Image.Image], prompt: str = None) -> List[Image.Image]:
        """Resample frames to match the target frame rate while preserving duration."""
        if not frames or self.source_fps == self.target_fps:
            return [frame.copy() for frame in frames]

        arrays = [np.asarray(frame).astype(np.float32) for frame in frames]
        num_frames = len(arrays)
        if num_frames == 1:
            return [Image.fromarray(arrays[0].astype(np.uint8))]

        duration = (num_frames - 1) / self.source_fps
        if duration <= 0:
            return [Image.fromarray(arr.astype(np.uint8)) for arr in arrays]

        target_count = max(1, int(round(duration * self.target_fps)) + 1)
        indices = np.linspace(0, num_frames - 1, target_count)

        resampled_frames: List[Image.Image] = []
        for idx in indices:
            lower = int(np.floor(idx))
            upper = min(int(np.ceil(idx)), num_frames - 1)
            if lower == upper:
                interp = arrays[lower]
            else:
                alpha = idx - lower
                interp = (1 - alpha) * arrays[lower] + alpha * arrays[upper]
            resampled_frames.append(Image.fromarray(np.clip(interp, 0, 255).astype(np.uint8)))
        return resampled_frames


class FrameSwap(VideoEditor):
    """Frame swap video editor."""
    
    def __init__(self, p: float = 0.25):
        """Initialize the frame swap video editor.

        Args:
            p (float, optional): The probability of swapping neighbor frames. Defaults to 0.25.
        """
        self.p = p
        
    def edit(self, frames: List[Image.Image], prompt: str = None) -> List[Image.Image]:
        """Swap adjacent frames with probability p.

        Args:
            frames (List[Image.Image]): The frames to swap.
            prompt (str, optional): The prompt for video editing. Defaults to None.

        Returns:
            List[Image.Image]: The swapped frames.
        """
        for i, frame in enumerate(frames):
            if i == 0:
                continue
            if random.random() >= self.p:
                frames[i - 1], frames[i] = frames[i], frames[i - 1]
        return frames


class FrameInterpolationAttack(VideoEditor):
    """Insert interpolated frames to alter temporal sampling density."""

    def __init__(self, interpolated_frames: int = 1):
        """Initialize the interpolation attack editor.

        Args:
            interpolated_frames (int, optional): Number of synthetic frames added between consecutive original frames. Defaults to 1.
        """
        if interpolated_frames < 0:
            raise ValueError("interpolated_frames must be non-negative")
        self.interpolated_frames = interpolated_frames

    def edit(self, frames: List[Image.Image], prompt: str = None) -> List[Image.Image]:
        """Insert interpolated frames between originals using linear blending."""
        if not frames or self.interpolated_frames == 0:
            return [frame.copy() for frame in frames]
        if len(frames) == 1:
            return [frames[0].copy()]

        arrays = [np.asarray(frame).astype(np.float32) for frame in frames]
        result: List[Image.Image] = []
        last_index = len(frames) - 1
        for idx in range(last_index):
            start = arrays[idx]
            end = arrays[idx + 1]
            result.append(frames[idx].copy())
            for insert_idx in range(1, self.interpolated_frames + 1):
                alpha = insert_idx / (self.interpolated_frames + 1)
                interp = (1 - alpha) * start + alpha * end
                result.append(Image.fromarray(np.clip(interp, 0, 255).astype(np.uint8)))
        result.append(frames[-1].copy())
        return result
    