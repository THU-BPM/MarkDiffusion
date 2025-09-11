from typing import List
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import numpy as np
from tqdm import tqdm
import cv2
import os
import subprocess
from utils.media_utils import pil_to_torch

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def dino_transform_Image(n_px):
    """DINO transform for PIL Images."""
    return Compose([
        Resize(size=n_px, antialias=False),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class VideoQualityAnalyzer:
    """Video quality analyzer base class."""

    def __init__(self):
        pass

    def analyze(self, frames: List[Image.Image]):
        """Analyze video quality.
        
        Args:
            frames: List of PIL Image frames representing the video
            
        Returns:
            Quality score(s)
        """
        raise NotImplementedError("Subclasses must implement analyze method")


class SubjectConsistencyAnalyzer(VideoQualityAnalyzer):
    """Analyzer for evaluating subject consistency across video frames using DINO features.
    
    This analyzer measures how consistently the main subject appears across frames by:
    1. Extracting DINO features from each frame
    2. Computing cosine similarity between consecutive frames and with the first frame
    3. Averaging these similarities to get a consistency score
    """
    
    def __init__(self, model_path: str = 'dino_vitb16', device: str = "cuda"):
        """Initialize the SubjectConsistencyAnalyzer.
        
        Args:
            model_path: Path to the pre-trained DINO model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load DINO model
        self.model = self._load_dino_model('dino_vitb16')
        self.model.eval()
        self.model.to(self.device)
        
        # Image transform for DINO
        self.transform = dino_transform_Image(224)
        
    def _load_dino_model(self, model_path: str):
        """Load DINO ViT-B/16 model.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Loaded DINO model
        """
        load_dict = {'repo_or_dir':'facebookresearch/dino:main',
                    # 'source':'local',
                    'model': model_path}
        model = torch.hub.load(**load_dict)
        return model
    
    def analyze(self, frames: List[Image.Image]) -> float:
        """Analyze subject consistency across video frames.
        
        Args:
            frames: List of PIL Image frames representing the video
            
        Returns:
            Subject consistency score (higher is better, range [0, 1])
        """
        if len(frames) < 2:
            return 1.0  # Single frame is perfectly consistent with itself
        
        video_sim = 0.0
        frame_count = 0
        
        # Process frames and extract features
        with torch.no_grad():
            for i, frame in enumerate(frames):
                # Transform and prepare frame
                frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
                
                # Extract features
                features = self.model(frame_tensor)
                features = F.normalize(features, dim=-1, p=2)
                
                if i == 0:
                    # Store first frame features
                    first_frame_features = features
                else:
                    # Compute similarity with previous frame
                    sim_prev = max(0.0, F.cosine_similarity(prev_features, features).item())
                    
                    # Compute similarity with first frame
                    sim_first = max(0.0, F.cosine_similarity(first_frame_features, features).item())
                    
                    # Average the two similarities
                    frame_sim = (sim_prev + sim_first) / 2.0
                    video_sim += frame_sim
                    frame_count += 1
                
                # Store current features as previous for next iteration
                prev_features = features
        
        # Return average similarity across all frame pairs
        if frame_count > 0:
            return video_sim / frame_count
        else:
            return 1.0


class MotionSmoothnessAnalyzer(VideoQualityAnalyzer):
    """Analyzer for evaluating motion smoothness in videos using AMT-S model.
    
    This analyzer measures motion smoothness by:
    1. Extracting frames at even indices from the video
    2. Using AMT-S model to interpolate between consecutive frames
    3. Comparing interpolated frames with actual frames to compute smoothness score
    
    The score represents how well the motion can be predicted/interpolated,
    with smoother motion resulting in higher scores.
    """
    
    def __init__(self, model_path: str = "model/amt/amt-s.pth", 
                 device: str = "cuda", niters: int = 1):
        """Initialize the MotionSmoothnessAnalyzer.
        
        Args:
            model_path: Path to the AMT-S model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
            niters: Number of interpolation iterations (default: 1)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.niters = niters
        
        # Initialize model parameters
        self._initialize_params()
        
        # Load AMT-S model
        self.model = self._load_amt_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
    def _initialize_params(self):
        """Initialize parameters for video processing."""
        if self.device.type == 'cuda':
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
        else:
            # Do not resize in cpu mode
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1
        
        # Time embedding for interpolation (t=0.5)
        self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)
        
    def _load_amt_model(self, model_path: str):
        """Load AMT-S model.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Loaded AMT-S model
        """
        # Import AMT-S model (note the hyphen in filename)
        import sys
        import importlib.util
        
        # Load the module with hyphen in filename
        spec = importlib.util.spec_from_file_location("amt_s", "/home/harry/workspace/MarkDiffusion/model/amt/networks/AMT-S.py")
        amt_s_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(amt_s_module)
        Model = amt_s_module.Model
        
        # Create model with default parameters
        model = Model(
            corr_radius=3,
            corr_lvls=4,
            num_flows=3
        )
        
        # Load checkpoint
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
        
        return model
    
    def _extract_frames(self, frames: List[Image.Image], start_from: int = 0) -> List[np.ndarray]:
        """Extract frames at even indices starting from start_from.
        
        Args:
            frames: List of PIL Image frames
            start_from: Starting index (default: 0)
            
        Returns:
            List of extracted frames as numpy arrays
        """
        extracted = []
        for i in range(start_from, len(frames), 2):
            # Convert PIL Image to numpy array
            frame_np = np.array(frames[i])
            extracted.append(frame_np)
        return extracted
    
    def _img2tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor.
        
        Args:
            img: Image as numpy array (H, W, C)
            
        Returns:
            Image tensor (1, C, H, W)
        """
        from model.amt.utils.utils import img2tensor
        return img2tensor(img)
    
    def _tensor2img(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image.
        
        Args:
            tensor: Image tensor (1, C, H, W)
            
        Returns:
            Image as numpy array (H, W, C)
        """
        from model.amt.utils.utils import tensor2img
        return tensor2img(tensor)
    
    def _check_dim_and_resize(self, tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Check dimensions and resize tensors if needed.
        
        Args:
            tensor_list: List of image tensors
            
        Returns:
            List of resized tensors
        """
        from model.amt.utils.utils import check_dim_and_resize
        return check_dim_and_resize(tensor_list)
    
    def _calculate_scale(self, h: int, w: int) -> float:
        """Calculate scaling factor based on available VRAM.
        
        Args:
            h: Height of the image
            w: Width of the image
            
        Returns:
            Scaling factor
        """
        scale = self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        return scale
    
    def _interpolate_frames(self, inputs: List[torch.Tensor], scale: float) -> List[torch.Tensor]:
        """Interpolate frames using AMT-S model.
        
        Args:
            inputs: List of input frame tensors
            scale: Scaling factor for processing
            
        Returns:
            List of interpolated frame tensors
        """
        from model.amt.utils.utils import InputPadder
        
        # Pad inputs
        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)
        
        # Perform interpolation for specified iterations
        for i in range(self.niters):
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs
        
        # Unpad outputs
        outputs = padder.unpad(*outputs)
        return outputs
    
    def _compute_frame_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute average absolute difference between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Average pixel difference
        """
        diff = cv2.absdiff(img1, img2)
        return np.mean(diff)
    
    def _compute_vfi_score(self, original_frames: List[np.ndarray], interpolated_frames: List[np.ndarray]) -> float:
        """Compute video frame interpolation score.
        
        Args:
            original_frames: Original video frames
            interpolated_frames: Interpolated frames
            
        Returns:
            VFI score (lower difference means better interpolation)
        """
        # Extract frames at odd indices for comparison
        ori_compare = self._extract_frames([Image.fromarray(f) for f in original_frames], start_from=1)
        interp_compare = self._extract_frames([Image.fromarray(f) for f in interpolated_frames], start_from=1)
        
        scores = []
        for ori, interp in zip(ori_compare, interp_compare):
            score = self._compute_frame_difference(ori, interp)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def analyze(self, frames: List[Image.Image]) -> float:
        """Analyze motion smoothness in video frames.
        
        Args:
            frames: List of PIL Image frames representing the video
            
        Returns:
            Motion smoothness score (higher is better, range [0, 1])
        """
        if len(frames) < 2:
            return 1.0  # Single frame has perfect smoothness
        
        # Convert PIL Images to numpy arrays
        np_frames = [np.array(frame) for frame in frames]
        
        # Extract frames at even indices
        frame_list = self._extract_frames(frames, start_from=0)
        
        # Convert to tensors
        inputs = [self._img2tensor(frame).to(self.device) for frame in frame_list]
        
        if len(inputs) <= 1:
            return 1.0  # Not enough frames for interpolation
        
        # Check dimensions and resize if needed
        inputs = self._check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        
        # Calculate scale based on available memory
        scale = self._calculate_scale(h, w)
        
        # Perform frame interpolation
        outputs = self._interpolate_frames(inputs, scale)
        
        # Convert outputs back to images
        output_images = [self._tensor2img(out) for out in outputs]
        
        # Compute VFI score
        vfi_score = self._compute_vfi_score(np_frames, output_images)
        
        # Normalize score to [0, 1] range (higher is better)
        # Original score is average pixel difference [0, 255], we normalize and invert
        normalized_score = (255.0 - vfi_score) / 255.0
        
        return normalized_score


class DynamicDegreeAnalyzer(VideoQualityAnalyzer):
    """Analyzer for evaluating dynamic degree (motion intensity) in videos using RAFT optical flow.
    
    This analyzer measures the amount and intensity of motion in videos by:
    1. Computing optical flow between consecutive frames using RAFT
    2. Calculating flow magnitude for each pixel
    3. Extracting top 5% highest flow magnitudes
    4. Determining if video has sufficient dynamic motion based on thresholds
    
    The score represents whether the video contains dynamic motion (1.0) or is mostly static (0.0).
    """
    
    def __init__(self, model_path: str = "model/raft/raft-things.pth",
                 device: str = "cuda", sample_fps: int = 8):
        """Initialize the DynamicDegreeAnalyzer.
        
        Args:
            model_path: Path to the RAFT model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
            sample_fps: Target FPS for frame sampling (default: 8)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_fps = sample_fps
        
        # Load RAFT model
        self.model = self._load_raft_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
    def _load_raft_model(self, model_path: str):
        """Load RAFT optical flow model.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Loaded RAFT model
        """
        from model.raft.core.raft import RAFT
        from easydict import EasyDict as edict
        
        # Configure RAFT arguments
        args = edict({
            "model": model_path,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False
        })
        
        # Create and load model
        model = RAFT(args)
        
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            # Remove 'module.' prefix if present (from DataParallel)
            new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(new_ckpt)
        
        return model
    
    def _extract_frames_for_flow(self, frames: List[Image.Image], target_fps: int = 8) -> List[torch.Tensor]:
        """Extract and prepare frames for optical flow computation.
        
        Args:
            frames: List of PIL Image frames
            target_fps: Target sampling rate (default: 8 fps)
            
        Returns:
            List of prepared frame tensors
        """
        # Estimate original FPS and calculate sampling interval
        # Assuming 30fps original video, adjust sampling to get ~8fps
        total_frames = len(frames)
        assumed_fps = 30  # Common video fps
        interval = max(1, round(assumed_fps / target_fps))
        
        # Sample frames at interval
        sampled_frames = []
        for i in range(0, total_frames, interval):
            frame = frames[i]
            # Convert PIL to numpy array
            frame_np = np.array(frame)
            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(frame_np.astype(np.uint8)).permute(2, 0, 1).float()
            frame_tensor = frame_tensor[None].to(self.device)
            sampled_frames.append(frame_tensor)
        
        return sampled_frames
    
    def _compute_flow_magnitude(self, flow: torch.Tensor) -> float:
        """Compute flow magnitude score from optical flow.
        
        Args:
            flow: Optical flow tensor (B, 2, H, W)
            
        Returns:
            Flow magnitude score
        """
        # Extract flow components
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
        u = flow_np[:, :, 0]
        v = flow_np[:, :, 1]
        
        # Compute flow magnitude
        magnitude = np.sqrt(np.square(u) + np.square(v))
        
        # Get top 5% highest magnitudes
        h, w = magnitude.shape
        magnitude_flat = magnitude.flatten()
        cut_index = int(h * w * 0.05)
        
        # Sort in descending order and take mean of top 5%
        top_magnitudes = np.sort(-magnitude_flat)[:cut_index]
        mean_magnitude = np.mean(np.abs(top_magnitudes))
        
        return mean_magnitude.item()
    
    def _determine_dynamic_threshold(self, frame_shape: tuple, num_frames: int) -> dict:
        """Determine thresholds for dynamic motion detection.
        
        Args:
            frame_shape: Shape of the frame tensor
            num_frames: Number of frames in the video
            
        Returns:
            Dictionary with threshold parameters
        """
        # Scale threshold based on image resolution
        scale = min(frame_shape[-2:])  # min of height and width
        magnitude_threshold = 6.0 * (scale / 256.0)
        
        # Scale count threshold based on number of frames
        count_threshold = round(4 * (num_frames / 16.0))
        
        return {
            "magnitude_threshold": magnitude_threshold,
            "count_threshold": count_threshold
        }
    
    def _check_dynamic_motion(self, flow_scores: List[float], thresholds: dict) -> bool:
        """Check if video has dynamic motion based on flow scores.
        
        Args:
            flow_scores: List of optical flow magnitude scores
            thresholds: Threshold parameters
            
        Returns:
            True if video has dynamic motion, False otherwise
        """
        magnitude_threshold = thresholds["magnitude_threshold"]
        count_threshold = thresholds["count_threshold"]
        
        # Count frames with significant motion
        motion_count = 0
        for score in flow_scores:
            if score > magnitude_threshold:
                motion_count += 1
            if motion_count >= count_threshold:
                return True
        
        return False
    
    def analyze(self, frames: List[Image.Image]) -> float:
        """Analyze dynamic degree (motion intensity) in video frames.
        
        Args:
            frames: List of PIL Image frames representing the video
            
        Returns:
            Dynamic degree score: 1.0 if video has dynamic motion, 0.0 if mostly static
        """
        if len(frames) < 2:
            return 0.0  # Cannot compute optical flow with less than 2 frames
        
        # Extract and prepare frames for optical flow
        prepared_frames = self._extract_frames_for_flow(frames, self.sample_fps)
        
        if len(prepared_frames) < 2:
            return 0.0
        
        # Determine thresholds based on video characteristics
        thresholds = self._determine_dynamic_threshold(
            prepared_frames[0].shape, 
            len(prepared_frames)
        )
        
        # Compute optical flow between consecutive frames
        flow_scores = []
        
        with torch.no_grad():
            for frame1, frame2 in zip(prepared_frames[:-1], prepared_frames[1:]):
                # Pad frames if necessary
                from model.raft.core.utils_core.utils import InputPadder
                padder = InputPadder(frame1.shape)
                frame1_padded, frame2_padded = padder.pad(frame1, frame2)
                
                # Compute optical flow
                _, flow_up = self.model(frame1_padded, frame2_padded, iters=20, test_mode=True)
                
                # Calculate flow magnitude score
                magnitude_score = self._compute_flow_magnitude(flow_up)
                flow_scores.append(magnitude_score)
        
        # Check if video has dynamic motion
        has_dynamic_motion = self._check_dynamic_motion(flow_scores, thresholds)
        
        # Return binary score: 1.0 for dynamic, 0.0 for static
        return 1.0 if has_dynamic_motion else 0.0


class BackgroundConsistencyAnalyzer(VideoQualityAnalyzer):
    """Analyzer for evaluating background consistency across video frames using CLIP features.
    
    This analyzer measures how consistently the background appears across frames by:
    1. Extracting CLIP visual features from each frame
    2. Computing cosine similarity between consecutive frames and with the first frame
    3. Averaging these similarities to get a consistency score
    
    Similar to SubjectConsistencyAnalyzer but focuses on overall visual consistency
    including background elements, making it suitable for detecting background stability.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """Initialize the BackgroundConsistencyAnalyzer.
        
        Args:
            model_name: CLIP model name (default: "ViT-B/32")
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        self.model, self.preprocess = self._load_clip_model(model_name)
        self.model.eval()
        self.model.to(self.device)
        
        # Image transform for CLIP (when processing tensor inputs)
        self.tensor_transform = self._get_clip_tensor_transform(224)
        
    def _load_clip_model(self, model_name: str):
        """Load CLIP model.
        
        Args:
            model_name: Name of the CLIP model to load
            
        Returns:
            Tuple of (model, preprocess_function)
        """
        import clip
        
        model, preprocess = clip.load(model_name, device=self.device)
        return model, preprocess
    
    def _get_clip_tensor_transform(self, n_px: int):
        """Get CLIP transform for tensor inputs.
        
        Args:
            n_px: Target image size
            
        Returns:
            Transform composition for tensor inputs
        """
        try:
            from torchvision.transforms import InterpolationMode
            BICUBIC = InterpolationMode.BICUBIC
        except ImportError:
            BICUBIC = Image.BICUBIC
            
        return Compose([
            Resize(n_px, interpolation=BICUBIC, antialias=False),
            CenterCrop(n_px),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def _prepare_images_for_clip(self, frames: List[Image.Image]) -> torch.Tensor:
        """Prepare PIL images for CLIP processing.
        
        Args:
            frames: List of PIL Image frames
            
        Returns:
            Batch tensor of preprocessed images
        """
        # Use CLIP's built-in preprocess for PIL images
        images = []
        for frame in frames:
            processed = self.preprocess(frame)
            images.append(processed)
        
        # Stack into batch tensor
        return torch.stack(images).to(self.device)
    
    def analyze(self, frames: List[Image.Image]) -> float:
        """Analyze background consistency across video frames.
        
        Args:
            frames: List of PIL Image frames representing the video
            
        Returns:
            Background consistency score (higher is better, range [0, 1])
        """
        if len(frames) < 2:
            return 1.0  # Single frame is perfectly consistent with itself
        
        # Prepare images for CLIP
        images = self._prepare_images_for_clip(frames)
        
        # Extract CLIP features
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1, p=2)
        
        video_sim = 0.0
        frame_count = 0
        
        # Compute similarity between frames
        for i in range(len(image_features)):
            image_feature = image_features[i].unsqueeze(0)
            
            if i == 0:
                # Store first frame features
                first_image_feature = image_feature
            else:
                # Compute similarity with previous frame
                sim_prev = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
                
                # Compute similarity with first frame
                sim_first = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
                
                # Average the two similarities
                frame_sim = (sim_prev + sim_first) / 2.0
                video_sim += frame_sim
                frame_count += 1
            
            # Store current features as previous for next iteration
            former_image_feature = image_feature
        
        # Return average similarity across all frame pairs
        if frame_count > 0:
            return video_sim / frame_count
        else:
            return 1.0
        
class ImagingQualityAnalyzer(VideoQualityAnalyzer):
    """Analyzer for evaluating imaging quality of videos.
    
    This analyzer measures the quality of videos by:
    1. Inputting frames into MUSIQ image quality predictor
    2. Determining if the video is blurry or has artifacts
    
    The score represents the quality of the video (higher is better).
    """
    def __init__(self, model_path: str = "model/musiq/musiq_spaq_ckpt-358bb6af.pth", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_musiq(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def _load_musiq(self, model_path: str):
        """Load MUSIQ model.
        
        Args:
            model_path: Path to the MUSIQ model checkpoint
            
        Returns:
            MUSIQ model
        """
        # if the model_path not exists
        # then makedir and wget
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            wget_command = ['wget', 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth', '-P', os.path.dirname(model_path)]
            subprocess.run(wget_command, check=True)
        
        from pyiqa.archs.musiq_arch import MUSIQ
        model = MUSIQ(pretrained_model_path=model_path)
        
        return model
    
    def _preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """Preprocess frames for MUSIQ model.
        
        Args:
            frames: List of PIL Image frames
            
        Returns:
            Preprocessed frames as tensor
        """
        frames = [pil_to_torch(frame, normalize=False) for frame in frames] # [(C, H, W)]
        frames = torch.stack(frames) # (T, C, H, W)
        
        _, _, h, w = frames.size()
        if max(h, w) > 512:
            scale = 512./max(h, w)
            frames = F.interpolate(frames, size=(int(scale * h), int(scale * w)), mode='bilinear', align_corners=False)
        
        return frames
    
    def analyze(self, frames: List[Image.Image]) -> float:
        """Analyze imaging quality of video frames.
        
        Args:
            frames: List of PIL Image frames representing the video
            
        Returns:
            Imaging quality score (higher is better, range [0, 1])
        """
        frame_tensor = self._preprocess_frames(frames)
        acc_score_video = 0.0
        for i in range(len(frame_tensor)):
            frame = frame_tensor[i].unsqueeze(0).to(self.device)
            score = self.model(frame)
            acc_score_video += float(score)
        return acc_score_video / (100 * len(frame_tensor))