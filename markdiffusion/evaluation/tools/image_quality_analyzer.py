from PIL import Image
from typing import List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
import lpips
import piq

class ImageQualityAnalyzer:
    """Base class for image quality analyzer."""
    
    def __init__(self):
        pass

    @abstractmethod
    def analyze(self):
        pass
    
class DirectImageQualityAnalyzer(ImageQualityAnalyzer):
    """Base class for direct image quality analyzer."""
    
    def __init__(self):
        pass

    def analyze(self, image: Image.Image, *args, **kwargs):
        pass
    
class ReferencedImageQualityAnalyzer(ImageQualityAnalyzer):
    """Base class for referenced image quality analyzer."""
    
    def __init__(self):
        pass

    def analyze(self, image: Image.Image, reference: Union[Image.Image, str], *args, **kwargs):
        pass
    
class GroupImageQualityAnalyzer(ImageQualityAnalyzer):
    """Base class for group image quality analyzer."""
    
    def __init__(self):
        pass

    def analyze(self, images: List[Image.Image], references: List[Image.Image], *args, **kwargs):
        pass
    
class RepeatImageQualityAnalyzer(ImageQualityAnalyzer):
    """Base class for repeat image quality analyzer."""
    
    def __init__(self):
        pass

    def analyze(self, images: List[Image.Image], *args, **kwargs):
        pass
    
class ComparedImageQualityAnalyzer(ImageQualityAnalyzer):
    """Base class for compare image quality analyzer."""
    
    def __init__(self):
        pass

    def analyze(self, image: Image.Image, reference: Image.Image, *args, **kwargs):
        pass

class InceptionScoreCalculator(RepeatImageQualityAnalyzer):
    """Inception Score (IS) calculator for evaluating image generation quality.
    
    Inception Score measures both the quality and diversity of generated images
    by evaluating how confidently an Inception model can classify them and how
    diverse the predictions are across the image set.
    
    Higher IS indicates better image quality and diversity (typical range: 1-10+).
    """
    
    def __init__(self, device: str = "cuda", batch_size: int = 32, splits: int = 1):
        """Initialize the Inception Score calculator.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            batch_size: Batch size for processing images
            splits: Number of splits for computing IS (default: 1). The splits must be divisible by the number of images for fair comparison.
                    For calculating the mean and standard error of IS, the splits should be set greater than 1.
                    If splits is 1, the IS is calculated on the entire dataset.(Avg = IS, Std = 0)
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.splits = splits
        self._load_model()
        
    def _load_model(self):
        """Load the Inception v3 model for feature extraction."""
        from torchvision import models, transforms
        
        # Load pre-trained Inception v3 model
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.aux_logits = False  # Disable auxiliary output
        self.model.eval()
        self.model.to(self.device)
        
        # Keep the original classification layer for proper predictions
        # No need to modify model.fc - it should output 1000 classes
        
        # Define preprocessing pipeline for Inception v3
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception v3 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet statistics
        ])
    
    def _get_predictions(self, images: List[Image.Image]) -> np.ndarray:
        """Extract softmax predictions from images using Inception v3.
        
        Args:
            images: List of PIL images to process
            
        Returns:
            Numpy array of shape (n_images, n_classes) containing softmax predictions
        """
        predictions_list = []
        
        # Process images in batches for efficiency
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                # Ensure RGB format
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                tensor = self.preprocess(img)
                batch_tensors.append(tensor)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Get predictions from Inception model
            with torch.no_grad():
                logits = self.model(batch_tensor)
                # Apply softmax to get probability distributions
                probs = F.softmax(logits, dim=1)
                predictions_list.append(probs.cpu().numpy())
        
        return np.concatenate(predictions_list, axis=0)
    
    def _calculate_inception_score(self, predictions: np.ndarray) -> tuple:
        """Calculate Inception Score from predictions.
        
        The IS is calculated as exp(KL divergence between conditional and marginal distributions).
        
        Args:
            predictions: Softmax predictions of shape (n_images, n_classes)
            
        Returns:
            Tuple of (mean_is, std_is) across splits
        """
        # Split predictions for more stable estimation
        n_samples = predictions.shape[0] # (n_images, n_classes)
        split_size = n_samples // self.splits

        splits = self.splits
        
        split_scores = []
        
        for split_idx in range(splits):
            # Get current split
            start_idx = split_idx * split_size
            end_idx = (split_idx + 1) * split_size if split_idx < splits - 1 else n_samples # Last split gets remaining samples
            split_preds = predictions[start_idx:end_idx]
            
            # Calculate marginal distribution p(y) - average across all samples
            p_y = np.mean(split_preds, axis=0)
            
            epsilon = 1e-16
            p_y_x_safe = split_preds + epsilon
            p_y_safe = p_y + epsilon
            kl_divergences = np.sum(
                p_y_x_safe * (np.log(p_y_x_safe / p_y_safe)), 
                axis=1)
            
            # Inception Score for this split is exp(mean(KL divergences))
            split_score = np.exp(np.mean(kl_divergences))
            split_scores.append(split_score)
        
        # Directly return the list of scores for each split
        return split_scores
    
    def analyze(self, images: List[Image.Image], *args, **kwargs) -> List[float]:
        """Calculate Inception Score for a set of generated images.
        
        Args:
            images: List of generated images to evaluate
            
        Returns:
            List[float]: Inception Score values for each split (higher is better, typical range: 1-10+)
        """
        if len(images) < self.splits:
            raise ValueError(f"Inception Score requires at least {self.splits} images (one per split)")
        
        if len(images) % self.splits != 0:
            raise ValueError(f"Inception Score requires the number of images to be divisible by the number of splits")
        
        # Get predictions from Inception model
        predictions = self._get_predictions(images)
        
        # Calculate Inception Score
        split_scores = self._calculate_inception_score(predictions)
        
        # Log the standard deviation for reference (but return only mean)
        mean_score = np.mean(split_scores)
        std_score = np.std(split_scores)
        if std_score > 0.5 * mean_score:
            print(f"Warning: High standard deviation in IS calculation: {mean_score:.2f} ± {std_score:.2f}")
        
        return split_scores
    
class CLIPScoreCalculator(ReferencedImageQualityAnalyzer):
    """CLIP score calculator for image quality analysis.
    
    Calculates CLIP similarity between an image and a reference.
    Higher scores indicate better semantic similarity.
    """
    
    def __init__(self, device: str = "cuda", model_name: str = "ViT-B/32", reference_source: str = "image"):
        """Initialize the CLIP Score calculator.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            model_name: CLIP model variant to use
            reference_source: The source of reference ('image' or 'text')
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.reference_source = reference_source
        self._load_model()
        
    def _load_model(self):
        """Load the CLIP model."""
        try:
            import clip
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("Please install the CLIP library: pip install git+https://github.com/openai/CLIP.git")
    
    def analyze(self, image: Image.Image, reference: Union[Image.Image, str], *args, **kwargs) -> float:
        """Calculate CLIP similarity between image and reference.
        
        Args:
            image: Input image to evaluate
            reference: Reference image or text for comparison
                - If reference_source is 'image': expects PIL Image
                - If reference_source is 'text': expects string
                
        Returns:
            float: CLIP similarity score (0 to 1)
        """
        
        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features based on reference source
        with torch.no_grad():
            # Encode image features
            img_features = self.model.encode_image(img_tensor)
            
            # Encode reference features based on source type
            if self.reference_source == 'text':
                if not isinstance(reference, str):
                    raise ValueError(f"Expected string reference for text mode, got {type(reference)}")
                
                # Tokenize and encode text
                text_tokens = clip.tokenize([reference]).to(self.device)
                ref_features = self.model.encode_text(text_tokens)
                
            elif self.reference_source == 'image':
                if not isinstance(reference, Image.Image):
                    raise ValueError(f"Expected PIL Image reference for image mode, got {type(reference)}")
                
                # Convert reference image to RGB if necessary
                if reference.mode != 'RGB':
                    reference = reference.convert('RGB')
                
                # Preprocess and encode reference image
                ref_tensor = self.preprocess(reference).unsqueeze(0).to(self.device)
                ref_features = self.model.encode_image(ref_tensor)
                
            else:
                raise ValueError(f"Invalid reference_source: {self.reference_source}. Must be 'image' or 'text'")
            
            # Normalize features
            img_features = F.normalize(img_features, p=2, dim=1)
            ref_features = F.normalize(ref_features, p=2, dim=1)
            
            # Calculate cosine similarity
            similarity = torch.cosine_similarity(img_features, ref_features).item()
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
        return similarity
    
class FIDCalculator(GroupImageQualityAnalyzer):
    """FID calculator for image quality analysis.
    
    Calculates Fréchet Inception Distance between two sets of images.
    Lower FID indicates better quality and similarity to reference distribution.
    """
    
    def __init__(self, device: str = "cuda", batch_size: int = 32, splits: int = 1):
        """Initialize the FID calculator.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            batch_size: Batch size for processing images
            splits: Number of splits for computing FID (default: 5). The splits must be divisible by the number of images for fair comparison.
                    For calculating the mean and standard error of FID, the splits should be set greater than 1.
                    If splits is 1, the FID is calculated on the entire dataset.(Avg = FID, Std = 0)
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.splits = splits
        self._load_model()
        
    def _load_model(self):
        """Load the Inception v3 model for feature extraction."""
        from torchvision import models, transforms
        
        # Load Inception v3 model
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, init_weights=False)
        inception.fc = nn.Identity()  # Remove final classification layer
        inception.aux_logits = False
        inception.eval()
        inception.to(self.device)
        self.model = inception
        
        # Define preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet statistics
        ])
    
    def _extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract features from a list of images.
        
        Args:
            images: List of PIL images
            
        Returns:
            Feature matrix of shape (n_images, 2048)
        """
        features_list = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                tensor = self.preprocess(img)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(batch_tensor) # (batch_size, 2048)
                features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0) # (n_images, 2048)
    
    def _calculate_fid(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate FID between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            float: FID score
        """
        from scipy.linalg import sqrtm
        
        # Calculate statistics
        mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        # Numerical stability
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)
    
    def analyze(self, images: List[Image.Image], references: List[Image.Image], *args, **kwargs) -> List[float]:
        """Calculate FID between two sets of images.
        
        Args:
            images: Set of images to evaluate
            references: Reference set of images
            
        Returns:
            List[float]: FID values for each split
        """
        if len(images) < 2 or len(references) < 2:
            raise ValueError("FID requires at least 2 images in each set")
        if len(images) % self.splits != 0 or len(references) % self.splits != 0:
            raise ValueError("FID requires the number of images to be divisible by the number of splits")
        
        fid_scores = []
        # Extract features
        features1 = self._extract_features(images)
        features2 = self._extract_features(references)
        
        # Calculate FID
        # for i in range(self.splits):
        #     start_idx = i * len(images) // self.splits
        #     end_idx = (i + 1) * len(images) // self.splits
        #     fid_scores.append(self._calculate_fid(features1[start_idx:end_idx], features2[start_idx:end_idx]))
        
        fid_scores = self._calculate_fid(features1, features2)
        
        return fid_scores
    
class LPIPSAnalyzer(RepeatImageQualityAnalyzer):
    """LPIPS analyzer for image quality analysis.
    
    Calculates perceptual diversity within a set of images.
    Higher LPIPS indicates more diverse/varied images.
    """
    
    def __init__(self, device: str = "cuda", net: str = "alex"):
        """Initialize the LPIPS analyzer.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            net: Network to use ('alex', 'vgg', or 'squeeze')
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net = net
        self._load_model()
        
    def _load_model(self) -> None:
        """
            Load the LPIPS model.
        """
        self.model = lpips.LPIPS(net=self.net)
        self.model.eval()
        self.model.to(self.device)
    
    def analyze(self, images: List[Image.Image], *args, **kwargs) -> float:
        """Calculate average pairwise LPIPS distance within a set of images.
        
        Args:
            images: List of images to analyze diversity
            
        Returns:
            float: Average LPIPS distance (diversity score)
        """
        if len(images) < 2:
            return 0.0  # No diversity with single image
        
        # Preprocess all images
        tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            tensor = lpips.im2tensor(np.array(img).astype(np.uint8)).to(self.device)  # Convert to tensor
            tensors.append(tensor)
        
        # Calculate pairwise LPIPS distances
        distances = []
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                with torch.no_grad():
                    distance = self.model.forward(tensors[i], tensors[j]).item()
                    distances.append(distance)
        
        # Return average distance as diversity score
        return np.mean(distances) if distances else 0.0
    
class PSNRAnalyzer(ComparedImageQualityAnalyzer):
    """PSNR analyzer for image quality analysis.
    
    Calculates Peak Signal-to-Noise Ratio between two images.
    Higher PSNR indicates better quality/similarity.
    """
    
    def __init__(self, max_pixel_value: float = 255.0):
        """Initialize the PSNR analyzer.
        
        Args:
            max_pixel_value: Maximum pixel value (255 for 8-bit images)
        """
        super().__init__()
        self.max_pixel_value = max_pixel_value
    
    def analyze(self, image: Image.Image, reference: Image.Image, *args, **kwargs) -> float:
        """Calculate PSNR between two images.
        
        Args:
            image (Image.Image): Image to evaluate
            reference (Image.Image): Reference image

        Returns:
            float: PSNR value in dB
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if reference.mode != 'RGB':
            reference = reference.convert('RGB')
        
        # Resize if necessary
        if image.size != reference.size:
            reference = reference.resize(image.size, Image.Resampling.BILINEAR)
        
        # Convert to numpy arrays
        img_array = np.array(image, dtype=np.float32)
        ref_array = np.array(reference, dtype=np.float32)
        
        # Calculate MSE
        mse = np.mean((img_array - ref_array) ** 2)
        
        # Avoid division by zero
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        psnr = 20 * np.log10(self.max_pixel_value / np.sqrt(mse))
        
        return float(psnr)


class NIQECalculator(DirectImageQualityAnalyzer):
    """Natural Image Quality Evaluator (NIQE) for no-reference image quality assessment.
    
    NIQE evaluates image quality based on deviations from natural scene statistics.
    It uses a pre-trained model of natural image statistics to assess quality without
    requiring reference images.
    
    Lower NIQE scores indicate better/more natural image quality (typical range: 2-8).
    """
    
    def __init__(self, 
                 model_path: str = "evaluation/tools/data/niqe_image_params.mat",
                 patch_size: int = 96,
                 sigma: float = 7.0/6.0,
                 C: float = 1.0):
        """Initialize NIQE calculator with pre-trained natural image statistics.
        
        Args:
            model_path: Path to the pre-trained NIQE model parameters (.mat file)
            patch_size: Size of patches for feature extraction (default: 96)
            sigma: Standard deviation for Gaussian window (default: 7/6)
            C: Constant for numerical stability in MSCN transform (default: 1.0)
        """
        super().__init__()
        self.patch_size = patch_size
        self.sigma = sigma
        self.C = C
        
        # Load pre-trained natural image statistics
        self._load_model_params(model_path)
        
        # Pre-compute gamma lookup table for AGGD parameter estimation
        self._precompute_gamma_table()
        
        # Generate Gaussian window for local mean/variance computation
        self.avg_window = self._generate_gaussian_window(3, self.sigma)
    
    def _load_model_params(self, model_path: str) -> None:
        """Load pre-trained NIQE model parameters from MAT file.
        
        Args:
            model_path: Path to the model parameters file
        """
        import scipy.io
        try:
            params = scipy.io.loadmat(model_path)
            self.pop_mu = np.ravel(params["pop_mu"])
            self.pop_cov = params["pop_cov"]
        except Exception as e:
            raise RuntimeError(f"Failed to load NIQE model parameters from {model_path}: {e}")
    
    def _precompute_gamma_table(self) -> None:
        """Pre-compute gamma function values for AGGD parameter estimation."""
        import scipy.special
        
        self.gamma_range = np.arange(0.2, 10, 0.001)
        a = scipy.special.gamma(2.0 / self.gamma_range)
        a *= a
        b = scipy.special.gamma(1.0 / self.gamma_range)
        c = scipy.special.gamma(3.0 / self.gamma_range)
        self.prec_gammas = a / (b * c)
    
    def _generate_gaussian_window(self, window_size: int, sigma: float) -> np.ndarray:
        """Generate 1D Gaussian window for filtering.
        
        Args:
            window_size: Half-size of the window (full size = 2*window_size + 1)
            sigma: Standard deviation of Gaussian
            
        Returns:
            1D Gaussian window weights
        """
        window_size = int(window_size)
        weights = np.zeros(2 * window_size + 1)
        weights[window_size] = 1.0
        sum_weights = 1.0
        
        sigma_sq = sigma * sigma
        for i in range(1, window_size + 1):
            tmp = np.exp(-0.5 * i * i / sigma_sq)
            weights[window_size + i] = tmp
            weights[window_size - i] = tmp
            sum_weights += 2.0 * tmp
        
        weights /= sum_weights
        return weights
    
    def _compute_mscn_transform(self, image: np.ndarray, extend_mode: str = 'constant') -> tuple:
        """Compute Mean Subtracted Contrast Normalized (MSCN) coefficients.
        
        MSCN transformation normalizes image patches by local mean and variance,
        making the coefficients more suitable for statistical modeling.
        
        Args:
            image: Input image array
            extend_mode: Boundary extension mode for filtering
            
        Returns:
            Tuple of (mscn_coefficients, local_variance, local_mean)
        """
        import scipy.ndimage
        
        assert len(image.shape) == 2, "Input must be grayscale image"
        h, w = image.shape
        
        # Allocate arrays for local statistics
        mu_image = np.zeros((h, w), dtype=np.float32)
        var_image = np.zeros((h, w), dtype=np.float32)
        image_float = image.astype(np.float32)
        
        # Compute local mean using separable Gaussian filtering
        scipy.ndimage.correlate1d(image_float, self.avg_window, 0, mu_image, mode=extend_mode)
        scipy.ndimage.correlate1d(mu_image, self.avg_window, 1, mu_image, mode=extend_mode)
        
        # Compute local variance
        scipy.ndimage.correlate1d(image_float**2, self.avg_window, 0, var_image, mode=extend_mode)
        scipy.ndimage.correlate1d(var_image, self.avg_window, 1, var_image, mode=extend_mode)
        
        # Variance = E[X^2] - E[X]^2
        var_image = np.sqrt(np.abs(var_image - mu_image**2))
        
        # MSCN transform
        mscn = (image_float - mu_image) / (var_image + self.C)
        
        return mscn, var_image, mu_image
    
    def _compute_aggd_features(self, coefficients: np.ndarray) -> tuple:
        """Compute Asymmetric Generalized Gaussian Distribution (AGGD) parameters.
        
        AGGD models the distribution of MSCN coefficients and their products,
        capturing shape and asymmetry characteristics.
        
        Args:
            coefficients: MSCN coefficients
            
        Returns:
            Tuple of (alpha, N, bl, br, left_std, right_std)
        """
        import scipy.special
        
        # Flatten coefficients
        coeffs_flat = coefficients.flatten()
        coeffs_squared = coeffs_flat * coeffs_flat
        
        # Separate left (negative) and right (positive) tail data
        left_data = coeffs_squared[coeffs_flat < 0]
        right_data = coeffs_squared[coeffs_flat >= 0]
        
        # Compute standard deviations for left and right tails
        left_std = np.sqrt(np.mean(left_data)) if len(left_data) > 0 else 0
        right_std = np.sqrt(np.mean(right_data)) if len(right_data) > 0 else 0
        
        # Estimate gamma (shape asymmetry parameter)
        if right_std != 0:
            gamma_hat = left_std / right_std
        else:
            gamma_hat = np.inf
        
        # Estimate r_hat (generalized Gaussian ratio)
        mean_abs = np.mean(np.abs(coeffs_flat))
        mean_squared = np.mean(coeffs_squared)
        
        if mean_squared != 0:
            r_hat = (mean_abs ** 2) / mean_squared
        else:
            r_hat = np.inf
        
        # Normalize r_hat using gamma
        rhat_norm = r_hat * (((gamma_hat**3 + 1) * (gamma_hat + 1)) / 
                            ((gamma_hat**2 + 1) ** 2))
        
        # Find best-fitting alpha by comparing with pre-computed values
        pos = np.argmin((self.prec_gammas - rhat_norm) ** 2)
        alpha = self.gamma_range[pos]
        
        # Compute AGGD parameters
        gam1 = scipy.special.gamma(1.0 / alpha)
        gam2 = scipy.special.gamma(2.0 / alpha)
        gam3 = scipy.special.gamma(3.0 / alpha)
        
        aggd_ratio = np.sqrt(gam1) / np.sqrt(gam3)
        bl = aggd_ratio * left_std   # Left scale parameter
        br = aggd_ratio * right_std   # Right scale parameter
        
        # Mean parameter
        N = (br - bl) * (gam2 / gam1)
        
        return alpha, N, bl, br, left_std, right_std
    
    def _compute_paired_products(self, mscn_coeffs: np.ndarray) -> tuple:
        """Compute products of adjacent MSCN coefficients in four orientations.
        
        These products capture dependencies between neighboring pixels.
        
        Args:
            mscn_coeffs: MSCN coefficient matrix
            
        Returns:
            Tuple of (horizontal, vertical, diagonal1, diagonal2) products
        """
        # Shift in four directions and compute products
        shift_h = np.roll(mscn_coeffs, 1, axis=1)      # Horizontal shift
        shift_v = np.roll(mscn_coeffs, 1, axis=0)      # Vertical shift
        shift_d1 = np.roll(shift_v, 1, axis=1)         # Main diagonal shift
        shift_d2 = np.roll(shift_v, -1, axis=1)        # Anti-diagonal shift
        
        # Compute products
        prod_h = mscn_coeffs * shift_h    # Horizontal pairs
        prod_v = mscn_coeffs * shift_v    # Vertical pairs
        prod_d1 = mscn_coeffs * shift_d1  # Diagonal pairs
        prod_d2 = mscn_coeffs * shift_d2  # Anti-diagonal pairs
        
        return prod_h, prod_v, prod_d1, prod_d2
    
    def _extract_subband_features(self, mscn_coeffs: np.ndarray) -> np.ndarray:
        """Extract statistical features from MSCN coefficients and their products.
        
        Args:
            mscn_coeffs: MSCN coefficient matrix
            
        Returns:
            Feature vector of length 18
        """
        # Extract AGGD parameters from MSCN coefficients
        alpha_m, N, bl, br, _, _ = self._compute_aggd_features(mscn_coeffs)
        
        # Compute paired products in four orientations
        prod_h, prod_v, prod_d1, prod_d2 = self._compute_paired_products(mscn_coeffs)
        
        # Extract AGGD parameters for each product orientation
        alpha1, N1, bl1, br1, _, _ = self._compute_aggd_features(prod_h)
        alpha2, N2, bl2, br2, _, _ = self._compute_aggd_features(prod_v)
        alpha3, N3, bl3, br3, _, _ = self._compute_aggd_features(prod_d1)
        alpha4, N4, bl4, br4, _, _ = self._compute_aggd_features(prod_d2)
        
        # Combine all features into feature vector
        # Note: For diagonal pairs in reference, bl3 is repeated twice (not br3)
        features = np.array([
            alpha_m, (bl + br) / 2.0,      # Shape and scale of MSCN
            alpha1, N1, bl1, br1,          # Vertical pairs (V)
            alpha2, N2, bl2, br2,          # Horizontal pairs (H)
            alpha3, N3, bl3, bl3,          # Diagonal pairs (D1) - note: bl3 repeated
            alpha4, N4, bl4, bl4,          # Anti-diagonal pairs (D2) - note: bl4 repeated
        ])
        
        return features
    
    def _extract_multiscale_features(self, image: np.ndarray) -> tuple:
        """Extract features at multiple scales.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (all_features, mean_features, sample_covariance)
        """
        h, w = image.shape
        
        # Check minimum size requirements
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(f"Image too small. Minimum size: {self.patch_size}x{self.patch_size}")
        
        # Ensure that the patch divides evenly into img
        hoffset = h % self.patch_size
        woffset = w % self.patch_size
        
        if hoffset > 0:
            image = image[:-hoffset, :]
        if woffset > 0:
            image = image[:, :-woffset]
        
        # Convert to float32 for processing
        image = image.astype(np.float32)
        
        # Downsample image by factor of 2 using PIL (as in reference)
        img_pil = Image.fromarray(image)
        size = tuple((np.array(img_pil.size) * 0.5).astype(int))
        img2 = np.array(img_pil.resize(size, Image.BICUBIC))
        
        # Compute MSCN transforms at two scales
        mscn1, _, _ = self._compute_mscn_transform(image)
        mscn1 = mscn1.astype(np.float32)
        
        mscn2, _, _ = self._compute_mscn_transform(img2)
        mscn2 = mscn2.astype(np.float32)
        
        # Extract features from patches at each scale
        feats_lvl1 = self._extract_patches_test_features(mscn1, self.patch_size)
        feats_lvl2 = self._extract_patches_test_features(mscn2, self.patch_size // 2)
        
        # Concatenate features from both scales
        feats = np.hstack((feats_lvl1, feats_lvl2))
        
        # Calculate mean and covariance
        sample_mu = np.mean(feats, axis=0)
        sample_cov = np.cov(feats.T)
        
        return feats, sample_mu, sample_cov
    
    def _extract_patches_test_features(self, mscn: np.ndarray, patch_size: int) -> np.ndarray:
        """Extract features from non-overlapping patches for test images.
        
        Args:
            mscn: MSCN coefficient matrix
            patch_size: Size of patches
            
        Returns:
            Array of patch features
        """
        h, w = mscn.shape
        patch_size = int(patch_size)
        
        # Extract non-overlapping patches
        patches = []
        for j in range(0, h - patch_size + 1, patch_size):
            for i in range(0, w - patch_size + 1, patch_size):
                patch = mscn[j:j + patch_size, i:i + patch_size]
                patches.append(patch)
        
        patches = np.array(patches)
        
        # Extract features from each patch
        patch_features = []
        for p in patches:
            patch_features.append(self._extract_subband_features(p))
        
        patch_features = np.array(patch_features)
        
        return patch_features
    
    def analyze(self, image: Image.Image, *args, **kwargs) -> float:
        """Calculate NIQE score for a single image.
        
        Args:
            image: Input image to evaluate
            
        Returns:
            float: NIQE score (lower is better, typical range: 2-8)
        """
        
        import scipy.linalg
        import scipy.special
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            if image.mode == 'RGB':
                # Convert RGB to grayscale as in reference: using 'LA' and taking first channel
                image = image.convert('LA')
                img_array = np.array(image)[:,:,0].astype(np.float32)
            else:
                image = image.convert('L')
                img_array = np.array(image, dtype=np.float32)
        else:
            img_array = np.array(image, dtype=np.float32)
        
        # Check minimum size requirements
        min_size = self.patch_size * 2 + 1
        if img_array.shape[0] < min_size or img_array.shape[1] < min_size:
            raise ValueError(f"Image too small. Minimum size: {min_size}x{min_size}")
        
        # Extract multi-scale features  
        all_features, sample_mu, sample_cov = self._extract_multiscale_features(img_array)
        
        # Compute distance from natural image statistics
        X = sample_mu - self.pop_mu
        
        # Calculate Mahalanobis-like distance
        # Use average of sample and population covariance as in reference
        covmat = (self.pop_cov + sample_cov) / 2.0
        
        # Compute pseudo-inverse for numerical stability
        pinv_cov = scipy.linalg.pinv(covmat)
        
        # Calculate NIQE score
        niqe_score = np.sqrt(np.dot(np.dot(X, pinv_cov), X))
        
        return float(niqe_score)


class SSIMAnalyzer(ComparedImageQualityAnalyzer):
    """SSIM analyzer for image quality analysis.
    
    Calculates Structural Similarity Index between two images.
    Higher SSIM indicates better quality/similarity.
    """
    
    def __init__(self, max_pixel_value: float = 255.0):
        """Initialize the SSIM analyzer.
        
        Args:
            max_pixel_value: Maximum pixel value (255 for 8-bit images)
        """
        super().__init__()
        self.max_pixel_value = max_pixel_value
        self.C1 = (0.01 * max_pixel_value) ** 2
        self.C2 = (0.03 * max_pixel_value) ** 2
        self.C3 = self.C2 / 2.0

    def analyze(self, image: Image.Image, reference: Image.Image, *args, **kwargs) -> float:
        """Calculate SSIM between two images.

        Args:
            image (Image.Image): Image to evaluate
            reference (Image.Image): Reference image

        Returns:
            float: SSIM value (0 to 1)
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if reference.mode != 'RGB':
            reference = reference.convert('RGB')
        
        # Resize if necessary
        if image.size != reference.size:
            reference = reference.resize(image.size, Image.Resampling.BILINEAR)
        
        # Convert to numpy arrays
        img_array = np.array(image, dtype=np.float32)
        ref_array = np.array(reference, dtype=np.float32)
        
        # Calculate means
        mu_x = np.mean(img_array)
        mu_y = np.mean(ref_array)
        
        # Calculate variances and covariance
        sigma_x = np.std(img_array)
        sigma_y = np.std(ref_array)
        sigma_xy = np.mean((img_array - mu_x) * (ref_array - mu_y))
        
        # Calculate SSIM
        luminance_mean=(2 * mu_x * mu_y + self.C1) / (mu_x**2 + mu_y**2 + self.C1)
        contrast=(2 * sigma_x * sigma_y + self.C2) / (sigma_x**2 + sigma_y**2 + self.C2)
        structure_comparison=(sigma_xy + self.C3) / (sigma_x * sigma_y + self.C3)
        ssim = luminance_mean * contrast * structure_comparison

        return float(ssim)
    

class BRISQUEAnalyzer(DirectImageQualityAnalyzer):
    """BRISQUE analyzer for no-reference image quality analysis.
    
    BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    evaluates perceptual quality of an image without requiring
    a reference. Lower BRISQUE scores indicate better quality.
    Typical range: 0 (best) ~ 100 (worst).
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor in range [0,1] with shape (1,C,H,W)."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # BCHW
        return tensor.to(self.device)

    def analyze(self, image: Image.Image, *args, **kwargs) -> float:
        """Calculate BRISQUE score for a single image.
        
        Args:
            image: PIL Image
        
        Returns:
            float: BRISQUE score (lower is better)
        """
        x = self._preprocess(image)
        with torch.no_grad():
            score = piq.brisque(x, data_range=1.0)  # piq expects [0,1]
        return float(score.item())
    
    
class VIFAnalyzer(ComparedImageQualityAnalyzer):
    """VIF (Visual Information Fidelity) analyzer using piq.
    
    VIF compares a distorted image with a reference image to 
    quantify the amount of visual information preserved.
    Higher VIF indicates better quality/similarity.
    Typical range: 0 ~ 1 (sometimes higher for good quality).
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor in range [0,1] with shape (1,C,H,W)."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # BCHW
        return tensor.to(self.device)

    def analyze(self, image: Image.Image, reference: Image.Image, *args, **kwargs) -> float:
        """Calculate VIF score between image and reference.
        
        Args:
            image: Distorted/test image (PIL)
            reference: Reference image (PIL)
        
        Returns:
            float: VIF score (higher is better)
        """
        x = self._preprocess(image)
        y = self._preprocess(reference)
        
        # Ensure same size (piq expects matching shapes)
        if x.shape != y.shape:
            _, _, h, w = x.shape
            y = torch.nn.functional.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            score = piq.vif_p(x, y, data_range=1.0)
        return float(score.item())
    
    
class FSIMAnalyzer(ComparedImageQualityAnalyzer):
    """FSIM (Feature Similarity Index) analyzer using piq.
    
    FSIM compares structural similarity between two images 
    based on phase congruency and gradient magnitude.
    Higher FSIM indicates better quality/similarity.
    Typical range: 0 ~ 1.
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor in range [0,1] with shape (1,C,H,W)."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # BCHW
        return tensor.to(self.device)

    def analyze(self, image: Image.Image, reference: Image.Image, *args, **kwargs) -> float:
        """Calculate FSIM score between image and reference.
        
        Args:
            image: Distorted/test image (PIL)
            reference: Reference image (PIL)
        
        Returns:
            float: FSIM score (higher is better)
        """
        x = self._preprocess(image)
        y = self._preprocess(reference)

        # Ensure same size
        if x.shape != y.shape:
            _, _, h, w = x.shape
            y = torch.nn.functional.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            score = piq.fsim(x, y, data_range=1.0)
        return float(score.item())