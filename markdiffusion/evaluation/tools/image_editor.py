from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
import os
import argparse
import sys
import numpy as np
import random


class ImageEditor:
    def __init__(self):
        pass
        
    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        pass

class JPEGCompression(ImageEditor):
    def __init__(self, quality: int = 95):
        super().__init__()
        self.quality = quality
        
    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        image.save(f"temp.jpg", quality=self.quality)
        compressed_image = Image.open(f"temp.jpg")
        os.remove(f"temp.jpg")
        return compressed_image
    
class Rotation(ImageEditor):
    def __init__(self, angle: int = 30, expand: bool = False):
        super().__init__()
        self.angle = angle       
        self.expand = expand     

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        return image.rotate(self.angle, expand=self.expand)

class CrSc(ImageEditor):
    """Crop-and-scale attack.

    `position` controls where the crop window is placed:
        - "center" (default): top-left corner at ((W-w)//2, (H-h)//2). Backward-compatible.
        - "random": offsets are sampled uniformly each call from [0, W-w] x [0, H-h].
        - tuple `(x_ratio, y_ratio)`: explicit normalized offsets in [0, 1] of the slack
          (W-w, H-h). e.g. (0.0, 0.0) = top-left, (1.0, 1.0) = bottom-right, (0.5, 0.5) = center.
    """

    def __init__(self, crop_ratio: float = 0.8, position="center"):
        super().__init__()
        self.crop_ratio = crop_ratio
        self.position = position
        if isinstance(position, str):
            if position not in {"center", "random"}:
                raise ValueError(f"position must be 'center', 'random', or a (x, y) tuple; got {position!r}")
        else:
            try:
                x_ratio, y_ratio = position
            except (TypeError, ValueError) as e:
                raise ValueError(f"position tuple must be (x_ratio, y_ratio); got {position!r}") from e
            if not (0.0 <= float(x_ratio) <= 1.0 and 0.0 <= float(y_ratio) <= 1.0):
                raise ValueError(f"position ratios must be in [0, 1]; got {position!r}")

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        width, height = image.size
        new_w = int(width * self.crop_ratio)
        new_h = int(height * self.crop_ratio)

        slack_w = max(0, width - new_w)
        slack_h = max(0, height - new_h)

        if self.position == "center":
            left = slack_w // 2
            top = slack_h // 2
        elif self.position == "random":
            left = random.randint(0, slack_w) if slack_w > 0 else 0
            top = random.randint(0, slack_h) if slack_h > 0 else 0
        else:
            x_ratio, y_ratio = self.position
            left = int(round(slack_w * float(x_ratio)))
            top = int(round(slack_h * float(y_ratio)))

        right = left + new_w
        bottom = top + new_h

        return image.crop((left, top, right, bottom)).resize((width, height))

class GaussianBlurring(ImageEditor):
    def __init__(self, radius: int = 2):
        super().__init__()
        self.radius = radius

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.radius))

class GaussianNoise(ImageEditor):
    def __init__(self, sigma: float = 25.0):
        super().__init__()
        self.sigma = sigma 

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        img = image.convert("RGB")
        arr = np.array(img).astype(np.float32)
        
        noise = np.random.normal(0, self.sigma, arr.shape)
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_arr)

class Brightness(ImageEditor):
    def __init__(self, factor: float = 1.2):
        super().__init__()
        self.factor = factor 

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(self.factor)

class Mask(ImageEditor):
    def __init__(self, mask_ratio: float = 0.1, num_masks: int = 5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        for _ in range(self.num_masks):
            max_mask_width = int(width * self.mask_ratio)
            max_mask_height = int(height * self.mask_ratio)
            
            mask_width = random.randint(max_mask_width // 2, max_mask_width)
            mask_height = random.randint(max_mask_height // 2, max_mask_height)
            
            x = random.randint(0, width - mask_width)
            y = random.randint(0, height - mask_height)
            
            draw.rectangle([x, y, x + mask_width, y + mask_height], fill='black')
        
        return img

class Overlay(ImageEditor):
    def __init__(self, num_strokes: int = 10, stroke_width: int = 5, stroke_type: str = 'random'):
        super().__init__()
        self.num_strokes = num_strokes
        self.stroke_width = stroke_width
        self.stroke_type = stroke_type

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        for _ in range(self.num_strokes):
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            num_points = random.randint(3, 8)
            points = [(start_x, start_y)]
            
            for i in range(num_points - 1):
                last_x, last_y = points[-1]
                max_step = min(width, height) // 4
                new_x = max(0, min(width, last_x + random.randint(-max_step, max_step)))
                new_y = max(0, min(height, last_y + random.randint(-max_step, max_step)))
                points.append((new_x, new_y))
            
            if self.stroke_type == 'random':
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            elif self.stroke_type == 'black':
                color = (0, 0, 0)
            elif self.stroke_type == 'white':
                color = (255, 255, 255)
            else:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            draw.line(points, fill=color, width=self.stroke_width)
        
        return img

class AdaptiveNoiseInjection(ImageEditor):
    def __init__(self, intensity: float = 0.5, auto_select: bool = True):
        super().__init__()
        self.intensity = intensity
        self.auto_select = auto_select
    
    def _analyze_image_features(self, img_array):
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        sobel_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        sobel_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edge_density = np.mean(sobel_x + sobel_y)
        
        kernel_size = 5
        texture_complexity = 0
        h, w = gray.shape
        for i in range(0, h - kernel_size, kernel_size):
            for j in range(0, w - kernel_size, kernel_size):
                patch = gray[i:i+kernel_size, j:j+kernel_size]
                texture_complexity += np.std(patch)
        texture_complexity /= ((h // kernel_size) * (w // kernel_size))
        
        return {
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'edge_density': edge_density,
            'texture_complexity': texture_complexity
        }
    
    def _select_noise_type(self, features):
        brightness = features['brightness_mean']
        edge_density = features['edge_density']
        texture = features['texture_complexity']
        
        if brightness < 80:
            return 'gaussian'
        elif edge_density > 30:
            return 'salt_pepper'
        elif texture > 20:
            return 'speckle'
        else:
            return 'poisson'
    
    def _add_gaussian_noise(self, img_array, sigma):
        noise = np.random.normal(0, sigma, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def _add_salt_pepper_noise(self, img_array, amount):
        noisy = img_array.copy()
        h, w = img_array.shape[:2]
        num_pixels = h * w
        
        num_salt = int(amount * num_pixels * 0.5)
        salt_coords_y = np.random.randint(0, h, num_salt)
        salt_coords_x = np.random.randint(0, w, num_salt)
        noisy[salt_coords_y, salt_coords_x] = 255
        
        num_pepper = int(amount * num_pixels * 0.5)
        pepper_coords_y = np.random.randint(0, h, num_pepper)
        pepper_coords_x = np.random.randint(0, w, num_pepper)
        noisy[pepper_coords_y, pepper_coords_x] = 0

        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def _add_poisson_noise(self, img_array):
        vals = len(np.unique(img_array))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img_array * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def _add_speckle_noise(self, img_array, variance):
        noise = np.random.randn(*img_array.shape) * variance
        noisy = img_array + img_array * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        img = image.convert("RGB")
        img_array = np.array(img).astype(np.float32)
        
        features = self._analyze_image_features(img_array)
        
        if self.auto_select:
            noise_type = self._select_noise_type(features)
            
            if noise_type == 'gaussian':
                sigma = 40 * self.intensity
                noisy_array = self._add_gaussian_noise(img_array, sigma)
            elif noise_type == 'salt_pepper':
                amount = 0.15 * self.intensity
                noisy_array = self._add_salt_pepper_noise(img_array, amount)
            elif noise_type == 'poisson':
                noisy_array = self._add_poisson_noise(img_array)
                blend_factor = min(0.8, self.intensity * 1.5)
                noisy_array = np.clip(
                    img_array * (1 - blend_factor) + noisy_array * blend_factor,
                    0, 255
                ).astype(np.uint8)
            else:
                variance = 0.5 * self.intensity
                noisy_array = self._add_speckle_noise(img_array, variance)
        else:
            weight = 0.25
            noisy_array = img_array.copy()
            
            gaussian = self._add_gaussian_noise(img_array, 30 * self.intensity)
            noisy_array = noisy_array * (1 - weight) + gaussian * weight
            
            salt_pepper = self._add_salt_pepper_noise(img_array, 0.08 * self.intensity)
            noisy_array = noisy_array * (1 - weight) + salt_pepper * weight
            
            poisson = self._add_poisson_noise(img_array)
            noisy_array = noisy_array * (1 - weight) + poisson * weight
            
            speckle = self._add_speckle_noise(img_array, 0.4 * self.intensity)
            noisy_array = noisy_array * (1 - weight) + speckle * weight

            noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_array)


class DiffusionPurification(ImageEditor):
    """Diffusion-based purification (regeneration) attack.

    Encodes the input image to latent space, injects Gaussian noise corresponding to
    a fraction of the diffusion schedule, then runs reverse denoising to obtain a
    regenerated image. Generative-watermark-friendly attack: the watermark survives
    only if it is robust to a partial round-trip through a diffusion model.
    Reference: Nie et al., "Diffusion Models for Adversarial Purification", ICML 2022.

    Args:
        diffusion_config: A `DiffusionConfig` providing `pipe`, `device`, and
            `num_inference_steps`. By default the purifier reuses
            `diffusion_config.pipe` (a `StableDiffusionPipeline`-like object).
        purification_strength: Fraction in (0, 1] of the diffusion schedule to use.
            Larger values inject more noise (stronger attack, lower fidelity).
        prompt: Optional text prompt for classifier-free guidance during denoising.
            Empty string by default (unconditional regeneration).
        purifier_pipe: Optional override for the pipeline used to purify; useful
            when the user wants the purifier to be a different model from the one
            that produced the watermarked image.
    """

    def __init__(self, diffusion_config, purification_strength: float = 0.3,
                 prompt: str = "", purifier_pipe=None):
        super().__init__()
        if not (0.0 < float(purification_strength) <= 1.0):
            raise ValueError(
                f"purification_strength must be in (0, 1]; got {purification_strength!r}"
            )
        self.diffusion_config = diffusion_config
        self.purification_strength = float(purification_strength)
        self.default_prompt = prompt
        self.pipe = purifier_pipe if purifier_pipe is not None else diffusion_config.pipe

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        import torch
        from markdiffusion.utils.media_utils import transform_to_model_format

        prompt = prompt if prompt is not None else self.default_prompt
        device = self.diffusion_config.device
        target_size = self.diffusion_config.image_size[0]

        # 1. Image -> tensor in [-1, 1], shape [1, 3, H, W]
        image_tensor = transform_to_model_format(image, target_size=target_size).unsqueeze(0).to(device)

        # 2. Encode prompt (classifier-free guidance disabled by default for purification)
        with torch.no_grad():
            prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                device=device,
                do_classifier_free_guidance=False,
                num_images_per_prompt=1,
            )
        image_tensor = image_tensor.to(prompt_embeds.dtype)

        # 3. Encode image to latent (matches utils.media_utils scaling factor)
        with torch.no_grad():
            latent = self.pipe.vae.encode(image_tensor).latent_dist.sample() * 0.18215

        # 4. Pick the timestep range and add noise
        scheduler = self.pipe.scheduler
        num_steps = self.diffusion_config.num_inference_steps
        scheduler.set_timesteps(num_steps, device=device)
        n_denoise = max(1, int(round(num_steps * self.purification_strength)))
        timesteps_to_use = scheduler.timesteps[-n_denoise:]
        t_start = timesteps_to_use[0]

        noise = torch.randn_like(latent)
        noisy_latent = scheduler.add_noise(latent, noise, t_start.unsqueeze(0))

        # 5. Reverse denoising loop
        x = noisy_latent
        with torch.no_grad():
            for t in timesteps_to_use:
                noise_pred = self.pipe.unet(x, t, encoder_hidden_states=prompt_embeds).sample
                x = scheduler.step(noise_pred, t, x).prev_sample

        # 6. Decode back to image, then to PIL
        with torch.no_grad():
            decoded = self.pipe.vae.decode(x / 0.18215, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        arr = (decoded[0].cpu().float().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

        out = Image.fromarray(arr)
        if out.size != image.size:
            out = out.resize(image.size)
        return out


class NeuralCodecCompression(ImageEditor):
    """Learned-image-codec compression (regeneration) attack.

    Re-encodes the image through a pretrained neural compression model, simulating
    the kind of distortion a downstream encoder would introduce at a target bitrate.
    Backed by `compressai` (an [optional] dependency).
    Reference: Cheng et al., "Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules", CVPR 2020.

    Args:
        quality: Quality level passed to compressai's pretrained zoo
            (typically 1-8 for cheng2020-anchor; higher = better fidelity, more bits).
        model_name: Any key from `compressai.zoo.image_models`. Defaults to
            'cheng2020-anchor'. Other useful choices: 'bmshj2018-factorized',
            'bmshj2018-hyperprior', 'mbt2018', 'cheng2020-attn'.
        device: Override the device for the codec; defaults to CUDA if available else CPU.
    """

    _MODEL_CACHE = {}  # (model_name, quality, device) -> nn.Module

    def __init__(self, quality: int = 5, model_name: str = "cheng2020-anchor",
                 device: str = None):
        super().__init__()
        self.quality = int(quality)
        self.model_name = model_name
        self.device = device

    def _get_model(self):
        import torch
        try:
            from compressai.zoo import image_models
        except ImportError as e:
            raise ImportError(
                "NeuralCodecCompression requires `compressai`. "
                "Install with: pip install -e '.[optional]' (or `pip install compressai`)."
            ) from e
        if self.model_name not in image_models:
            raise ValueError(
                f"Unknown compressai model {self.model_name!r}; "
                f"available: {sorted(image_models)}"
            )
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = (self.model_name, self.quality, device)
        net = self._MODEL_CACHE.get(cache_key)
        if net is None:
            net = image_models[self.model_name](quality=self.quality, pretrained=True)
            net = net.eval().to(device)
            self._MODEL_CACHE[cache_key] = net
        return net, device

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        import torch
        from torchvision import transforms

        net, device = self._get_model()
        original_size = image.size

        # compressai models expect input dimensions divisible by 64.
        w, h = original_size
        new_w = max(64, (w // 64) * 64)
        new_h = max(64, (h // 64) * 64)
        if (new_w, new_h) != (w, h):
            input_image = image.resize((new_w, new_h))
        else:
            input_image = image

        x = transforms.ToTensor()(input_image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = net(x)
            x_hat = out["x_hat"].clamp(0, 1)

        out_pil = transforms.ToPILImage()(x_hat[0].cpu())
        if out_pil.size != original_size:
            out_pil = out_pil.resize(original_size)
        return out_pil
