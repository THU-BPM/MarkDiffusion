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
    def __init__(self, crop_ratio: float = 0.8):
        super().__init__()
        self.crop_ratio = crop_ratio  

    def edit(self, image: Image.Image, prompt: str = None) -> Image.Image:
        width, height = image.size
        new_w = int(width * self.crop_ratio)
        new_h = int(height * self.crop_ratio)
        
        left = (width - new_w) // 2
        top = (height - new_h) // 2
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
