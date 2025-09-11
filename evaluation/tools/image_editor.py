from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import os
import argparse
import sys
import numpy as np


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
