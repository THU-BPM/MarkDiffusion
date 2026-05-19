Implementing Custom Algorithms
==============================

This guide shows you how to implement your own watermarking algorithm in MarkDiffusion.

Architecture Overview
---------------------

MarkDiffusion uses a modular architecture where each watermarking algorithm extends a base class. To add a new algorithm, you need to:

1. Create algorithm class extending ``BaseWatermark``
2. Implement required methods
3. Add configuration file
4. (Optional) Create visualizer
5. (Optional) Create detector

Base Watermark Class
--------------------

All watermarking algorithms inherit from ``BaseWatermark``:

.. code-block:: python

   from markdiffusion.watermark.base import BaseWatermark

   class MyWatermark(BaseWatermark):
       """Your custom watermarking algorithm."""
       
       def __init__(self, algorithm_config, diffusion_config):
           super().__init__(algorithm_config, diffusion_config)
           # Initialize your algorithm-specific parameters
           self.watermark_strength = algorithm_config.get('watermark_strength', 1.0)
           self.key = algorithm_config.get('key', 0)
       
       def generate_watermarked_media(self, input_data, **kwargs):
           """Generate watermarked image/video."""
           # Implement watermark embedding
           pass
       
       def detect_watermark_in_media(self, media, **kwargs):
           """Detect watermark in media."""
           # Implement watermark detection
           pass
       
       def get_data_for_visualize(self, media):
           """Get data for visualization."""
           # Return data needed for visualization
           pass

Required Methods
----------------

generate_watermarked_media
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates watermarked media from a prompt or input.

.. code-block:: python

   def generate_watermarked_media(self, input_data, **kwargs):
       """
       Generate watermarked media.
       
       Args:
           input_data: Text prompt or input data
           **kwargs: Additional generation parameters
           
       Returns:
           PIL.Image or list of PIL.Image (for video)
       """
       # Get diffusion pipeline
       pipe = self.diffusion_config.pipe
       
       # Modify latents or noise for watermarking
       # Option 1: Modify initial noise
       latents = self._get_watermarked_latents(
           shape=(1, 4, 64, 64),
           seed=self.diffusion_config.gen_seed
       )
       
       # Option 2: Use callback to modify during generation
       def callback(step, timestep, latents_callback):
           return self._embed_watermark_step(latents_callback, timestep)
       
       # Generate with watermark
       with torch.no_grad():
           output = pipe(
               prompt=input_data,
               latents=latents,
               num_inference_steps=self.diffusion_config.num_inference_steps,
               guidance_scale=self.diffusion_config.guidance_scale,
               callback=callback,
               **kwargs
           )
       
       return output.images[0]

detect_watermark_in_media
~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects watermark in an image or video.

.. code-block:: python

   def detect_watermark_in_media(self, media, **kwargs):
       """
       Detect watermark in media.
       
       Args:
           media: PIL.Image or list of PIL.Image
           **kwargs: Additional detection parameters
           
       Returns:
           dict: Detection result with score and metadata
       """
       # Convert image to tensor
       if isinstance(media, Image.Image):
           image_tensor = self._image_to_tensor(media)
       
       # Invert to latent space
       latents = self._invert_image(image_tensor)
       
       # Extract watermark signal
       watermark_signal = self._extract_watermark_signal(latents)
       
       # Calculate detection score
       score = self._calculate_detection_score(watermark_signal)
       
       # Return detection result
       return {
           'detected': score > self.detection_threshold,
           'score': float(score),
           'threshold': self.detection_threshold,
           'confidence': float(score) / self.detection_threshold
       }

get_data_for_visualize
~~~~~~~~~~~~~~~~~~~~~~

Provides data for visualization.

.. code-block:: python

   def get_data_for_visualize(self, media):
       """
       Get visualization data.
       
       Args:
           media: Generated watermarked media
           
       Returns:
           dict: Data for visualization
       """
       return {
           'original_image': media,
           'watermark_pattern': self._get_watermark_pattern(),
           'latents': self._invert_image(media),
           'fft_spectrum': self._compute_fft(media),
           'detection_score': self.detect_watermark_in_media(media)['score'],
           # Add any other data needed for visualization
       }

Example: Simple Frequency Domain Watermark
-------------------------------------------

Here's a complete example of a simple frequency domain watermarking algorithm:

.. code-block:: python

   import torch
   import torch.fft as fft
   import numpy as np
   from PIL import Image
   from markdiffusion.watermark.base import BaseWatermark

   class FrequencyWatermark(BaseWatermark):
       """Simple frequency domain watermarking."""
       
       def __init__(self, algorithm_config, diffusion_config):
           super().__init__(algorithm_config, diffusion_config)
           self.strength = algorithm_config.get('strength', 0.1)
           self.radius = algorithm_config.get('radius', 10)
           self.key = algorithm_config.get('key', 0)
           self.threshold = algorithm_config.get('threshold', 0.5)
           
           # Generate watermark pattern
           torch.manual_seed(self.key)
           self.watermark_pattern = torch.randn(1, 4, 64, 64)
       
       def _embed_watermark_in_latents(self, latents):
           """Embed watermark in frequency domain."""
           # Apply FFT
           latents_fft = fft.fft2(latents)
           
           # Create ring mask
           h, w = latents.shape[-2:]
           y, x = torch.meshgrid(
               torch.arange(h) - h//2,
               torch.arange(x) - w//2,
               indexing='ij'
           )
           distance = torch.sqrt(x**2 + y**2)
           ring_mask = ((distance > self.radius - 2) & 
                       (distance < self.radius + 2)).float()
           
           # Embed watermark in frequency domain
           watermark_fft = fft.fft2(self.watermark_pattern.to(latents.device))
           latents_fft = latents_fft + self.strength * watermark_fft * ring_mask
           
           # Inverse FFT
           watermarked_latents = fft.ifft2(latents_fft).real
           return watermarked_latents
       
       def generate_watermarked_media(self, input_data, **kwargs):
           """Generate watermarked image."""
           pipe = self.diffusion_config.pipe
           
           # Generate initial noise
           generator = torch.Generator(device=pipe.device)
           generator.manual_seed(self.diffusion_config.gen_seed)
           
           latents = torch.randn(
               (1, 4, 64, 64),
               generator=generator,
               device=pipe.device,
               dtype=pipe.dtype
           )
           
           # Embed watermark
           latents = self._embed_watermark_in_latents(latents)
           
           # Generate image
           with torch.no_grad():
               output = pipe(
                   prompt=input_data,
                   latents=latents,
                   num_inference_steps=self.diffusion_config.num_inference_steps,
                   guidance_scale=self.diffusion_config.guidance_scale,
                   **kwargs
               )
           
           return output.images[0]
       
       def detect_watermark_in_media(self, media, **kwargs):
           """Detect watermark in image."""
           # Invert image to latents
           from markdiffusion.inversions.ddim_inversion import DDIMInversion
           
           inversion = DDIMInversion(self.diffusion_config)
           latents = inversion.invert(media)
           
           # Extract frequency domain
           latents_fft = fft.fft2(latents)
           watermark_fft = fft.fft2(self.watermark_pattern.to(latents.device))
           
           # Calculate correlation in frequency domain
           correlation = torch.sum(latents_fft * torch.conj(watermark_fft)).real
           score = torch.sigmoid(correlation / 1000).item()
           
           return {
               'detected': score > self.threshold,
               'score': score,
               'threshold': self.threshold
           }
       
       def get_data_for_visualize(self, media):
           """Get visualization data."""
           from markdiffusion.inversions.ddim_inversion import DDIMInversion
           
           inversion = DDIMInversion(self.diffusion_config)
           latents = inversion.invert(media)
           latents_fft = fft.fft2(latents)
           
           return {
               'original_image': media,
               'latents': latents,
               'latents_fft': torch.abs(latents_fft),
               'watermark_pattern': self.watermark_pattern,
               'watermark_fft': torch.abs(fft.fft2(self.watermark_pattern)),
               'detection_result': self.detect_watermark_in_media(media)
           }

Configuration File
------------------

Create a JSON configuration file for your algorithm:

``config/FrequencyWatermark.json``:

.. code-block:: json

   {
     "algorithm_name": "FrequencyWatermark",
     "strength": 0.1,
     "radius": 10,
     "key": 42,
     "threshold": 0.5,
     "description": "Simple frequency domain watermarking"
   }

Register Algorithm
------------------

Register your algorithm in ``watermark/auto_watermark.py``:

.. code-block:: python

   from markdiffusion.watermark.frequency.frequency_watermark import FrequencyWatermark

   class AutoWatermark:
       ALGORITHM_MAP = {
           'TR': TreeRingWatermark,
           'GS': GaussianShadingWatermark,
           # ... existing algorithms
           'FW': FrequencyWatermark,  # Add your algorithm
       }

Custom Visualizer
-----------------

Create a custom visualizer for your algorithm:

.. code-block:: python

   from markdiffusion.visualize.base import BaseVisualizer
   import matplotlib.pyplot as plt

   class FrequencyWatermarkVisualizer(BaseVisualizer):
       """Visualizer for FrequencyWatermark."""
       
       def __init__(self, data_for_visualization):
           super().__init__(data_for_visualization)
       
       def draw_latents(self, ax=None):
           """Draw latent representation."""
           if ax is None:
               fig, ax = plt.subplots(figsize=(6, 6))
           
           latents = self.data['latents'][0, 0].cpu().numpy()
           im = ax.imshow(latents, cmap='viridis')
           ax.set_title('Latent Representation')
           ax.axis('off')
           plt.colorbar(im, ax=ax)
           
           return ax.figure if ax.figure else fig
       
       def draw_frequency_spectrum(self, ax=None):
           """Draw frequency spectrum."""
           if ax is None:
               fig, ax = plt.subplots(figsize=(6, 6))
           
           fft_data = self.data['latents_fft'][0, 0].cpu().numpy()
           fft_shifted = np.fft.fftshift(np.log(fft_data + 1))
           
           im = ax.imshow(fft_shifted, cmap='hot')
           ax.set_title('Frequency Spectrum')
           ax.axis('off')
           plt.colorbar(im, ax=ax)
           
           return ax.figure if ax.figure else fig
       
       def draw_watermark_pattern(self, ax=None):
           """Draw watermark pattern."""
           if ax is None:
               fig, ax = plt.subplots(figsize=(6, 6))
           
           pattern = self.data['watermark_pattern'][0, 0].cpu().numpy()
           im = ax.imshow(pattern, cmap='RdBu_r')
           ax.set_title('Watermark Pattern')
           ax.axis('off')
           plt.colorbar(im, ax=ax)
           
           return ax.figure if ax.figure else fig
       
       def visualize(self, rows=1, cols=3, methods=None, **kwargs):
           """Create comprehensive visualization."""
           if methods is None:
               methods = [
                   'draw_latents',
                   'draw_frequency_spectrum',
                   'draw_watermark_pattern'
               ]
           
           fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
           axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
           
           for i, method_name in enumerate(methods):
               if i < len(axes):
                   method = getattr(self, method_name)
                   method(ax=axes[i])
           
           plt.tight_layout()
           return fig

Register Visualizer
-------------------

Register your visualizer in ``visualize/auto_visualization.py``:

.. code-block:: python

   from markdiffusion.visualize.frequency.frequency_visualizer import FrequencyWatermarkVisualizer

   class AutoVisualizer:
       VISUALIZER_MAP = {
           'TR': TreeRingVisualizer,
           'GS': GaussianShadingVisualizer,
           # ... existing visualizers
           'FW': FrequencyWatermarkVisualizer,  # Add your visualizer
       }

Testing Your Algorithm
----------------------

Create tests for your algorithm:

.. code-block:: python

   import unittest
   from markdiffusion.watermark.auto_watermark import AutoWatermark
   from markdiffusion.utils.diffusion_config import DiffusionConfig

   class TestFrequencyWatermark(unittest.TestCase):
       """Test cases for FrequencyWatermark."""
       
       def setUp(self):
           """Set up test fixtures."""
           # Initialize diffusion config
           self.diffusion_config = DiffusionConfig(...)
           
           # Load algorithm
           self.watermark = AutoWatermark.load(
               'FW',
               algorithm_config='config/FrequencyWatermark.json',
               diffusion_config=self.diffusion_config
           )
       
       def test_generation(self):
           """Test watermarked image generation."""
           prompt = "A test image"
           image = self.watermark.generate_watermarked_media(prompt)
           
           self.assertIsNotNone(image)
           self.assertEqual(image.size, (512, 512))
       
       def test_detection(self):
           """Test watermark detection."""
           prompt = "A test image"
           image = self.watermark.generate_watermarked_media(prompt)
           result = self.watermark.detect_watermark_in_media(image)
           
           self.assertTrue(result['detected'])
           self.assertGreater(result['score'], self.watermark.threshold)
       
       def test_robustness(self):
           """Test robustness to attacks."""
           from markdiffusion.evaluation.tools.image_editor import JPEGCompression
           
           prompt = "A test image"
           image = self.watermark.generate_watermarked_media(prompt)
           
           # Apply JPEG compression
           editor = JPEGCompression(quality=75)
           compressed = editor.edit_image(image)
           
           # Detect after attack
           result = self.watermark.detect_watermark_in_media(compressed)
           self.assertTrue(result['detected'])

   if __name__ == '__main__':
       unittest.main()

Best Practices
--------------

1. **Modularity**: Keep algorithm logic separate from diffusion pipeline
2. **Configuration**: Use JSON configs for all hyperparameters
3. **Documentation**: Add docstrings and comments
4. **Testing**: Write comprehensive tests
5. **Visualization**: Provide meaningful visualizations
6. **Efficiency**: Optimize for GPU when possible
7. **Robustness**: Test against common attacks

Advanced Topics
---------------

Custom Inversion
~~~~~~~~~~~~~~~~

Implement custom latent inversion:

.. code-block:: python

   from markdiffusion.inversions.base_inversion import BaseInversion

   class CustomInversion(BaseInversion):
       def invert(self, image):
           """Custom inversion logic."""
           # Implement your inversion method
           pass

Training Custom Models
~~~~~~~~~~~~~~~~~~~~~~

If your algorithm requires trained models:

.. code-block:: python

   def train_watermark_model(config):
       """Train watermark embedding model."""
       # Set up training
       model = WatermarkEncoder()
       optimizer = torch.optim.Adam(model.parameters())
       
       # Training loop
       for epoch in range(config['num_epochs']):
           for batch in dataloader:
               loss = compute_loss(model, batch)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
       
       # Save model
       torch.save(model.state_dict(), 'ckpts/my_watermark.pth')

Contributing
------------

Once your algorithm is working:

1. Fork the MarkDiffusion repository
2. Create a feature branch
3. Add your algorithm with tests and documentation
4. Submit a pull request

See :doc:`../contributing` for detailed contribution guidelines.

Next Steps
----------

- :doc:`evaluation_pipelines` - Create custom evaluation pipelines
- :doc:`configuration` - Advanced configuration options
- :doc:`../api/watermark` - Watermark API reference
- :doc:`../contributing` - Contribution guidelines

