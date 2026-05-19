Watermarking Workflow
=====================

This guide explains the complete workflow for watermarking images and videos with MarkDiffusion.

Basic Workflow
--------------

The watermarking process consists of three main stages:

1. **Configuration** - Set up the diffusion model and watermarking algorithm
2. **Generation** - Generate watermarked media
3. **Detection** - Detect and verify watermarks

Configuration
-------------

Diffusion Model Setup
~~~~~~~~~~~~~~~~~~~~~

First, configure your diffusion model:

.. code-block:: python

   import torch
   from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
   from markdiffusion.utils.diffusion_config import DiffusionConfig

   # Device selection
   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Load model
   model_id = "stabilityai/stable-diffusion-2-1"
   scheduler = DPMSolverMultistepScheduler.from_pretrained(
       model_id, 
       subfolder="scheduler"
   )
   pipe = StableDiffusionPipeline.from_pretrained(
       model_id, 
       scheduler=scheduler
   ).to(device)

   # Create configuration
   diffusion_config = DiffusionConfig(
       scheduler=scheduler,
       pipe=pipe,
       device=device,
       image_size=(512, 512),
       num_inference_steps=50,
       guidance_scale=7.5,
       gen_seed=42,
       inversion_type="ddim"
   )

DiffusionConfig Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - scheduler
     - Scheduler
     - Diffusion scheduler (e.g., DDPM, DDIM, DPM)
   * - pipe
     - Pipeline
     - Diffusion pipeline object
   * - device
     - str
     - Device to run on ('cuda' or 'cpu')
   * - image_size
     - tuple
     - Output image size (height, width)
   * - num_inference_steps
     - int
     - Number of denoising steps
   * - guidance_scale
     - float
     - Classifier-free guidance scale
   * - gen_seed
     - int
     - Random seed for reproducibility
   * - inversion_type
     - str
     - Type of latent inversion ('ddim' or 'exact')

Algorithm Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Each watermarking algorithm has its own configuration file:

.. code-block:: python

   from markdiffusion.watermark.auto_watermark import AutoWatermark

   # Load watermark with configuration
   watermark = AutoWatermark.load(
       'GS',  # Algorithm name
       algorithm_config='markdiffusion/config/GS.json',  # Config file path
       diffusion_config=diffusion_config
   )

Configuration File Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example ``GS.json`` configuration:

.. code-block:: json

   {
     "algorithm_name": "GS",
     "secret_key": 42,
     "message_length": 256,
     "embed_dim": 4,
     "watermark_strength": 1.0,
     "detection_threshold": 0.5
   }

You can customize these parameters based on your requirements.

Generation
----------

Image Generation
~~~~~~~~~~~~~~~~

Basic image generation with watermark:

.. code-block:: python

   # Single image generation
   prompt = "A beautiful landscape with mountains"
   watermarked_image = watermark.generate_watermarked_media(prompt)
   
   # Save image
   watermarked_image.save("output.png")
   
   # Display image
   watermarked_image.show()

Video Generation
~~~~~~~~~~~~~~~~

For video watermarking:

.. code-block:: python

   from diffusers import DiffusionPipeline

   # Setup video pipeline
   video_pipe = DiffusionPipeline.from_pretrained(
       "cerspense/zeroscope_v2_576w",
       torch_dtype=torch.float16
   ).to(device)

   video_config = DiffusionConfig(
       scheduler=video_pipe.scheduler,
       pipe=video_pipe,
       device=device,
       image_size=(576, 320),
       num_inference_steps=40,
       guidance_scale=7.5,
       gen_seed=42,
       num_frames=16
   )

   # Load video watermark
   video_watermark = AutoWatermark.load(
       'VideoShield',
       algorithm_config='config/VideoShield.json',
       diffusion_config=video_config
   )

   # Generate watermarked video
   prompt = "A cat walking in a garden"
   frames = video_watermark.generate_watermarked_media(prompt)

   # Save frames
   import os
   os.makedirs("video_output", exist_ok=True)
   for i, frame in enumerate(frames):
       frame.save(f"video_output/frame_{i:04d}.png")

Detection
---------

Basic Detection
~~~~~~~~~~~~~~~

Detect watermark in a generated image:

.. code-block:: python

   # Detect watermark
   detection_result = watermark.detect_watermark_in_media(watermarked_image)
   
   print(f"Detection result: {detection_result}")

Batch Detection
~~~~~~~~~~~~~~~

Detect watermarks in multiple images:

.. code-block:: python

   import os
   from PIL import Image

   # Load images
   image_dir = "watermarked_images"
   results = {}

   for filename in os.listdir(image_dir):
       if filename.endswith(('.png', '.jpg', '.jpeg')):
           img_path = os.path.join(image_dir, filename)
           img = Image.open(img_path)
           
           # Detect watermark
           result = watermark.detect_watermark_in_media(img)
           results[filename] = result

   # Print results
   for filename, result in results.items():
       print(f"{filename}: {result}")

Detection with Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some algorithms support custom detection parameters:

.. code-block:: python

   detection_result = watermark.detect_watermark_in_media(
       watermarked_image,
       num_inference_steps=50,
       guidance_scale=1.0,
       detection_threshold=0.6  # Custom threshold
   )

Video Detection
~~~~~~~~~~~~~~~

Detect watermarks in video frames:

.. code-block:: python

   # Detect in all frames
   detection_result = video_watermark.detect_watermark_in_media(frames)
   
   # Frame-by-frame detection
   frame_results = []
   for i, frame in enumerate(frames):
       result = video_watermark.detect_watermark_in_media(frame)
       frame_results.append(result)
       print(f"Frame {i}: {result}")

Watermark Removal Prevention
-----------------------------

Testing Against Attacks
~~~~~~~~~~~~~~~~~~~~~~~

Verify watermark persistence after attacks:

.. code-block:: python

   from markdiffusion.evaluation.tools.image_editor import (
       JPEGCompression, GaussianBlurring, Rotation
   )

   # Original detection
   original_result = watermark.detect_watermark_in_media(watermarked_image)
   print(f"Original: {original_result}")

   # After JPEG compression
   jpeg_editor = JPEGCompression(quality=75)
   compressed_image = jpeg_editor.edit_image(watermarked_image)
   jpeg_result = watermark.detect_watermark_in_media(compressed_image)
   print(f"After JPEG: {jpeg_result}")

   # After blur
   blur_editor = GaussianBlurring(kernel_size=3)
   blurred_image = blur_editor.edit_image(watermarked_image)
   blur_result = watermark.detect_watermark_in_media(blurred_image)
   print(f"After blur: {blur_result}")

   # After rotation
   rotation_editor = Rotation(angle=15)
   rotated_image = rotation_editor.edit_image(watermarked_image)
   rotation_result = watermark.detect_watermark_in_media(rotated_image)
   print(f"After rotation: {rotation_result}")

Next Steps
----------

- :doc:`visualization` - Visualize watermarking mechanisms
- :doc:`evaluation` - Evaluate watermark quality and robustness
- :doc:`algorithms` - Learn about specific algorithms

