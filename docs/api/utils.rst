Configuration and Utilities
===========================

This page documents the configuration and utility APIs for MarkDiffusion.

DiffusionConfig
---------------

The ``DiffusionConfig`` class configures the diffusion model parameters for watermarking.

.. py:class:: utils.diffusion_config.DiffusionConfig

   Configuration class for diffusion model settings.
   
   :param scheduler: Diffusion scheduler (e.g., DPMSolverMultistepScheduler)
   :param pipe: Diffusion pipeline (e.g., StableDiffusionPipeline)
   :param device: Device to run on ('cuda' or 'cpu')
   :param image_size: Size of generated images (tuple, e.g., (512, 512))
   :param num_inference_steps: Number of denoising steps (default: 50)
   :param guidance_scale: Classifier-free guidance scale (default: 7.5)
   :param gen_seed: Random seed for generation (default: 42)
   :param inversion_type: Type of inversion ('ddim' or 'exact', default: 'ddim')
   :param num_frames: Number of frames for video (optional, for video watermarks)
   :param fps: Frames per second for video (optional, for video watermarks)

**Example Usage:**

.. code-block:: python

   from markdiffusion.utils.diffusion_config import DiffusionConfig
   from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
   import torch
   
   # Initialize diffusion components
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   scheduler = DPMSolverMultistepScheduler.from_pretrained(
       "model_path", subfolder="scheduler"
   )
   pipe = StableDiffusionPipeline.from_pretrained(
       "model_path", scheduler=scheduler
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

**For Video Watermarks:**

.. code-block:: python

   # Video configuration
   video_diffusion_config = DiffusionConfig(
       scheduler=scheduler,
       pipe=video_pipe,
       device=device,
       image_size=(512, 512),
       num_frames=16,
       fps=8,
       num_inference_steps=50,
       guidance_scale=7.5,
       gen_seed=42,
       inversion_type="ddim"
   )

.. note::
   Most parameters have sensible defaults. You primarily need to provide the scheduler, 
   pipeline, and device. Other parameters can be adjusted based on your specific requirements.
