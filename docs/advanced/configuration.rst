Advanced Configuration
======================

This guide covers advanced configuration options for MarkDiffusion.

Diffusion Configuration
-----------------------

Complete DiffusionConfig Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.utils.diffusion_config import DiffusionConfig

   diffusion_config = DiffusionConfig(
       # Required parameters
       scheduler=scheduler,
       pipe=pipe,
       device='cuda',
       
       # Image/Video dimensions
       image_size=(512, 512),         # Output image size (H, W)
       num_frames=16,                  # For video generation
       
       # Generation parameters
       num_inference_steps=50,         # Denoising steps
       guidance_scale=7.5,             # CFG scale
       gen_seed=42,                    # Random seed
       
       # Inversion parameters
       inversion_type="ddim",          # 'ddim' or 'exact'
       num_inversion_steps=50,         # Inversion steps
       
       # Advanced options
       eta=0.0,                        # DDIM eta parameter
       use_fp16=True,                  # Use half precision
       enable_xformers=True,           # Use xformers optimization
   )

Scheduler Options
~~~~~~~~~~~~~~~~~

Different schedulers offer different trade-offs:

.. code-block:: python

   from diffusers import (
       DDPMScheduler,
       DDIMScheduler,
       PNDMScheduler,
       LMSDiscreteScheduler,
       DPMSolverMultistepScheduler,
       EulerDiscreteScheduler,
       EulerAncestralDiscreteScheduler,
   )

   # DDPM - Original diffusion scheduler
   scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

   # DDIM - Faster, deterministic
   scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

   # DPM-Solver - Fast, high quality
   scheduler = DPMSolverMultistepScheduler.from_pretrained(
       model_id, 
       subfolder="scheduler",
       algorithm_type="dpmsolver++",
       solver_order=2
   )

   # Euler - Good for artistic generation
   scheduler = EulerDiscreteScheduler.from_pretrained(
       model_id,
       subfolder="scheduler"
   )

Algorithm Configuration
-----------------------

Tree-Ring Configuration
~~~~~~~~~~~~~~~~~~~~~~~

``markdiffusion/config/TR.json``:

.. code-block:: json

   {
     "algorithm_name": "TR",
     "key": 42,
     "w_radius": 10,
     "w_channel": 0,
     "w_strength": 2.0,
     "detection_threshold": 0.5,
     "fft_type": "2d",
     "ring_width": 4
   }

**Parameters:**

- ``key``: Watermark key (seed for pattern generation)
- ``w_radius``: Ring radius in frequency domain
- ``w_channel``: Channel index to embed watermark (0-3)
- ``w_strength``: Watermark embedding strength
- ``detection_threshold``: Detection score threshold
- ``fft_type``: FFT type ('2d' or '1d')
- ``ring_width``: Width of the ring pattern

Gaussian-Shading Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``markdiffusion/config/GS.json``:

.. code-block:: json

   {
     "algorithm_name": "GS",
     "secret_key": 42,
     "message_length": 256,
     "embed_dim": 4,
     "watermark_strength": 1.0,
     "detection_threshold": 0.5,
     "use_adaptive_strength": false,
     "adaptive_strength_params": {
       "min_strength": 0.5,
       "max_strength": 2.0,
       "sensitivity": 0.1
     }
   }

**Parameters:**

- ``secret_key``: Secret key for watermark generation
- ``message_length``: Length of watermark message in bits
- ``embed_dim``: Embedding dimension (latent channels)
- ``watermark_strength``: Base embedding strength
- ``detection_threshold``: Detection score threshold
- ``use_adaptive_strength``: Enable adaptive strength based on content
- ``adaptive_strength_params``: Parameters for adaptive strength

ROBIN Configuration
~~~~~~~~~~~~~~~~~~~

``markdiffusion/config/ROBIN.json``:

.. code-block:: json

   {
     "algorithm_name": "ROBIN",
     "generator_path": "ckpts/robin/watermark_generator.pth",
     "watermark_message": "10101010...",
     "embedding_strength": 1.0,
     "adversarial_weight": 0.1,
     "detection_threshold": 0.5,
     "use_adversarial_training": true,
     "adversarial_attacks": ["jpeg", "blur", "noise"],
     "attack_params": {
       "jpeg_quality": 75,
       "blur_kernel": 3,
       "noise_std": 0.05
     }
   }

Video Watermark Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``config/VideoShield.json``:

.. code-block:: json

   {
     "algorithm_name": "VideoShield",
     "model_path": "ckpts/videoshield/model.pth",
     "watermark_key": 42,
     "temporal_consistency_weight": 0.5,
     "frame_embed_interval": 1,
     "use_optical_flow": true,
     "detection_threshold": 0.5,
     "batch_size": 4
   }

Environment Variables
---------------------

Set environment variables for configuration:

.. code-block:: bash

   # CUDA settings
   export CUDA_VISIBLE_DEVICES=0,1
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

   # Model cache directory
   export HF_HOME=/path/to/huggingface/cache
   export TORCH_HOME=/path/to/torch/cache

   # Optimization flags
   export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Mac M1/M2

In Python:

.. code-block:: python

   import os

   # Set environment variables
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'
   os.environ['TOKENIZERS_PARALLELISM'] = 'false'

Memory Optimization
-------------------

Reduce Memory Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable CPU offloading
   pipe.enable_model_cpu_offload()

   # Enable sequential CPU offloading (more aggressive)
   pipe.enable_sequential_cpu_offload()

   # Enable attention slicing
   pipe.enable_attention_slicing(slice_size="auto")

   # Enable VAE slicing
   pipe.enable_vae_slicing()

   # Use fp16
   pipe = StableDiffusionPipeline.from_pretrained(
       model_id,
       torch_dtype=torch.float16,
       variant="fp16"
   )

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   for batch in dataloader:
       optimizer.zero_grad()
       
       with autocast():
           loss = compute_loss(model, batch)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()

Speed Optimization
------------------

Enable xFormers
~~~~~~~~~~~~~~~

.. code-block:: python

   # Install xformers
   # pip install xformers

   # Enable memory efficient attention
   pipe.enable_xformers_memory_efficient_attention()

Compile Model (PyTorch 2.0+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch

   # Compile model for faster inference
   pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple prompts in parallel
   prompts = ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]
   
   with torch.no_grad():
       images = pipe(
           prompt=prompts,
           num_inference_steps=50,
           guidance_scale=7.5
       ).images

Multi-GPU Configuration
-----------------------

Data Parallel
~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn

   # Wrap model in DataParallel
   if torch.cuda.device_count() > 1:
       pipe.unet = nn.DataParallel(pipe.unet)
       pipe.vae = nn.DataParallel(pipe.vae)

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP

   # Initialize process group
   dist.init_process_group(backend='nccl')
   local_rank = int(os.environ['LOCAL_RANK'])
   
   # Move model to GPU
   model = model.to(local_rank)
   
   # Wrap in DDP
   model = DDP(model, device_ids=[local_rank])

Custom Model Loading
--------------------

Load Custom Checkpoint
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from diffusers import StableDiffusionPipeline

   # Load from local checkpoint
   pipe = StableDiffusionPipeline.from_pretrained(
       "/path/to/local/checkpoint",
       torch_dtype=torch.float16,
       safety_checker=None,  # Disable safety checker
       requires_safety_checker=False
   )

   # Load specific components
   from diffusers import UNet2DConditionModel

   unet = UNet2DConditionModel.from_pretrained(
       "/path/to/checkpoint",
       subfolder="unet"
   )
   pipe.unet = unet

LoRA Support
~~~~~~~~~~~~

.. code-block:: python

   # Load LoRA weights
   pipe.load_lora_weights("/path/to/lora/weights")

   # Adjust LoRA scale
   pipe.set_lora_scale(0.8)

   # Unload LoRA
   pipe.unload_lora_weights()

Logging Configuration
---------------------

Set Up Logging
~~~~~~~~~~~~~~

.. code-block:: python

   import logging

   # Configure logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('markdiffusion.log'),
           logging.StreamHandler()
       ]
   )

   logger = logging.getLogger(__name__)

   # Use logger
   logger.info("Starting watermark generation")
   logger.warning("Low detection score")
   logger.error("Generation failed")

Weights & Biases Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import wandb

   # Initialize W&B
   wandb.init(
       project="markdiffusion",
       config={
           "algorithm": "GS",
           "num_samples": 1000,
           "num_inference_steps": 50
       }
   )

   # Log metrics
   wandb.log({
       "detection_rate": 0.95,
       "avg_psnr": 42.5,
       "robustness_jpeg": 0.87
   })

   # Log images
   wandb.log({"watermarked_image": wandb.Image(image)})

   # Finish run
   wandb.finish()

Configuration Files
-------------------

YAML Configuration
~~~~~~~~~~~~~~~~~~

Create a comprehensive config file:

``config/experiment.yaml``:

.. code-block:: yaml

   experiment:
     name: "gs_evaluation"
     seed: 42
     device: "cuda"

   model:
     name: "stabilityai/stable-diffusion-2-1"
     scheduler: "DPMSolverMultistep"
     dtype: "float16"

   watermark:
     algorithm: "GS"
     config_file: "markdiffusion/config/GS.json"
     parameters:
       watermark_strength: 1.0
       detection_threshold: 0.5

   generation:
     num_inference_steps: 50
     guidance_scale: 7.5
     image_size: [512, 512]

   evaluation:
     dataset: "StableDiffusionPrompts"
     num_samples: 1000
     attacks:
       - name: "JPEG"
         quality: 75
       - name: "Blur"
         kernel_size: 3
     metrics:
       - "PSNR"
       - "SSIM"
       - "LPIPS"

Load YAML Config
~~~~~~~~~~~~~~~~

.. code-block:: python

   import yaml

   def load_config(config_path):
       with open(config_path, 'r') as f:
           config = yaml.safe_load(f)
       return config

   # Load and use config
   config = load_config('config/experiment.yaml')

   # Access config values
   algorithm_name = config['watermark']['algorithm']
   num_samples = config['evaluation']['num_samples']

Command-Line Arguments
----------------------

Using argparse
~~~~~~~~~~~~~~

.. code-block:: python

   import argparse

   def parse_args():
       parser = argparse.ArgumentParser(
           description="MarkDiffusion Evaluation"
       )
       
       # Model arguments
       parser.add_argument('--model_id', type=str, 
                          default='stabilityai/stable-diffusion-2-1')
       parser.add_argument('--device', type=str, default='cuda')
       
       # Watermark arguments
       parser.add_argument('--algorithm', type=str, required=True,
                          choices=['TR', 'GS', 'ROBIN', 'SEAL'])
       parser.add_argument('--config', type=str, required=True)
       
       # Generation arguments
       parser.add_argument('--num_inference_steps', type=int, default=50)
       parser.add_argument('--guidance_scale', type=float, default=7.5)
       parser.add_argument('--seed', type=int, default=42)
       
       # Evaluation arguments
       parser.add_argument('--num_samples', type=int, default=100)
       parser.add_argument('--output_dir', type=str, default='outputs')
       
       return parser.parse_args()

   # Use arguments
   args = parse_args()
   print(f"Using algorithm: {args.algorithm}")

Best Practices
--------------

1. **Version Control**: Keep config files in version control
2. **Documentation**: Document all config parameters
3. **Validation**: Validate config values before use
4. **Defaults**: Provide sensible default values
5. **Reproducibility**: Save full config with results
6. **Modularity**: Separate configs for different components

Example Complete Setup
----------------------

.. code-block:: python

   import torch
   import yaml
   import logging
   from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
   from markdiffusion.watermark.auto_watermark import AutoWatermark
   from markdiffusion.utils.diffusion_config import DiffusionConfig

   def setup_experiment(config_path):
       """Set up complete experiment from config."""
       
       # Load config
       with open(config_path) as f:
           config = yaml.safe_load(f)
       
       # Set up logging
       logging.basicConfig(level=logging.INFO)
       logger = logging.getLogger(__name__)
       
       # Set seed
       seed = config['experiment']['seed']
       torch.manual_seed(seed)
       
       # Load model
       device = config['experiment']['device']
       model_id = config['model']['name']
       
       scheduler = DPMSolverMultistepScheduler.from_pretrained(
           model_id, subfolder="scheduler"
       )
       
       pipe = StableDiffusionPipeline.from_pretrained(
           model_id,
           scheduler=scheduler,
           torch_dtype=torch.float16 if config['model']['dtype'] == 'float16' else torch.float32
       ).to(device)
       
       # Enable optimizations
       pipe.enable_xformers_memory_efficient_attention()
       pipe.enable_attention_slicing()
       
       # Create diffusion config
       diffusion_config = DiffusionConfig(
           scheduler=scheduler,
           pipe=pipe,
           device=device,
           image_size=tuple(config['generation']['image_size']),
           num_inference_steps=config['generation']['num_inference_steps'],
           guidance_scale=config['generation']['guidance_scale'],
           gen_seed=seed
       )
       
       # Load watermark
       watermark = AutoWatermark.load(
           config['watermark']['algorithm'],
           config['watermark']['config_file'],
           diffusion_config
       )
       
       logger.info(f"Experiment setup complete: {config['experiment']['name']}")
       
       return watermark, config

   # Use it
   watermark, config = setup_experiment('config/experiment.yaml')

Next Steps
----------

- :doc:`custom_algorithms` - Implement custom algorithms
- :doc:`evaluation_pipelines` - Create custom evaluation pipelines
- :doc:`../api/watermark` - API reference

