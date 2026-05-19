Custom Evaluation Pipelines
============================

Learn how to create custom evaluation pipelines for your specific needs.

Pipeline Architecture
---------------------

MarkDiffusion's evaluation system consists of:

- **Pipelines**: Orchestrate the evaluation workflow
- **Datasets**: Provide test data
- **Editors**: Apply transformations/attacks
- **Analyzers**: Calculate metrics
- **Calculators**: Aggregate results

Creating Custom Datasets
-------------------------

Inherit from Base Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.evaluation.dataset import BaseDataset
   from PIL import Image
   import os

   class CustomDataset(BaseDataset):
       """Custom dataset for evaluation."""
       
       def __init__(self, data_dir, max_samples=None):
           super().__init__(max_samples=max_samples)
           self.data_dir = data_dir
           self._load_data()
       
       def _load_data(self):
           """Load data from directory."""
           self.prompts = []
           self.reference_images = []
           
           for filename in os.listdir(self.data_dir):
               if filename.endswith('.txt'):
                   # Load prompt
                   with open(os.path.join(self.data_dir, filename)) as f:
                       self.prompts.append(f.read().strip())
                   
                   # Load corresponding image
                   img_file = filename.replace('.txt', '.png')
                   img_path = os.path.join(self.data_dir, img_file)
                   if os.path.exists(img_path):
                       self.reference_images.append(Image.open(img_path))
           
           # Limit samples if specified
           if self.max_samples:
               self.prompts = self.prompts[:self.max_samples]
               self.reference_images = self.reference_images[:self.max_samples]
       
       def __len__(self):
           return len(self.prompts)
       
       def __getitem__(self, idx):
           return {
               'prompt': self.prompts[idx],
               'reference_image': self.reference_images[idx] if idx < len(self.reference_images) else None
           }

Custom Image Editors/Attacks
-----------------------------

Create Custom Attack
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.evaluation.tools.image_editor import BaseImageEditor
   import torch
   import torch.nn.functional as F

   class CustomAttack(BaseImageEditor):
       """Custom image attack."""
       
       def __init__(self, attack_strength=0.1, **kwargs):
           super().__init__(**kwargs)
           self.attack_strength = attack_strength
       
       def edit_image(self, image):
           """Apply custom attack to image."""
           # Convert PIL to tensor
           img_tensor = self._pil_to_tensor(image)
           
           # Apply your custom transformation
           # Example: Add adversarial noise
           noise = torch.randn_like(img_tensor) * self.attack_strength
           attacked = torch.clamp(img_tensor + noise, 0, 1)
           
           # Convert back to PIL
           attacked_image = self._tensor_to_pil(attacked)
           return attacked_image
       
       @staticmethod
       def _pil_to_tensor(image):
           """Convert PIL image to tensor."""
           import torchvision.transforms as transforms
           transform = transforms.ToTensor()
           return transform(image)
       
       @staticmethod
       def _tensor_to_pil(tensor):
           """Convert tensor to PIL image."""
           import torchvision.transforms as transforms
           transform = transforms.ToPILImage()
           return transform(tensor)

Advanced Attack Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class AdversarialAttack(BaseImageEditor):
       """Adversarial attack against watermark detection."""
       
       def __init__(self, watermark, epsilon=0.1, num_steps=10, **kwargs):
           super().__init__(**kwargs)
           self.watermark = watermark
           self.epsilon = epsilon
           self.num_steps = num_steps
       
       def edit_image(self, image):
           """Generate adversarial example."""
           img_tensor = self._pil_to_tensor(image).unsqueeze(0)
           img_tensor.requires_grad = True
           
           # Gradient-based attack
           for _ in range(self.num_steps):
               # Forward pass
               detection_result = self.watermark.detect_watermark_in_media(
                   self._tensor_to_pil(img_tensor.squeeze(0))
               )
               score = detection_result['score']
               
               # Backward pass (minimize detection score)
               score_tensor = torch.tensor(score, requires_grad=True)
               score_tensor.backward()
               
               # Update image
               with torch.no_grad():
                   img_tensor -= self.epsilon * img_tensor.grad.sign()
                   img_tensor = torch.clamp(img_tensor, 0, 1)
                   img_tensor.grad.zero_()
           
           return self._tensor_to_pil(img_tensor.squeeze(0))

Custom Analyzers
----------------

Create Quality Analyzer
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.evaluation.tools.image_quality_analyzer import BaseImageQualityAnalyzer
   import torch
   import torch.nn as nn

   class CustomQualityAnalyzer(BaseImageQualityAnalyzer):
       """Custom image quality analyzer."""
       
       def __init__(self, device='cuda'):
           super().__init__()
           self.device = device
           # Initialize any models if needed
           self.model = self._load_model()
       
       def _load_model(self):
           """Load quality assessment model."""
           # Load pre-trained model or define your own
           model = YourQualityModel()
           model.eval()
           model.to(self.device)
           return model
       
       def analyze(self, image1, image2=None):
           """
           Analyze image quality.
           
           Args:
               image1: Primary image
               image2: Reference image (optional)
           
           Returns:
               float: Quality score
           """
           # Convert to tensor
           img1_tensor = self._prepare_image(image1)
           
           with torch.no_grad():
               if image2 is None:
                   # No-reference quality assessment
                   score = self.model(img1_tensor)
               else:
                   # Full-reference quality assessment
                   img2_tensor = self._prepare_image(image2)
                   score = self.model(img1_tensor, img2_tensor)
           
           return float(score.item())
       
       def _prepare_image(self, image):
           """Prepare image for model input."""
           import torchvision.transforms as transforms
           
           transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
           ])
           
           img_tensor = transform(image).unsqueeze(0).to(self.device)
           return img_tensor

Perceptual Quality Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class PerceptualQualityAnalyzer(BaseImageQualityAnalyzer):
       """Perceptual quality based on deep features."""
       
       def __init__(self, device='cuda'):
           super().__init__()
           self.device = device
           # Use pre-trained VGG for perceptual features
           from torchvision.models import vgg16
           self.vgg = vgg16(pretrained=True).features[:23].eval().to(device)
           
       def analyze(self, image1, image2):
           """Calculate perceptual distance."""
           # Extract features
           feat1 = self._extract_features(image1)
           feat2 = self._extract_features(image2)
           
           # Calculate distance
           distance = torch.nn.functional.mse_loss(feat1, feat2)
           return float(distance.item())
       
       def _extract_features(self, image):
           """Extract perceptual features."""
           import torchvision.transforms as transforms
           
           transform = transforms.Compose([
               transforms.Resize((224, 224)),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
           ])
           
           img_tensor = transform(image).unsqueeze(0).to(self.device)
           
           with torch.no_grad():
               features = self.vgg(img_tensor)
           
           return features

Custom Evaluation Pipeline
---------------------------

Basic Custom Pipeline
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.evaluation.pipelines.detection import BaseDetectionPipeline

   class CustomEvaluationPipeline:
       """Custom evaluation pipeline."""
       
       def __init__(self, dataset, attacks, analyzers, show_progress=True):
           self.dataset = dataset
           self.attacks = attacks
           self.analyzers = analyzers
           self.show_progress = show_progress
       
       def evaluate(self, watermark, **kwargs):
           """Run comprehensive evaluation."""
           from tqdm import tqdm
           
           results = {
               'generation': [],
               'detection': [],
               'robustness': {},
               'quality': {}
           }
           
           # Initialize robustness results
           for attack_name in self.attacks:
               results['robustness'][attack_name] = []
           
           # Iterate through dataset
           iterator = tqdm(self.dataset) if self.show_progress else self.dataset
           
           for data in iterator:
               prompt = data['prompt']
               ref_img = data.get('reference_image')
               
               # 1. Generate watermarked image
               wm_image = watermark.generate_watermarked_media(prompt)
               results['generation'].append(wm_image)
               
               # 2. Detect watermark
               detection = watermark.detect_watermark_in_media(wm_image)
               results['detection'].append(detection['score'])
               
               # 3. Test robustness
               for attack_name, attack in self.attacks.items():
                   attacked_image = attack.edit_image(wm_image)
                   detection_after = watermark.detect_watermark_in_media(attacked_image)
                   results['robustness'][attack_name].append(detection_after['score'])
               
               # 4. Analyze quality
               if ref_img:
                   for analyzer_name, analyzer in self.analyzers.items():
                       if analyzer_name not in results['quality']:
                           results['quality'][analyzer_name] = []
                       score = analyzer.analyze(wm_image, ref_img)
                       results['quality'][analyzer_name].append(score)
           
           # Aggregate results
           aggregated = self._aggregate_results(results)
           return aggregated
       
       def _aggregate_results(self, results):
           """Aggregate evaluation results."""
           import numpy as np
           
           aggregated = {
               'detection_rate': np.mean(results['detection']),
               'robustness': {},
               'quality': {}
           }
           
           # Aggregate robustness
           for attack_name, scores in results['robustness'].items():
               aggregated['robustness'][attack_name] = {
                   'mean': np.mean(scores),
                   'std': np.std(scores),
                   'min': np.min(scores),
                   'max': np.max(scores)
               }
           
           # Aggregate quality
           for metric_name, scores in results['quality'].items():
               aggregated['quality'][metric_name] = {
                   'mean': np.mean(scores),
                   'std': np.std(scores)
               }
           
           return aggregated

Advanced Pipeline with Caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pickle
   import os
   from pathlib import Path

   class CachedEvaluationPipeline:
       """Evaluation pipeline with result caching."""
       
       def __init__(self, dataset, attacks, analyzers, 
                    cache_dir='eval_cache', use_cache=True):
           self.dataset = dataset
           self.attacks = attacks
           self.analyzers = analyzers
           self.cache_dir = Path(cache_dir)
           self.use_cache = use_cache
           self.cache_dir.mkdir(exist_ok=True)
       
       def evaluate(self, watermark, **kwargs):
           """Run evaluation with caching."""
           # Generate cache key
           cache_key = self._generate_cache_key(watermark, kwargs)
           cache_file = self.cache_dir / f"{cache_key}.pkl"
           
           # Check cache
           if self.use_cache and cache_file.exists():
               print(f"Loading cached results from {cache_file}")
               with open(cache_file, 'rb') as f:
                   return pickle.load(f)
           
           # Run evaluation
           results = self._run_evaluation(watermark, **kwargs)
           
           # Save to cache
           with open(cache_file, 'wb') as f:
               pickle.dump(results, f)
           
           return results
       
       def _generate_cache_key(self, watermark, kwargs):
           """Generate unique cache key."""
           import hashlib
           
           # Create string representation
           key_str = f"{watermark.__class__.__name__}"
           key_str += f"_dataset{len(self.dataset)}"
           key_str += f"_attacks{list(self.attacks.keys())}"
           key_str += str(sorted(kwargs.items()))
           
           # Hash it
           return hashlib.md5(key_str.encode()).hexdigest()
       
       def _run_evaluation(self, watermark, **kwargs):
           """Actually run the evaluation."""
           # Implementation similar to CustomEvaluationPipeline
           pass

Batch Evaluation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BatchEvaluationPipeline:
       """Evaluate multiple algorithms in batch."""
       
       def __init__(self, algorithms, dataset, attacks, analyzers):
           self.algorithms = algorithms
           self.dataset = dataset
           self.attacks = attacks
           self.analyzers = analyzers
       
       def evaluate_all(self, **kwargs):
           """Evaluate all algorithms."""
           results = {}
           
           for algo_name, watermark in self.algorithms.items():
               print(f"\n{'='*50}")
               print(f"Evaluating {algo_name}")
               print('='*50)
               
               pipeline = CustomEvaluationPipeline(
                   self.dataset, self.attacks, self.analyzers
               )
               results[algo_name] = pipeline.evaluate(watermark, **kwargs)
           
           return results
       
       def compare_results(self, results):
           """Generate comparison report."""
           import pandas as pd
           
           # Create comparison dataframe
           comparison_data = []
           
           for algo_name, result in results.items():
               row = {
                   'Algorithm': algo_name,
                   'Detection_Rate': result['detection_rate'],
               }
               
               # Add robustness metrics
               for attack, metrics in result['robustness'].items():
                   row[f'Rob_{attack}'] = metrics['mean']
               
               # Add quality metrics
               for metric, values in result['quality'].items():
                   row[f'Quality_{metric}'] = values['mean']
               
               comparison_data.append(row)
           
           df = pd.DataFrame(comparison_data)
           return df

Progressive Evaluation
~~~~~~~~~~~~~~~~~~~~~~

For long-running evaluations with progress saving:

.. code-block:: python

   class ProgressiveEvaluationPipeline:
       """Evaluation with progress checkpointing."""
       
       def __init__(self, dataset, attacks, analyzers, 
                    checkpoint_dir='checkpoints', checkpoint_freq=10):
           self.dataset = dataset
           self.attacks = attacks
           self.analyzers = analyzers
           self.checkpoint_dir = Path(checkpoint_dir)
           self.checkpoint_freq = checkpoint_freq
           self.checkpoint_dir.mkdir(exist_ok=True)
       
       def evaluate(self, watermark, resume=True, **kwargs):
           """Run evaluation with checkpointing."""
           checkpoint_file = self.checkpoint_dir / "progress.pkl"
           
           # Load checkpoint if resuming
           if resume and checkpoint_file.exists():
               with open(checkpoint_file, 'rb') as f:
                   state = pickle.load(f)
               start_idx = state['current_idx']
               results = state['results']
               print(f"Resuming from sample {start_idx}")
           else:
               start_idx = 0
               results = self._initialize_results()
           
           # Run evaluation
           for idx in range(start_idx, len(self.dataset)):
               data = self.dataset[idx]
               
               # Process sample
               sample_result = self._process_sample(watermark, data, **kwargs)
               self._update_results(results, sample_result)
               
               # Save checkpoint
               if (idx + 1) % self.checkpoint_freq == 0:
                   state = {
                       'current_idx': idx + 1,
                       'results': results
                   }
                   with open(checkpoint_file, 'wb') as f:
                       pickle.dump(state, f)
                   print(f"Checkpoint saved at sample {idx + 1}")
           
           # Final results
           final_results = self._finalize_results(results)
           
           # Clean up checkpoint
           if checkpoint_file.exists():
               checkpoint_file.unlink()
           
           return final_results

Example Usage
-------------

Using Custom Pipeline
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.watermark.auto_watermark import AutoWatermark
   
   # Create custom dataset
   dataset = CustomDataset('my_data', max_samples=100)
   
   # Define attacks
   attacks = {
       'JPEG': JPEGCompression(quality=75),
       'Blur': GaussianBlurring(kernel_size=3),
       'Custom': CustomAttack(attack_strength=0.1)
   }
   
   # Define analyzers
   analyzers = {
       'PSNR': PSNRAnalyzer(),
       'CustomQuality': CustomQualityAnalyzer()
   }
   
   # Create pipeline
   pipeline = CustomEvaluationPipeline(
       dataset=dataset,
       attacks=attacks,
       analyzers=analyzers,
       show_progress=True
   )
   
   # Load watermark
   watermark = AutoWatermark.load('GS', 'markdiffusion/config/GS.json', diffusion_config)
   
   # Run evaluation
   results = pipeline.evaluate(watermark)
   
   # Print results
   print("\n=== Evaluation Results ===")
   print(f"Detection Rate: {results['detection_rate']:.4f}")
   print("\nRobustness:")
   for attack, metrics in results['robustness'].items():
       print(f"  {attack}: {metrics['mean']:.4f} ± {metrics['std']:.4f}")
   print("\nQuality:")
   for metric, values in results['quality'].items():
       print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")

Batch Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load multiple algorithms
   algorithms = {}
   for algo_name in ['TR', 'GS', 'ROBIN', 'SEAL']:
       algorithms[algo_name] = AutoWatermark.load(
           algo_name,
           f'config/{algo_name}.json',
           diffusion_config
       )
   
   # Batch evaluation
   batch_pipeline = BatchEvaluationPipeline(
       algorithms=algorithms,
       dataset=dataset,
       attacks=attacks,
       analyzers=analyzers
   )
   
   # Run evaluations
   all_results = batch_pipeline.evaluate_all()
   
   # Compare results
   comparison_df = batch_pipeline.compare_results(all_results)
   print("\n=== Algorithm Comparison ===")
   print(comparison_df.to_string(index=False))
   
   # Save to CSV
   comparison_df.to_csv('algorithm_comparison.csv', index=False)

Best Practices
--------------

1. **Modular Design**: Keep components independent and reusable
2. **Caching**: Cache expensive operations (generation, detection)
3. **Progress Tracking**: Use tqdm and checkpoints for long evaluations
4. **Error Handling**: Handle failures gracefully
5. **Logging**: Log detailed information for debugging
6. **Reproducibility**: Set seeds and document parameters
7. **Scalability**: Design for large-scale experiments

Next Steps
----------

- :doc:`custom_algorithms` - Implement custom watermarking algorithms
- :doc:`configuration` - Advanced configuration options
- :doc:`../api/evaluation` - Evaluation API reference

