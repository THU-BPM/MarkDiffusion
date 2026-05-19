Evaluation
==========

MarkDiffusion provides comprehensive evaluation tools to assess watermark performance across three key dimensions: detectability, robustness, and output quality.

Overview
--------

Evaluation Dimensions
~~~~~~~~~~~~~~~~~~~~~

1. **Detectability** - How reliably can watermarks be detected?
2. **Robustness** - How well do watermarks survive attacks?
3. **Quality** - How much do watermarks affect output quality?

Evaluation Components
~~~~~~~~~~~~~~~~~~~~~

- **Pipelines** - Automated evaluation workflows
- **Tools** - Individual evaluation metrics and attack methods
- **Analyzers** - Quality assessment modules
- **Calculators** - Detection performance metrics

Detectability Evaluation
------------------------

Basic Detection Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.evaluation.dataset import StableDiffusionPromptsDataset
   from markdiffusion.evaluation.pipelines.detection import (
       WatermarkedMediaDetectionPipeline,
       UnWatermarkedMediaDetectionPipeline,
       DetectionPipelineReturnType
   )
   from markdiffusion.evaluation.tools.success_rate_calculator import (
       DynamicThresholdSuccessRateCalculator
   )

   # Create dataset
   dataset = StableDiffusionPromptsDataset(max_samples=200)

   # Setup pipelines
   watermarked_pipeline = WatermarkedMediaDetectionPipeline(
       dataset=dataset,
       media_editor_list=[],  # No attacks
       show_progress=True,
       return_type=DetectionPipelineReturnType.SCORES
   )

   unwatermarked_pipeline = UnWatermarkedMediaDetectionPipeline(
       dataset=dataset,
       media_editor_list=[],
       show_progress=True,
       return_type=DetectionPipelineReturnType.SCORES
   )

   # Configure detection
   detection_kwargs = {
       "num_inference_steps": 50,
       "guidance_scale": 1.0,
   }

   # Evaluate
   watermarked_scores = watermarked_pipeline.evaluate(
       watermark, detection_kwargs=detection_kwargs
   )
   unwatermarked_scores = unwatermarked_pipeline.evaluate(
       watermark, detection_kwargs=detection_kwargs
   )

   # Calculate metrics
   calculator = DynamicThresholdSuccessRateCalculator(
       labels=['watermarked'] * len(watermarked_scores) + 
              ['unwatermarked'] * len(unwatermarked_scores),
       target_fpr=0.01  # Target false positive rate
   )

   results = calculator.calculate(watermarked_scores, unwatermarked_scores)
   print(f"TPR at 1% FPR: {results['tpr']:.4f}")
   print(f"AUC: {results['auc']:.4f}")

Detection Metrics
~~~~~~~~~~~~~~~~~

Available detection metrics:

- **TPR** (True Positive Rate) - Watermark detection rate
- **FPR** (False Positive Rate) - False alarm rate
- **TNR** (True Negative Rate) - Correct rejection rate
- **FNR** (False Negative Rate) - Miss rate
- **Accuracy** - Overall detection accuracy
- **Precision** - Positive predictive value
- **Recall** - Same as TPR
- **F1-Score** - Harmonic mean of precision and recall
- **AUC** (Area Under ROC Curve) - Overall performance

Fixed Threshold Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.evaluation.tools.success_rate_calculator import (
       FundamentalSuccessRateCalculator
   )

   # Use fixed threshold
   calculator = FundamentalSuccessRateCalculator(
       threshold=0.5  # Fixed detection threshold
   )

   results = calculator.calculate(watermarked_scores, unwatermarked_scores)
   print(f"Accuracy: {results['accuracy']:.4f}")
   print(f"F1-Score: {results['f1_score']:.4f}")

Robustness Evaluation
---------------------

Image Attacks
~~~~~~~~~~~~~

Test watermark robustness against various image attacks:

.. code-block:: python

   from markdiffusion.evaluation.tools.image_editor import (
       JPEGCompression,
       GaussianBlurring,
       GaussianNoise,
       Rotation,
       CrSc,  # Crop and Scale
       Brightness,
       Mask,
       Overlay,
       AdaptiveNoiseInjection
   )

   # Define attacks
   attacks = {
       'JPEG-90': JPEGCompression(quality=90),
       'JPEG-75': JPEGCompression(quality=75),
       'JPEG-50': JPEGCompression(quality=50),
       'Blur-3': GaussianBlurring(kernel_size=3),
       'Blur-5': GaussianBlurring(kernel_size=5),
       'Noise-0.01': GaussianNoise(std=0.01),
       'Noise-0.05': GaussianNoise(std=0.05),
       'Rotate-15': Rotation(angle=15),
       'Rotate-45': Rotation(angle=45),
       'CropScale-0.75': CrSc(crop_ratio=0.75),
       'Brightness-1.2': Brightness(factor=1.2),
       'Mask': Mask(num_masks=5, mask_size=50),
       'Overlay': Overlay(num_strokes=10),
       'AdaptiveNoise': AdaptiveNoiseInjection(noise_type='gaussian')
   }

   # Evaluate against each attack
   robustness_results = {}
   for attack_name, attack_editor in attacks.items():
       print(f"\nEvaluating: {attack_name}")
       
       pipeline = WatermarkedMediaDetectionPipeline(
           dataset=dataset,
           media_editor_list=[attack_editor],
           show_progress=True,
           return_type=DetectionPipelineReturnType.SCORES
       )
       
       scores = pipeline.evaluate(watermark, detection_kwargs=detection_kwargs)
       avg_score = sum(scores) / len(scores) if scores else 0
       robustness_results[attack_name] = avg_score

   # Print results
   print("\n=== Robustness Results ===")
   for attack, score in sorted(robustness_results.items(), 
                                key=lambda x: x[1], reverse=True):
       print(f"{attack:20s}: {score:.4f}")

Video Attacks
~~~~~~~~~~~~~

Test video watermark robustness:

.. code-block:: python

   from markdiffusion.evaluation.tools.video_editor import (
       MPEG4Compression,
       FrameAverage,
       FrameSwap,
       VideoCodecAttack,
       FrameRateAdapter,
       FrameInterpolationAttack
   )

   # Define video attacks
   video_attacks = {
       'MPEG4': MPEG4Compression(quality=20),
       'FrameAvg': FrameAverage(window_size=3),
       'FrameSwap': FrameSwap(swap_probability=0.1),
       'FPS-15': FrameRateAdapter(target_fps=15),
       'Interpolate': FrameInterpolationAttack(factor=2)
   }

   # Evaluate video robustness
   from markdiffusion.evaluation.dataset import VBenchDataset
   
   video_dataset = VBenchDataset(max_samples=50)
   video_robustness_results = {}

   for attack_name, attack_editor in video_attacks.items():
       pipeline = WatermarkedMediaDetectionPipeline(
           dataset=video_dataset,
           media_editor_list=[attack_editor],
           show_progress=True,
           return_type=DetectionPipelineReturnType.SCORES
       )
       
       scores = pipeline.evaluate(video_watermark, detection_kwargs=detection_kwargs)
       video_robustness_results[attack_name] = sum(scores) / len(scores)

Combined Attacks
~~~~~~~~~~~~~~~~

Test against multiple simultaneous attacks:

.. code-block:: python

   # Combine multiple attacks
   combined_attacks = [
       JPEGCompression(quality=75),
       GaussianBlurring(kernel_size=3),
       Rotation(angle=10)
   ]

   pipeline = WatermarkedMediaDetectionPipeline(
       dataset=dataset,
       media_editor_list=combined_attacks,  # All attacks applied
       show_progress=True,
       return_type=DetectionPipelineReturnType.SCORES
   )

   scores = pipeline.evaluate(watermark, detection_kwargs=detection_kwargs)
   print(f"Combined attack score: {sum(scores)/len(scores):.4f}")

Quality Evaluation
------------------

Image Quality Metrics
~~~~~~~~~~~~~~~~~~~~~

Direct Quality Analysis
^^^^^^^^^^^^^^^^^^^^^^^

For single image quality metrics:

.. code-block:: python

   from markdiffusion.evaluation.pipelines.image_quality_analysis import (
       DirectImageQualityAnalysisPipeline,
       QualityPipelineReturnType
   )
   from markdiffusion.evaluation.tools.image_quality_analyzer import (
       NIQECalculator, BRISQUEAnalyzer
   )

   pipeline = DirectImageQualityAnalysisPipeline(
       dataset=dataset,
       watermarked_image_editor_list=[],
       unwatermarked_image_editor_list=[],
       analyzers=[NIQECalculator(), BRISQUEAnalyzer()],
       show_progress=True,
       return_type=QualityPipelineReturnType.MEAN_SCORES
   )

   results = pipeline.evaluate(watermark)
   print(f"NIQE (watermarked): {results['watermarked']['NIQE']:.4f}")
   print(f"NIQE (unwatermarked): {results['unwatermarked']['NIQE']:.4f}")
   print(f"BRISQUE (watermarked): {results['watermarked']['BRISQUE']:.4f}")

Referenced Quality Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For metrics requiring reference images or text:

.. code-block:: python

   from markdiffusion.evaluation.pipelines.image_quality_analysis import (
       ReferencedImageQualityAnalysisPipeline
   )
   from markdiffusion.evaluation.tools.image_quality_analyzer import CLIPScoreCalculator
   from markdiffusion.evaluation.dataset import MSCOCODataset

   mscoco_dataset = MSCOCODataset(max_samples=100)

   pipeline = ReferencedImageQualityAnalysisPipeline(
       dataset=mscoco_dataset,
       watermarked_image_editor_list=[],
       unwatermarked_image_editor_list=[],
       analyzers=[CLIPScoreCalculator()],
       unwatermarked_image_source='generated',
       reference_image_source='natural',
       show_progress=True,
       return_type=QualityPipelineReturnType.MEAN_SCORES
   )

   results = pipeline.evaluate(watermark)
   print(f"CLIP Score: {results['CLIPScore']:.4f}")

Compared Quality Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

Compare watermarked vs unwatermarked images:

.. code-block:: python

   from markdiffusion.evaluation.pipelines.image_quality_analysis import (
       ComparedImageQualityAnalysisPipeline
   )
   from markdiffusion.evaluation.tools.image_quality_analyzer import (
       PSNRAnalyzer, SSIMAnalyzer, LPIPSAnalyzer,
       VIFAnalyzer, FSIMAnalyzer
   )

   pipeline = ComparedImageQualityAnalysisPipeline(
       dataset=dataset,
       watermarked_image_editor_list=[],
       unwatermarked_image_editor_list=[],
       analyzers=[
           PSNRAnalyzer(),
           SSIMAnalyzer(),
           LPIPSAnalyzer(),
           VIFAnalyzer(),
           FSIMAnalyzer()
       ],
       show_progress=True,
       return_type=QualityPipelineReturnType.MEAN_SCORES
   )

   results = pipeline.evaluate(watermark)
   print(f"PSNR: {results['PSNR']:.2f} dB")
   print(f"SSIM: {results['SSIM']:.4f}")
   print(f"LPIPS: {results['LPIPS']:.4f}")
   print(f"VIF: {results['VIF']:.4f}")
   print(f"FSIM: {results['FSIM']:.4f}")

Group Quality Analysis
^^^^^^^^^^^^^^^^^^^^^^

Metrics requiring sets of images:

.. code-block:: python

   from markdiffusion.evaluation.pipelines.image_quality_analysis import (
       GroupImageQualityAnalysisPipeline
   )
   from markdiffusion.evaluation.tools.image_quality_analyzer import (
       FIDCalculator, InceptionScoreCalculator
   )

   pipeline = GroupImageQualityAnalysisPipeline(
       dataset=mscoco_dataset,
       watermarked_image_editor_list=[],
       unwatermarked_image_editor_list=[],
       analyzers=[FIDCalculator(), InceptionScoreCalculator()],
       unwatermarked_image_source='generated',
       reference_image_source='natural',
       show_progress=True,
       return_type=QualityPipelineReturnType.MEAN_SCORES
   )

   results = pipeline.evaluate(watermark)
   print(f"FID: {results['FID']:.2f}")
   print(f"IS: {results['InceptionScore']:.2f}")

Repeat Quality Analysis
^^^^^^^^^^^^^^^^^^^^^^^

For diversity evaluation:

.. code-block:: python

   from markdiffusion.evaluation.pipelines.image_quality_analysis import (
       RepeatImageQualityAnalysisPipeline
   )

   pipeline = RepeatImageQualityAnalysisPipeline(
       dataset=StableDiffusionPromptsDataset(max_samples=10),
       prompt_per_image=20,  # Generate 20 images per prompt
       watermarked_image_editor_list=[],
       unwatermarked_image_editor_list=[],
       analyzers=[LPIPSAnalyzer()],
       show_progress=True,
       return_type=QualityPipelineReturnType.MEAN_SCORES
   )

   results = pipeline.evaluate(watermark)
   print(f"Average LPIPS (diversity): {results['LPIPS']:.4f}")

Video Quality Metrics
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.evaluation.dataset import VBenchDataset
   from markdiffusion.evaluation.pipelines.video_quality_analysis import (
       DirectVideoQualityAnalysisPipeline
   )
   from markdiffusion.evaluation.tools.video_quality_analyzer import (
       SubjectConsistencyAnalyzer,
       BackgroundConsistencyAnalyzer,
       MotionSmoothnessAnalyzer,
       DynamicDegreeAnalyzer,
       ImagingQualityAnalyzer
   )

   # Evaluate different video quality dimensions
   dimensions = {
       'subject_consistency': SubjectConsistencyAnalyzer(device='cuda'),
       'background_consistency': BackgroundConsistencyAnalyzer(device='cuda'),
       'motion_smoothness': MotionSmoothnessAnalyzer(device='cuda'),
       'dynamic_degree': DynamicDegreeAnalyzer(device='cuda'),
       'imaging_quality': ImagingQualityAnalyzer(device='cuda')
   }

   video_quality_results = {}
   for dim_name, analyzer in dimensions.items():
       video_dataset = VBenchDataset(max_samples=50, dimension=dim_name)
       
       pipeline = DirectVideoQualityAnalysisPipeline(
           dataset=video_dataset,
           watermarked_video_editor_list=[],
           unwatermarked_video_editor_list=[],
           watermarked_frame_editor_list=[],
           unwatermarked_frame_editor_list=[],
           analyzers=[analyzer],
           show_progress=True,
           return_type=QualityPipelineReturnType.MEAN_SCORES
       )
       
       results = pipeline.evaluate(video_watermark)
       video_quality_results[dim_name] = results

   # Print results
   print("\n=== Video Quality Results ===")
   for dim, score in video_quality_results.items():
       print(f"{dim}: {score}")

Comprehensive Evaluation
------------------------

Full Evaluation Suite
~~~~~~~~~~~~~~~~~~~~~

Evaluate all aspects together:

.. code-block:: python

   def comprehensive_evaluation(watermark_algo, dataset, attacks):
       """Run comprehensive evaluation on a watermark algorithm."""
       results = {
           'detectability': {},
           'robustness': {},
           'quality': {}
       }
       
       # 1. Detectability
       print("=== Detectability Evaluation ===")
       watermarked_pipeline = WatermarkedMediaDetectionPipeline(
           dataset=dataset, media_editor_list=[], show_progress=True,
           return_type=DetectionPipelineReturnType.SCORES
       )
       unwatermarked_pipeline = UnWatermarkedMediaDetectionPipeline(
           dataset=dataset, media_editor_list=[], show_progress=True,
           return_type=DetectionPipelineReturnType.SCORES
       )
       
       wm_scores = watermarked_pipeline.evaluate(watermark_algo)
       unwm_scores = unwatermarked_pipeline.evaluate(watermark_algo)
       
       calculator = DynamicThresholdSuccessRateCalculator(target_fpr=0.01)
       det_results = calculator.calculate(wm_scores, unwm_scores)
       results['detectability'] = det_results
       
       # 2. Robustness
       print("\n=== Robustness Evaluation ===")
       for attack_name, attack in attacks.items():
           pipeline = WatermarkedMediaDetectionPipeline(
               dataset=dataset, media_editor_list=[attack],
               show_progress=True,
               return_type=DetectionPipelineReturnType.SCORES
           )
           scores = pipeline.evaluate(watermark_algo)
           results['robustness'][attack_name] = sum(scores) / len(scores)
       
       # 3. Quality
       print("\n=== Quality Evaluation ===")
       quality_pipeline = ComparedImageQualityAnalysisPipeline(
           dataset=dataset,
           watermarked_image_editor_list=[],
           unwatermarked_image_editor_list=[],
           analyzers=[PSNRAnalyzer(), SSIMAnalyzer(), LPIPSAnalyzer()],
           show_progress=True,
           return_type=QualityPipelineReturnType.MEAN_SCORES
       )
       quality_results = quality_pipeline.evaluate(watermark_algo)
       results['quality'] = quality_results
       
       return results

   # Run evaluation
   attacks = {
       'JPEG-75': JPEGCompression(quality=75),
       'Blur': GaussianBlurring(kernel_size=3),
       'Noise': GaussianNoise(std=0.05),
       'Rotation': Rotation(angle=15)
   }

   eval_results = comprehensive_evaluation(watermark, dataset, attacks)

   # Print summary
   print("\n" + "="*50)
   print("COMPREHENSIVE EVALUATION SUMMARY")
   print("="*50)
   print(f"\nDetectability:")
   print(f"  TPR @ 1% FPR: {eval_results['detectability']['tpr']:.4f}")
   print(f"  AUC: {eval_results['detectability']['auc']:.4f}")
   print(f"\nRobustness:")
   for attack, score in eval_results['robustness'].items():
       print(f"  {attack}: {score:.4f}")
   print(f"\nQuality:")
   print(f"  PSNR: {eval_results['quality']['PSNR']:.2f} dB")
   print(f"  SSIM: {eval_results['quality']['SSIM']:.4f}")
   print(f"  LPIPS: {eval_results['quality']['LPIPS']:.4f}")

Compare Multiple Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   algorithms = ['TR', 'GS', 'ROBIN', 'SEAL']
   comparison_results = {}

   for algo_name in algorithms:
       print(f"\n{'='*50}")
       print(f"Evaluating {algo_name}")
       print('='*50)
       
       # Load algorithm
       algo = AutoWatermark.load(
           algo_name,
           algorithm_config=f'config/{algo_name}.json',
           diffusion_config=diffusion_config
       )
       
       # Evaluate
       results = comprehensive_evaluation(algo, dataset, attacks)
       comparison_results[algo_name] = results

   # Print comparison table
   import pandas as pd
   
   # Create comparison dataframe
   comparison_data = []
   for algo, results in comparison_results.items():
       row = {
           'Algorithm': algo,
           'TPR@1%FPR': results['detectability']['tpr'],
           'AUC': results['detectability']['auc'],
           'PSNR': results['quality']['PSNR'],
           'SSIM': results['quality']['SSIM'],
       }
       for attack in attacks:
           row[f'Rob_{attack}'] = results['robustness'][attack]
       comparison_data.append(row)
   
   df = pd.DataFrame(comparison_data)
   print("\n" + "="*100)
   print("ALGORITHM COMPARISON")
   print("="*100)
   print(df.to_string(index=False))

Best Practices
--------------

Sample Size Selection
~~~~~~~~~~~~~~~~~~~~~

- **Quick test**: 50-100 samples
- **Standard evaluation**: 200-500 samples
- **Publication**: 1000+ samples

Next Steps
----------

- :doc:`algorithms` - Algorithm-specific evaluation tips
- :doc:`../api/evaluation` - Evaluation API reference

