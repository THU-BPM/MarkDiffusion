Evaluation API
==============

This page documents the evaluation APIs for testing watermark robustness and quality.

Datasets
--------

Dataset classes for loading prompts and test data.

MSCOCODataset
~~~~~~~~~~~~~

.. py:class:: evaluation.dataset.MSCOCODataset

   Dataset for loading MS-COCO captions and images.
   
   :param parquet_file: Path to the parquet file containing COCO data
   :param max_samples: Maximum number of samples to load (optional)

StableDiffusionPromptsDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: evaluation.dataset.StableDiffusionPromptsDataset

   Dataset for loading text prompts for Stable Diffusion.
   
   :param parquet_file: Path to the parquet file containing prompts
   :param max_samples: Maximum number of samples to load (optional)

VBenchDataset
~~~~~~~~~~~~~

.. py:class:: evaluation.dataset.VBenchDataset

   Dataset for loading VBench video prompts.
   
   :param prompt_file: Path to the text file containing prompts
   :param max_samples: Maximum number of samples to load (optional)

Evaluation Pipelines
--------------------

Detection Pipelines
~~~~~~~~~~~~~~~~~~~

.. py:class:: evaluation.pipelines.detection.WatermarkedMediaDetectionPipeline

   Pipeline for evaluating detection performance on watermarked media.
   
   **Key Methods:**
   
   - ``run(watermark, dataset, **kwargs)`` - Run detection evaluation
   - ``get_results()`` - Get evaluation results

.. py:class:: evaluation.pipelines.detection.UnWatermarkedMediaDetectionPipeline

   Pipeline for evaluating false positive rate on unwatermarked media.
   
   **Key Methods:**
   
   - ``run(watermark, dataset, **kwargs)`` - Run detection evaluation
   - ``get_results()`` - Get evaluation results

Quality Analysis Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: evaluation.pipelines.image_quality_analysis.DirectImageQualityAnalysisPipeline

   Pipeline for analyzing image quality directly without reference.
   
   **Key Methods:**
   
   - ``run(watermark, dataset, quality_analyzers, **kwargs)`` - Run quality analysis
   - ``get_results()`` - Get analysis results

.. py:class:: evaluation.pipelines.video_quality_analysis.DirectVideoQualityAnalysisPipeline

   Pipeline for analyzing video quality.
   
   **Key Methods:**
   
   - ``run(watermark, dataset, quality_analyzers, **kwargs)`` - Run quality analysis
   - ``get_results()`` - Get analysis results

Evaluation Tools
----------------

Image Attacks/Editors
~~~~~~~~~~~~~~~~~~~~~

Common image attack methods for testing watermark robustness:

.. py:class:: evaluation.tools.image_editor.JPEGCompression

   JPEG compression attack.
   
   :param quality: JPEG quality (0-100)

.. py:class:: evaluation.tools.image_editor.GaussianBlur

   Gaussian blur attack.
   
   :param kernel_size: Size of the Gaussian kernel

.. py:class:: evaluation.tools.image_editor.GaussianNoise

   Gaussian noise attack.
   
   :param std: Standard deviation of the noise

.. py:class:: evaluation.tools.image_editor.Rotation

   Rotation attack.
   
   :param angle: Rotation angle in degrees

.. py:class:: evaluation.tools.image_editor.CenterCrop

   Center crop attack.
   
   :param crop_ratio: Ratio of image to keep (0-1)

Quality Analyzers
~~~~~~~~~~~~~~~~~

Image quality metrics:

.. py:class:: evaluation.tools.image_quality_analyzer.PSNRAnalyzer

   Peak Signal-to-Noise Ratio analyzer.

.. py:class:: evaluation.tools.image_quality_analyzer.SSIMAnalyzer

   Structural Similarity Index analyzer.

Video quality metrics:

.. py:class:: evaluation.tools.video_quality_analyzer.SubjectConsistencyAnalyzer

   Video subject consistency analyzer.

Success Rate Calculator
~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: evaluation.tools.success_rate_calculator.DynamicThresholdSuccessRateCalculator

   Calculate detection success rates with dynamic thresholds.

**Example Usage:**

.. code-block:: python

   from markdiffusion.evaluation.dataset import MSCOCODataset
   from markdiffusion.evaluation.pipelines.detection import WatermarkedMediaDetectionPipeline
   from markdiffusion.evaluation.tools.image_editor import JPEGCompression
   from markdiffusion.evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
   
   # Load dataset
   dataset = MSCOCODataset('dataset/mscoco/mscoco.parquet', max_samples=100)
   
   # Create pipeline
   pipeline = WatermarkedMediaDetectionPipeline(
       attack=JPEGCompression(quality=50),
       success_rate_calculator=DynamicThresholdSuccessRateCalculator()
   )
   
   # Run evaluation
   pipeline.run(watermark, dataset)
   results = pipeline.get_results()
   print(results)

.. code-block:: python

   from markdiffusion.evaluation.pipelines.image_quality_analysis import DirectImageQualityAnalysisPipeline
   from markdiffusion.evaluation.dataset import StableDiffusionPromptsDataset
   from markdiffusion.evaluation.tools.image_quality_analyzer import PSNRAnalyzer, SSIMAnalyzer
   
   # Load dataset
   dataset = StableDiffusionPromptsDataset('dataset/prompts.parquet', max_samples=50)
   
   # Create pipeline with quality analyzers
   pipeline = DirectImageQualityAnalysisPipeline(
       quality_analyzers=[PSNRAnalyzer(), SSIMAnalyzer()]
   )
   
   # Run analysis
   pipeline.run(watermark, dataset)
   results = pipeline.get_results()
   print(results)

.. note::
   For detailed evaluation examples and workflows, see :doc:`../user_guide/evaluation`.
