Watermark API
=============

This page documents the core watermarking APIs that users directly interact with.

AutoWatermark
-------------

The ``AutoWatermark`` class is the primary interface for watermarking operations.

.. autoclass:: watermark.auto_watermark.AutoWatermark
   :members: load, generate_watermarked_media, generate_unwatermarked_media, detect_watermark_in_media, get_data_for_visualize

**Key Methods:**

- ``load(algorithm_name, algorithm_config, diffusion_config)`` - Load a watermarking algorithm
- ``generate_watermarked_media(input_data, **kwargs)`` - Generate watermarked media (image or video)
- ``generate_unwatermarked_media(input_data, **kwargs)`` - Generate clean media without watermark
- ``detect_watermark_in_media(media, **kwargs)`` - Detect watermark in media
- ``get_data_for_visualize(media, **kwargs)`` - Get data for visualization

**Supported Algorithms:**

- Image watermarks: ``TR``, ``GS``, ``PRC``, ``RI``, ``SEAL``, ``ROBIN``, ``WIND``, ``GM``, ``SFW``
- Video watermarks: ``VideoShield``, ``VideoMark``

**Example Usage:**

.. code-block:: python

   from markdiffusion.watermark.auto_watermark import AutoWatermark
   from markdiffusion.utils.diffusion_config import DiffusionConfig

   # Load a watermark algorithm
   watermark = AutoWatermark.load('TR', 
                                  algorithm_config='markdiffusion/config/TR.json',
                                  diffusion_config=diffusion_config)

   # Generate watermarked image
   watermarked_image = watermark.generate_watermarked_media("A sunset over mountains")

   # Detect watermark
   result = watermark.detect_watermark_in_media(watermarked_image)
   print(result)

.. note::
   For algorithm-specific implementation details, please refer to the 
   :doc:`../user_guide/algorithms` page.
