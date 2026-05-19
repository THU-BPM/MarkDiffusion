Visualization
=============

MarkDiffusion provides powerful visualization tools to understand how watermarking algorithms work.

Overview
--------

Visualization helps you:

- Understand watermarking mechanisms
- Debug watermarking issues
- Present results in papers/reports
- Educate users about watermarking

The visualization module creates insightful plots showing:

- Watermark patterns in frequency domain
- Latent representations
- Watermark bits and reconstruction
- Spatial and frequency domain comparisons

Basic Usage
-----------

Simple Visualization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from markdiffusion.watermark.auto_watermark import AutoWatermark
   from markdiffusion.visualize.auto_visualization import AutoVisualizer

   # Generate watermarked image
   watermark = AutoWatermark.load('GS', 'markdiffusion/config/GS.json', diffusion_config)
   watermarked_image = watermark.generate_watermarked_media(prompt)

   # Get visualization data
   viz_data = watermark.get_data_for_visualize(watermarked_image)

   # Create visualizer
   visualizer = AutoVisualizer.load('GS', data_for_visualization=viz_data)

   # Generate visualization
   fig = visualizer.visualize(
       rows=2,
       cols=2,
       methods=['draw_watermark_bits', 'draw_reconstructed_watermark_bits',
                'draw_inverted_latents', 'draw_inverted_latents_fft']
   )

   # Save figure
   fig.savefig('visualization.png', dpi=300, bbox_inches='tight')

Visualization Methods by Algorithm
-----------------------------------

Tree-Ring (TR)
~~~~~~~~~~~~~~

Tree-Ring visualizations show frequency domain patterns:

.. code-block:: python

   visualizer = AutoVisualizer.load('TR', data_for_visualization=viz_data)
   
   fig = visualizer.visualize(
       rows=2,
       cols=3,
       methods=[
           'draw_init_latents',           # Initial latent representation
           'draw_init_latents_fft',       # FFT of initial latents
           'draw_watermark_pattern',      # Ring pattern
           'draw_watermarked_latents',    # Watermarked latents
           'draw_watermarked_latents_fft',# FFT showing rings
           'draw_generated_image'         # Final image
       ]
   )

**Available Methods:**

- ``draw_init_latents`` - Initial latent vectors
- ``draw_init_latents_fft`` - Frequency spectrum of initial latents
- ``draw_watermark_pattern`` - The ring pattern overlay
- ``draw_watermarked_latents`` - Latents after watermark injection
- ``draw_watermarked_latents_fft`` - Frequency domain with rings visible
- ``draw_generated_image`` - Final generated image

Gaussian-Shading (GS)
~~~~~~~~~~~~~~~~~~~~~

Gaussian-Shading visualizations show bit-level watermark information:

.. code-block:: python

   visualizer = AutoVisualizer.load('GS', data_for_visualization=viz_data)
   
   fig = visualizer.visualize(
       rows=2,
       cols=2,
       methods=[
           'draw_watermark_bits',              # Original watermark bits
           'draw_reconstructed_watermark_bits', # Extracted bits
           'draw_inverted_latents',            # Inverted latent codes
           'draw_inverted_latents_fft'         # Frequency analysis
       ]
   )

**Available Methods:**

- ``draw_watermark_bits`` - Original watermark message bits
- ``draw_reconstructed_watermark_bits`` - Extracted watermark bits
- ``draw_bit_accuracy_heatmap`` - Bit-wise accuracy visualization
- ``draw_inverted_latents`` - Inverted latent representation
- ``draw_inverted_latents_fft`` - FFT of inverted latents
- ``draw_generated_image`` - Final watermarked image

ROBIN
~~~~~

ROBIN visualizations show adversarially optimized patterns:

.. code-block:: python

   visualizer = AutoVisualizer.load('ROBIN', data_for_visualization=viz_data)
   
   fig = visualizer.visualize(
       rows=2,
       cols=2,
       methods=[
           'draw_watermark_message',
           'draw_extracted_message',
           'draw_perturbation_pattern',
           'draw_frequency_analysis'
       ]
   )

**Available Methods:**

- ``draw_watermark_message`` - Original message
- ``draw_extracted_message`` - Detected message
- ``draw_perturbation_pattern`` - Adversarial perturbation
- ``draw_frequency_analysis`` - Frequency domain analysis
- ``draw_robustness_map`` - Spatial robustness heatmap

GaussMarker (GM)
~~~~~~~~~~~~~~~~

GaussMarker shows dual-domain watermarking:

.. code-block:: python

   visualizer = AutoVisualizer.load('GM', data_for_visualization=viz_data)
   
   fig = visualizer.visualize(
       rows=2,
       cols=3,
       methods=[
           'draw_spatial_watermark',
           'draw_frequency_watermark',
           'draw_combined_watermark',
           'draw_gnr_output',
           'draw_detection_map',
           'draw_generated_image'
       ]
   )

**Available Methods:**

- ``draw_spatial_watermark`` - Spatial domain component
- ``draw_frequency_watermark`` - Frequency domain component
- ``draw_combined_watermark`` - Combined dual-domain
- ``draw_gnr_output`` - GNR network output
- ``draw_detection_map`` - Detection confidence map

VideoShield
~~~~~~~~~~~

Video visualizations show temporal patterns:

.. code-block:: python

   visualizer = AutoVisualizer.load('VideoShield', data_for_visualization=viz_data)
   
   fig = visualizer.visualize(
       rows=3,
       cols=4,
       methods=[
           'draw_frame_sequence',        # All frames
           'draw_temporal_watermark',    # Watermark over time
           'draw_optical_flow',          # Motion patterns
           'draw_consistency_map'        # Temporal consistency
       ]
   )

**Available Methods:**

- ``draw_frame_sequence`` - Video frame grid
- ``draw_temporal_watermark`` - Watermark evolution over frames
- ``draw_optical_flow`` - Motion flow visualization
- ``draw_consistency_map`` - Temporal consistency
- ``draw_frame_differences`` - Inter-frame differences

Advanced Visualization
----------------------

Custom Layout
~~~~~~~~~~~~~

Create custom visualization layouts:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Create custom figure
   fig, axes = plt.subplots(2, 3, figsize=(15, 10))
   
   # Get visualizer
   visualizer = AutoVisualizer.load('TR', data_for_visualization=viz_data)
   
   # Draw on specific axes
   visualizer.draw_init_latents(ax=axes[0, 0])
   visualizer.draw_init_latents_fft(ax=axes[0, 1])
   visualizer.draw_watermark_pattern(ax=axes[0, 2])
   visualizer.draw_watermarked_latents(ax=axes[1, 0])
   visualizer.draw_watermarked_latents_fft(ax=axes[1, 1])
   visualizer.draw_generated_image(ax=axes[1, 2])
   
   # Adjust layout
   plt.tight_layout()
   fig.savefig('custom_visualization.png', dpi=300)

Comparing Methods
~~~~~~~~~~~~~~~~~

Visualize multiple algorithms side-by-side:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   algorithms = ['TR', 'GS', 'ROBIN']
   fig, axes = plt.subplots(len(algorithms), 3, figsize=(15, len(algorithms)*4))
   
   for i, algo in enumerate(algorithms):
       # Load watermark
       wm = AutoWatermark.load(algo, f'config/{algo}.json', diffusion_config)
       img = wm.generate_watermarked_media(prompt)
       viz_data = wm.get_data_for_visualize(img)
       
       # Visualize
       visualizer = AutoVisualizer.load(algo, data_for_visualization=viz_data)
       visualizer.draw_generated_image(ax=axes[i, 0])
       visualizer.draw_inverted_latents_fft(ax=axes[i, 1])
       
       # Detection heatmap
       detection = wm.detect_watermark_in_media(img)
       axes[i, 2].text(0.5, 0.5, f"{algo}\nScore: {detection.get('score', 'N/A')}",
                       ha='center', va='center', fontsize=14)
       axes[i, 2].axis('off')
   
   plt.tight_layout()
   fig.savefig('algorithm_comparison.png', dpi=300)

Next Steps
----------

- :doc:`evaluation` - Evaluate watermark performance
- :doc:`algorithms` - Learn about algorithm internals
- :doc:`../api/visualization` - Visualization API reference

