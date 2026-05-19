Visualization API
=================

This page documents the visualization APIs for analyzing watermarking mechanisms.

AutoVisualizer
--------------

The ``AutoVisualizer`` class provides unified visualization for all watermarking algorithms.

**Key Methods:**

.. py:class:: visualize.auto_visualization.AutoVisualizer

   .. py:staticmethod:: load(algorithm_name, data_for_visualization)
   
      Load a visualizer for the specified algorithm.
      
      :param algorithm_name: Name of the watermarking algorithm (e.g., 'TR', 'GS', 'PRC')
      :param data_for_visualization: Data obtained from ``get_data_for_visualize()``
      :return: An AutoVisualizer instance

   .. py:method:: visualize(rows, cols, methods, method_kwargs=None, save_path=None, **kwargs)
   
      Generate visualization figures.
      
      :param rows: Number of rows in the figure grid
      :param cols: Number of columns in the figure grid
      :param methods: List of visualization methods to use
      :param method_kwargs: List of keyword arguments for each method
      :param save_path: Path to save the figure (optional)
      :return: matplotlib figure object

**Example Usage:**

.. code-block:: python

   from markdiffusion.visualize.auto_visualization import AutoVisualizer
   
   # Get visualization data from watermark
   data_for_vis = watermark.get_data_for_visualize(watermarked_image)
   
   # Load visualizer
   visualizer = AutoVisualizer.load('TR', data_for_visualization=data_for_vis)
   
   # Create visualization
   fig = visualizer.visualize(
       rows=1,
       cols=5,
       methods=['draw_pattern_fft', 'draw_orig_latents_fft', 
                'draw_watermarked_image', 'draw_inverted_latents_fft', 
                'draw_inverted_pattern_fft'],
       save_path='visualization.pdf'
   )

**Available Visualization Methods:**

Each algorithm has specific visualization methods. Common methods include:

- ``draw_watermarked_image`` - Display the watermarked image
- ``draw_orig_latents`` / ``draw_orig_latents_fft`` - Original latent representations
- ``draw_inverted_latents`` / ``draw_inverted_latents_fft`` - Inverted latent representations
- Algorithm-specific methods (e.g., ``draw_pattern_fft`` for Tree-Ring, ``draw_watermark_bits`` for Gaussian-Shading)

.. note::
   For detailed visualization examples, see :doc:`../user_guide/visualization`.
