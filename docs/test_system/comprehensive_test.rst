Comprehensive Testing
====================

MarkDiffusion provides a complete testing suite that covers all functionality in the repository. 
This comprehensive test module is located in the ``test/`` directory.

Test Coverage Statistics
------------------------

- **Total Test Cases**: 672 unit tests
- **Code Coverage**: 94.73%
- **Coverage Scope**: Nearly all functional modules
- **Uncovered Code**: Primarily exception handling and edge case logic

The test suite is designed to ensure the reliability and correctness of all components in the MarkDiffusion toolkit.

Test Structure
--------------

The test suite is organized into several test files:


**test_watermark_algorithms.py**
   Tests for initialization, generation, detection, inversion, and visualization of all watermark algorithms.

**test_pipelines.py**
   Tests for evaluation pipelines including detection and quality analysis.

**test_dataset.py**
   Tests for dataset classes (StableDiffusionPromptsDataset, MSCOCODataset, VBenchDataset).

**test_utils.py**
   Tests for utility functions used throughout the toolkit.

**test_exceptions.py**
   Tests for custom exception classes.

**test_image_editor.py**
   Tests for image editing modules (rotation, cropping, noise, etc.).

**test_video_editor.py**
   Tests for video editing modules.

**test_edge_cases.py**
   Tests for some edge cases.

Running the Tests
-----------------

.. note::

   For detailed instructions on running tests, including specific test commands, markers, 
   and advanced usage options, please refer to the ``test/README.md`` file in the repository.
   
   This document provides only a high-level overview of the testing system.

Quick Start
~~~~~~~~~~~

Install test dependencies and run the complete test suite:

.. code-block:: bash

   pip install -r test/requirements-test.txt
   pytest test -v --cov=. --cov-report=html --cov-report=term-missing --html=report.html


For more detailed information about running tests, please refer to the ``test/README.md`` file 
in the repository.

