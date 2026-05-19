Contributing to MarkDiffusion
=============================

We welcome contributions from the community! This guide will help you get started.

.. note::
   Please read our `Code of Conduct <../code_of_conduct.md>`_ and the detailed 
   `Contributing Guidelines <../contributing.md>`_ in the repository root before contributing.

Overview
--------

This document provides technical guidelines for contributing to MarkDiffusion. For general contribution 
workflow (forking, cloning, creating branches, submitting PRs), please refer to the 
`Contributing Guidelines <../contributing.md>`_ in the repository root.

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

We follow PEP 8 style guidelines. Please ensure your code:

- Uses 4 spaces for indentation
- Has descriptive variable and function names
- Includes docstrings for all public functions and classes
- Stays under 100 characters per line when practical

Example:

.. code-block:: python

   def generate_watermarked_media(self, input_data, **kwargs):
       """
       Generate watermarked media from input.
       
       Args:
           input_data (str): Text prompt or input data
           **kwargs: Additional generation parameters
           
       Returns:
           PIL.Image: Watermarked image
           
       Examples:
           >>> watermark = AutoWatermark.load('GS', 'markdiffusion/config/GS.json', config)
           >>> image = watermark.generate_watermarked_media("A sunset")
       """
       # Implementation
       pass

Documentation
~~~~~~~~~~~~~

All new features should include:

- Docstrings following Google or NumPy style
- Type hints for function arguments and returns
- Usage examples in docstrings
- Updates to relevant documentation pages

Example with type hints:

.. code-block:: python

   from typing import Union, Dict, Any
   from PIL import Image

   def detect_watermark_in_media(
       self,
       media: Union[Image.Image, list],
       **kwargs: Any
   ) -> Dict[str, Any]:
       """
       Detect watermark in media.
       
       Args:
           media: PIL Image or list of PIL Images (for video)
           **kwargs: Additional detection parameters
           
       Returns:
           Dictionary containing detection results with keys:
               - detected (bool): Whether watermark was detected
               - score (float): Detection confidence score
               - threshold (float): Detection threshold used
       """
       pass

Testing
~~~~~~~

All new code should include comprehensive tests. We use pytest for testing.

.. note::

   For detailed testing instructions, including test structure, markers, and running options, 
   please refer to the ``test/README.md`` file in the repository.
   
   For test code examples, please refer to the existing test files in the ``test/`` directory:
   
   - ``test/test_watermark_algorithms.py`` - Watermark algorithm tests
   - ``test/test_pipelines.py`` - Pipeline tests
   - ``test/test_image_editor.py`` - Image editing tests
   - ``test/test_video_editor.py`` - Video editing tests
   - ``test/test_utils.py`` - Utility function tests
   - ``test/test_dataset.py`` - Dataset tests

Run tests:

.. code-block:: bash

   # Install test dependencies
   pip install -r test/requirements-test.txt
   
   # Run all tests with coverage and HTML report
   pytest test -v --cov=. --cov-report=html --cov-report=term-missing --html=report.html
   
   # Test all algorithms
   pytest test/test_watermark_algorithms.py -v
   
   # Test a specific algorithm
   pytest test/test_watermark_algorithms.py -v --algorithm TR
   
   # Test all pipelines
   pytest test/test_pipelines.py -v

For more test commands and options, see ``test/README.md``.

Contribution Process
--------------------

Adding a New Algorithm
~~~~~~~~~~~~~~~~~~~~~~

To add a new watermarking algorithm:

1. **Create algorithm directory structure**

   .. code-block:: bash

      watermark/my_algorithm/
      ├── __init__.py
      ├── my_algorithm.py
      detection/my_algorithm/
      ├── __init__.py
      ├── my_algorithm_detection.py
      visualize/my_algorithm/
      ├── __init__.py
      ├── my_algorithm_visualizer.py
      config/
      ├── MyAlgorithm.json
      test/
      ├── test_my_algorithm.py

2. **Implement the algorithm**

   Implement the watermark generation and detection logic.

3. **Add configuration**

   Create ``config/MyAlgorithm.json`` with algorithm parameters.

4. **Register algorithm**

   Add to ``watermark/auto_watermark.py``:

   .. code-block:: python

      from markdiffusion.watermark.my_algorithm.my_algorithm import MyAlgorithm

      class AutoWatermark:
          ALGORITHM_MAP = {
              # ... existing algorithms
              'MA': MyAlgorithm,
          }

5. **Write tests**

   Create comprehensive tests in ``test/test_my_algorithm.py``.

6. **Update documentation**

   - Add algorithm description to ``docs/user_guide/algorithms.rst``
   - Update ``docs/index.rst`` to list the new algorithm
   - Add usage examples

7. **Submit pull request**

   See Pull Request Guidelines below.

Adding Evaluation Tools
~~~~~~~~~~~~~~~~~~~~~~~

To add a new evaluation tool:

1. **Implement the tool**

   For image attacks:

   .. code-block:: python

      from markdiffusion.evaluation.tools.image_editor import BaseImageEditor

      class MyAttack(BaseImageEditor):
          def __init__(self, param1, param2, **kwargs):
              super().__init__(**kwargs)
              self.param1 = param1
              self.param2 = param2
          
          def edit_image(self, image):
              # Implement attack
              return modified_image

   For quality metrics:

   .. code-block:: python

      from markdiffusion.evaluation.tools.image_quality_analyzer import BaseImageQualityAnalyzer

      class MyMetric(BaseImageQualityAnalyzer):
          def analyze(self, image1, image2=None):
              # Implement metric
              return score

2. **Add tests**

3. **Update documentation**

4. **Submit pull request**

Submission Checklist
--------------------

Before submitting your contribution, please ensure you have completed the following:

Type of Change
~~~~~~~~~~~~~~

Please indicate the type of your contribution:

- 🐛 **Bug fix** (non-breaking change which fixes an issue)
- ✨ **New feature** (non-breaking change which adds functionality)
- 📝 **Documentation update** (changes to documentation only)
- 💥 **Breaking change** (fix or feature that would cause existing functionality to not work as expected)
- 🔧 **Refactor/Optimization** (code improvement without changing logic)

Pre-Submission Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~

- [ ] I have read and followed the Contributing Guidelines
- [ ] I have performed a self-review of my own code
- [ ] My code follows the project's coding style/standards (PEP 8)
- [ ] I have added/updated necessary comments and documentation
- [ ] I have added corresponding test cases (if new feature or bug fix)
- [ ] All local tests pass successfully

**Testing**

.. code-block:: bash

   # Install test dependencies
   pip install -r test/requirements-test.txt
   
   # Run all tests with coverage
   pytest test -v --cov=. --cov-report=html --cov-report=term-missing --html=report.html

For detailed testing instructions, see ``test/README.md``.

**Code Style**

.. code-block:: bash

   # Check code style
   flake8 watermark/ detection/ evaluation/ visualize/
   black --check watermark/ detection/ evaluation/ visualize/
   
   # Auto-format code
   black watermark/ detection/ evaluation/ visualize/

**Documentation**

- Update relevant documentation files
- Add usage examples where appropriate
- Ensure all docstrings are complete and follow the project style

**Pull Request**

For the complete pull request process and guidelines, please refer to the 
`Contributing Guidelines <../CONTRIBUTING.md>`_ in the repository root.

Additional Information
----------------------

**Community Guidelines**

All participants are expected to follow our `Code of Conduct <../code_of_conduct.md>`_. 
Please be respectful, constructive, and help create a welcoming environment for everyone.

**Reporting Issues**

For bug reports and feature requests, please use the appropriate templates configured in the GitHub repository.

**Questions?**

If you have questions:

- Open a discussion on GitHub
- Check existing issues and pull requests
- Review the documentation at https://markdiffusion.readthedocs.io

**Contact**

For major contributions or collaborations:

- GitHub: https://github.com/THU-BPM/MarkDiffusion
- Email: panly24@mails.tsinghua.edu.cn

Thank you for contributing to MarkDiffusion!

