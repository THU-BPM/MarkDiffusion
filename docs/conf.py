# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'MarkDiffusion'
copyright = '2025, MarkDiffusion Team'
author = 'MarkDiffusion Team'

# The full version, including alpha/beta/rc tags
release = '1.0.2'
version = '1.0.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx_copybutton',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'advanced/*']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#5F259F',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'custom.css',
]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs. This file should be a Windows icon file (.ico) being 16x16 or 32x32 pixels large.
html_favicon = None

# -- Extension configuration -------------------------------------------------

# autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Mock imports for modules that are not needed for documentation
autodoc_mock_imports = [
    'cv2',
    'diffusers',
    'ujson',
    'torch',
    'torchvision',
    'transformers',
    'datasets',
    'galois',
    'ldpc',
    'pycryptodome',
    'Crypto',
    'sentence_transformers',
    'sklearn',
    'scipy',
    'tqdm',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.axes',
    'numpy',
    'PIL',
    'pillow',
    'pil',
]

# Suppress warnings
suppress_warnings = ['app.add_directive', 'app.add_node']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# autosummary settings
autosummary_generate = True

# intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    # 'numpy': ('https://numpy.org/doc/stable/', None),  # Commented out due to server issues
    'PIL': ('https://pillow.readthedocs.io/en/stable/', None),
}

# Timeout for intersphinx
intersphinx_timeout = 10

# copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# nbsphinx settings
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

