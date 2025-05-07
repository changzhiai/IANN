# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


project = 'IANN'
copyright = '2025, Changzhi Ai'
author = 'Changzhi Ai'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # if you're using Google or NumPy docstrings
    'sphinx_rtd_theme',
]

autodoc_mock_imports = ["asap3", "e3nn", "torch", "torch_geometric", "opt_einsum_fx"]

autodoc_class_signature = 'mixed'