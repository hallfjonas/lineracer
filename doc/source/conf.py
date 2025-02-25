# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lineracer'
copyright = '2025, Ephraim Eckl, Jonas Hall'
author = 'Ephraim Eckl, Jonas Hall'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Source code dir relative to this file

extensions = [
    'sphinx.ext.autodoc',       # Core library for html generation from docstrings
    'sphinx.ext.autosummary',   # Create neat summary tables
    'sphinx.ext.napoleon',      # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',      # Add a link to the Python source code for classes, functions etc.

]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
