# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


project = 'IANN'
copyright = '2024, Changzhi Ai'
author = 'Changzhi Ai'
release = '0.1.3'

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

# Sphinx copies html_logo/html_favicon into the output itself, so those need no
# html_static_path. The path below is deliberately the _static/css subdirectory
# rather than _static itself: pointing it at _static would copy every file in
# there, duplicating the ~4 MB of figures that the figure directive already
# copies into _images/. Sphinx copies the *contents* of each static path into
# _static/, so _static/css/custom.css lands at _static/custom.css.
html_static_path = ["_static/css"]
html_css_files = ["custom.css"]

# The sidebar variant, purpose-built for the one width it is rendered at. It is
# the white-ink version because the theme's nav header is a mid blue -- the
# navy-ink one on that background is legible but muddy -- and its tagline is
# split over two lines. That is what makes a tagline viable here at all: a
# single line is the widest element, so it would widen the viewBox from 405 to
# 676 units, and since the <img> width is fixed in CSS that shrinks the icon and
# wordmark while still rendering the tagline at ~8 px. Two lines, neither wider
# than "IANN", keep the box at 405x160 and render at ~12 px.
#
# Its rendered size is set in _static/css/custom.css, not here and not in the
# SVG: the theme clamps the logo to the sidebar width, so the file's own width
# attribute has no effect.
html_logo = "_static/logo/iann-logo-dark.svg"
html_favicon = "_static/logo/iann-favicon.svg"

html_theme_options = {
    # The logo already contains the wordmark, and the theme prints the project
    # name above it by default -- which renders "IANN" twice in the sidebar.
    "logo_only": True,

    # Read the Docs version switcher. "attached" docks the flyout into the
    # sidebar; the default "hidden" leaves it to float over the page corner.
    # This only renders on Read the Docs -- the menu is injected by the
    # readthedocs-addons script at serve time, so a local build shows nothing.
    "flyout_display": "attached",
    "version_selector": True,
    # English only, so a language menu would offer a single choice.
    "language_selector": False,
}


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # if you're using Google or NumPy docstrings
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',  # show the source code of the current module
]

# Every module is named iann.*, so without this the Python Module Index files
# them all under a single "i" and the alphabet jumpbox is useless. Sorting on the
# part after the prefix groups them by their own initial instead.
modindex_common_prefix = ["iann."]

autodoc_mock_imports = [
    "asap3", "e3nn", "torch", "torch_geometric", "opt_einsum_fx",
    "cuequivariance", "cuequivariance_torch",  # optional use_cue backend
]

autodoc_class_signature = 'mixed'

# Model constructors take dozens of parameters through **kwargs, so autodoc would
# render an unreadable signature. Collapse it to "(...)" and let the prose in
# engine_models.rst carry the parameters instead. Keep this list in step with the
# autoclass entries in api.rst -- a model added there but omitted here renders its
# full signature.
_COLLAPSE_SIGNATURE = (
    "PaiNN", "NequIP", "MACE", "EquiformerV2", "EquiformerV3",
    "Allegro", "UMA", "FastPot",
)

def process_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "class" and any(cls in name for cls in _COLLAPSE_SIGNATURE):
        return "(...)", None

def setup(app):
    app.connect("autodoc-process-signature", process_signature)