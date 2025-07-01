# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import (
    Path,
)

docs_source_dir = Path(__file__).resolve()
root_dir = docs_source_dir.parents[1]

# Add the 'src' directory to sys.path
sys.path.insert(
    0,
    root_dir / "src",
)

for p in sys.path:
    print(p)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tss-ml"
copyright = "2025, Ted Langhorst"
author = "Ted Langhorst"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # autodocument
    "sphinx.ext.napoleon",  # google and numpy doc string support
]

templates_path = ["_templates"]
exclude_patterns = [
    "**.ipynb_checkpoints",
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

# -- Napoleon autodoc options -------------------------------------------------
napoleon_numpy_docstring = True
