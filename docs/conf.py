# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the source package importable during doc builds
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Customer Churn Prediction"
copyright = "2026, Group 2"
author = "Group 2"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # auto-generate docs from docstrings
    "sphinx.ext.autosummary",    # generate summary tables
    "sphinx.ext.napoleon",       # support Google / NumPy docstring style
    "sphinx.ext.viewcode",       # add links to highlighted source code
    "sphinx.ext.intersphinx",    # link to external docs (pandas, sklearn…)
]

autosummary_generate = True       # auto-create stub .rst files
autodoc_member_order = "bysource" # keep the order from the source file
autodoc_typehints = "description" # render type hints in the description

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}
