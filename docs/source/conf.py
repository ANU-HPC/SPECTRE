# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SPEC4Py'
copyright = '2025, Add Authors'
author = 'Add Authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'autoapi.extension',
    'sphinxcontrib.bibtex',
]

templates_path = ['_templates']
exclude_patterns = []

autoapi_dirs = ['../../spectre']
# This can be set to True to keep the generated .rst files for debugging
autoapi_keep_files = False

bibtex_bibfiles = ['references.bib']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
