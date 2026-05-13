# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SPECTRE'
copyright = '2025, EPFL - Swiss Plasma Center'
author = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'autoapi.extension',
    'sphinxcontrib.bibtex',
    'myst_nb',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = []

autoapi_dirs = ['../../spectre']
# This can be set to True to keep the generated .rst files for debugging
autoapi_keep_files = False

bibtex_bibfiles = ['references.bib']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

nb_execution_mode = "off"

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
]
