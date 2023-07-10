# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 'Lussac'
copyright = '2023, Aurélien Wyngaard'
author = 'Aurélien Wyngaard'
release = 'v2.0.0b1'


# -- General configuration ---------------------------------------------------

extensions = [
	'sphinx.ext.duration',
	'sphinxemoji.sphinxemoji'
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = [sphinx_rtd_theme.get_html_theme_path()]
