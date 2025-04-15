# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import date
import subprocess
import os

# Doxygen
if not os.path.exists( "_build/xml/" ):
    subprocess.call( 'doxygen ../Doxyfile.in', 
                     shell=True )

project = 'Machine Learning Compilers'
copyright = f'{date.today().year}, Lucas Obitz, Luca-Philipp Grumbach'
author = 'Lucas Obitz, Luca-Philipp Grumbach'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'breathe'
]

templates_path = [
    '_templates'
]
exclude_patterns = [
    '_build',
    '.DS_Store'
]

highlight_language = 'c++'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "source_repository": "https://github.com/Shad00Z/machine-learning-compilers/",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_static_path = [
    '_static'
]

# -- Breathe configuration -------------------------------------------------

breathe_projects = {
    "Machine Learning Compilers": "_build/xml/",
}

breathe_default_project = "Machine Learning Compilers"
breathe_default_members = (
    'members', 
    'undoc-members'
)
