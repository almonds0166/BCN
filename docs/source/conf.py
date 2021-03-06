# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
from datetime import datetime

# -- Project information -----------------------------------------------------

project = "BCN"
copyright = "2021, Madison Landry"
author = "Madison Landry"

# The full version, including alpha/beta/rc tags
from bcn import __version__
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
   "sphinx.ext.napoleon",
   "sphinx.ext.intersphinx",
   "sphinx.ext.autodoc",
   "sphinx_autodoc_typehints",
   "sphinx.ext.todo",
   "sphinxcontrib.katex",
]
intersphinx_mapping = {
   "python": ("https://docs.python.org/3", None),
   "torch": ("https://pytorch.org/docs/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = "alabaster"
html_theme = "sphinx_book_theme" # https://sphinx-themes.org/sample-sites/sphinx-book-theme/

today = datetime.utcnow().strftime(
   "<time datetime=\"%Y-%m-%dT%H:%MZ\">" \
   "%b %d at %H:%M UTC"
   "</time>"
)
html_theme_options = {
   "home_page_in_toc": True,
   "toc_title": "Jump to",
   "repository_url": "https://github.com/almonds0166/BCN",
   "path_to_docs": "docs/",
   "use_repository_button": True,
   #"use_edit_page_button": True,
   "extra_navbar": f"<p>Last built {today}</p>", # extra footer on left sidebar
}

html_title = f"{project} v{release} docs"
show_navbar_depth = 3

default_role = "py:obj" # so I can type `blah` instead of :class:`blah`

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]