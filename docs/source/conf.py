import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "Metoryx"
copyright = "2025, h-terao"
author = "h-terao"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "special-members": "__init__, __call__",
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

html_theme = "shibuya"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
