# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "regional-mom6"
copyright = "2024, COSIMA community and outside contributors"
author = "COSIMA community and outside contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_mock_imports = ["xesmf"]

# Theming options
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/COSIMA/regional-mom6",
    "use_repository_button": True,
}

# Disable demo notebook execution by nbsphinx (and therefore readthedocs notebook execution)
nbsphinx_execute = "never"
