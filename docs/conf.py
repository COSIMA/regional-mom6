# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MOM6 Regional"
copyright = "2023, Ashley Barnes & COSIMA Contributors"
author = "Ashley Barnes & COSIMA Contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
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
