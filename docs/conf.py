import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Regional MOM6"
copyright = "2023, Ashley Barnes & COSIMA Contributors"
author = "Ashley Barnes & COSIMA Contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "nbsphinx",
    "rtds_action",
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

# RTDS action
rtds_action_github_repo = "COSIMA/regional-mom6"
# path relative to conf.py to put the rendered notebooks
rtds_action_path = "demo_notebooks"
rtds_action_artifact_prefix = "notebooks-for-"
rtds_action_github_token = os.environ["GITHUB_TOKEN"]
