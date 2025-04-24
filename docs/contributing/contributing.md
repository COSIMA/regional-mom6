Instructions for Contributors
=============================

Before you submit a [pull request](https://github.com/COSIMA/regional-mom6/pulls) it's always a
good idea to run the tests locally and catch any potential bugs/errors that might have been
introduced. Also, sometimes it's also a good idea to build the documentation locally to see
how new docstrings or any new bits of documentation that you may have added look like.


## Testing

To run the tests from a local clone of the repository we first need to create a conda
environment with all the required dependencies.

We create the environment by calling

```{code-block} bash
conda env create --prefix ./env --file environment-ci.yml
```

from the repository's local clone main directory. Then we activate it via

```{code-block} bash
conda activate ./env
```

We then install both the package in this environment as well as the `pytest` package:

```{code-block} bash
python -m pip install .
python -m pip install pytest
```

Now we can run the tests with

```{code-block} bash
python -m pytest tests/
```

If we also want to run the doctests (that is, the tests that appear as examples in
various docstrings), we can use

```{code-block} bash
python -m pytest --doctest-modules tests/ regional_mom6/
```

## Documentation

To build the docs from a local clone of the repository we first need to create a conda
environment after we first navigate to the `docs` directory of our local repository clone.

```{code-block} bash
cd docs
conda create --name docs-env --file requirements.txt
```

We activate this environment and install the package itself as an editable install (`pip install -e`).

```{code-block} bash
conda activate docs-env
pip install -e ..
```

Now we can build the docs via `make`:

```{code-block} bash
make html
```

and upon successful build, we preview the documentation by opening `_build/html/index.html`
in our favorite browser.

Alternatively, instead of creating a conda environment, we can install the required
dependencies for the docs via `pip`; the rest is same, that is

```{code-block} bash
cd docs
pip install -r requirements.txt
pip install -e ..
make html
```
