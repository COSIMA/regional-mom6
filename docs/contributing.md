Instructions for Contributors
=============================

Before you submit a [pull request](https://github.com/COSIMA/regional-mom6/pulls) it's always a
good idea to run the tests locally and catch any potential bugs/errors that might have been
introduced. Also, sometimes it's also a good idea to build the documentation locally to see
how new docstrings or any new bits of documentation that you may have added look like.


## Testing

To run the tests from a local clone of the repository we first need to create a conda
environment with all the dependencies required. From the repositories local clone main
directory do

```{code-block} bash
conda env create --prefix ./env --file environment-ci.yml
```

and then activate it

```{code-block} bash
conda activate ./env
```

Now we need to install the package in this environment as well as `pytest`

```{code-block} bash
python -m pip install .
python -m pip install pytest
```

Now we can run the test with

```{code-block} bash
python -m pytest tests/
```

## Documentation

To build the docs from a local clone of the repository we first need to create a conda
environment. Navigate to the `docs` directory of your local repository clone (e.g., `cd docs`)
and then 

```{code-block} bash
cd docs
conda create --name docs-env --file requirements.txt
```

Then activate this environment and install the package itself as an editable install (`pip install -e`).

```{code-block} bash
conda activate docs-env
pip install -e ..
```

Now we can make the docs

```{code-block} bash
make html
```

and open `_build/html/index.html` in your favorite browser.

Alternatively, we can install the dependencies needed for the docs via `pip`; the rest is same, that is

```{code-block} bash
cd docs
pip install -r requirements.txt
pip install -e ..
make html
```
