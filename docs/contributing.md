Instructions for Contributors
=============================


## Documentation

To build the docs from a local clone of the repository we first need to create a conda
environment. Navigate to the `docs` directory of your local repository clone (e.g., `cd docs`)
and then 

```{bash}
cd docs
conda create --name my-docs-env-name --file requirements.txt
```

Then activate this environment and run `make`:

```{bash}
conda activate my-docs-env-name
make html
```

and open `_build/html/index.html` in your favorite browser.

Alternatively, we can also install the dependencies needed for the docs via `pip`.

```bash
cd docs
pip install -r docs/requirements.txt
make html
```
