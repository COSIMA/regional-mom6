Installation
============

At the moment you can install the package via `pip` from
GitHub. Before this, the binary `esmpy` dependency is required. This
is easiest to install using Conda. To do so, first create a custom
Conda environment, or activate an existing environment into which you
want to install `esmpy` and `regional_mom6`. Then install `emspy`:

```{code-block} bash
conda install -c conda-forge esmpy
```

Alternatively, it's possible to follow the [Installing ESMPy from
Source](https://earthsystemmodeling.org/esmpy_doc/release/latest/html/install.html#installing-esmpy-from-source)
instructions to do this in a Conda-free way. With `esmpy` available, you can then install
`regional_mom6` via pip. If your environment doesn't yet have pip, then `conda install pip` should do the job.

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

This will install the latest version of `regional_mom6` plus any required dependencies.
`esmpy` won't be installed as a dependency and that's why need to install it separately.

Alternatively, you can also install a particular tag or git commit using, e.g.,

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git@v0.X.X
```

or

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```
