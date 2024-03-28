Installation
============

We can install `regional_mom6` via `pip` from GitHub. A prerequisite of is that the binary `esmpy`
dependency is first installed. The easiest way to install `esmpy` is using Conda.

First we create a custom Conda environment, or activate an existing environment into which we
will install `esmpy` and `regional_mom6`. Then install `emspy` via:

```{code-block} bash
conda install -c conda-forge esmpy
```

Alternatively, to install `esmpy` in a Conda-free way, follow the instructions for [installing ESMPy from
source](https://earthsystemmodeling.org/esmpy_doc/release/latest/html/install.html#installing-esmpy-from-source).
With `esmpy` available, we can then install `regional_mom6` via pip. (If we don't have have pip, then `conda install pip`
should do the job.)

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

The above installs the version of `regional_mom6` (plus any required dependencies) that corresponds
to the latest commit in GitHub. `esmpy` won't be installed as a dependency and that's why need to
install it separately.

We can also install `regional_mom6` from a particular tag or git commit using, e.g.,

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git@v0.X.X
```

or

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```
