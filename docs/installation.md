Installation
============

A prerequisite is the binary `esmpy` dependency, which provides regridding capabilities. The easiest way to install `esmpy` is using Conda.
We encourage creating a new or using an existing conda environment, into which we install `esmpy` and `regional_mom6`.
Install `emspy` via:

```bash
conda install -c conda-forge esmpy
```

Alternatively, to install `esmpy` in a Conda-free way, follow the instructions for [installing ESMPy from
source](https://earthsystemmodeling.org/esmpy_doc/release/latest/html/install.html#installing-esmpy-from-source).
With `esmpy` available, we can then install `regional_mom6` via pip. (If we don't have have pip, then
`conda install pip` should do the job.)

Then install `regional-mom6` via [`conda`](https://anaconda.org/conda-forge/regional-mom6):

```bash
conda install conda-forge::regional-mom6
```

or via [`pip`](https://badge.fury.io/py/regional-mom6):

```bash
pip install regional-mom6
```

The above installs the version of `regional-mom6` (plus any required dependencies) that corresponds to the latest tagged release of the package.

`esmpy` won't be installed as a dependency and that's why need to install it separately.

Alternatively, we can install directly via Github, e.g., 

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

to get the version that corresponds to the latest commit in GitHub.
We can also install `regional-mom6` from a particular git commit using, e.g.,

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```
