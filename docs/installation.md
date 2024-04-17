# Installation

We encourage creating a new or using an existing conda environment.

## Easy, clean, one liner via conda

The easiest way to install `regional-mom6` is via [`conda`](https://anaconda.org/conda-forge/regional-mom6).

```bash
conda install conda-forge::regional-mom6
```

That's it -- now enjoy!

## "*But I want pip, can't I install with pip*?"

To install via `pip` is a bit more cumbersome.

A prerequisite is the binary `esmpy` dependency, which provides regridding capabilities.
The easiest way to install `esmpy` is via conda:

```bash
conda install -c conda-forge esmpy
```

Alternatively, to install `esmpy` in a Conda-free way, follow the instructions for [installing ESMPy from
source](https://earthsystemmodeling.org/esmpy_doc/release/latest/html/install.html#installing-esmpy-from-source).
With `esmpy` available, we can then install `regional_mom6` via pip. (If we don't have have pip, then
`conda install pip` should do the job.)

With `esmpy` installed we can now install `regional-mom6` via [`pip`](https://pypi.org/project/regional-mom6/):

```bash
pip install regional-mom6
```

The above installs the version of `regional-mom6` (plus any required dependencies) that corresponds to the latest tagged release of the package.

## "*I want to live on the edge! I want the latest developments*"

To install `regional-mom6` directly via GitHub using `pip`, first install `esmpy` as described above. Then:

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

to get the version that corresponds to the latest commit in GitHub.
Alternatively, install the version that corresponds to a particular git commit using

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```
