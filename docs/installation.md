Installation
============

#### Easy, clean, one liner

The easiest way is to install `regional-mom6` via [`conda`](https://anaconda.org/conda-forge/regional-mom6).
We encourage creating a new or using an existing conda environment and then simply

```bash
conda install conda-forge::regional-mom6
```

That's it -- now enjoy!

#### "*But I want `pip`, can't I install with `pip`*?"

We can install via `pip` but it's a bit more cumbersome.
Again, we encourage creating a new or using an existing conda environment.

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

#### "*I'd like to be on the cutting edge of the development*?"

Alternatively, we can install directly `regional-mom6` directly via GitHub using `pip`.
First install `esmpy` as described above and then:

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

to get the version that corresponds to the latest commit in GitHub.
Or, install the version that corresponds to a particular git commit using (for example)

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```
