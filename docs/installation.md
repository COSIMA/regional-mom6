Installation
============

You can install the package via `pip` directly from Github.

To do so, first we create a custom conda environment or activate an environment we already have.
Then we first need to install `emspy`:

```{code-block} bash
conda install -c conda-forge esmpy
```

and then you can install `regional_mom6` via pip.

(If your environment don't have pip then `conda install pip` should do the job.)

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

This will install the latest version of `regional_mom6` plus any required dependencies.
(`esmpy` won't be installed as a dependency and that's why need to install it separately.)

Alternatively, you can also install a particular tag or git commit using, e.g.,

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git@v0.X.X
```

or

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```
