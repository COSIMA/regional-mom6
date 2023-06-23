# regional_mom6

*Python package for automatic generation of regional configurations for the [Modular Ocean Model 6](https://github.com/mom-ocean/MOM6).*

[![codecov](https://codecov.io/gh/COSIMA/regional-mom6/branch/master/graph/badge.svg?token=7OEZ1UZRY4)](https://codecov.io/gh/COSIMA/regional-mom6) [![Documentation Status](https://readthedocs.org/projects/regional-mom6/badge/?version=latest)](https://regional-mom6.readthedocs.io/en/latest/?badge=latest) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Users just need to provide some information about where, when, and how big their domain is and also where raw input forcing files are.
The package sorts out all the boring details and creates a set of MOM6-friendly input files along with setup directories ready to go!

Check out the the [documentation](https://regional-mom6.readthedocs.io/en/latest/) and browse through the [demos](https://regional-mom6.readthedocs.io/en/latest/demos.html).


### Installation

At the moment you can install the package via `pip` from Github. To do so, you can first create a custom
conda environment or activate an environment you already have. Then we need to install `emspy`

```bash
conda install -c conda-forge esmpy
```

and then you can install `regional_mom6` via pip.

(If your environment don't have pip then `conda install pip` should do the job.)

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

This will install the latest version of `regional_mom6` plus any required dependencies.
(`esmpy` won't be installed as a dependency and that's why need to install it separately.)

Alternatively, you can also install a particular tag or git commit using, e.g.,

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@v0.X.X
```

or

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```
