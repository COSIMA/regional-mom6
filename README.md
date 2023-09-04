# regional_mom6

*Python package for automatic generation of regional configurations for the [Modular Ocean Model 6](https://github.com/mom-ocean/MOM6).*

[![Repo status](https://www.repostatus.org/badges/latest/active.svg?style=flat-square)](https://www.repostatus.org/#active) [![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://mit-license.org) [![codecov](https://codecov.io/gh/COSIMA/regional-mom6/branch/master/graph/badge.svg?token=7OEZ1UZRY4)](https://codecov.io/gh/COSIMA/regional-mom6) [![Documentation Status](https://readthedocs.org/projects/regional-mom6/badge/?version=latest)](https://regional-mom6.readthedocs.io/en/latest/?badge=latest) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Users just need to provide some information about where, when, and how big their domain is and also where raw input forcing files are. The package sorts out all the boring details and creates a set of MOM6-friendly input files along with setup directories ready to go! 

Although the resultant regional configuration will likely still need some tweaking, the idea behind this package is that the model should help the user sidestep some of the tricky issues with getting the model to run in the first place. This removes some of the steep learning curve for people new to working with the model.

Limitations: Currently the package supports only one horizontal grid type (that is equally spaced in longitude); there are plans to add more grid options. We have designed the package in a way that it is modular so, for example, one needs to implement just another method for a different type of grid and the rest should be good to go.

If you find this package useful and have any suggestions please feel free to open an [issue](https://github.com/COSIMA/regional-mom6/issues) or a [discussion](https://github.com/COSIMA/regional-mom6/discussions). We'd love to have [new contributors](https://regional-mom6.readthedocs.io/en/latest/contributing.html) and we are very keen to help you out along the way!

## What you need to get started
1. A cool idea for a new regional MOM6 domain
2. A working MOM6 executable on a machine of your choice. 
3. A bathymetry file that at least covers your domain
4. 3D ocean forcing files *of any resolution* on your choice of A, B or C Arakawa grid
5. Surface forcing files (eg ERA or JRA reanalysis)

Check out the the [documentation](https://regional-mom6.readthedocs.io/en/latest/) and browse through the [demos](https://regional-mom6.readthedocs.io/en/latest/demos.html).

## Installation

At the moment you can install the package via `pip` from
GitHub. Before this, the binary `esmpy` dependency is required. This
is easiest to install using Conda. To do so, first create a custom
Conda environment, or activate an existing environment into which you
want to install `esmpy` and `regional_mom6`. Then install `emspy`:

```bash
conda install -c conda-forge esmpy
```

Alternatively, it's possible to follow the [Installing ESMPy from
Source](https://earthsystemmodeling.org/esmpy_doc/release/latest/html/install.html#installing-esmpy-from-source)
instructions to do this in a Conda-free way. With `esmpy` available, you can then install
`regional_mom6` via pip. If your environment doesn't yet have pip, then `conda install pip` should do the job.

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

This will install the latest version of `regional_mom6` plus any required dependencies.
`esmpy` won't be installed as a dependency and that's why need to install it separately.

Alternatively, you can also install a particular tag or git commit using, e.g.,

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@v0.X.X
```

or

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```


## Getting started

The two example notebooks walk you through how to use the package on two different sets of input datasets. Make sure that you can get at least one of these working on your setup with your MOM6 executable, and then try to modify them to apply to your domain with your bathymethry / forcing / boundaries. 
