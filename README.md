# regional_mom6

*Python package for automatic generation of regional configurations for the [Modular Ocean Model 6](https://github.com/mom-ocean/MOM6).*

[![Repo status](https://www.repostatus.org/badges/latest/active.svg?style=flat-square)](https://www.repostatus.org/#active) [![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://mit-license.org) [![codecov](https://codecov.io/gh/COSIMA/regional-mom6/branch/main/graph/badge.svg?token=7OEZ1UZRY4)](https://codecov.io/gh/COSIMA/regional-mom6) [![Documentation Status](https://readthedocs.org/projects/regional-mom6/badge/?version=latest)](https://regional-mom6.readthedocs.io/en/latest/?badge=latest) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Users just need to provide some information about where, when, and how big their domain is and also where raw input forcing files are. The package sorts out all the boring details and creates a set of MOM6-friendly input files along with setup directories ready to go! 

The idea behind this package is that it should let the user sidestep some of the tricky issues with getting the model to run in the first place. This removes some of the steep learning curve for people new to working with MOM6. Note that the resultant model configuration might still need some tweaking (e.g., fiddling with timestep to avoid CFL-related numerical stability issues or fiddling with bathymetry to deal with very narrow fjords or channels that may exist).

**Features**

- Automatic grid generation at chosen vertical and horizontal grid spacing.
- Automatic removal of non-advective cells from the bathymetry that cause the model to crash.
- Handle slicing across 'seams' in of the forcing input datasets (e.g., when the regional configuration spans the longitude 180 of a global dataset that spans [-180, 180]).
- Handles metadata encoding.
- Modifies pre-made configuration files to match your experiment
- Handles interpolation and interpretation of input data. Limited pre-processing of your forcing data required!

Limitations: Currently the package only comes with one function for generating a horizontal grid, namely one that's equally spaced in longitude and latitude. However, users can provide their own grid, or ideally open a PR with their desired grid generation function and we'll include it as an option! Further, only boundary segments parallel to longitude or latitude lines are currently supported. 

If you find this package useful and have any suggestions please feel free to open an [issue](https://github.com/COSIMA/regional-mom6/issues) or a [discussion](https://github.com/COSIMA/regional-mom6/discussions). We'd love to have [new contributors](https://regional-mom6.readthedocs.io/en/latest/contributing.html) and we are very keen to help you out along the way!

## What you need to get started:
1. a cool idea for a new regional MOM6 domain,
2. a working MOM6 executable on a machine of your choice, 
3. a bathymetry file that at least covers your domain,
4. 3D ocean forcing files *of any resolution* on your choice of A, B, or C Arakawa grid,
5. surface forcing files (e.g., from ERA or JRA reanalysis), and
6. [GFDL's FRE tools](https://github.com/NOAA-GFDL/FRE-NCtools) be downloaded and compiled on the machine you are using.

Check out the [documentation](https://regional-mom6.readthedocs.io/en/latest/) and browse through the [demos](https://regional-mom6.readthedocs.io/en/latest/demos.html).

## Installation

We can install `regional_mom6` via `pip` from GitHub. A prerequisite is the binary `esmpy`
dependency, which provides regridding capabilities. The easiest way to install `esmpy` is using Conda.
We encourage using a new or existing conda environment, into which we will install `esmpy` and `regional_mom6`. 
Then install `emspy` via:

```bash
conda install -c conda-forge esmpy
```

Alternatively, to install `esmpy` in a Conda-free way, follow the instructions for [installing ESMPy from
source](https://earthsystemmodeling.org/esmpy_doc/release/latest/html/install.html#installing-esmpy-from-source).
With `esmpy` available, we can then install `regional_mom6` via pip. (If we don't have have pip, then
`conda install pip` should do the job.)

```{code-block} bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

The above installs the version of `regional_mom6` (plus any required dependencies) that corresponds
to the latest commit in GitHub. `esmpy` won't be installed as a dependency and that's why need to
install it separately.

We can also install `regional_mom6` from a particular tag or git commit using, e.g.,

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@v0.X.X
```

or

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```

## MOM6 Configuration and Version Requirements

The package and demos assume a coupled MOM6-SIS2 configuration, but also work for MOM6 ocean-only configuration after appropriate changes in the `input.nml` and `MOM_input` files.

Additionally, regional configurations require that the MOM6 executable _must_ be compiled with **symmetric memory**.

The current release of this package assumes the latest source code of all components needed to run MOM6 as of
January 2024. A forked version of the [`setup-mom6-nci`](https://github.com/ashjbarnes/setup-mom6-nci) repository
contains scripts for compiling MOM6 and, furthermore, its [`src`](https://github.com/ashjbarnes/setup-mom6-nci/tree/setup-mom6/src)
directory lists the particular commits that were used to compile MOM6 and its submodules for this package.

Note that the commits used for MOM6 submodules (e.g., Flexible Modelling System (FMS), coupler, SIS2) are _not_
necessarily those used by the GFDL's [`MOM6_examples`](https://github.com/NOAA-GFDL/MOM6-examples) repository.

## Getting started


The [example notebooks](https://regional-mom6.readthedocs.io/en/latest/demos.html) walk you through how to use
the package using two different sets of input datasets.
Please ensure that you can get at least one of these working on your setup with your MOM6 executable before trying modify the example to suit your domain with your bathymethry, forcing, and boundary conditions.

You can download the notebooks [from Github](https://github.com/COSIMA/regional-mom6/tree/ncc/installation/demos) or by clicking on the download <img width="22" alt="download" src="https://github.com/COSIMA/regional-mom6/assets/7112768/2c1ae149-c6a8-4395-ab09-2f77588008d9"> button, e.g., at the top-right of the [regional tasmania forced by ERA5 example](https://regional-mom6.readthedocs.io/en/latest/demo_notebooks/reanalysis-forced.html).

