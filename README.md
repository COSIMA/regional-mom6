# regional_mom6

*Python package for automatic generation of regional configurations for the [Modular Ocean Model 6](https://github.com/mom-ocean/MOM6).*

[![Repo status](https://www.repostatus.org/badges/latest/active.svg?style=flat-square)](https://www.repostatus.org/#active)
[![conda forge](https://img.shields.io/conda/vn/conda-forge/regional-mom6.svg)](https://anaconda.org/conda-forge/regional-mom6)
[![pypi](https://badge.fury.io/py/regional-mom6.svg)](https://badge.fury.io/py/regional-mom6)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://mit-license.org)
[![codecov](https://codecov.io/gh/COSIMA/regional-mom6/branch/main/graph/badge.svg?token=7OEZ1UZRY4)](https://codecov.io/gh/COSIMA/regional-mom6)
[![Documentation Status](https://readthedocs.org/projects/regional-mom6/badge/?version=latest)](https://regional-mom6.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Users just need to provide some information about where, when, and how big their domain is and also where raw input forcing files are. The package sorts out all the boring details and creates a set of MOM6-friendly input files along with setup directories ready to go! 

The idea behind this package is that it should let the user sidestep some of the tricky issues with getting the model to run in the first place. This removes some of the steep learning curve for people new to working with MOM6. Note that the resultant model configuration might still need some tweaking (e.g., fiddling with timestep to avoid CFL-related numerical stability issues or fiddling with bathymetry to deal with very narrow fjords or channels that may exist).

Limitations: Currently the package supports only one horizontal grid type (that is equally spaced in longitude); there are plans to add more grid options. We have designed the package in a way that it is modular so, for example, one needs to implement just another method for a different type of grid and the rest should be good to go.

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
Or, install the version that corresponds to a particular git commit using

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```

## MOM6 Configuration and Version Requirements

The package and demos assume a coupled SIS2-MOM6 configuration.
The examples could work for an ocean-only MOM6 run but this has not been tested. 

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

**Note**

The `xesmf` the package attempts to regrid in parallel, and if it's not able to do so, it throws a warning and
runs in serial. You also get a print out of the relevant `mpirun` command which you could use as a backup.
Depending on your setup of your machine, you may need to write scripts that implement the package to access more
computational resources than might be available, e.g., on the HPC machine of you are working on.
