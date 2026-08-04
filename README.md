# regional-mom6

*Python package for automatic generation of regional configurations for the [Modular Ocean Model version 6](https://github.com/mom-ocean/MOM6) (MOM6)*

[![Repo status](https://www.repostatus.org/badges/latest/active.svg?style=flat-square)](https://www.repostatus.org/#active)
[![conda forge](https://img.shields.io/conda/vn/conda-forge/regional-mom6.svg)](https://anaconda.org/conda-forge/regional-mom6)
[![pypi](https://badge.fury.io/py/regional-mom6.svg)](https://badge.fury.io/py/regional-mom6)
[![Documentation Status](https://readthedocs.org/projects/regional-mom6/badge/?version=latest)](https://regional-mom6.readthedocs.io/en/latest/?badge=latest)

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://mit-license.org)
[![codecov](https://codecov.io/gh/COSIMA/regional-mom6/branch/main/graph/badge.svg?token=7OEZ1UZRY4)](https://codecov.io/gh/COSIMA/regional-mom6)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![status](https://joss.theoj.org/papers/d396435c09aae4c2f4b62cdbc1493c1e/status.svg)](https://joss.theoj.org/papers/d396435c09aae4c2f4b62cdbc1493c1e)


## Features
- Generates multiple types of horizontal and vertical grid utilising NCAR's [MOM6_forge](https://github.com/NCAR/mom6_forge/tree/280ab8f8321d4033d41e4b05f1645fe511cd552f)
- Removes non-advective cells from the bathymetry that cause the model to crash.
- Interpolates input data, which can be on any Arakawa grid at any resolution. No pre-processing of forcing datasets is generally required.
- Converts ERA5 surface data to fields appropriate for MOM6 surface forcing
- Handle slicing across 'seams' in of the forcing input datasets (e.g., when the regional
  configuration includes longitude 180 and the forcing longitude is defined in [-180, 180]).
- Handles metadata encoding.
- Creates directory structure with the configuration files as expected by MOM6.
- Produces MOM6 namelist files matching your experiment

Regional-mom6 is designed to be machine agnostic as much as possible, meaning that as long as you have a working MOM6 executable on your computer, this package gets you most of the way towards running your MOM6 configuration. However, additional support is available for the two main institutions who use and maintain regional-mom6: [COSIMA](https://cosima.org.au/) and NCAR's [CROCODILE project](https://github.com/CROCODILE-CESM/). 

Check out the [documentation](https://regional-mom6.readthedocs.io/en/latest/) and try the [demos](https://regional-mom6.readthedocs.io/en/latest/demos.html).

## For COSIMA / Gadi users

There's an [example notebook](https://github.com/COSIMA/regional-mom6/blob/main/demos/ACCESS-rOM3-demo.ipynb) that's designed specifically for Gadi users. This is the best place to start! Aside from the paths defined in this notebook which are gadi-specific, the other important part is right at the end: the [`.setup_rom3()`](https://github.com/COSIMA/regional-mom6/blob/1c48b714ac9ae55de9b3e92f0efb8aa37a0ec0cb/regional_mom6/regional_mom6.py#L1921) method will set up the [ACCESS-NRI supported](https://access-om3-configs.access-hive.org.au/latest/contributing/Overview/) version of MOM6 ready to run with the [Payu workflow manager](https://github.com/payu-org/payu)

## For people using the Community Earth System Model Framework (CESM)

CESM users should check out the [CrocoDash](https://github.com/CROCODILE-CESM/CrocoDash) package wraps regional-mom6 (among other things) to set up regional models within the CESM framework. 

## For users outside Australia and the U.S

This package can still be used to set up your model! The only catch is that you need to supply an executable built on your machine. We maintain a machine agnostic [demo](https://regional-mom6.readthedocs.io/en/latest/demos.html) on how to use regional-mom6 for everything short of _running_ the model.

## We want to hear from you

If you have any suggestions please feel free to open an [issue](https://github.com/COSIMA/regional-mom6/issues) or start a [discussion](https://github.com/COSIMA/regional-mom6/discussions). We welcome any [new contributors](https://regional-mom6.readthedocs.io/en/latest/contributing/contributing.html) and we are very keen to help you out along the way! 


## Installation

We encourage creating a new or using an existing conda environment.

#### Easy, clean, one liner via conda

The easiest way to install `regional-mom6` is via [`conda`](https://anaconda.org/conda-forge/regional-mom6).

```bash
conda install conda-forge::regional-mom6
```

That's it -- now enjoy!

#### "*But I want pip, can't I install with pip*?"

To install via `pip` is a bit more cumbersome.

A prerequisite is the binary `esmpy` dependency, which provides re-gridding capabilities.
The easiest way to install `esmpy` is via conda:

```bash
conda install -c conda-forge esmpy
```

Alternatively, to install `esmpy` in a Conda-free way, follow the instructions for [installing ESMPy from
source](https://earthsystemmodeling.org/esmpy_doc/release/latest/html/install.html#installing-esmpy-from-source).
With `esmpy` available, we can then install `regional-mom6` via pip. (If we don't have have pip, then
`conda install pip` should do the job.)

With `esmpy` installed we can now install `regional-mom6` via [`pip`](https://pypi.org/project/regional-mom6/):

```bash
pip install regional-mom6
```

The above installs the version of `regional-mom6` (plus any required dependencies) that corresponds to the latest tagged release of the package.

#### "*I want to live on the edge! I want the latest developments*"

To install `regional-mom6` directly from the [GitHub repository](https://github.com/COSIMA/regional-mom6/) using `pip`, first install `esmpy` as described above. Then:

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git
```

to get the version that corresponds to the latest commit in GitHub.
Alternatively, install the version that corresponds to a particular git commit using, for example,

```bash
pip install git+https://github.com/COSIMA/regional-mom6.git@061b0ef80c7cbc04de0566df329c4ea472002f7e
```


## Getting started

The [example notebooks](https://regional-mom6.readthedocs.io/en/latest/demos.html) walk you through how to use
the package using two different sets of input datasets.
Please ensure that you can get at least one of these working on your setup with your MOM6 executable before trying to modify the example to suit your domain with your bathymetry, forcing, and boundary conditions.

You can download the notebooks [from Github](https://github.com/COSIMA/regional-mom6/tree/main/demos) or by clicking on the download <img width="22" alt="download" src="https://github.com/COSIMA/regional-mom6/assets/7112768/2c1ae149-c6a8-4395-ab09-2f77588008d9"> button, e.g., at the top-right of the [regional Tasmania forced by ERA5 example](https://regional-mom6.readthedocs.io/en/latest/demo_notebooks/reanalysis-forced.html).

## Citing

If you use regional-mom6 in research, teaching, or other activities, we would be grateful
if you could mention regional-mom6 and cite our paper in JOSS:

> Barnes et al., (2024). regional-mom6: A Python package for automatic generation of regional configurations for the Modular Ocean Model 6. _Journal of Open Source Software_, **9(100)**, 6857, doi:[10.21105/joss.06857](https://doi.org/10.21105/joss.06857).

The bibtex entry for the paper is:

```bibtex
@article{regional-mom6-JOSS,
  doi = {10.21105/joss.06857},
  url = {https://doi.org/10.21105/joss.06857},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {100},
  pages = {6857},
  author = {Ashley J. Barnes and Navid C. Constantinou and Angus H. Gibson and Andrew E. Kiss and Chris Chapman and John Reily and Dhruv Bhagtani and Luwei Yang},
  title = {{regional-mom6: A Python package for automatic generation of regional configurations for the Modular Ocean Model 6}},
  journal = {Journal of Open Source Software}
}
```
