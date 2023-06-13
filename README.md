# A regional domain generator for Modular Ocean Model 6

[![codecov](https://codecov.io/gh/COSIMA/mom6-regional/branch/master/graph/badge.svg?token=7OEZ1UZRY4)](https://codecov.io/gh/COSIMA/mom6-regional)

The `regional_library.py` file contains functions and classes that do most of the legwork in setting up a regional domain in MOM6.
Just give it some information about where, when and how big, and point it in the direction of your raw input files, and it'll sort out the boring details to create MOM6-friendly input files.

Note: We are aiming to convert this repository into a python package and distribute it via `pip` and `conda`. At the moment, there's still some legacy stuff -- hold on to your chairs as we clean up the repo and the pipelines for generating regional mom6 configurations.

Check out the [demo notebook](https://nbviewer.org/github/COSIMA/mom6-regional-scripts/blob/master/demo.ipynb)!
