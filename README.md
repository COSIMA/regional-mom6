# A regional domain generator for Modular Ocean Model 6

[![codecov](https://codecov.io/gh/COSIMA/mom6-regional/branch/master/graph/badge.svg?token=7OEZ1UZRY4)](https://codecov.io/gh/COSIMA/mom6-regional)

`mom6_regional` contains functions and classes that do most of the legwork in setting up a regional domain in MOM6.
Just provide it with some information about where, when and how big, and point it in the direction of your raw input files, and it'll sort out the boring details to create MOM6-friendly input files. Check out the [demo notebook](https://nbviewer.org/github/COSIMA/mom6-regional-scripts/blob/master/demo.ipynb)!

### Python package
This repository is being converted into a python package that will be distributed via `pip` and `conda`. At the moment, there are still som legacy stuff -- hold on to your chairs as we clean up the repo and the pipelines for generating regional MOM6 configurations.
