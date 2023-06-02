# A regional domain generator for Modular Ocean Model 6

The `regional_library.py` file contains functions and classes that do most of the legwork in setting up a regional domain in MOM6.
Just give it some information about where, when and how big, and point it in the direction of your raw input files, and it'll sort out the boring details to create MOM6-friendly input files.

Note: We are aiming to convert this into a python package and disctribute it via `pip` and `conda`. At the moment, there's still some legacy stuff in this repo for now while we get the last of the functionality into the pipeline.

Check out the [demo notebook](https://nbviewer.org/github/COSIMA/mom6-regional-scripts/blob/master/demo.ipynb)!
