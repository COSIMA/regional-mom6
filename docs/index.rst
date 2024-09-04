Regional MOM6 Documentation
===========================

`regional-mom6`_ is a Python package for automatic generation of regional configurations for the `Modular Ocean Model version 6`_ (MOM6).

In brief...
-----------

Users just need to provide some information about where, when, and how big their domain
is and also where raw input forcing files are. The package sorts out all the boring details
and creates a set of MOM6-friendly input files along with setup directories ready to go! 

The idea behind this package is that it should let the user sidestep some of the tricky
issues with getting the model to run in the first place. This removes some of the steep
learning curve for people new to working with MOM6. Note that the resultant model configuration
might still need some tweaking (e.g., fiddling with timestep to avoid CFL-related numerical
stability issues or fiddling with bathymetry to deal with very narrow fjords or channels that may exist).


Features
--------

- Automatic grid generation at chosen vertical and horizontal grid spacing.
- Automatic removal of non-advective cells from the bathymetry that cause the model to crash.
- Handle slicing across 'seams' in of the forcing input datasets (e.g., when the regional
  configuration includes longitude 180 and the forcing longitude is defined in [-180, 180]).
- Handles metadata encoding.
- Creates directory structure with the configuration files as expected by MOM6.
- Handles interpolation and interpretation of input data. No pre-processing of forcing datasets
  is required. (In some cases, slicing the forcing dataset before helps with hitting limitations
  related to the machine's available memory.)


Limitations
------------

- Only generates regional horizontal grids with uniform spacing in longitude and latitude.
  However, users can provide their own non-uniform grid, or ideally `open a pull request`_
  with a method that generates other types of horizontal grids.
- Only supports boundary segments that are parallel to either lines of constant longitude or lines of
  constant latitude.


What you need to get started
----------------------------

1. a cool idea for a new regional MOM6 domain,
2. a working MOM6 executable on a machine of your choice, 
3. a bathymetry file that at least covers your domain,
4. 3D ocean forcing files *of any resolution* on your choice of A, B, or C Arakawa grid,
5. surface forcing files (e.g., from ERA or JRA reanalysis), and
6. `GFDL's FRE tools <https://github.com/NOAA-GFDL/FRE-NCtools>`_ downloaded and compiled on the machine you are using.

Browse through the `demos <demos.html>`_.

Citing
------

If you use regional-mom6 in research, teaching, or other activities, we would be grateful 
if you could mention regional-mom6 and cite our paper in JOSS:

    Barnes et al., (2024). regional-mom6: A Python package for automatic generation of regional configurations for the Modular Ocean Model 6. *Journal of Open Source Software*, **9(100)**, 6857, doi:`10.21105/joss.06857 <https://doi.org/10.21105/joss.06857>`_.

The bibtex entry for the paper is:

.. code-block:: bibtex

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


.. _regional-mom6: https://github.com/COSIMA/regional-mom6
.. _Modular Ocean Model version 6: https://github.com/mom-ocean/MOM6
.. _open a pull request: https://github.com/COSIMA/regional-mom6/pulls


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   demos
   mom6-file-structure-primer
   api
   contributing
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
