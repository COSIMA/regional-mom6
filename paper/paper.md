---
title: 'regional-mom6: Automatic generation of regional configurations for the Modular Ocean Model v6 in Python'

tags:
  - Python
  - ocean modeling
  - regional modeling
  - mom6
authors:
  - name: Ashley J. Barnes
    orcid: 0000-0003-3165-8676
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Navid C. Constantinou
    orcid: 0000-0002-8149-4094
    affiliation: "1, 2, 3"
  - name: Angus H. Gibson
    orcid: 0000-0001-7577-3604
    affiliation: 1
  - name: Andrew E. Kiss
    orcid: 0000-0001-8960-9557
    affiliation: "1, 2"
  - name: Chris Chapman
    orcid: 0000-0002-6030-1951
    affiliation: 4
  - name: John Reilly
    affiliation: 5
  - name: Dhruv Bhagtani
    orcid: 0000-0002-1222-375X
    affiliation: "1, 2"
affiliations:
 - name: Australian National University, Australia
   index: 1
 - name: ARC Centre of Excellence for Climate Extremes, Australia
   index: 2
 - name: ARC Centre of Excellence for the Weather of the 21st Century, Australia
   index: 3
 - name: CSIRO Environment, Hobart, Tasmania, Australia
   index: 4
 - name: University of Tasmania, Australia
   index: 5

date: 28 March 2024
bibliography: paper.bib
---


# Summary

`regional-mom6` is a Python package that provides an easy and versatile way to set up regional configurations of the Modular Ocean Model version 6 (MOM6).

In the ocean, fast and small-scale motions (from ~100m to ~100km varying at time scales of hours to days) play an important role in shaping the large-scale ocean circulation and climate (length scales ~10,000km varying at decadal time scales) [@Melet2022ch2; @deLavergne2022ch3; @Gula2022ch8].
Despite the increase in computational power and the use of graphical processing units that can bring breakthrough performance and speedup [@silvestri2023oceananigansjl], there are always processes, boundary, or forcing features that are smaller than the model's grid spacing and, thus, remain unresolved in operational ocean models.
Regional ocean models can be run at higher resolutions while limiting the required computational resources.

A regional ocean model simulates the ocean only in a prescribed region, which is a subset of the global ocean.
To do that, we need to apply open boundary conditions at the region's boundaries, that is, we need to impose conditions that mimic the oceanic flow that we are not simulating [@Orlanski1976].
For example, \autoref{fig:tasman} shows the surface currents from a regional ocean simulation of the Tasman sea that was configured using the `regional-mom6` package.
The boundaries of the domain depicted in \autoref{fig:tasman} are forced with the ocean flow from a global ocean reanalysis product.
Higher-resolution regional ocean models improve the representation of smaller-scale motions, such as tidal beams, mixing, mesoscale and sub-mesoscale circulation, as well as the oceanic response to smaller-scale bathymetric or coastal features (such as headlands, islands, sea-mounts, or submarine canyons) and surface forcing (such as atmospheric fronts and convective storms).
Regional modelling further allows for the "downscaling" of coarse-resolution global ocean or climate models, permitting the representation of the variation in local conditions that might otherwise be contained within only a few (or even a single!) model grid cells in a global model.

MOM6 is a widely-used open-source, general circulation ocean--sea ice model, written in Fortran [@Adcroft2019MOM6].
MOM6 contains several improvements over its predecessor MOM5 [@griffies2014elements], including the implementation of the Arbitrary-Lagrangian-Eulerian vertical coordinates [@bleck2002gvc; @griffies2020ALE], more efficient tracer advection schemes, and state-of-the art parameterizations of sub-grid scale physics.
Pertinent for our discussion, MOM6 provides support for open boundary conditions and thus is becoming popular for regional ocean modeling studies (see, e.g., @gmd-16-6943-2023, @egusphere-2024-394) in addition to global configurations.
However, setting up a regional configuration for MOM6 can be challenging, time consuming, and often involves using several programming languages, a few different tools, and also manually editing/tweaking some input files.
The `regional-mom6` package overcomes these difficulties, automatically generating a regional MOM6 configuration of the user's choice with relatively simple domain geometry, that is, rectangular domains.

![A snapshot of the surface ocean currents from a regional ocean simulation of the Tasman sea using MOM6. The simulation is forced by the GLORYS and ERA5 reanalysis datasets and configured with a horizontal resolution of 1/80th degree and 100 vertical levels (see @tasmantides for the source code). \label{fig:tasman}](tasman_speed.png){ width=80% }

The `regional-mom6` package takes as input various datasets that contain the ocean initial condition, the boundary forcing (ocean and atmosphere) for the regional domain, and the seafloor topography.
The input datasets can be on the Arakawa A, B, or C grids [@arakawa1977computational]; the package performs the appropriate interpolation using `xESMF` [@xesmf] under the hood, to put the everything on the C grid required by MOM6.
This base grid for the regional configuration can be constructed in two ways,
either by the user defining a desired resolution and choosing between pre-configured options,
or by the user providing pre-existing horizontal and/or vertical MOM6 grids.
The user can use MOM6's Arbitrary-Lagrangian-Eulerian vertical coordinates [@griffies2020ALE], regardless of the native vertical coordinates of the boundary forcing input.
The package automates the re-gridding of all the required forcing input, takes care of all the metadata encoding, generates the regional grid, and ensures that the final input files are in the format expected by MOM6.
Additionally, the tricky case of a regional configuration that includes the 'seam' in the longitude of the raw input data (e.g. a 10ᵒ-wide regional configuration centred at Fiji (178ᵒE) and forced by input with native longitude coordinate in the range 180ᵒW--180ᵒE) is handled automatically, removing the need for any preprocessing of the input data.
This automation allows users to set up a regional MOM6 configuration using only Python and from the convenience of a single Jupyter notebook.
@Herzfeld2011 provide rules of thumb to guide the user in setting regional grid parameters such as the resolution.


`regional-mom6` is installable via `conda`, it is continuously tested, and comes with extensive documentation including tutorials and examples for setting up regional MOM6 configurations using publicly-available forcing and bathymetry datasets (namely, the GLORYS dataset for ocean boundary forcing [@glorys], the ERA5 reanalysis for atmospheric forcing [@era5], and the GEBCO dataset for seafloor topography [@gebco]).

With the entire process for setting up a regional configuration streamlined to run within a Jupyter notebook, the package dramatically reduces the barrier-to-entry for first-time users, or those without a strong background in Fortran, experience in compiling and running scripts in terminals, and manipulating netCDF files.
Besides making regional modelling with MOM6 more accessible, our package can automate the generation of multiple experiments (e.g., a series of perturbation experiments), saving time and effort, and improving reproducibility. 

We designed `regional-mom6` with automation of regional configurations in mind.
However, the package's code design and modularity make more complex configurations possible since users can use their own custom-made grids with more complex boundaries and construct the boundary forcing terms one by one.


# Statement of need

The learning curve for setting up a regional ocean model can be steep, and it is not obvious for a new user what inputs are required, nor the appropriate format.
In the case of MOM6, there are several tools scattered in Github repositories, for example those collected in Earth System Modeling Group grid tools [@gridtools].
Also, there exist several regional configuration examples (e.g., [**ADD 1-2 CITATIONS OF REPOS HERE?**]) but they are hardcoded for particular domains, specific input files, and work only on specific high-performance computing machines.

Until now there has been no one-stop-shop for users to learn how to get a regional MOM6 configuration up and running.
Users are required to use several tools in several programming languages and then modify -- sometimes by hand -- some of the input metadata to bring everything into the format that MOM6 expects.
Many parts of this process are not documented, requiring users to dig into the MOM6 Fortran source code.
Other ocean models have packages to aid in regional configuration setup, for example `Pyroms` [@pyroms] for the Regional Oceanic Modelling System (ROMS; @shchepetkin2005regional) and `MITgcm_python` [@mitgcmpy] for the Massachusetts Institute of Technology General Circulation Model (MITgcm; @marshall1997finite).
With MOM6's growing user base for regional applications, there is a need for a platform that walks users through regional domain configuration from start to finish and, ideally, automates the process on the way.
Other than reducing the barrier-to-entry, automating the regional configuration renders the workflow much more reproducible; see discussion by @polton2023reproducible.
`regional-mom6` precisely meets these needs.

By having a shared set of tools that the community can work with and contribute to, this package also facilitates collaboration and knowledge-sharing between different research groups.
Using a shared framework for setting up regional models, it is easier to compare and contrast examples of different experiments and allows for users to gain intuition for generating their chosen domain.

`regional-mom6` package can also be used for educational purposes, for example as part of course curricula.
With the technically-challenging aspects of setting up a regional configuration now being automated by the `regional-mom6` package, students can set up and run simple MOM6 regional configurations and also change parameters like the model's resolution or the forcing, run again, and see how these parameters affect the ocean flow.

# Acknowledgements

We thank the Consortium for Ocean–Sea Ice Modeling in Australia ([cosima.org.au](https://cosima.org.au)), Josué Martínez-Moreno, and Callum Shakespeare for useful discussions during the development of this package.
We acknowledge support from the Australian Research Council under DECRA Fellowship DE210100749 (N.C.C.) and grant LP200100406 (A.E.K.).
We would also like to acknowledge the code and notes by James Simkins, Andrew Ross, and Rob Cermak, which helped us to troubleshoot and improve the algorithms in our package.

# References
