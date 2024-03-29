---
title: 'regional_mom6: Automatic generation of regional configurations for the Modular Ocean Model 6 in Python'

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
    affiliation: "1, 2"
  - name: Angus H. Gibson
    orcid: 0000-0001-7577-3604
    affiliation: 1
  - name: Chris Chapman
    orcid: 0000-0002-6030-1951
    affiliation: 3
  - name: Dhruv Bhagtani
    orcid: 0000-0002-1222-375X
    affiliation: "1, 2"
  - name: John Reily
    affiliation: 4
  - name: Andrew E. Kiss
    orcid: 0000-0001-8960-9557
    affiliation: "1, 2"
affiliations:
 - name: Australian National University, Australia
   index: 1
 - name: ARC Centre of Excellence in Climate Extremes, Australia
   index: 2
 - name: CSIRO Oceans and Atmosphere, Hobart, Tasmania, Australia
   index: 3
 - name: University of Tasmania, Australia
   index: 4

date: 28 March 2024
bibliography: paper.bib
---


# Summary

Modular Ocean Model 6 (MOM6) is a widely-used general circulation open-source ocean-sea ice-ice shelf model developed mainly at the Geophysical Fluid Dynamics Laboratory (GFDL) [@Adcroft2019MOM6].
Among other improvements over its predecessor MOM5 (citation), MOM6 allows open boundary conditions and thus it is becoming popular also for regional ocean modeling studies (see, e.g., @gmd-16-6943-2023, @egusphere-2024-394).
However, setting up a regional configuration for MOM6 can be challenging and time consuming and often involves using several programming languages, a few different tools, and also editing/fiddling some input files by hand.
The `regional_mom6` python package automates the procedure of generating a regional MOM6 configuration.

The `regional_mom6` package takes as input various datasets that containing the ocean initial condition, the boundary forcing (ocean and atmosphere) for the regional domain, and the bathymetry.
The inputs datasets can be on the Arakawa A, B, or C grids; the package performs the appropriate interpolation using `xESMF` [@xesmf] under the hood, to put the everything on the C grid required by MOM6.
Thus, the package automates the re-gridding of all the required forcing input, takes care of all the metadata encoding, generates the regional grid, and deals with few other miscellaneous steps required.
This allows users to setup a regional MOM6 configurations using only Python and from a single Jupyter notebook.

<!-- The `regional_mom6` package takes raw files containing the initial condition, the boundary forcing, and bathymetry.
These inputs can be on the Arakawa A, B, or C grids, and the package performs the appropriate interpolation using `xESMF` [@xesmf] onto the C grid required by MOM6.
This base grid can either be constructed based on the user's desired resolution and choice of pre-configured options, or the user can provide their own horizontal or vertical grids.
In either case, the package then handles the coordinates, dimensions, metadata and encoding to ensure that the final input files are in the format expected by MOM6.
Additionally, the tricky case of a `seam' in the longitude of the raw input data (for instance at -180 and 180) is handled automatically, removing the need for any preprocessing of the data. 
The package also comes with pre-configured run directories, which can be automatically copied and modified to match the user's experiment.
Subsequently, a user need only copy a demo notebook, modify the longitude, latitude and resolution, and simply by running the notebook from start to finish will generate all they need for running a MOM6 experiment in their domain of interest. -->

`regional_mom6` is continuously tested and comes with an extensive documentation that also includes documented tutorials/examples for setting up regional MOM6 configurations using publicly-available forcing and bathymetry datasets (namely, the GLORYS dataset for ocean boundary forcing [@glorys], the ERA5 for the atmospheric forcing [@era5], and the GEBCO dataset for bathymetry [@gebco]).

Having the entire process for setting up a regional configuration running in a Jupyter notebook dramatically reduces the barrier of entry for first-time users, or those without a strong background in FORTRAN, experience in compiling and running scripts in terminals, and manipulating netCDF files.
Besides making regional modelling with MOM6 more accessible, our package can automate the generation of multiple experiments (e.g., series of perturbation experiments), saving time, effort, and improving reproducibility. 

We designed `regional_mom6` with automation of regional configurations in mind.
However, the package's code design and modularity allows makes more complex configurations possible since users can use their own custom-made grids with more complex boundaries and construct the boundary forcing terms one by one.

<!-- As more advanced use cases emerge, users can contribute their grid generation functions as well as example configuration files and notebooks.  -->

![A snapshot of surface speed in MOM6 simulation in the Tasman sea. The source code can be found at https://github.com/ashjbarnes/tasman-tides. \label{fig:example}](tasman-speed.png){ width=80% }.

Figure \autoref{fig:example} is an example of a domain set up with the `regional_mom6` package. In this case, the forcing datasets are GLORYS and ERA5, with an 80th degree horizontal resolution and 100 vertical levels. 

# Statement of need

The learning curve for setting up a regional ocean model can be quite steep.
In the case of MOM6, there are several tools scattered in Github repositories, like, for example, those collected in ESMG's grid tools [@gridtools].
Also, there exist several regional configuration example but that are hardcoded for particular domains, specific input files, and work only on certain HPC machines.

There is no one-stop-shop for users to learn how to get a regional MOM6 configuration up and running.
Users are required to use several tools and in several programming languages and then modify --sometimes by hand-- some of the input metadata to bring everything in the format that MOM6 expects.
Often parts of this process are not documented bringing users down to their knees since they end up having to figure it out a lot of things by themselves, e.g., by deciphering FORTRAN code.
Other ocean models have packages to aid in regional configuration setup.
Characteristically, `Pyroms` [@pyroms] for the Regional Oceanic Modelling System (ROMS; @shchepetkin2005regional) and `MITgcm_python` [@mitgcmpy] for the Massachusetts Institute of Technology General Circulation Model (MITgcm; @marshall1997finite).
With MOM6's growing user base for regional applications, there is a need for a platform that walks users through regional domain configuration from from start to finish and, ideally, automates the process on the way.
`regional_mom6` comes to fill in precisely this need.

<!-- A package also provides a standardised way of setting up regional models, allowing for more efficient troubleshooting. 
This is particularly important as the MOM6 boundary code is still under active development, meaning that an old example found Github may not work as intended with a newer executable.
Currently, it is difficult to discern what the best model settings are for a particular experiment with a given MOM6 executable. 
However, having different releases of a python package tied to releases of the MOM6 executable will help users avoid difficult to diagnose compatibility errors between the MOM6 codebase, input file formats and parameter files. -->

By having a shared set of tools that the community can work with and contribute to, this package also facilitates collaboration and knowledge-sharing between different research groups.
<!-- For instance, the Australian ocean modelling community built a set of tools known as the COSIMA Cookbook (cite github repo).
Alongside the tools grew a set of contributed examples for post-processing and analysis of model outputs. -->
Using a shared framework for setting up regional models, it is easier to compare and contrast examples of different experiments and allows for user to gain intuition for generating their chosen domain.

Another potential advantage of a package that allows users to automatically obtain regional configurations of MOM6 is in education.
With the technically-challenging aspects of setting up a regional configuration now being automated by `regional_mom6` package, students can setup and run simple MOM6 regional configurations and also change parameters like resolution or forcing, run again, and see how these parameters affect the ocean flow.

# Acknowledgements

We thank the Consortium for Ocean–Sea Ice Modeling in Australia ([cosima.org.au](https://cosima.org.au)) for useful discussions during the development of this package.
N.C.C. acknowledges funding from the Australian Research Council under DECRA Fellowship DE210100749.
We would also like to acknowledge the code and notes by James Simkins, Andrew Ross, and Rob Cermak, which helped us to troubleshoot and improve the algorithms in our package.

# References
