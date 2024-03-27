---
title: 'regional_mom6: A Python package for automatic generation of regional configurations for the Modular Ocean Model 6'

tags:
  - Python
  - ocean modeling
  - regional
  - mom6
authors:
  - name: Ashley J. Barnes
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 16 February 2024
bibliography: paper.bib
---


# Summary

Modular Ocean Model 6 (MOM6) is a widely used general circulation ocean model developed at the Geophysical Fluid Dynamics Laboratory (GFDL) [@Adcroft2019MOM6].
Among other improvements on its predecessor MOM5, this iteration permits open boundary conditions, and MOM6 is subsequently growing in popularity for high resolution regional modelling.
However, setting up a regional domain can be challenging and time consuming for new users even for the simplest rectangular domains.
The `regional_mom6` python package automates much of the regridding, metadata encoding, grid generation and other miscellaneous steps, allowing models to be up and running more quickly.

The `regional_mom6` package takes raw files containing the initial condition, forcing and bathymetry.
These inputs can be on the Arakawa A,B or C grids, and the package performs the appropriate interpolation using `xESMF` (citation needed) onto the C grid required by MOM6.
This base grid can either be constucted based on the user's desired resolution and choice of pre-configured options, or the user can provide their own horizontal or vertical grids.
In either case, the package then handles the coordinates, dimensions, metadata and encoding to ensure that the final input files are in formats expected by MOM6.
The package also comes with pre-configured run directories, which can be automatically copied and modified to match the user's experiment.
Subsequently, a user need only copy a demo notebook, modify the longitude, latitude and resolution, and simply by running the notebook from start to finish will generate all they need for running a MOM6 experiment in their domain of interest.

Although `regional_mom6` was desined to automate the setup as much as possible to aid first time users, it can also be used for more advanced configurations.
The modular desin of the code means that users can use their own custom grids and set up boundaries one-by-one to accommodate more complex domain shapes.

# Statement of need

The learning curve for setting up a regional ocean model can be quite steep.
In the case of MOM6, there are several tools scattered around github like those collected in [ESMG's grid tools](https://github.com/ESMG/gridtools), as well as examples hardcoded for particular domains, input files and hardware.
However, there is no one-stop-shop to learn how to get a regional MOM6 model up and running, meaning that a newcomer must collect many disparate pieces of information from around the internet unless they are able to get help.
Other models have packages to aid in domain setup like [pyroms](https://github.com/ESMG/pyroms) for ROMS and [MITgcm_python](https://github.com/knaughten/mitgcm_python) for MITgcm [@marshall1997finite].
With MOM6's growing user base for regional applications, there is a need for a platform that walks users through regional domain setup from from start to finish, and ideally helps with some of the time consuming parts of the process that ought to be automated.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We thank the Consortium for Ocean–Sea Ice Modeling in Australia ([cosima.org.au](https://cosima.org.au)) for useful discussions during the development of this package.
N.C.C. acknowledges funding from the Australian Research Council under DECRA Fellowship DE210100749.

# References
