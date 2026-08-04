## regional-mom6 Code structure

The role of the *regional-mom6* is to populate an `input` directory with all of the input files in the table above, and the accompanying text files in the `run` directory, based on the user's specifications.

This is done by encapsulating most of the user's preferences in an initial *Experiment* class, which on the initialisation step in the figure below, either generates the grids, or reads in those provided by the user.

If automatic generation is chosen, then the package creates a horizontal grid with lines of constant latitude and longitude, spaced equally in degrees by the horizontal resolution specified by the user. The vertical grid is specified by three conditions: the number of vertical layers, the depth of the lowest layer and the ratio in thickness between the top and bottom layers. The depth should be chosen to match the deepest point in the domain, and the ratio chosen based on whether the research question requires more surface or bottom resolution (e.g. mixed layer dynamics vs topographic interactions). A hyperbolic tangent function then smoothly varies the layer thicknesses between the surface and bottom, matching the user's specifications.

![A schematic of regional-mom6's structure, and its usage split into three steps. The pink "Segment" and "Experiment" boxes represent classes, with class objects attached to them with dark lines. These class objects are shown in blue, and also correspond to files on disk, except for the "settings", which are written to the three grey text files in step 3. The green boxes show the raw input files that are re-gridded. Pink arrows represent methods of each class.](schematic.png)

*Figure: A schematic of regional-mom6's structure and basic usage.*

Aside from the grid specifications, in the initialisation step the user also specifies the minimum allowed depth still considered "ocean", the number of open boundaries, their orientations and which tidal constituents to include. On the initialisation of the *experiment* object, the `input` directory is generated, into which the remaining MOM6 files will be deposited alongside the grids.

The next steps, denoted "re-gridding" in the figure above, are to run a series of methods of the *Experiment* instance to generate the bathymetry, initial conditions and boundary forcing files. Each of these requires three things:

- The path to where the base dataset is stored
- The variable names in this dataset, mapped to the MOM6 variables (e.g. `"y"`:`"latitude"`)
- The Arakawa grid that the source dataset is on, so *regional-mom6* knows how to interpolate it.

With this information, the data is passed onto the interpolation functions to be put onto the model grids. The interpolation functions must first look up the destination grids, as in the table above, and subset the supergrid accordingly. Then, the *XESMF* library is called to conservatively re-grid the data. This can be computationally expensive, and for large domains careful thought in memory management and parallel processing is required, as discussed in the chunking section.

If the model grid is rotated, or any boundary is curved or angled such that the grid does not align with longitude or latitude lines, then velocities — including tidal velocity phases and amplitudes — are rotated appropriately at this step.

If enabled, tidal forcing is included both in the momentum equations by MOM6, and applied at the boundaries. Unlike the other boundary forcing, the tidal forcing is not a timeseries, but rather the complex phases and amplitudes of the TPXO tidal data, re-gridded at the boundaries. The appropriate tidal velocities and surface height anomalies are then retrieved by the model at each time-step, and added to the boundary forcing timeseries of surface height and velocity.

After re-gridding, the bathymetry file needs some additional care before it's ready for use. Firstly, any points in the bathymetry that are shallower than the minimum depth parameter specified on initialisation are set to zero depth. This way, they will be counted as land when MOM6 determines the land-sea mask. Secondly, lakes as well as bays that are connected to the open ocean by a one grid cell wide opening are found and removed, as these constitute numerically unstable, "non-advective" cells on a C-grid. Effectively, a situation can arise whereby water is advected into the bay but not out again, causing runaway sea surface height. An optional additional step is to remove channels that are one grid cell wide. While not inherently unstable on a C-grid, they have been known to cause stability issues in some domains.

Finally, the metadata and encoding of the data are modified to match the specification required by MOM6, and the finished data are saved into the `input` folder, ready for use by the model. With the heavy computation complete, *regional-mom6* now writes all of the configuration and settings files, and saves them in the `run` directory. During the operations so far, the *experiment* instance has kept track of all of the parameter changes that need to be passed to the model.

The "setup" step in the figure above represents running the "set-up run directory" method. This causes a set of default configuration files to be copied to the `run` directory, alongside a layout and override file which have been generated by the package specifically for the user's experiment. The layout file contains details of the grid and CPU layout for parallel processing, and the override file contains every MOM6 parameter that has been changed to run this experiment. All of these parameters kept track of by the *Experiment* instance can be exported as a dictionary for easy sharing and reproducing of experiments, as long as the other party has access to the same forcing datasets.

## regional-mom6 and MOM6_forge

As of 2026, *regional-mom6* is used in conjunction with NCAR's *mom6_forge* package, which now handles all of the grid generation and bathymetry manipulation.
There is little difference to the front end of the package, since *regional-mom6* simply wraps *mom6_forge* grid and bathymetry objects under the hood.
The integration of these two packages is backwards compatible, with all of the original grid and bathymetry functionality of *regional-mom6* simply moved into *mom6_forge*.