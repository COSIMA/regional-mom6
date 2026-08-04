# regional-mom6 workflow

# Anatomy of a regional MOM6 model

Running a regional MOM6 ocean model requires similar components to a global model, with some modifications and extra files necessary to handle the open boundary conditions.

The *regional-mom6* package puts text files into a `run` directory, and data files into an `input` directory. The text files specify all of the model parameters and settings. Typically, default values are copied from an existing model run that's been sufficiently tested for realism and stability, and then some parameters are modified to match the specifics of the new model.

Input files are more challenging to adapt, requiring care and often substantial computational resources to produce. Except for idealised models where a user might artificially create bathymetry or forcing files, these input files are based on large scientific datasets for bathymetry, tidal forcing, atmospheric conditions at the sea surface and the ocean state itself. Each data source often comes from a different organisation with different variable names, data encoding conventions and grids. Before being used in the model, all of these input datasets must be cut out and modified to match the user's chosen model grid and MOM6's set of conventions.

Arguably, the most important parts of a new model are its horizontal and vertical grids. MOM6 uses an Arakawa C grid (Arakawa & Lamb, 1977), which places tracer quantities at the centre of each rectangular cell ("h-points"), and meridional and zonal velocities at the northern and eastern sides ("q points") of each cell respectively. The horizontal grid is referred to as a "supergrid", as it is the only file which contains information for every single "q" and "h" point, so has twice as many points in each dimension as any other input file.

When run in its flagship Adaptive Lagrangian-Eulerian mode, MOM6 technically only allows the user to specify a "target" vertical grid, to which the model aims to nudge its internal adaptive grid. For the purposes of setting up the model, one can treat this "target" grid as one with fixed vertical depths.

Once the grids and data sources have been decided on, one then generates all of the input files needed for their regional MOM6 model, which are summarised in the table below.

## Table: Summary of required input files for a regional MOM6 model

*Along with the grid points they must be on. "xh" and "yh" refer to zonal tracer and meridional tracer points respectively, and likewise for the velocity "q" points. Vertical grids are less restrictive — as long as they are given as depth levels, MOM6 will interpolate them at runtime. While the open boundary tracers can also be on a generic horizontal grid, we re-grid them to the model grid to reduce overhead.*

| Name | Description | Horizontal Grid |
|---|---|---|
| hgrid | Horizontal grid file containing all h- and q-points | All h & q points |
| vcoord | Vertical coordinate file of target depths | n/a |
| bathymetry | Seafloor topography | xh, yh |
| init_tracers | Initial condition for tracers. Must at least contain temperature and salinity. | xh, yh |
| init_vels | Initial meridional (v) and zonal (u) velocities | u: xq, yh<br>v: xh, yq |
| OBC_segment_n | Timeseries of forcing data open boundary condition at the *n*th boundary segment. Typically includes velocities, tracers and tidal amplitude & phase. The number of segments is unlimited, but must be along a single slice of the super-grid along one axis. | Any depth-level grid, all h & q points on axis parallel to segment |
| Various surface forcing files | Surface fluxes (wind, rain, solar radiation, river runoff etc.) | Any curvilinear grid.<br>MOM6 interpolates online |