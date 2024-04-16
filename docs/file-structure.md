MOM6 file structure
============

This section describes the various directories and files that `regional-mom6` package produces.
A better understanding of what these files do will help with troubleshooting and more advanced customisations.

## The `run` directory

The directory, specified by the `mom_run_dir` path keyword argument in the `experiment` class, contains only text files that configure MOM6 and are used at model initialisation.
You can see examples of these files in the `premade_run_directories`.
These files are:

* `input.nml`:
  High-level information that is passed directly to each component of your MOM6 setup.
  The paths of to the `SIS` and `MOM` input directories and outputs are included.
  The `coupler` section turns on or off different model components, and specifies how long to run the experiment for. 

* `diag_table`:
  The diagnostics to save from your model. You can't keep everything! Consider the things that are most important for your experiment - you can fill up disk space very fast if you save too much. Different lines in the diagnostic table either specify a new output *file* and its associated characteristics, or a new output *variable* and the matching file that it should go in (which needs to already have been specified!). If you're not sure which diagnostics to pick, you can run the model for 1 hour and look in the output folder. Here there'll be a file called `available_diags` which lists every possible diagnostic for your model configuration. Here it will also tell you which grids you're allowed to output them on. Aside from the native model grid, you can create your own set of vertical coordinates. To output on your custom vertical coordinate, create a netcdf containing all of the vertical points (be they depths or densities) go to the `MOM_input` file and specify additional diagnostic coordinates there. Then, you can pick these coordinates in the diag table.

Documentation as to how to format the file can be found [here](https://mom6.readthedocs.io/en/dev-gfdl/api/generated/pages/Diagnostics.html). 

`data_table`
The data table is read by the coupler to provide different model components with inputs. For example, for our ocean only model runs, atmospheric forcing data ought to be provided. This can either be a constant value, or a dataset as in the reanalysis-forced demo. As more model components are included, the data table has less of a role to play. However, unless you want risk freezing or boiling the ocean you'll usually need to provide solar fluxes at a minimum! 

Documentation as to how to format the file can be found [here](https://mom6.readthedocs.io/en/dev-gfdl/forcing.html). 

`MOM_input / SIS_input`
These files provide the basic settings for the core MOM and SIS code. The settings themselves are reasonably well documented. After running the experiment for a short amount of time, you can find a `MOM_parameter_doc.all` file which lists every possible setting your can modify for your experiment. The MOM_regional package can copy and modify a default set of input files to work with your experiment. There's too much in these files to explain here. The aforementioned vertical diagnostic coordinates are specified here, as are all of the different parameterisation schemes and hyperparameters used by the model. Some really important ones are the timesteps which will likely need to be fiddled with to get your model running quickly but stably. However, it can be more helpful to specify these in the `MOM_override` file instead. 

Another important part section for regional modelling is the specification of open boundary segments. You need to include a separate line for each boundary in your domain, and specify any additional tracers that need be included.

`MOM_override`
This file serves to override settings chosen in other input files. This is helpful for making a distinction between the thing you're fiddling with and 95% of the settings that you'll always be leaving alone. For instance, you might need to temporarily change your baroclinic (`DT`), thermal (`DT_THERM`) or baroclinic (`DT_BT`) timesteps, or are doing perturbation experiments that requires you to switch between different bathymetry files.

`config file`
This file is machine dependent and environment dependent. For instance, if you're using Australia's National Computational Infrastructure (NCI), then you're likely using the `payu` framework, and you'll have a `config.yml` file. Regardless of what it looks like, this file should contain information that points to the executable, your input directory (aka the `mom_input_dir` you specified), the computational resources you'd like to request and other various settings. 

The package does come with a premade `config.yml` file for payu users which is automatically copied and modified when the appropriate flag is passed to the `setup_rundir` method. If you find this package useful and you use a different machine, I'd encourage you to provide an example config file for your institution! Then this could be copied into.


## The run directory
This is the folder referred to by the `mom_input_dir` path. Here we have mostly NetCDF files that are read by MOM6 at runtime. These files can be big, so it's usually helpful to store them somewhere where disk space isn't an issue. 

`hgrid.nc`
This is the horizontal grid that the model runs on. Known as the 'supergrid', it contains twice as many x and y points as you might expect. This is because *all* points on the Arakawa C grid are included. Since you're running a regional experiment, you'll be using the 'symmetric memory' configuration of the MOM6 executable. This means that the horizontal grids boundary must be entirely composed of cell edge points (like those used by velocities). So, if you have a model with 300 x cells, the `nx` dimension will be 601 wide. 

The `nx` and `ny` points are where data is stored, whereas `nxp` and `nyp` here define the spaces between points used to compute area. The x and y variables in `hgrid` refer to the longitude and latitude. Importantly, x and y both depend on `nyx` and `nyp` meaning that the grid doesn't have to follow lines of constant latitude or longitude. If you make your own custom horizontal and vertical grids, you can simply set `read_existing_grid` to `True` when creating the experiment object.

`vcoord.nc`
This specifies the values of the vertical coordinate. By default this package sets up a `z*` vertical coordinate but others can be provided and the `MOM_input` file adjusted appropriately. If you want to customise the vertical coordinate, you can initialise an `experiment` object to begin with, then modify and re-save the `vcoord.nc`. You can provide more vertical coordinates (giving them a different name of course) for diagnostic purposes. These allow your diagnostics to be remapped and output on this coordinate at runtime. 

`bathymetry.nc`
Fairly self explanatory, but can be the source of some difficulty. The package automatically attempts to remove "non advective cells". These are small enclosed lakes at the boundary that can cause numerical problems whereby water might flow in but have no way to flow out. Likewise, there can be issues with very shallow (only 1 or 2 layers) or very narrow (1 cell wide) channels. If your model runs for a while but then gets extreme sea surface height values, it could be caused by an unlucky combination of boundary and bathymetry.

Another thing to note is that the bathymetry interpolation can be computationally intensive. If using a high resolution dataset like GEBCO and a large domain, you might not be able to execute the `.setup_bathymetry` method in a Jupyter notebook if such notebooks have restricted computational capacity. Instructions for running the interpolation via `mpirun` are printed on execution of the `.setup_bathymetry` method in case this is an issue. 

`forcing/init_*.nc`
These are the initial conditions bunched into velocities, tracers and the free surface height (`eta`). 

`forcing/forcing_segment*`
These are the boundary forcing segments, numbered the same way as in MOM_input. The dimensions and coordinates are fairly confusing, and getting them wrong can likewise cause some cryptic error messages! These boundaries don't have to follow lines of constant longitude and latitude, but it is much easier to set things up if they do. For an example of a curved boundary, see this [Northwest Atlantic experiment](https://github.com/jsimkins2/nwa25/tree/main).