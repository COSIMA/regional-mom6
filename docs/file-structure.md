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
  The diagnostics to save from your model run.
  Choose wisely the quantities that are relevant to your experiment and the analysis you plan to do otherwise you can fill up disk space very fast.
  Different lines in the * `diag_table` either specify a new output *file* and its associated characteristics, or a new output *variable* and the matching file that it should go in (which needs to already have been specified).
  If uncertain regarding which diagnostics to pick, try running the model for a short period (e.g., 1 hour) and look in the output folder.
  There, you'll find a file `available_diags` that lists every available diagnostic for your model configuration also mentioning  which grids the quantity can be output on.
  Aside from the native model grid, we can create our own custom vertical coordinates to output on.
  To output on a custom vertical coordinate, create a netCDF that contains all of the vertical points (in the coordinate of your choice) and then edit the `MOM_input` file to specify additional diagnostic coordinates.
  After that, we are able to select the custom vertical coordinate in the `diag_table`.

  Instructions for how to format the `diag_table` are included in the [MOM6 documentation](https://mom6.readthedocs.io/en/dev-gfdl/api/generated/pages/Diagnostics.html).

* `data_table`
  The data table is read by the coupler to provide different model components with inputs.
  With more model components we need more inputs.

  Instructions for how to format the `data_table` are included in the [MOM6 documentation](https://mom6.readthedocs.io/en/dev-gfdl/forcing.html). 

* `MOM_input / SIS_input`
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