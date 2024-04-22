A primer on MOM6 file structure
===============================

Here we describe the various directories and files that the `regional-mom6` package produces, and provide
useful insights that, hopefully, will help users deal with troubleshooting and more advanced customisations.

## `run` directory

The directory, specified by the `mom_run_dir` path keyword argument in the `experiment` class, contains only text files that configure MOM6 and are used at model initialisation.
You can see examples of these files in [`demos/premade_run_directories`](https://github.com/COSIMA/regional-mom6/tree/main/demos/premade_run_directories).
These files are:

* `input.nml`:
  High-level information that is passed directly to each component of your MOM6 setup.
  The paths of the `SIS` and `MOM` input directories and outputs are included.
  The `coupler` section turns on or off different model components, and specifies how long to run the experiment for. 

* `diag_table`:
  The diagnostics to output at model runtime.
  We need to choose wisely the quantities/output frequency that are relevant to our experiment and the
  analysis we plan to do otherwise the size of output can quickly become excessive.
  Each line in the `diag_table` either specifies a new output *file* and its associated characteristics,
  or a new output *variable* and the matching file that it should go in (which needs to already have been
  specified).
  If uncertain of the available diagnostics, we can run the model for a short period (e.g., 1 hour) and then
  look in the output directory for `available_diags` that lists every available diagnostic for our
  model configuration also mentioning which grids the quantity can be output on.
  Aside from the native model grid, we can create our own custom vertical coordinates to output on.
  To output on a custom vertical coordinate, create a netCDF file that contains all of the vertical points
  (in the coordinate of your choice) and then edit the `MOM_input` file to specify additional diagnostic
  coordinates.
  After that, we can use this custom vertical coordinate in the `diag_table`.

  Instructions for how to format the `diag_table` are included in the [MOM6 documentation](https://mom6.readthedocs.io/en/dev-gfdl/api/generated/pages/Diagnostics.html).

* `data_table`:
  The data table is read by the coupler to provide the different model components with inputs.

  Instructions for how to format the `data_table` are included in the [MOM6 documentation](https://mom6.readthedocs.io/en/dev-gfdl/forcing.html). 

* `field_table`:
  The field table defines tracer fields.

* `MOM_input / SIS_input`:
  Basic settings for the core MOM and SIS code with reasonably good documentation.
  After running the experiment for a short amount of time, you can find a `MOM_parameter_doc.all` file which lists every possible setting your can modify for your experiment.
  The `regional-mom6` package can copy and modify a default set of input files to work with your experiment.
  There's too much in these files to explain here.
  The aforementioned vertical diagnostic coordinates are specified here, as are all of the different parameterisation schemes and hyperparameters used by the model.
  Some of these parameters are important, e.g., the timesteps, which will likely need to be fiddled with to get your model running quickly but stably.
  However, it can be more helpful to specify these in the `MOM_override` file instead. 

  Another important part section for regional modelling is the specification of open boundary segments.
  A separate line for each boundary in our domain is included and also any additional tracers need to be specified here.

* `MOM_override`:
  This file serves to override settings chosen in other input files. This is helpful for making a distinction between the thing you're fiddling with and 95% of the settings that you'll always be leaving alone. For instance, you might need to temporarily change your baroclinic (`DT`), tracer (`DT_THERM`) or barotropic (`DT_BT`) timesteps, or are doing perturbation experiments that require you to switch between different bathymetry files.

* `MOM_layout`:
  This file provides information on the model grid, processor layout and I/O layout.

* `config` file:
  This file is machine dependent and environment dependent. For instance, if you're using Australia's National Computational Infrastructure (NCI), then you're likely using the [`payu`](https://payu.readthedocs.io/en/latest/) framework, and you'll have a `config.yaml` file. Regardless of what it looks like, this file should contain information that points to the executable, your input directory (aka the `mom_input_dir` you specified), the computational resources you'd like to request and other various settings. 

  The package does come with a premade `config.yaml` file for payu users which is automatically copied and modified when the appropriate flag is passed to the `setup_rundir` method. If you find this package useful and you use a different machine, I'd encourage you to provide an example config file for your institution!

## `input` directory

The `mom_input_dir` directory stores mostly netCDF files that are read by MOM6 at runtime.
These files can be big, so it is usually helpful to store them somewhere without any disk limitations. 

* `hgrid.nc`
  The horizontal grid that the model runs on. Known as the 'supergrid', it contains twice as many points in each
  horizontal dimension as one would expect from the domain extent and the chosen resolution. This is because *all*
  points on the Arakawa C grid are included: both velocity and tracer points live in the 'supergrid'. For a regional
  configuration, we need to use the 'symmetric memory' configuration of the MOM6 executable. This implies that the
  horizontal grid's boundary must be entirely composed of cell edge points (i.e. those used by velocities). Therefore,
  for example, a model configuration that is 20 degrees wide in longitude and has 0.5 degrees longitudinal resolution, would have 40 cells in the `x` dimension and thus a supergrid with `nx = 41`. 

  The `nx` and `ny` points are where data is stored, whereas `nxp` and `nyp` here define the spaces between points
  used to compute area. The `x` and `y` variables in `hgrid` refer to the longitude and latitude. Importantly, `x`
  and `y` are both two-dimensional (they both depend on both `nx` and `ny`) meaning that the grid does not have
  to follow lines of constant latitude or longitude. Users who create their own own custom horizontal and vertical
  grids can set `read_existing_grid` to `True` when creating an experiment.

* `vcoord.nc`
  The values of the vertical coordinate. By default, `regional-mom6` sets up a `z*` vertical coordinate, but other
  coordinates may be provided after appropriate adjustments in the `MOM_input` file. Users who would like to
  customise the vertical coordinate can initialise an `experiment` object to begin with, then modify the `vcoord.nc`
  file and save. Users can provide additional vertical coordinates (under different names) for diagnostic purposes.
  These additional vertical coordinates allow diagnostics to be remapped and output during the model run. 

* `bathymetry.nc`
  Fairly self-explanatory, but can be the source of some difficulty. The package automatically attempts to remove "non-advective cells". These are small enclosed lakes at the boundary that can cause numerical problems whereby water might flow in but have no way to flow out. Likewise, there can be issues with very shallow (only 1 or 2 layers) or very narrow (1 cell wide) channels. If your model runs for a while but then gets extreme sea surface height values, it could be caused by an unlucky combination of boundary and bathymetry.

  Another thing to note is that the bathymetry interpolation can be computationally intensive. For a high-resolution
  dataset like GEBCO and a large domain, one might not be able to execute the `.setup_bathymetry` method within
  a Jupyter notebook. In that case, instructions for running the interpolation via `mpirun` will be printed upon
  executing the `setup_bathymetry` method.

* `forcing/init_*.nc`
  The initial conditions bunched into velocities, tracers, and the free surface height. 

* `forcing/forcing_segment*`
  The boundary forcing segments, numbered the same way as in `MOM_input`. The dimensions and coordinates are fairly
  confusing, and getting them wrong can likewise cause some cryptic error messages! These boundaries do not have to
  follow lines of constant longitude and latitude, but it is much easier to set things up if they do. For an example
  of a curved boundary, see this [Northwest Atlantic experiment](https://github.com/jsimkins2/nwa25/tree/main).
