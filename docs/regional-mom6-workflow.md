# regional-mom6 workflow

regional-mom6 sets up all the data and files for running a basic regional configuration case of MOM6.
This includes:

1. Run files like ``MOM_override``, ``MOM_input``, and ``diag_table``.
2. Boundary condition files like velocity, tracers, tides.
3. Basic input files like horizontal grid (``hgrid``) and the bathymetry.
4. Initial condition files.

regional-mom6 provides the user methods for grabbing and organising the files.

regional-mom6 organizes all files into two directories: an ``input`` directory and a ``run`` directory.
The input directory includes all of the data we need for our regional case. The run directory includes all of the parameters and outputs (``diags``) we want for our model. Please see the structure primer document for more information. The rest of the directories include the data like the initial and the boundary conditions.

Therefore, to start for the user to use regional-mom6, they should have two (empty or not) directories for the input and run files, as well as directories for their input data.

The user may also need to provide a path to ``FRE_tools``. (This depends on the HPC machine used, e.g., on Australia's Gadi this is required, on NCAR-Derecho/Casper this is not required.)

regional_mom6.regional_mom6.experiment

To create all these files, regional-mom6 uses a class called {meth}`experiment <regional_mom6.regional_mom6.experiment>` that holds all of the parameters and methods. Users can follow a few quick steps to setup their cases:
1. Initialise an experiment with all the directories and parameters wanted. The initalisation can also create the horizontal grid and vertical coordinate or read two files called ``hgrid.nc`` and ``vcoord.nc`` from the ``input`` directory.
2. Call different ``setup_this_and_that`` methods to setup all the data needed for the case (bathymetry, initial condition, velocity, tracers, tides).
3. Finally, call {meth}`setup_run_directory <regional_mom6.regional_mom6.experiment.setup_run_directory>` to setup the run files like ``MOM_override`` for their cases.
1. Based on how MOM6 is configured on the machine used, there may be follow-up steps unique to each situation. regional-mom6 provides all of what the user needs to run MOM6.

There are a few convenience functions to help support the process.
1. Very light read and write config file functions to easily save experiments
2. A ``change_MOM_parameter`` method to adjust ``MOM_parameter`` values from within Python.
