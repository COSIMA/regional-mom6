# Regional MOM6 Workflow

Regional MOM6(RM6) sets up all the data and files for running a basic regional case of MOM6.

It includes:

1. Run Files like MOM_override, MOM_input, diag_table
2. BC File like velocity, tracers, tides
3. Basic input files like hgrid & bathymetry.
4. Initial Condition files


To set up a case with all the files, RM6 has its own way of grabbing and organizing files for the user.

RM6 organizes all files into two directories, a "input" directory, and a "run" directory. The input directory includes all of the data we need for our regional case. The run directory includes all of the parameters and outputs (diags) we want for our model. Please see the structure primer document for more information.

The other folders are the data directories. RM6 needs the user to collect the initial condition & boundary condition data and put them into a folder(s). 

Therefore, to start for the user to use RM6, they should have two (empty or not) directories for the input and run files, as well as directories for their input data. 

Depending on the computer used (NCAR-Derecho/Casper doesn't need this), the user may also need to provide a path to FRE_tools. 

To create all these files, RM6 using a class called "Experiment" to hold all of the parameters and functions. Users can follow a few quick steps to setup their cases:
1. Initalize the experiment object with all the directories and parameters wanted. The initalization can also create the hgrid and vertical coordinate or accept two files called "hgrid.nc" and "vcoord.nc" in the input directory.
2. Call different "setup..." functions to setup all the data needed for the case (bathymetry, initial condition, velocity, tracers, tides).
3. Finally, call "setup_run_directory" to setup the run files like MOM_override for their cases.
4. Based on how MOM6 is setup on the computer, there are follow-up steps unique to each situation. RM6 provides all of what the user needs to run MOM6.


There are a few convience functions to help support the process.
1. Very light read and write config file functions to easily save experiments
2. A change_MOM_parameter function to easily adjust MOM parameter values from python.