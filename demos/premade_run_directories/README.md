# Premade Run Directories

**UPDATE IN 2026**
All of the premande run directories here are for FMS versions of mom6. If you're looking to run the ACCESS-NRI supported ACCESS-regional-OM3 (rOM3) model, look at the demo notebook for setting up rOM3. This pulls the configuration files from an [ACCESS-NRI repository](https://github.com/ACCESS-NRI/access-om3-configs/tree/helen/dev-rM_generic_jra_iaf), so that the regional-mom6 package always gets the most up to date executable etc. Previously, these configuration files (MOM_input, config.yaml etc.) lived in regional mom6.

The premade run directories here are left for backwards compatability, but be warmed that FMS configurations won't be as well maintained given that NCAR users and now most Australian users run with NUOPC rather than FMS. 

These directories are used for the demo notebooks, and can be used as templates for setting up a new experiment. The [documentation](https://regional-mom6.readthedocs.io/en/latest/mom6-file-structure-primer.html) explains what all the files are for.

The `common_files` folder contains all of the required files for a regional MOM6 run, including a `data_table` with constant surface forcing as a placeholder. The other two directories offer different choices for the surface forcing, affecting the `data_table`.
