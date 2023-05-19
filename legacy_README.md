# Rather than deleting for now I'm just leaving the original readme here. 

It's very out of date and only relevant to the 'mom6_regional_scripts.py' library

# mom6-regional-scripts
Scripts for cutting out regional boundary forcing for MOM6. These are
based on the more ad-hoc setups from our first attempts at setting up
regional simulations. First, there was [mom6-panan], which was fairly
straightforward, as it only needs the northern boundary, and is
zonally periodic. Originally, this was also scaled up to double the
resolution, which complicated things slightly. The next step toward
generalising the input pipeline was [mom6-eac]. This uses the same
resolution as the forcing, but there are four boundaries to deal with.

Here, the aim is to make the input pipeline general, reproducible, and
extensible: we should be able to target any domain with a minimum of
fuss.

[mom6-panan]: https://github.com/cosima/mom6-panan


## Pipeline steps
There are a few steps in the pipeline, from the processing of input
data to the generation of the boundary forcing segments
themselves. We'll start with an overall description of the pipeline
and the steps involved.

UPDATE 16/09
There's now a jupyter notebook that steps through this pipeline linearly. This contains some additional documentation too.

### Create model grid
Once the target domain has been decided, a MOM6-compatible grid needs
to be generated. For the easiest interoperability with the forcing, we
can simply subset the global grid used for the input data. With the
grid in the *supergrid* (double-resolution, describing cell centres
and corners) format, we can use `make_hgrid` from [FRE-NCtools]. At
this point, we should also consider generating (or subsetting) the
topography for the region. This can be used with the
`make_solo_mosaic`, `make_quick_mosaic` and `check_mask` tools to
generate the remaining required inputs.

[FRE-NCtools]: https://github.com/NOAA-GFDL/FRE-NCtools

### Input preparation
The main input for this process is one year of daily output from
ACCESS-OM2-01. We need five fields on the boundaries, and optionally
another few fields for the initial condition. On the boundaries, we
need temperature, salt, velocities, and sea surface height
(eta). These are contained in the following files:
- `ocean_daily.nc` (contains eta)
- `ocean_daily_3d_temp.nc`
- `ocean_daily_3d_salt.nc`
- `ocean_daily_3d_u.nc`
- `ocean_daily_3d_v.nc`

As an initial condition, we should have temperature and salt through
the entire domain. Velocities may also be used, but it may be worth
letting the model spin those up itself.

After locating the input files, we do some preprocessing to subset the
entire input field down to only our target domain. One part of this
step is to do with the JRA55 *repeat year forcing* (RYF). Instead of
beginning at the start of the year, the forcing dataset starts in
May. To line the boundary forcing up with the atmospheric forcing, we
shift the selected data: select a year starting in May, but move the
first four months to the start. Because this is a temporary dataset
and we value efficient processing over longevity, the temporary
domains are output in the [zarr] format.

[zarr]: https://zarr.readthedocs.io

### Interpolate initial condition
At this stage, the pieces are in place to be able to interpolate the
initial condition onto the regional grid. In fact, for a tracer-only
initial condition, nothing needs to be done if the input grid is
aligned exactly with the regional grid: the tracer points are
consistent between the Arakawa B- and C-staggerings, and we have just
cut out our exact domain in the previous step. If velocities are
required, they must be interpolated from the B-grid corners to the
cell faces for MOM6's C-grid.

### Segment interpolation
The open boundary forcing data needs to be on the model *supergrid*
(double-resolution), so we need to interpolate it up from the input
dataset. This is a pretty straightforward linear interpolation. We
also make sure to fill in all the NaNs from the input: MOM6 doesn't
mask off any points vertically, so if we don't do this, we could
introduce NaNs into our computational domain and crash the model.

### Runoff regridding
Most of the atmospheric forcing fields are interpolated onto the model
domain on the fly. Because of this, we don't even have to do anything
in most cases. However, the runoff field is an exception: because it
expects to run in a global domain, non-zero runoff is routed to the
nearest wet cell to its source location. In a regional domain, this
means that a majority of the world's runoff ends up at a single point
on the domain boundary! The runoff regridding is a smarter offline
tool that makes sure the online algorithm doesn't have any data
outside the domain to erroneously place on the boundaries.


## Running the scripts
With an understand for what the scripts do, and how they work, let's
look now at actually using them to generate a set of inputs to force a
regional model.

### Preparing the environment
On Gadi, the `conda/analysis3-unstable` environment contains all the
packages that are needed to run all the steps. Simply load the corresponding module:

```shell
module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
```

Otherwise, create a new virtual environment and install into it the
dependencies. It's possible that this approach will fail, because
xesmf requires ESMPy which isn't available through PyPI, and must be
installed manually.

```shell
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Preparing the inputs
There are three key input datasets that are needed for this process:

1. `ocean_hgrid.nc`, which can be extracted from ACCESS-OM2 or
   generated from scratch
2. topography, which may similarly be extracted from an existing
   dataset or generated fresh
3. the boundary forcing data

Along with these, it's also important to know the boundaries of your
regional domain in terms of both its latitude/longitude, and its
indices in the boundary forcing data. The former is useful for doing
coarse subsetting with xarray, and the latter for extracting the
domain with the NCO tools.

Using the `ocean_hgrid.nc` and the topography, a grid mosaic,
including a land mask can be generated:

```shell
# only add periodx for a periodic domain!
make_solo_mosaic --num_tiles 1 --dir . --mosaic_name ocean_mosaic \
    --tile_file ocean_hgrid.nc --periodx 360

make_quick_mosaic --input_mosaic ocean_mosaic.nc --mosaic_name grid_spec \
    --ocean_topog ocean_topog.nc
```

### Running the scripts
With everything in place, it's just a matter of modifying and running
the scripts in sequence. Start with `prepare_segments.py`. This makes
the very broad assumption that you'd like to force your model with
ACCESS-OM2 at 0.1 degree, from the *01deg_jra55v13_ryf9091*
experiment. If this is the case, the configurable variables exist at
the top of the file, and may be changed to suit your domain. In the
default configuration, we also expect that `prepare_segments.py` is
followed by `brushcut_xesmf_zarr.py` in the same batch job. For this
reason, the temporary segments that are cut out are saved in zarr
format into `PBS_JOBFS`. If you'd like to keep the segments for
longer, make sure to output them to a non-ephemeral disk!

Once the segments are generated, it's time to cut out the open
boundary forcing. The `brushcut_xesmf_zarr.py` script needs to know
the base path where the segments are located, and the path to the
`ocean_hgrid.nc` for your regional domain. There are two other
requirements for this script: it will save both the regridding weights
and the output segments into the *current directory*. Ensure you run
it from a location where you have sufficient storage, i.e. not your
home directory. As an optimisation, if you have run the script before
on the same domain, you may reuse the regridding weights to save
regenerating them.

We currently don't handle the case where the grid for the initial
condition isn't coincident with the regional grid. As such, we make
the assumption that tracers are colocated and their initial condition
can be handled by the model online through the ALE core. If we added
the ability to remap tracers onto the model's horizontal grid, it
would go here. However, if you also want a velocity initial condition,
but the data is on a B-grid, you can use the `interp_initial.py`
script to interpolate it from cell corners to edges, to be compatible
with C-grid staggering. This needs a bit of manual work with NCO to
cut out the initial fields. Then the script only needs to know where
those initial fields are, and your `ocean_hgrid.nc` location. It will
also create its output in the *current directory*, so either change
this, or make sure you run it from a sensible location.

Finally, there's the `regrid_runoff.py` script that's important if you
care about runoff forcing. It needs the `ocean_mask.nc` that was
generated as part of the mosaic, the `ocean_hgrid.nc`, and the runoff
forcing file, e.g. from the JRA55v13 dataset. As before, the output
will be generated in the *current directory*.

Once the set of scripts has been run, you're ready to configure your
model! See [mom6-panan] and [mom6-eac] for some examples.
