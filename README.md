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
[mom6-eac]: https://github.com/cosima/mom6-eac/


## Pipeline steps
There are a few steps in the pipeline, from the processing of input
data to the generation of the boundary forcing segments
themselves. We'll start with an overall description of the pipeline
and the steps involved.

## Create model grid
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
ACESS-OM2-01. We need five fields on the boundaries, and optionally
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
