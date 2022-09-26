from itertools import cycle
import os

import dask
import pandas as pd
import xarray as xr

from dask.diagnostics import ProgressBar

"""
Configurable variables:

base:
  This is the base directory for temporarily holding the subset of the
  forcing data in zarr format. We default to the local compute node
  scratch disk, ``PBS_JOBFS``. For more experimental setups, or if you
  don't want to run the entire pipeline in one go, this could point
  to, e.g. ``/scratch``.

t:
  This is the selection of output directories whose data fully
  overlaps the desired forcing period. This makes the assumption that
  you're using the *01deg_jra55v13_ryf9091* experiment to force your
  model. If this isn't the case, you'll have to load in your data
  differently, e.g. using the COSIMA Cookbook.

run_year:
  This is the actual year of output data we want to use for our
  forcing. It will be selected from the output files that were
  concatenated together according to the ``t`` parameter.

"""

def time_rotate(d,run_year = 2170):
    before_start_time = f"{run_year}-12-31"
    after_end_time = f"{run_year+1}-01-01"

    left = d.sel(time=slice(after_end_time, None))
    left["time"] = pd.date_range("1991-01-01 12:00:00", periods=120)

    right = d.sel(time=slice(None, before_start_time))
    right["time"] = pd.date_range("1991-05-01 12:00:00", periods=245)

    return xr.concat([left, right], "time")

def prepare_segments(xextent,yextent,run_year = 2170):
    base = os.getenv("PBS_JOBFS")
    t = range(1077, 1082)

    # Everything that follows shouldn't need further configuration, if you're using the
    # same experiment

    t = range(1077, 1082) 
    base = os.getenv("PBS_JOBFS")

    surface_tracer_vars = ["temp", "salt"]
    line_tracer_vars = ["eta_t"]
    surface_velocity_vars = ["u", "v"]
    surface_vars = surface_tracer_vars + surface_velocity_vars

    chunks = {
        "T": {"time": 1, "st_ocean": 7, "yt_ocean": 300, "xt_ocean": None},
        "U": {"time": 1, "st_ocean": 7, "yu_ocean": 300, "xu_ocean": None},
    }

    in_datasets = {}
    for var, staggering in list(zip(surface_tracer_vars, cycle("T"))) + list(
        zip(surface_velocity_vars, cycle("U"))
    ):
        d = xr.open_mfdataset(
            [
                f"/g/data/ik11/outputs/access-om2-01/01deg_jra55v13_ryf9091/output{i}/ocean/ocean_daily_3d_{var}.nc"
                for i in t
            ],
            chunks=chunks[staggering],
            combine="by_coords",
            parallel=False,
        )[var]
        in_datasets[var] = staggering, d

    # line datasets, assume they all come from ocean_daily
    d_2d = xr.open_mfdataset(
        [
            f"/g/data/ik11/outputs/access-om2-01/01deg_jra55v13_ryf9091/output{i}/ocean/ocean_daily.nc"
            for i in t
        ],
        chunks={"time": 1, "yt_ocean": 300, "xt_ocean": None},
        combine="by_coords",
        parallel=False,
    )[line_tracer_vars]

    d_tracer = xr.merge([d for s, d in in_datasets.values() if s == "T"] + [d_2d])
    d_velocity = xr.merge([d for s, d in in_datasets.values() if s == "U"])

    # time slicing

    d_tracer = time_rotate(d_tracer.sel(time=slice(f"{run_year}-05-01", f"{run_year+1}-04-30")))
    d_velocity = time_rotate(d_velocity.sel(time=slice(f"{run_year}-05-01", f"{run_year+1}-04-30")))

    # reduce selection around target latitude
    # and remove spatial chunks (required for xesmf)
    d_tracer = d_tracer.sel(yt_ocean=slice(yextent[0] - 1, yextent[1] + 1), xt_ocean=slice(xextent[0] - 1,xextent[1] + 1)).chunk(
        {"yt_ocean": None, "xt_ocean": None}
    )
    d_velocity = d_velocity.sel(yu_ocean=slice(yextent[0] - 1, yextent[1] + 1), xu_ocean=slice(xextent[0] - 1,xextent[1] + 1)).chunk(
        {"yu_ocean": None, "xu_ocean": None}
    )

    with ProgressBar():
        d_tracer.to_zarr(
            f"{base}/tracer.zarr",
            encoding={"time": {"dtype": "double", "units": "days since 1900-01-01 12:00:00", "calendar": "noleap"}},
            mode = "w"
        )

    with ProgressBar():
        d_velocity.to_zarr(
            f"{base}/velocity.zarr",
            encoding={"time": {"dtype": "double", "units": "days since 1900-01-01 12:00:00", "calendar": "noleap"}},
            mode = "w"
        )
        