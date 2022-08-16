from itertools import cycle
import os

import dask
import pandas as pd
import xarray as xr

from dask.diagnostics import ProgressBar

base = os.getenv("PBS_JOBFS")

t = range(1077, 1082)

surface_tracer_vars = ["temp", "salt"]
line_tracer_vars = ["eta_t"]
surface_velocity_vars = ["u", "v"]
surface_vars = surface_tracer_vars + surface_velocity_vars

chunks = {
    "T": {"time": 1, "st_ocean": 7, "yt_ocean": 300, "xt_ocean": None},
    "U": {"time": 1, "st_ocean": 7, "yu_ocean": 300, "xu_ocean": None},
}

def time_rotate(d):
    left = d.sel(time=slice("2171-01-01", None))
    left["time"] = pd.date_range("1991-01-01 12:00:00", periods=120)

    right = d.sel(time=slice(None, "2170-12-31"))
    right["time"] = pd.date_range("1991-05-01 12:00:00", periods=245)

    return xr.concat([left, right], "time")

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

d_tracer = time_rotate(d_tracer.sel(time=slice("2170-05-01", "2171-04-30")))
d_velocity = time_rotate(d_velocity.sel(time=slice("2170-05-01", "2171-04-30")))

# reduce selection around target latitude
# and remove spatial chunks (required for xesmf)
d_tracer = d_tracer.sel(yt_ocean=slice(-49, -5), xt_ocean=slice(-217, -183)).chunk(
    {"yt_ocean": None, "xt_ocean": None}
)
d_velocity = d_velocity.sel(yu_ocean=slice(-49, -5), xu_ocean=slice(-217, -183)).chunk(
    {"yu_ocean": None, "xu_ocean": None}
)

with ProgressBar():
    d_tracer.to_zarr(
        f"{base}/tracer.zarr",
        encoding={"time": {"dtype": "double", "units": "days since 1900-01-01 12:00:00", "calendar": "noleap"}},
    )

with ProgressBar():
    d_velocity.to_zarr(
        f"{base}/velocity.zarr",
        encoding={"time": {"dtype": "double", "units": "days since 1900-01-01 12:00:00", "calendar": "noleap"}},
    )
