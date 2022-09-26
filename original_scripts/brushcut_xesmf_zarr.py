"""
Convert 0.1° ACCESS 3D daily outputs of temperature, salt and velocity
into the "brushcutter" format suitable for MOM6 open boundary forcing.
"""

from itertools import cycle
import os

import dask.array as da
import dask.bag as db
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from dask.distributed import Client, worker_client
from dask.diagnostics import ProgressBar

"""
Configurable variables:

base:
  This is the base directory that holds the data from the
  *prepare_segments* script. The value of this variable should match
  between these two scripts.

hgrid_path:
  This points to the ocean_hgrid.nc file for your regional domain.

weights_exist:
  If you have run this script before, and have the regridding weights
  saved, you can save a bit of time by skipping their
  regeneration. Otherwise, leave this as false.
"""

base = os.getenv("PBS_JOBFS")
hgrid_path = "/g/data/x77/ahg157/inputs/mom6/eac-01/hgrid_01.nc"
weights_exist = False

# Everything that follows shouldn't need further configuration

surface_tracer_vars = ["temp", "salt"]
line_tracer_vars = ["eta_t"]
surface_velocity_vars = ["u", "v"]
surface_vars = surface_tracer_vars + surface_velocity_vars

def input_datasets():
    # open target grid dataset
    # we interpolate onto the hgrid
    dg = xr.open_dataset(hgrid_path)

    d_tracer = xr.open_zarr(f"{base}/tracer.zarr")
    d_velocity = xr.open_zarr(f"{base}/velocity.zarr")

    return dg, d_tracer, d_velocity

def interp_segment(segment):
    dg, d_tracer, d_velocity = input_datasets()
    i, edge = segment

    dg_segment = dg.isel(**edge)
    # interpolation grid
    dg_out = xr.Dataset(
        {
            "lat": (["location"], dg_segment.y.squeeze().data),
            "lon": (["location"], dg_segment.x.squeeze().data),
        }
    )

    # segment suffix
    seg = f"segment_{i+1:03}"
    seg_dir = ["nx", "nx", "ny", "ny"][i]
    seg_alt = ["ny", "ny", "nx", "nx"][i]
    alt_axis = [2, 2, 3, 3][i]

    # create the regridding weights between our grids
    # note: reuse_weights should be False unless the weights files
    #       do indeed exist!
    regridder_tracer = xe.Regridder(
        d_tracer.rename(xt_ocean="lon", yt_ocean="lat"),
        dg_out,
        "bilinear",
        locstream_out=True,
        reuse_weights=weights_exist,
        filename=f"bilinear_tracer_weights_{seg}.nc",
    )
    regridder_velocity = xe.Regridder(
        d_velocity.rename(xu_ocean="lon", yu_ocean="lat"),
        dg_out,
        "bilinear",
        locstream_out=True,
        reuse_weights=weights_exist,
        filename=f"bilinear_velocity_weights_{seg}.nc",
    )

    # now we can apply it to input DataArrays:
    segment_out = xr.merge([regridder_tracer(d_tracer), regridder_velocity(d_velocity)])
    del segment_out["lon"]
    del segment_out["lat"]
    segment_out["temp"] -= 273.15

    # fill in NaNs
    segment_out = (
        segment_out
        .ffill("st_ocean")
        .interpolate_na("location")
        .ffill("location")
        .bfill("location")
    )

    # fix up all the coordinate metadata
    segment_out = segment_out.rename(location=f"{seg_dir}_{seg}")
    for var in surface_vars:
        segment_out[var] = segment_out[var].rename(st_ocean=f"nz_{seg}_{var}")
        segment_out = segment_out.rename({var: f"{var}_{seg}"})
        segment_out[f"nz_{seg}_{var}"] = np.arange(segment_out[f"nz_{seg}_{var}"].size)

    for var in line_tracer_vars:
        segment_out = segment_out.rename({var: f"{var}_{seg}"})

    # segment coordinates (x, y, z)
    segment_out[f"{seg_dir}_{seg}"] = np.arange(segment_out[f"{seg_dir}_{seg}"].size)
    segment_out[f"{seg_alt}_{seg}"] = [0]

    # lat/lon/depth/dz
    segment_out[f"lon_{seg}"] = ([f"ny_{seg}", f"nx_{seg}"], dg_segment.x.data)
    segment_out[f"lat_{seg}"] = ([f"ny_{seg}", f"nx_{seg}"], dg_segment.y.data)

    # reset st_ocean so it's not an index coordinate
    segment_out = segment_out.reset_index("st_ocean").reset_coords("st_ocean_")
    depth = segment_out["st_ocean_"]
    depth.name = "depth"
    depth["st_ocean"] = np.arange(depth["st_ocean"].size)
    del segment_out["st_ocean_"]

    # some fiddling to do dz in the same way as brushcutter, while making xarray happy
    dz = depth.diff("st_ocean")
    dz.name = "dz"
    dz = xr.concat([dz, dz[-1]], dim="st_ocean")
    dz["st_ocean"] = depth["st_ocean"]

    encoding_dict = {
        "time": {
            "dtype": "double",
            "units": "days since 1900-01-01 12:00:00",
            "calendar": "noleap",
        },
        f"nx_{seg}": {
            "dtype": "int32",
        },
        f"ny_{seg}": {
            "dtype": "int32",
        },
    }

    for var in line_tracer_vars:
        v = f"{var}_{seg}"

        segment_out[v] = segment_out[v].expand_dims(
            f"{seg_alt}_{seg}", axis=alt_axis - 1
        )

        encoding_dict[v] = {
            "_FillValue": netCDF4.default_fillvals["f8"],
        }

    for var in surface_vars:
        v = f"{var}_{seg}"

        # add the y dimension
        segment_out[v] = segment_out[v].expand_dims(
            f"{seg_alt}_{seg}", axis=alt_axis
        )
        segment_out[f"dz_{v}"] = (
            ["time", f"nz_{v}", f"ny_{seg}", f"nx_{seg}"],
            da.broadcast_to(
                dz.data[None, :, None, None],
                segment_out[v].shape,
                chunks=(1, None, None, None),
            ),
        )

        s = list(segment_out[v].shape)
        s[0] = 1 # chunked in time
        s[1] = 11 # a little bit of vertical chunking

        encoding_dict[v] = {
            "_FillValue": netCDF4.default_fillvals["f8"],
            "zlib": True,
            "chunksizes": tuple(s),
        }
        encoding_dict[f"dz_{v}"] = {
            "_FillValue": netCDF4.default_fillvals["f8"],
            "zlib": True,
            "chunksizes": tuple(s),
        }
        encoding_dict[f"nz_{seg}_{var}"] = {
            "dtype": "int32"
        }

    with ProgressBar():
        segment_out.load().to_netcdf(f"forcing_obc_{seg}.nc", encoding=encoding_dict, unlimited_dims="time")

if __name__ == "__main__":
    c = Client(local_directory=os.getenv("PBS_JOBFS"))

    for seg in enumerate([{"nyp": [0]}, {"nyp": [-1]}, {"nxp": [0]}, {"nxp": [-1]}]):
        interp_segment(seg)
