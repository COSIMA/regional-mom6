# JR 2023-01-12
# compute Tasman Sea SST index from NOAA OISST v2.1

# load python modules
print("loading modules")
import netCDF4
import xarray as xr
import xesmf as xe
from itertools import cycle
import os
import dask
import numpy as np
import pandas as pd
import dask.array as da
import dask.bag as db
from pykdtree.kdtree import KDTree
from dask.diagnostics import ProgressBar
import subprocess

from timeit import default_timer as timer
import sys

from dask.distributed import Client
from datetime import timedelta
# import glob
# from xmhw.xmhw import detect

# Import scripts
# from regional_model_scripts import input_datasets, prepare_segments, interp_segment, time_rotate, regrid_runoff, sel_hgrid_indices

base = os.getenv("PBS_JOBFS")

def print_run_time(time):
    print(f"Elapsed (wall) time: {str(timedelta(seconds=time))}")
    
def input_datasets(path):
    # open target grid dataset
    # we interpolate onto the hgrid
    dg = xr.open_dataset(path + "hgrid.nc")

    d_tracer = xr.open_zarr(f"{path}/tracer.zarr")
    d_velocity = xr.open_zarr(f"{path}/velocity.zarr")

    return dg, d_tracer, d_velocity

def time_rotate(d,run_year = 2170):
    before_start_time = f"{run_year}-12-31"
    after_end_time = f"{run_year+1}-01-01"

    left = d.sel(time=slice(after_end_time, None))
    left["time"] = pd.date_range("1991-01-01 12:00:00", periods=120)

    right = d.sel(time=slice(None, before_start_time))
    right["time"] = pd.date_range("1991-05-01 12:00:00", periods=245)

    return xr.concat([left, right], "time")

def sel_hgrid_indices(field,extent):
    """
    Inputs: 
        field    xarray.dataarray   the existing hgrid lon or lat to be cut down to size
        extent   list               [min,max] the lat OR lon

    Returns:
        numpy array containing the start and end indices needed to slice the hgrid/

    Function finds the indices corresponding to the start and end of some coordinate range such that the hgrid starts and ends with q points rather than t points. 
    Useful for cutting out hgrid automatically. Note that this doesn't work near the poles in the northern hemisphere.
    
    It rounds the input field so that 208.99999999 == 209, giving nice even numbers of points between whole number lat/lon bounds
    
    This function is lazily implemented. Handling the edge cases was easiest to do without vectorising, but there should be numpy functions that would make this less inefficient.

    Written by Ashley Barnes for use with the regional scripts notebook
    """
    temp = False

    indices = []

    for i in range(field.shape[0]):
        if round(field[i].values.reshape(1)[0],6) >= extent[0] and temp == False:
            indices.append(2 * i)
            temp = True
        elif round(field[i].values.reshape(1)[0],6) > extent[1] and temp == True:
            indices.append((i-1) * 2 + 1) ## This goes back a step to find the last t point that was in the bounds. Then goes forward one so that the final point to slice is on the corner
            temp = False
            break
    
    return np.array(indices)

# set up a Dask cluster
if __name__ == '__main__':
    '''
    This python script will simply run the "prepare_segments" and "interp_segments" scripts that Angus and Ashley have written, within a PBS job. 
    What are the inputs?
        Basically, this expects that you are running from a directory that has all the precursor files already there. Then:
            prepare_segments(xextent, yextent)
            interp_segments(segment, path, base=os.getenv('PBS_JOBFS')...
    '''
    
    om2path = "/g/data/ik11/inputs/access-om2/input_08022019/mom_01deg/" ## Access om2_01 input for topography and hgrid
    initpath = "/g/data/ik11/outputs/access-om2-01/01deg_jra55v13_ryf9091/output1077" ## Access om2_01 output for BCs and IC
    #toolpath = "/g/data/v45/ahg157/tools/" ## Compiled tools needed for construction of mask tables

    yextent = [-48, 10]
    xextent = [-217 , -184]
    expt_name = "mom6_003"

    # path = f"/g/data/v45/jr5971/mom6_prep/{expt_name}/"
    path = f'/scratch/v45/jr5971/mom6/regional_configs/{expt_name}/'
    
    
    start = timer()
    
    worker_dir=os.getenv('PBS_JOBFS')
    if not worker_dir:
        worker_dir=os.getenv('TMPDIR')
    if not worker_dir:
        worker_dir="/tmp"
    client = Client(local_directory=worker_dir)
    print(client.ncores)

    
    #######################################################################################
#     # run prepare segments and save in temporary memory
    print('running prepare segments')
    # prepare_segments(xextent, yextent)
    surface_tracer_vars = ["temp", "salt"]
    line_tracer_vars = ["eta_t"]
    surface_velocity_vars = ["u", "v"]
    surface_vars = surface_tracer_vars + surface_velocity_vars

    chunks = {
        "T": {"time": 1, "st_ocean": 7, "yt_ocean": 300, "xt_ocean": None},
        "U": {"time": 1, "st_ocean": 7, "yu_ocean": 300, "xu_ocean": None},
    }
    
    t = range(1077, 1082)

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
    
    run_year = 2170

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
    
    #######################################
    uni_chunks_T = {'time':-1, 'st_ocean':15, 'yt_ocean':45, 'xt_ocean':70}
    uni_chunks_U = {'time':-1, 'st_ocean':15, 'yu_ocean':45, 'xu_ocean':70}
    d_tracer = d_tracer.chunk(uni_chunks_T)
    d_velocity = d_velocity.chunk(uni_chunks_U)
    #######################################
    print('saving tracer datasets to scratch')
    with ProgressBar():
        d_tracer.to_zarr(
            f"{path}/tracer.zarr",
            encoding={"time": {"dtype": "double", "units": "days since 1900-01-01 12:00:00", "calendar": "noleap"}},
            mode = "w"
        )
    print('saving velocity datasets to scratch')
    with ProgressBar():
        d_velocity.to_zarr(
            f"{path}/velocity.zarr",
            encoding={"time": {"dtype": "double", "units": "days since 1900-01-01 12:00:00", "calendar": "noleap"}},
            mode = "w"
        )
    print('finished prepare_segments')


    #for seg in enumerate([{"nyp": [0]}]): #, {"nyp": [-1]}, {"nxp": [0]}, {"nxp": [-1]}]):
    
    ###########################################################################################################
    ########### INTERP SEGMENTS ######################
    ###########################################################################################################
    print('starting interp segments')
    
    weights_exist = False
    
    
    for seg in enumerate([{"nyp":[0]},{"nyp": [-1]}, {"nxp": [0]}, {"nxp": [-1]}]):
        surface_tracer_vars = ["temp", "salt"]
        line_tracer_vars = ["eta_t"]
        surface_velocity_vars = ["u", "v"]
        surface_vars = surface_tracer_vars + surface_velocity_vars
        
        print('loading input datasets for segmentation')
        dg, d_tracer, d_velocity = input_datasets(path)
        i, edge = seg 

        dg_segment = dg.isel(**edge)
        # interpolation grid
        print('creating output grid at higher resolution (dg_out)')
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
        print('setting up regridder for both tracer and velocity')
        regridder_tracer = xe.Regridder(
            d_tracer.rename(xt_ocean="lon", yt_ocean="lat"),
            dg_out,
            "bilinear",
            locstream_out=True,
            reuse_weights=weights_exist,
            filename= path + f"weights/bilinear_tracer_weights_{seg}.nc",
        )
        regridder_velocity = xe.Regridder(
            d_velocity.rename(xu_ocean="lon", yu_ocean="lat"),
            dg_out,
            "bilinear",
            locstream_out=True,
            reuse_weights=weights_exist,
            filename=path + f"weights/bilinear_velocity_weights_{seg}.nc",
        )

        d_tracer = d_tracer.compute()
        d_velocity = d_velocity.compute()

        ######################################
        #print('chunking segments to be one block in x and y')
        #d_tracer = d_tracer.chunk({'time':None,'yt_ocean':-1,
                                  # 'xt_ocean':-1,'st_ocean':None})
        #d_velocity = d_velocity.chunk({'time':None,'yu_ocean':-1,
                                   #    'xu_ocean':-1,'st_ocean':None})

        ######################################

        # now we can apply it to input DataArrays:
        print('regridding 10th degree to 30th degree for tracers and velocities')
        segment_out = xr.merge([regridder_tracer(d_tracer), regridder_velocity(d_velocity)])
        # del segment_out["lon"] # May not need this??
        # del segment_out["lat"]
        segment_out["temp"] -= 273.15

        # fill in NaNs
        print('filling in NaNs (land cells)')
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
        # segment_out = segment_out.reset_index("st_ocean").reset_coords("st_ocean_")
        depth = segment_out["st_ocean"]
        depth.name = "depth"
        depth["st_ocean"] = np.arange(depth["st_ocean"].size)
        # del segment_out["st_ocean_"]

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
            segment_out["time"] = segment_out["time"].assign_attrs({"modulo":" "}) ## Add modulo attribute for MOM6 to treat as repeat forcing
            print('saving to netcdf in scratch')
            segment_out.to_netcdf(path + f"forcing/forcing_obc_{seg}.nc", encoding=encoding_dict, unlimited_dims="time")
            print(f'finished {seg}')
            # interp_segment(seg, path)     


