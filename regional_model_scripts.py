import numpy as np
from itertools import cycle
import os
from pykdtree.kdtree import KDTree
import dask.array as da
import dask.bag as db
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import subprocess
from dask.distributed import Client, worker_client
from dask.diagnostics import ProgressBar

"""
These functions allow the user to prepare the necessary input files MOM6 regional experiment with boundary forcing. To see how they're used and in what order, refer to the accompanying Jupyter notebook. The "Prepare Segments" function assumes that you're cutting your forcing data from an ACCESS-OM2 style grid, but the rest of the functions should be a bit more generic.
"""


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


base = os.getenv("PBS_JOBFS")


## Below are the functions for preparing the boundary segments for the brushcutter



def time_rotate(d,run_year = 2170):
    before_start_time = f"{run_year}-12-31"
    after_end_time = f"{run_year+1}-01-01"

    left = d.sel(time=slice(after_end_time, None))
    left["time"] = pd.date_range("1991-01-01 12:00:00", periods=120)

    right = d.sel(time=slice(None, before_start_time))
    right["time"] = pd.date_range("1991-05-01 12:00:00", periods=245)

    return xr.concat([left, right], "time")

def prepare_segments(xextent,yextent,run_year = 2170,t = range(1077, 1082),base = os.getenv("PBS_JOBFS")):
    """
    Given lon/lat bounds, this function cuts out and pre-processes ACCESS-om2-01 model outputs ready for the brushcutter functions that will turn them into MOM6 inputs. This function is hard-coded to work with ACCESS-om2-01 conventions, including naming and the 

    Originally written by Angus Gibson, minor updates September 2022 by Ashley Barnes. 
    Args

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

    Returns: 
    Nothing. Saves zarr files to memory to be read by brushcutter


    """
    
    

    # Everything that follows shouldn't need further configuration, if you're using the
    # same experiment


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
        



### below are the 'brushcutter' functions

def input_datasets(path):
    # open target grid dataset
    # we interpolate onto the hgrid
    dg = xr.open_dataset(path + "hgrid.nc")

    d_tracer = xr.open_zarr(f"{os.getenv('PBS_JOBFS')}/tracer.zarr")
    d_velocity = xr.open_zarr(f"{os.getenv('PBS_JOBFS')}/velocity.zarr")

    return dg, d_tracer, d_velocity

def interp_segment(segment,path,weights_exist = False,base = os.getenv("PBS_JOBFS")):
    """
    Convert 0.1° ACCESS 3D daily outputs of temperature, salt and velocity
    into the "brushcutter" format suitable for MOM6 open boundary forcing.

    Originally written by Angus Gibson, minor updates September 2022 by Ashley Barnes. 

    Args:

    base:
    This is the base directory that holds the data from the
    *prepare_segments* script. The value of this variable should match
    between these two scripts.

    path:
    This points to the hgrid.nc file for your regional domain.

    weights_exist:
    If you have run this script before, and have the regridding weights
    saved, you can save a bit of time by skipping their
    regeneration. Otherwise, leave this as false.

    returns:
    Nothing. Saves brushcut files to path/forcing
    """

    surface_tracer_vars = ["temp", "salt"]
    line_tracer_vars = ["eta_t"]
    surface_velocity_vars = ["u", "v"]
    surface_vars = surface_tracer_vars + surface_velocity_vars

    dg, d_tracer, d_velocity = input_datasets(path)
    i, edge = segment

    dg_segment = dg.isel(**edge)
    # interpolation grid
    dg_out = xr.Dataset(
        {
            "lat": (["location"], dg_segment.y.squeeze().data),
            "lon": (["location"], dg_segment.x.squeeze().data),
        }
    ).set_coords(["lat","lon"])

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

    # now we can apply it to input DataArrays:
    segment_out = xr.merge([regridder_tracer(d_tracer), regridder_velocity(d_velocity)])

    print("before fixing up metadata")
    print(segment_out)


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
    # segment_out = segment_out.reset_index("st_ocean").reset_coords("st_ocean_")
    depth = segment_out["st_ocean"]
    depth.name = "depth"
    depth["st_ocean"] = np.arange(depth["st_ocean"].size)
    del segment_out["st_ocean"]

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
        segment_out.load().to_netcdf(path + f"forcing/forcing_obc_{seg}.nc", encoding=encoding_dict, unlimited_dims="time",engine="netcdf4")





ocean_mask_path = "ocean_mask.nc"
hgrid_path = "hgrid_01.nc"
runoff_path = "runoff_box.nc"
output_path = "runoff_regrid.nc"


# Everything that follows shouldn't need further configuration

def regrid_runoff(ocean_mask_path,hgrid_path,runoff_path,output_path,xextent,yextent):
    """
    Regrids the runoff across the domain to the new coastlines. Written by Angus Gibson

    Args:

    ocean_mask_path:
    This points to the path to your ``ocean_mask.nc``, which is created
    from ``ocean_hgrid.nc`` and your topography using the
    ``make_solo_mosaic`` tool.

    hgrid_path:
    This points to the ocean_hgrid.nc file for your regional domain.

    runoff_path:
    This points to the runoff forcing file.

    Returns:
    nothing. Saves regrided runoff to output_path
    """
    # ocean mask and supergrid (reduced to tracer points) for the target grid
    dm = xr.open_dataset(ocean_mask_path).rename({"nx": "longitude", "ny": "latitude"})
    dg = (
      xr.open_dataset(hgrid_path)
      .isel(nxp=slice(1, None, 2), nyp=slice(1, None, 2))
      .rename({"nyp": "latitude", "nxp": "longitude"})
    )

    # merge areas to get full cell area
    area = dg.area
    area["ny"] = area.ny // 2
    area["nx"] = area.nx // 2
    area = (
      area
      .stack(cell=["ny", "nx"])
      .groupby("cell")
      .sum()
      .unstack("cell")
    )

    # calculate coastal mask
    cst = xr.zeros_like(dm.mask)
    for dim in ["longitude", "latitude"]:
        for off in [-1, 1]:
          cst = xr.where((dm.mask > 0) & (dm.mask.shift(**{dim: off}) == 0), 1, cst)

    # indices of coast points -- Nx2, first column are y indices, then x indices
    cst_pts = np.vstack(np.nonzero(cst)).T
    # coords of coast points -- Nx2, first column are latitudes, then longitudes
    cst_coords = xr.concat((dg.y, dg.x + 360), "d").data.reshape(2, -1).T[np.flatnonzero(cst)]
    cst_areas = area.data.flatten()[np.flatnonzero(cst)]

    kd = KDTree(cst_coords)

    # open the runoff section and construct its corner points
    dr = xr.open_dataset(runoff_path).sel(latitude = slice(yextent[0],yextent[1]),
         longitude = slice(xextent[0] + 360,xextent[1] + 360)) ## need to add 360 since xextent is between -280 -> 80
    print(dr)
    res = 0.25
    lons = np.arange(dr.longitude[0] - res/2, dr.longitude[-1] + res, res)
    lats = np.arange(dr.latitude[0] - res/2,  dr.latitude[-1] + res, res)

    # source coords for remapping
    runoff_coords = np.c_[np.meshgrid(dr.latitude, dr.longitude, indexing="ij")].reshape(2, -1).T
    # coords for cell area calculation
    corner_lat, corner_lon = np.meshgrid(np.deg2rad(lats), np.deg2rad(lons), indexing="ij")
    Re = 6378.137e3
    runoff_areas = np.abs(
      ((corner_lon[1:,1:] - corner_lon[:-1,:-1]) * Re**2) * (np.sin(corner_lat[1:,1:]) - np.sin(corner_lat[:-1,:-1]))
    )

    # nearest coastal point for every runoff point
    _, nearest_cst = kd.query(runoff_coords)

    # create output DataArray
    runoff = xr.DataArray(
      0.0,
      {"time": dr.time, "latitude": dg.y.isel(longitude=0), "longitude": dg.x.isel(latitude=0)},
      ["time", "latitude", "longitude"]
    )
    runoff.name = "friver"
    runoff.time.attrs["modulo"] = " "

    ind_y = xr.DataArray(cst_pts[:,0], dims="coast")
    ind_x = xr.DataArray(cst_pts[:,1], dims="coast")

    for i in range(dr.time.size):
        # list of nearest coast point (on target grid), with the source data
        dat = np.c_[nearest_cst, (dr.friver[i].data * runoff_areas).flatten()]
        dat = dat[dat[:,0].argsort()] # sort by coast point idx

        # group by destination point
        cst_point, split_idx = np.unique(dat[:,0], return_index=True)
        cst_point = cst_point.astype(int)
        split_idx = split_idx[1:]

        # sum per destination point
        dat_cst = [x.sum() for x in np.split(dat[:,1], split_idx)]

        # assign the target value
        runoff[i, ind_y[cst_point], ind_x[cst_point]] = dat_cst / cst_areas[cst_point]

    runoff.to_netcdf(output_path, unlimited_dims="time")


## Chris's function for generating hgrid from scratch

def Create_Base_Grid(output_directory,grid_dx=1/50,grid_dy=1/50,FRE_tools_dir='/g/data/ul08/FRE_tools/bin/bin'):
    
    west_longitude_limit = 0
    east_longitude_limit = 360

    south_latitude_limit = -90
    north_latitude_limit = 90
    
    
    n_lon = int( (east_longitude_limit-west_longitude_limit)/grid_dx )
    n_lat = int( (north_latitude_limit-south_latitude_limit)/grid_dy ) 
    
    grid_type = 'regular_lonlat_grid'

    input_args = " --grid_type " +  grid_type 
    input_args = input_args + " --nxbnd 2 --nybnd 2" #
    input_args = input_args + f' --xbnd {west_longitude_limit},{east_longitude_limit}' #.format(yes_votes, percentage) 
    input_args = input_args + f' --ybnd {south_latitude_limit},{north_latitude_limit}'
    input_args = input_args + f' --nlon {n_lon}, --nlat {n_lat}'
    input_args = input_args + " center c_cell"

    try:
        print("MAKE HGRID",subprocess.run([FRE_tools_dir + '/make_hgrid'] + input_args.split(" "),cwd = output_directory),sep = "\n")
        return 0
    except:
        print('Make_hgrid failed')
        return -9