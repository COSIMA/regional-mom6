import numpy as np
from itertools import cycle
import os
import dask.array as da
import dask.bag as db
import numpy as np
import xarray as xr
import xesmf as xe
import subprocess
from scipy.ndimage import binary_fill_holes
import netCDF4
from dask.distributed import Client, worker_client
from dask.diagnostics import ProgressBar
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

def nicer_slicer(data,xextent,xcoords,buffer = 2):
    """
    Slices longitudes, handling periodicity and 'seams' where the data wraps around (commonly either at -180 -> 180 or -270 -> 90)

    Parameters:
        data (xarray dataset)          The global data you want to slice in longitude
        xextent (tuple)                The target longitudes you want to slice to. This must be either start negative and progress to positive, or be entirely positive
        x (str or list of str)         The name of the longitude dimension in your xarray or list of names


    The algorithm works in five steps:

    Determine whether we need to add or subtract 360 to get the middle of the xextent to lie within data's lonitude range (hereby oldx)

    Shift the dataset so that its midpoint matches the midpoint of xextent (up to a muptiple of 360). Now, the modified oldx doesn't increase monotonically W to E since the 'seam' has moved.

    Fix oldx to make it monatonically increasing again. This uses the information we have about the way the dataset was shifted/rolled

    Slice the data index-wise. We know that |xextent| / 360 multiplied by the number of discrete longitude points will give the total width of our slice, and we've already set the midpoint to be the middle of the
    target domain. Here we add a buffer region on either side if we need it for interpolation.

    Finally re-add the right multiple of 360 so the whole domain matches the target.

    """
    
    if isinstance(xcoords,str):
        xcoords = [xcoords]

    for x in xcoords:

        mp_target = np.mean(xextent) ## Midpoint of target domain


        ## Find a corresponding value for the intended domain midpint in our data. Assuming here that data has equally spaced longitude values spanning 360deg
        for i in range(-1,2,1):
            if (data[x][0] <= mp_target + 360 * i <= data[x][-1]):

                _mp_target = mp_target + 360 * i ## Shifted version of target midpoint. eg, could be -90 vs 270. i keeps track of what multiple of 360 we need to shift entire grid by to match midpoint

                mp_data = data[x][data[x].shape[0]//2].values ## Midpoint of the data

                shift = -1 * (data[x].shape[0] * (_mp_target - mp_data)) // 360
                shift = int(shift)                   ## This is the number of indices between the data midpoint, and the target midpoint. Sign indicates direction needed to shift

                new_data = data.roll({x:1 * shift},roll_coords=True)   ## Shifts data so that the midpoint of the target domain is the middle of the data for easy slicing

                new_x = new_data[x].values           ## Create a new longitude coordinate. We'll modify this to remove any seams (jumps like -270 -> 90) 

                ## Take the 'seam' of the data, and either backfill or forward fill based on whether the data was shifted F or west
                if shift > 0:
                    new_seam_index = shift

                    new_x[0:new_seam_index] -= 360

                if shift < 0:
                    new_seam_index = data[x].shape[0] + shift

                    new_x[new_seam_index:] += 360

                new_x -= i * 360 ## Use this to recentre the midpoint to match that of target domain
            

                new_data = new_data.assign_coords({x:new_x})

                ## Choose the number of x points to take from the middle, including a buffer. Use this to index the new global dataset

                num_xpoints = int(data[x].shape[0]* (mp_target - xextent[0]))// 360 + buffer * 2 ## The extra 8 is a buffer region

        data = new_data.isel({x: slice(data[x].shape[0]//2 - num_xpoints,data[x].shape[0]//2 + num_xpoints)})

    return data


# Ashley, written March 2023
def motu_requests(xextent, yextent, daterange, outfolder, usr, pwd,segs,url = "https://my.cmems-du.eu/motu-web/Motu", serviceid = "GLOBAL_MULTIYEAR_PHY_001_030-TDS",productid = "cmems_mod_glo_phy_my_0.083_P1D-m",buffer = 0.3):
    """
    Generates motu data request for each specified boundary, as well as for the initial condition. By default pulls the GLORYS reanalysis dataset.
    
    Arguments:
        xextent,yextent     list containing extreme values of lon/lat coordinates for rectangular domain
        daterange           start and end dates of boundary forcing window. Format %Y-%m-%d %H:%M:%S
        usr,pwd             username and password for your cmems account
        segs                list of the cardinal directions for your boundary forcing. Must be a list even if there's only one
        outfolder           folder to dump the downloaded files
        url etc             strings that point to the dataset you want to use for your forcing files

    Returns:
        A string containing newline delimited motu requests ready to run

    """
    if type(segs) == str:
        return f"\nprintf 'processing {segs} segment' \npython -m motuclient --motu {url} --service-id {serviceid} --product-id {productid} --longitude-min {xextent[0]} --longitude-max {xextent[1]} --latitude-min {yextent[0]} --latitude-max {yextent[1]} --date-min {daterange[0]} --date-max {daterange[1]} --depth-min 0.49 --depth-max 6000 --variable so --variable thetao --variable vo --variable zos --variable uo --out-dir {outfolder} --out-name {segs}_unprocessed --user '{usr}' --pwd '{pwd}'\n"
    
    ## Buffer pads out our boundaries a small amount to allow for interpolation
    xextent,yextent = np.array(xextent) , np.array(yextent)
    script = "#!/bin/bash\n\n"
    for seg in segs:
        if seg == "east":
                script += motu_requests(
                        [xextent[1] - buffer,xextent[1] + buffer],
                        yextent, daterange, outfolder, usr, pwd, seg,url=url,serviceid=serviceid,productid=productid
                )
        if seg == "west":
                script += motu_requests(
                        [xextent[0] - buffer,xextent[0] + buffer],
                        yextent, daterange, outfolder, usr, pwd, seg,url=url,serviceid=serviceid,productid=productid
                )
        if seg == "north":
                script += motu_requests(
                        xextent,
                        [yextent[1] - buffer,yextent[1] + buffer], 
                        daterange, outfolder, usr, pwd, seg,url=url,serviceid=serviceid,productid=productid
                )

        if seg == "south":
                script += motu_requests(
                        xextent,
                        [yextent[0] - buffer,yextent[0] + buffer], 
                        daterange, outfolder, usr, pwd, seg,url=url,serviceid=serviceid,productid=productid
                    ) 
    ## Now handle the initial condition
    script += motu_requests(
            xextent + np.array([-1 * buffer,buffer]),
            yextent + np.array([-1 * buffer,buffer]), 
            [daterange[0], dt.datetime.strptime(daterange[0],"%Y-%m-%d %H:%M:%S") + dt.timedelta(hours=1)], ## For initial condition just take one day
            outfolder, usr, pwd, "ic",url=url,serviceid=serviceid,productid=productid
    )
    return script

def dz(npoints,ratio,target_depth,min_dz = 0.0001,tolerence = 1):
    """
    Recursive function takes the target depth, the ratio between largest and smallest layer thicknesses, and generates a hyperbolic tangent thickness profile for the experiment.
    Iterates to find the mininum depth value which gives the target depth within some tolerence

            Parameters:
                npoints (int):           Number of vertical points
                ratio (float):           Ratio of largest to smallest layer thickness. Negative values mean higher resolution is at bottom
                target_depth (float):    Maximum depth of a layer
                min_dz (float):          Starting point for iteration
                tolerence (float):       How close to target depth you want to get

            Returns:
                numpy array of layer thicknesses

    """
    profile = min_dz + 0.5 *  (np.abs(ratio) * min_dz - min_dz) * (1 + np.tanh(2 * np.pi * (np.arange(npoints) - npoints // 2) / npoints))
    tot = np.sum(profile)
    if np.abs(tot - target_depth) < tolerence:
        if ratio > 0:
            return profile

        return profile[::-1]

    err_ratio = target_depth / tot

    return dz(npoints,ratio,target_depth,min_dz * err_ratio)

# Borrowed from grid tools (GFDL)
def angle_between(v1, v2, v3):
    """Returns angle v2-v1-v3 i.e betweeen v1-v2 and v1-v3."""
    # vector product between v1 and v2
    px = v1[1] * v2[2] - v1[2] * v2[1]
    py = v1[2] * v2[0] - v1[0] * v2[2]
    pz = v1[0] * v2[1] - v1[1] * v2[0]
    # vector product between v1 and v3
    qx = v1[1] * v3[2] - v1[2] * v3[1]
    qy = v1[2] * v3[0] - v1[0] * v3[2]
    qz = v1[0] * v3[1] - v1[1] * v3[0]

    ddd = (px * px + py * py + pz * pz) * (qx * qx + qy * qy + qz * qz)
    ddd = (px * qx + py * qy + pz * qz) / np.sqrt(ddd)
    angle = np.arccos( ddd )
    return angle

# Borrowed from grid tools (GFDL)
def quad_area(lat, lon):
    """Returns area of spherical quad (bounded by great arcs)."""
    # x,y,z are 3D coordinates
    d2r = np.deg2rad(1.)
    x = np.cos(d2r * lat) * np.cos(d2r * lon)
    y = np.cos(d2r * lat) * np.sin(d2r * lon)
    z = np.sin(d2r * lat)
    c0 = (x[ :-1, :-1], y[ :-1, :-1], z[ :-1, :-1])
    c1 = (x[ :-1,1:  ], y[ :-1,1:  ], z[ :-1,1:  ])
    c2 = (x[1:  ,1:  ], y[1:  ,1:  ], z[1:  ,1:  ])
    c3 = (x[1:  , :-1], y[1:  , :-1], z[1:  , :-1])
    a0 = angle_between(c1, c0, c2)
    a1 = angle_between(c2, c1, c3)
    a2 = angle_between(c3, c2, c0)
    a3 = angle_between(c0, c3, c1)
    return a0 + a1 + a2 + a3 - 2. * np.pi



def rectangular_hgrid(x,y):
    """
    Given an array of latitudes and longitudes, constructs a working hgrid with all the metadata. X must be evenly spaced, y can be scaled to your hearts content (eg if you want to ensure equal sized cells)

    LIMITATIONS: This is hard coded to only take x and y on perfectly rectangular grid. Rotated grid needs to be handled separately. Make sure both x and y are monatonically increasing.
    
    Parameters:
        `x (array)'         Array of all longitude points on supergrid. Assumes even spacing in x
        `y (array)'         Likewise for latitude. 

    Returns:
        horizontal grid with all the bells and whistles that MOM6 / FMS wants
    """

    #! Hardcoded for grid that lies on lat/lon lines. Rotated grid must be handled separately

    res = x[1] - x[0] #! Replace this if you deviate from rectangular grid!!
    
    R = 6371000 # This is the exact radius of the Earth if anyone asks 

    # dx = pi R cos(phi) / 180. Note dividing resolution by two because we're on the supergrid
    dx = np.broadcast_to(
        np.pi * R * np.cos(np.pi * y / 180) * 0.5 * res / 180,
        (x.shape[0] - 1,y.shape[0])
        ).T

    # dy  = pi R delta Phi / 180. Note dividing dy by 2 because we're on supergrid
    dy = np.broadcast_to(
        R * np.pi * 0.5 * np.diff(y) / 180,
        (x.shape[0],y.shape[0] - 1)
        ).T

    X , Y = np.meshgrid(x,y)
    area = quad_area(Y,X) * 6371e3 ** 2

    attrs = {
        "tile":{
            "standard_name" :"grid_tile_spec",
            "geometry" :"spherical",
            "north_pole" :"0.0 90.0",
            "discretization" :"logically_rectangular",
            "conformal" :"true"
        },
        "x":{
            "standard_name" : "geographic_longitude",
            "units" : "degree_east"
        },
        "y":{        
            "standard_name" : "geographic_latitude",
            "units" : "degree_north"
        },
        "dx":{
            "standard_name":"grid_edge_x_distance",
            "units":"metres",
        },
        "dy":{
            "standard_name":"grid_edge_y_distance",
            "units":"metres",
        },
        "area":{
            "standard_name":"grid_cell_area",
            "units":"m2",
        },
        "angle_dx":{
            "standard_name":"grid_vertex_x_angle_WRT_geographic_east",
            "units":"degrees_east",
        },
        "arcx": {
            "standard_name":"grid_edge_x_arc_type",
            "north_pole" :"0.0 90.0",
        }
        
    }


    return xr.Dataset(
        {"tile":((),np.array(b'tile1', dtype='|S255'),attrs["tile"]),
        "x":(["nyp","nxp"],X,attrs["x"]),
        "y":(["nyp","nxp"],Y,attrs["y"]),
        "dx":(["nyp","nx"],dx,attrs["dx"]),
        "dy":(["ny","nxp"],dy,attrs["dy"]),
        "area":(["ny","nx"],area,attrs["area"]),
        "angle_dx":(["nyp","nxp"],X * 0,attrs["angle_dx"]),
        "arcx":((),np.array(b'small_circle', dtype='|S255'),attrs["arcx"])
        })
class experiment:
    """
    Knows everything about your regional experiment! Methods in this class will generate the various input files you need to generate a MOM6 experiment forced with Open Boundary Conditions. It's written agnostic to your choice of boundary forcing,topography and surface forcing - you need to tell it what your variables are all called via mapping dictionaries where keys are mom6 variable / coordinate names, and entries are what they're called in your dataset. 
    """

    def __init__(self, xextent,yextent,daterange,resolution,vlayers,dz_ratio,depth,mom_run_dir,mom_input_dir,toolpath):
        try:
            os.mkdir(mom_run_dir)
        except:
            pass

        try:
            os.mkdir(mom_input_dir)
        except:
            pass
        self.xextent = xextent
        self.yextent = yextent
        self.daterange = [dt.datetime.strptime(daterange[0],"%Y-%m-%d %H:%M:%S"),
                          dt.datetime.strptime(daterange[1],"%Y-%m-%d %H:%M:%S")]
        self.res = resolution
        self.vlayers = vlayers
        self.dz_ratio = dz_ratio
        self.depth = depth
        self.mom_run_dir = mom_run_dir
        self.mom_input_dir = mom_input_dir
        self.toolpath = toolpath
        self.hgrid = self._make_hgrid()
        self.vgrid = self._make_vgrid()
        # if "temp" not in os.listdir(inputdir):
        #     os.mkdir(inputdir + "temp")


        if "weights" not in os.listdir(self.mom_input_dir):
            os.mkdir(mom_input_dir + "weights")
        if "forcing" not in os.listdir(self.mom_input_dir):
            os.mkdir(self.mom_input_dir + "forcing")

        # create a simlink from input directory to run directory and vv
        subprocess.run(f"ln -s {self.mom_input_dir} {self.mom_run_dir}/inputdir",shell=True)
        subprocess.run(f"ln -s {self.mom_run_dir} {self.mom_input_dir}/rundir",shell=True)
        
        return 
    
    def _make_hgrid(self,method = "even_spacing"):
        """
        Sets up hgrid based on users specification of domain. Default behaviour leaves latitude and longitude evenly spaced. 
        If user specifies a resolution of 0.1 degrees, longitude is spaced this way and latitude spaced with 0.1 cos(mean_latitude). This way, grids in the 
        centre of the domain are perfectly square, but decrease monatonically in area away from the equator 
        """
        if method == "even_spacing":
            # longitudes will just be evenly spaced, based only on resolution and bounds
            x = np.linspace(self.xextent[0],self.xextent[1],int((self.xextent[1] - self.xextent[0])/(self.res / 2)) + 1)

            # Latitudes evenly spaced by dx * cos(mean_lat)
            res_y = self.res * np.cos(np.mean(self.yextent) * np.pi / 180) 
            y = np.linspace(self.yextent[0],self.yextent[1],int((self.yextent[1] - self.yextent[0])/(res_y / 2)) + 1)
            hgrid = rectangular_hgrid(x,y) 
            hgrid.to_netcdf(self.mom_input_dir + "/hgrid.nc")
            

            return hgrid


    def _old_make_hgrid(self):
        """
        Generates a mom6 hgrid ready to go. Makes a basic grid first, then uses FRE tools to create a full hgrid with all the metadata.
        """
        ## Total number of qpoints in x
        qpoints_x = int((self.xextent[1] - self.xextent[0])/self.res) + 1
        qpoints_y = int((self.yextent[1] - self.yextent[0])/self.res) + 1
        if qpoints_x == 0 or qpoints_y == 0:
            raise ValueError("Please ensure domain extents match chosen resolution")
        Xq = np.linspace(self.xextent[0],self.xextent[1],qpoints_x)
        Yq = np.linspace(self.yextent[0],self.yextent[1],qpoints_y)

        Xt = np.linspace(
            self.xextent[0] + self.res/2,
            self.xextent[1] - self.res/2,
            qpoints_x - 1
            )

        Yt = np.Xt = np.linspace(
            self.yextent[0] + self.res/2,
            self.yextent[1] - self.res/2,
            qpoints_y - 1
            )

        # broadcast to meshgrid
        Xt, Yt = np.meshgrid(Xt, Yt)
        Xq, Yq = np.meshgrid(Xq, Yq)

        # create output dataset
        ds = xr.Dataset({
            "grid_lon": (['grid_yc', 'grid_xc'], Xq),
            'grid_lat': (['grid_yc', 'grid_xc'], Yq),
            'grid_lont': (['grid_yt', 'grid_xt'], Xt),
            'grid_latt': (['grid_yt', 'grid_xt'], Yt),
        })
        ds.to_netcdf(self.mom_input_dir + "/grid.nc")

        ## Generate the hgrid with fretools. Need to generalise later to not rely on random scripts!
        args = "--grid_type from_file --my_grid_file grid.nc".split(" ")
        print("FRE TOOLS: Make hgrid \n\n",subprocess.run([self.toolpath + "make_hgrid/make_hgrid"] + args, cwd=self.mom_input_dir))
        subprocess.run(["mv","horizontal_grid.nc","hgrid.nc"],cwd=self.mom_input_dir)

        ## Make Solo Mosaic
        args = "--num_tiles 1 --dir . --mosaic_name ocean_mosaic --tile_file hgrid.nc".split(" ")
        print("FRE TOOLS: Make solo mosaic\n\n",subprocess.run([self.toolpath + "make_solo_mosaic/make_solo_mosaic"] + args, cwd=self.mom_input_dir), sep="\n")
        return xr.open_dataset(self.mom_input_dir + "/hgrid.nc")
    
    def _make_vgrid(self):
        """
        Generates a vertical grid based on the number of layers and vertical ratio specified at the class level.
        """
        thickness = dz(self.vlayers + 1,self.dz_ratio,self.depth)
        vcoord = xr.Dataset(
            {"zi":("zi",np.cumsum(thickness)),
             "zl":("zl",(np.cumsum(thickness) + 0.5 * thickness)[0:-1])} ## THIS MIGHT BE WRONG REVISIT
        )
        vcoord["zi"].attrs = {"units":"meters"}
        vcoord.to_netcdf(self.mom_input_dir + "/vcoord.nc")

        return vcoord


    def ocean_forcing(self,path,varnames,boundaries = None,gridtype = "A",vcoord_type = "height"):
        """
        Reads in the forcing files that force the ocean at boundaries (if specified) and for initial condition

        Parameters:
            `path (str)`                   path to directory where the forcing files are stored. Files should be named north_segment_unprocessed for each boundary and ic_unprocessed for the ic
            varnames (dict)              dictionary that maps the mom6 variable / coordinate names to what they're called in this dataset. See documentation for expected format.
            boundaries (list of str)     List of the cardinal directions of included boundaries in anticlockwise order
            gridtype (str)               input is A,B or C type grid. Gets converted to mom6's C grid
            vcoord_type (str)            the type of vertical coordinate used in the forcing files. Either 'height' or 'thickness'.
        """

        ## Do initial condition

        ## pull out the initial velocity on MOM5's Bgrid

        ic_raw = xr.open_dataset(path + "/ic_unprocessed.nc")

        if varnames["time"] in ic_raw.dims:
            ic_raw = ic_raw.isel({varnames["time"] : 0})

        ## Separate out tracers from two velocity fields of IC
        try:
            ic_raw_tracers = ic_raw[[varnames["tracers"][i] for i in varnames["tracers"]]]
        except:
            print("Error in reading in initial condition tracers. Terminating")
            raise ValueError
        try:
            ic_raw_u = ic_raw[varnames["u"]]
            ic_raw_v = ic_raw[varnames["v"]]
        except:
            print("Error in reading in initial condition velocities. Terminating")
            raise ValueError
        try:
            ic_raw_eta = ic_raw[varnames["eta"]]
        except:
            print("Error in reading in initial condition tracers. Terminating")
            raise ValueError
        

        ## Rename all coordinates to cgrid convention
        if gridtype == "A":
            ic_raw_tracers = ic_raw_tracers.rename({varnames["x"]:"lon",varnames["y"]:"lat"})
            ic_raw_u = ic_raw_u.rename({varnames["x"]:"lon",varnames["y"]:"lat"})
            ic_raw_v = ic_raw_v.rename({varnames["x"]:"lon",varnames["y"]:"lat"})
            ic_raw_eta = ic_raw_eta.rename({varnames["x"]:"lon",varnames["y"]:"lat"})

        if gridtype == "B":
            ic_raw_tracers = ic_raw_tracers.rename({varnames["xh"]:"lon",varnames["yh"]:"lat"})
            ic_raw_eta = ic_raw_eta.rename({varnames["xh"]:"lon",varnames["yh"]:"lat"})
            ic_raw_u = ic_raw_u.rename({varnames["xq"]:"lon",varnames["yq"]:"lat"})
            ic_raw_v = ic_raw_v.rename({varnames["xq"]:"lon",varnames["yq"]:"lat"})

        if gridtype == "C":
            ic_raw_tracers = ic_raw_tracers.rename({varnames["xh"]:"lon",varnames["yh"]:"lat"})
            ic_raw_eta = ic_raw_eta.rename({varnames["xh"]:"lon",varnames["yh"]:"lat"})
            ic_raw_u = ic_raw_u.rename({varnames["xq"]:"lon",varnames["yh"]:"lat"})
            ic_raw_v = ic_raw_v.rename({varnames["xh"]:"lon",varnames["yq"]:"lat"})


        ## Construct the xq,yh and xh yq grids
        ugrid = self.hgrid[["x","y"]].isel(nxp=slice(None, None, 2), nyp=slice(1, None, 2)).rename({"x": "lon", "y": "lat"}).set_coords(["lat","lon"])
        vgrid = self.hgrid[["x","y"]].isel(nxp=slice(1, None, 2), nyp=slice(None, None, 2)).rename({"x": "lon", "y": "lat"}).set_coords(["lat","lon"])

        ## Construct the cell centre grid for tracers (xh,yh). 
        tgrid = xr.Dataset(
            {"lon":(["lon"],self.hgrid.x.isel(nxp=slice(1, None, 2), nyp=1).values),
            "lat":(["lat"],self.hgrid.y.isel(nxp=1, nyp=slice(1, None, 2)).values)
                    }
        )


        ### Drop NaNs to be re-added later
        # NaNs are from the land mask. When we interpolate onto a new grid, need to put in the new land mask. If NaNs left in, land mask stays the same

        ic_raw_tracers = ic_raw_tracers.interpolate_na("lon",method = "nearest").ffill("lon").bfill("lon")
        ic_raw_eta = ic_raw_eta.interpolate_na("lon",method = "nearest").ffill("lon").bfill("lon")
        ic_raw_u = ic_raw_u.interpolate_na("lon",method = "nearest").ffill("lon").bfill("lon")
        ic_raw_v = ic_raw_v.interpolate_na("lon",method = "nearest").ffill("lon").bfill("lon")

        ## Make our three horizontal regrideers 
        regridder_u = xe.Regridder(
            ic_raw_u, ugrid, "bilinear",
        )
        regridder_v = xe.Regridder(
            ic_raw_v, vgrid, "bilinear",
        )

        regridder_t = xe.Regridder(
            ic_raw_tracers, tgrid, "bilinear",
        )

        print("INITIAL CONDITIONS")
        ## Regrid all fields horizontally.
        print("Regridding Velocities...",end="")
        vel_out = xr.merge(
            [
            regridder_u(ic_raw_u).rename({"lon": "xq", "lat": "yh", "nyp": "ny",varnames["zl"]:"zl"}).rename("u"),
            regridder_v(ic_raw_v).rename({"lon": "xh", "lat": "yq", "nxp": "nx",varnames["zl"]:"zl"}).rename("v")
            ]
        )
        print("Done.\nRegridding Tracers...")
        tracers_out = xr.merge(
            [regridder_t(ic_raw_tracers[varnames["tracers"][i]]).rename(i) for i in varnames["tracers"]]
            ).rename({"lon": "xh", "lat": "yh",varnames["zl"]:"zl"})
        print("Done.\nRegridding Free surface...")
        
        eta_out = regridder_t(ic_raw_eta).rename({"lon": "xh", "lat": "yh"}).rename("eta_t") ## eta_t is the name set in MOM_input by default

        ## Return attributes to arrays

        vel_out.u.attrs = ic_raw_u.attrs 
        vel_out.v.attrs = ic_raw_v.attrs 
        vel_out.xq.attrs = ic_raw_u.lon.attrs
        vel_out.yq.attrs = ic_raw_v.lat.attrs
        vel_out.yh.attrs = ic_raw_u.lat.attrs
        vel_out.yh.attrs = ic_raw_v.lon.attrs
        vel_out.zl.attrs = ic_raw_u[varnames["zl"]].attrs
 
        tracers_out.xh.attrs = ic_raw_tracers.lon.attrs
        tracers_out.yh.attrs = ic_raw_tracers.lat.attrs
        tracers_out.zl.attrs = ic_raw_tracers[varnames["zl"]].attrs
        for i in varnames["tracers"]:
            tracers_out[i].attrs = ic_raw_tracers[varnames["tracers"][i]].attrs

        eta_out.xh.attrs = ic_raw_tracers.lon.attrs
        eta_out.yh.attrs = ic_raw_tracers.lat.attrs
        eta_out.attrs = ic_raw_eta.attrs 


        if np.min(tracers_out["temp"].isel({"zl":0})) > 100:
            tracers_out["temp"] -= 273.15

        ## Regrid the fields vertically

        ### NEED TO FIX THE HANDLING OF THICKNESS INPUT. will result in smaller number of vertical layers

        if vcoord_type == "thickness":
            tracers_out["zl"] = tracers_out["zl"].diff("zl")
            dz = tracers_out[self.z].diff(self.z)
            dz.name = "dz"
            dz = xr.concat([dz,dz[-1]],dim = self.z)


        tracers_out = tracers_out.interp({'zl':self.vgrid.zl.values})
        vel_out = vel_out.interp({'zl':self.vgrid.zl.values})

        print("Saving outputs... ",end="")
        vel_out.fillna(0).to_netcdf(
            self.mom_input_dir + "forcing/init_vel.nc",
            mode = "w",
            encoding={
                "u": {"_FillValue": netCDF4.default_fillvals["f4"]},
                "v": {"_FillValue": netCDF4.default_fillvals["f4"]}        
            },
        )


        tracers_out.to_netcdf(
            self.mom_input_dir + "forcing/init_tracers.nc",
            mode = "w",
            encoding = {'xh': {'_FillValue': None},
                        'yh': {'_FillValue': None},
                        "zl": {"_FillValue": None},
                        'temp': {'_FillValue': -1e20,'missing_value': -1e20},
                        'salt': {'_FillValue': -1e20,'missing_value': -1e20}
                        },
        )
        eta_out.to_netcdf(
            self.mom_input_dir + "forcing/init_eta.nc",
            mode = "w",
            encoding = {'xh': {'_FillValue': None},
                        'yh': {'_FillValue': None},
                        'eta_t':{'_FillValue':None}
                        },
        )
        print("done.")

        self.ic_eta = eta_out
        self.ic_tracers = tracers_out
        self.ic_vels = vel_out

        if boundaries == None:
            return

        print("BRUSHCUT BOUNDARIES")

        ## Generate a rectangular OBC domain. This is the default configuration. For fancier domains, need to use the segment class manually
        for i,o in enumerate(boundaries):
            print(f"Processing {o}...",end="")
            seg = segment(
                self.hgrid,
                f"{path}/{o.lower()}_unprocessed.nc", # location of raw boundary
                f"{self.mom_input_dir}",           # Save path
                varnames,
                "segment_{:03d}".format(i+1),
                o.lower(), # orienataion
                self.daterange[0],
                gridtype,
                vcoord_type
            )

            seg.brushcut()
            print("Done.")

    def bathymetry(self,bathy_path,varnames,fill_channels = False,minimum_layers = 3,maketopog = True):
        """
        Cuts out and interpolates chosen bathymetry, then fills inland lakes. Optionally fills narrow channels, although this is less of an issue for C grid based models like MOM6. Output saved to the input folder for your experiment. 

        Parameters:
            bathy_path (str)            Path to chosen bathymetry file. Should be a netcdf that contains your region of interest
            varnames (dict)             Dictionary mapping the coordinate and variable names of interest. Eg: {'xh':'lon','yh':'lat','elevation':'depth'}
            fill_channels (bool)   Whether or not to fill in diagonal channels. This removes more narrow inlets, but can also connect extra islands to land. 
            minimum layers (bool)   The minimum depth allowed as an integer number of layers. 3 layers means that anything shallower than the 3rd layer is deemed land
            maketopog (bool)            If true, runs fre tools to make topography. If False, reads in existing topog file and proceeds with hole filling
        """




        ## Determine whether we need to adjust bathymetry longitude to match model grid. 
        # 
        # eg if bathy is 0->360 but self.hgrid is -180->180, longitude slice becomes 
        ## 

        # if bathy[varnames["xh"]].values[0] < 0:



        if maketopog == True:
            bathy = xr.open_dataset(bathy_path,chunks="auto")[varnames["elevation"]]
            
            bathy = bathy.sel({
                varnames["yh"]:slice(self.yextent[0],self.yextent[1])
            }
            ).astype("float")

            bathy = nicer_slicer(bathy,self.xextent,varnames["xh"])


            bathy.attrs['missing_value'] = -1e20 # This is what FRE tools expects I guess?
            bathy.to_netcdf(f"{self.mom_input_dir}bathy_original.nc", engine='netcdf4')


        #     #! New code to test: Can we regrid first to pass make_topog a smaller dataset to handle?
        #     tgrid = xr.Dataset(
        #     {"lon":(["lon"],self.hgrid.x.isel(nxp=slice(1, None, 2), nyp=1).values),
        #     "lat":(["lat"],self.hgrid.y.isel(nxp=1, nyp=slice(1, None, 2)).values)
        #             }
        # )
        #     regridder_t = xe.Regridder(
        #         bathy, tgrid, "bilinear",
        #     )

        #     bathy_regrid.to_netcdf(f"{self.mom_input_dir}bathy_regrid.nc", engine='netcdf4')
        #     #! End new test code

            ## Now pass bathymetry through the FRE tools


            ## Make Topog
            args = f"--mosaic ocean_mosaic.nc --topog_type realistic --topog_file bathy_original.nc --topog_field {varnames['elevation']} --scale_factor -1 --output topog_raw.nc".split(" ")
            print(
                "FRE TOOLS: make topog parallel\n\n",
                subprocess.run(["/g/data/v45/jr5971/FRE-NCtools/build3_up_MAXXGRID/tools/make_topog/make_topog_parallel"] + args,cwd = self.mom_input_dir)
            )



        ## reopen topography to modify
        topog = xr.open_dataset(self.mom_input_dir + "topog_raw.nc")

        ## Remove inland lakes
        
        min_depth = self.vgrid.zi[minimum_layers]


        ocean_mask = topog.copy(deep = True).depth.where(topog.depth <= min_depth , 1)
        land_mask = np.abs(ocean_mask - 1)
        changed = True ## keeps track of whether solution has converged or not

        forward = True ## only useful for iterating through diagonal channel removal. Means iteration goes SW -> NE

        while changed == True:

            ## First fill in all lakes. This uses a scipy function where it fills holes made of 0's within a field of 1's 
            land_mask[:,:] = binary_fill_holes(land_mask.data)
            ## Get the ocean mask instead of land- easier to remove channels this way
            ocean_mask = np.abs(land_mask - 1)

            ## Now fill in all one-cell-wide channels
            newmask = xr.where(ocean_mask * (land_mask.shift(nx = 1) + land_mask.shift(nx = -1)) == 2,1,0)
            newmask += xr.where(ocean_mask * (land_mask.shift(ny = 1) + land_mask.shift(ny = -1)) == 2,1,0)

            if fill_channels == True:

                ## fill in all one-cell-wide horizontal channels
                newmask = xr.where(ocean_mask * (land_mask.shift(nx = 1) + land_mask.shift(nx = -1)) == 2,1,0)
                newmask += xr.where(ocean_mask * (land_mask.shift(ny = 1) + land_mask.shift(ny = -1)) == 2,1,0)
                ## Diagonal channels 
                if forward == True:
                    ## horizontal channels
                    newmask += xr.where((ocean_mask * ocean_mask.shift(nx = 1)) * (land_mask.shift({"nx":1,"ny":1}) + land_mask.shift({"ny":-1})) == 2,1,0) ## up right & below
                    newmask += xr.where((ocean_mask * ocean_mask.shift(nx = 1)) * (land_mask.shift({"nx":1,"ny":-1}) + land_mask.shift({"ny":1})) == 2,1,0) ## down right & above
                    ## Vertical channels
                    newmask += xr.where((ocean_mask * ocean_mask.shift(ny = 1)) * (land_mask.shift({"nx":1,"ny":1}) + land_mask.shift({"nx":-1})) == 2,1,0) ## up right & left
                    newmask += xr.where((ocean_mask * ocean_mask.shift(ny = 1)) * (land_mask.shift({"nx":-1,"ny":1}) + land_mask.shift({"nx":1})) == 2,1,0) ## up left & right

                    forward = False

                if forward == False:
                    ## Horizontal channels
                    newmask += xr.where((ocean_mask * ocean_mask.shift(nx = -1)) * (land_mask.shift({"nx":-1,"ny":1}) + land_mask.shift({"ny":-1})) == 2,1,0) ## up left & below
                    newmask += xr.where((ocean_mask * ocean_mask.shift(nx = -1)) * (land_mask.shift({"nx":-1,"ny":-1}) + land_mask.shift({"ny":1})) == 2,1,0) ## down left & above
                    ## Vertical channels
                    newmask += xr.where((ocean_mask * ocean_mask.shift(ny = -1)) * (land_mask.shift({"nx":1,"ny":-1}) + land_mask.shift({"nx":-1})) == 2,1,0) ## down right & left
                    newmask += xr.where((ocean_mask * ocean_mask.shift(ny = -1)) * (land_mask.shift({"nx":-1,"ny":-1}) + land_mask.shift({"nx":1})) == 2,1,0) ## down left & right

                    forward = True

              
            newmask = xr.where(newmask > 0 , 1,0)
            changed = np.max(newmask) == 1
            land_mask += newmask

        # land_mask.to_netcdf(self.mom_input_dir + "land_mask.nc")
        ocean_mask = np.abs(land_mask - 1)
        # ocean_mask.to_netcdf(self.mom_input_dir + "ocean_mask.nc")

        # ocean_mask = ocean_mask.where(ocean_mask == 1,-1e30)

        topog["depth"] *= ocean_mask


        topog["depth"] = topog["depth"].where(topog["depth"] != 0, np.nan)

        topog.expand_dims({'ntiles':1}).to_netcdf(self.mom_input_dir + "topog_deseas.nc",mode = "w",encoding={"depth":{'_FillValue': None}} )

        subprocess.run("mv topog_deseas.nc topog.nc",shell=True,cwd=self.mom_input_dir)
        

        ## Now run the remaining FRE tools to construct masks based on our topography

        args = "--num_tiles 1 --dir . --mosaic_name ocean_mosaic --tile_file hgrid.nc".split(" ")
        print("MAKE SOLO MOSAIC",subprocess.run(
            self.toolpath + "make_solo_mosaic/make_solo_mosaic --num_tiles 1 --dir . --mosaic_name ocean_mosaic --tile_file hgrid.nc",
             shell=True,
             cwd = self.mom_input_dir),sep = "\n\n")



        print("QUICK MOSAIC" , subprocess.run(
            self.toolpath + "make_quick_mosaic/make_quick_mosaic --input_mosaic ocean_mosaic.nc --mosaic_name grid_spec --ocean_topog topog.nc",
            shell=True
            ,cwd = self.mom_input_dir),sep = "\n\n")

        self.processor_mask((10,10))
        return 

    def processor_mask(self,layout):
            """
            Just a wrapper for FRE Tools check_mask. User provides processor layout tuple of processing units.
            """

            if "topog.nc" not in os.listdir(self.mom_input_dir):
                print("No topography file! Need to run make_bathymetry first")
                return
            try:            
                os.remove("mask_table*") ## Removes old mask table so as not to clog up inputdir
            except:
                pass
            print("CHECK MASK" , subprocess.run(
                self.toolpath + f"check_mask/check_mask --grid_file ocean_mosaic.nc --ocean_topog topog.nc --layout {layout[0]},{layout[1]} --halo 4",
                shell=True,
                cwd = self.mom_input_dir))
            return
    
    


class segment:
    """
    Class to turn raw boundary segment data into MOM6 boundary segments. 
    """
    def __init__(self, hgrid,infile, outfolder,varnames,seg_name,orientation, startdate, gridtype="A",vcoord_type = "height",time_units = "days"):
        """
        Boundary segments should only contain the necessary data for that segment. No horizontal chunking is done here, so big fat segments will process slowly.

        Data should be at daily temporal resolution, iterating upwards from the provided startdate. Function ignores the time metadata and puts it on Julian calendar. 


        hgrid:        xarray        the horizontal grid used for domain
        infolder:     string        path to the raw, unprocessed boundary segment
        outfolder:    string        path to folder where the model inputs will be stored
        varnames:     dictionary    Mapping between the variable / dimension names and standard naming convension of this pipeline. eg {"xq":"longitude,"yh":"latitude","salt":"salinity...}. Key "tracers" points to nested dictionary of tracers to include in boundary
        orientation:  string        Cardinal direction (lowercase) of the boundary segment
        gridtype:     string        A,B or C type grid
        seg_name:     string        Name of the segment. Something like 'segment_001'
        vcoord_type:  string        Vertical coordinate is either interfacial 'height' or layer 'thickness'. Handles appropriately
        time_units:   string        The units used by raw forcing file. eg. hours, days
        """

        ## Store coordinate names
        if gridtype == "A":
            self.x = varnames["x"]
            self.y = varnames["y"]
            
        elif gridtype in ("B","C"):
            self.xq = varnames["xq"]
            self.xh = varnames["xh"]
            self.yq = varnames["yq"]
            self.yh = varnames["yh"]

        ## Store velocity names 
        self.u = varnames["u"]
        self.v = varnames["v"]
        self.z = varnames["zl"]
        self.eta = varnames["eta"]
        self.time = varnames["time"]
        self.startdate = startdate
        ## Store tracer names
        self.tracers = varnames["tracers"]
        self.time_units = time_units


        ## Store other data
        self.infile = infile
        self.outfolder = outfolder
        self.orientation = orientation.lower() ## might not be needed? NSEW
        self.grid = gridtype
        self.hgrid = hgrid
        self.seg_name = seg_name
        self.vcoord_type = vcoord_type

    def brushcut(self,ryf = False):
        ### Implement brushcutter scheme on single segment ### 
        # print(self.infile + f"/{self.orientation}_segment_unprocessed")
        # rawseg = xr.open_dataset(self.infile,decode_times=False,engine="netcdf4")
        rawseg = xr.open_dataset(self.infile, decode_times=True)
        # rawseg = xr.open_dataset(self.infile,decode_times=False,chunks={self.time:30,self.z:25})

        ## Depending on the orientation of the segment, cut out the right bit of the hgrid 
        ## and define which coordinate is along or into the segment
        if self.orientation == "north":
            hgrid_seg = self.hgrid.isel(nyp = [-1])
            perpendicular = "ny"
            parallel = "nx"

        if self.orientation == "south":
            hgrid_seg = self.hgrid.isel(nyp = [0])
            perpendicular = "ny"
            parallel = "nx"
    

        if self.orientation == "east":
            hgrid_seg = self.hgrid.isel(nxp = [-1])
            perpendicular = "nx"
            parallel = "ny"

        if self.orientation == "west":
            hgrid_seg = self.hgrid.isel(nxp = [0])
            perpendicular = "nx"
            parallel = "ny"

        ## Need to keep track of which axis the 'main' coordinate corresponds to for later on when re-adding the 'secondary' axis
        if perpendicular == "ny":
            axis1 = 3
            axis2 = 2
        else:
            axis1 = 2
            axis2 = 3

        

        ## Grid for interpolating our fields
        interp_grid = xr.Dataset(
            {
                "lat": ([f"{parallel}_{self.seg_name}"], hgrid_seg.y.squeeze().data),
                "lon": ([f"{parallel}_{self.seg_name}"], hgrid_seg.x.squeeze().data),
            }
        ).set_coords(["lat","lon"])


        if self.grid == "A":
            ## In this case velocities and tracers all on same points
            regridder = xe.Regridder(
                rawseg[self.u].rename({self.x:"lon", self.y:"lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder + f"weights/bilinear_velocity_weights_{self.orientation}.nc",
            )
            segment_out = xr.merge([
                regridder(rawseg[
                [self.u,self.v,self.eta] + [self.tracers[i] for i in self.tracers]
                ])])
            

        if self.grid =="B":
            ## All tracers on one grid, all velocities on another
            regridder_velocity = xe.Regridder(
                rawseg[self.u].rename({self.xq:"lon", self.yq:"lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename= self.outfolder + f"weights/bilinear_velocity_weights_{self.orientation}.nc",
            )

            regridder_tracer = xe.Regridder(
                rawseg[self.tracers["salt"]].rename({self.xh:"lon", self.yh:"lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder + f"weights/bilinear_tracer_weights_{self.orientation}.nc",
            )


            segment_out = xr.merge([
                regridder_velocity(rawseg[[self.u,self.v]]), 
                regridder_tracer(rawseg[[self.eta] + [self.tracers[i] for i in self.tracers]]), 
                 ])
            
        if self.grid =="C":
            ## All tracers on one grid, all velocities on another
            regridder_uvelocity = xe.Regridder(
                rawseg[self.u].rename({self.xq:"lon", self.yh:"lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename= self.outfolder + f"weights/bilinear_uvelocity_weights_{self.orientation}.nc",
            )

            regridder_vvelocity = xe.Regridder(
                rawseg[self.v].rename({self.xh:"lon", self.yq:"lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename= self.outfolder + f"weights/bilinear_vvelocity_weights_{self.orientation}.nc",
            )

            regridder_tracer = xe.Regridder(
                rawseg[self.tracers["salt"]].rename({self.xh:"lon", self.yh:"lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder + f"weights/bilinear_tracer_weights_{self.orientation}.nc",
            )


            segment_out = xr.merge([
                regridder_vvelocity(rawseg[[self.v]]), 
                regridder_uvelocity(rawseg[[self.u]]), 
                regridder_tracer(rawseg[[self.eta] + [self.tracers[i] for i in self.tracers]]), 
                 ])

        ## segment out now contains our interpolated boundary.
        ## Now, we need to fix up all the metadata and save
        
        del segment_out["lon"]
        del segment_out["lat"]
        ## Convert temperatures to celsius # use pint
        if np.min(segment_out[self.tracers["temp"]].isel({self.time:0,self.z:0})) > 100:
            segment_out[self.tracers["temp"]] -= 273.15

        # fill in NaNs
        segment_out = (
            segment_out
            .ffill(self.z)
            .interpolate_na(f"{parallel}_{self.seg_name}")
            .ffill(f"{parallel}_{self.seg_name}")
            .bfill(f"{parallel}_{self.seg_name}")
        )


        ##### FIX UP COORDINATE METADATA #####
        ## OLD: Use 1950 reference
        # start_jd50 = (self.startdate - dt.datetime.strptime("1950-01-01 00:00:00","%Y-%m-%d %H:%M:%S")).days
        # time = np.arange(
        #     start_jd50,
        #     start_jd50 + rawseg[self.time].shape[0]  ## Time is just range of days from start of window until end in Julian day offset from 1950 epoch
        # )

        #! This only works for RYF or shorter IAF runs. 
        #  We'd need a better way to split up forcing files into separate chunks if you wanted to run one year at a time. 
        time = np.arange(
            0, #! Indexing everything from start of experiment = simple but maybe counterintutive? 
            segment_out[self.time].shape[0], ## Time is indexed from start date of window
            dtype = float
        )

        

        segment_out = segment_out.assign_coords({"time":time})

        segment_out.time.attrs = {"calendar":"julian","units":f"{self.time_units} since {self.startdate}","modulo":" "}
        # Dictionary we built for encoding the netcdf at end
        encoding_dict = {
            "time": {
                "dtype": "double",
            },
            f"nx_{self.seg_name}": {
                "dtype": "int32",
            },
            f"ny_{self.seg_name}": {
                "dtype": "int32",
            },
        }


        ### Generate our dz variable. This needs to be in layer thicknesses
        if self.vcoord_type == "height":
            dz = segment_out[self.z].diff(self.z)
            dz.name = "dz"
            dz = xr.concat([dz,dz[-1]],dim = self.z)

        else:
            dz = segment_out[self.z]
            dz.name = "dz"
        del segment_out[self.z]


        # Here, keep in mind that 'var' keeps track of the mom6 variable names we want, and self.tracers[var] will return the name of the variable from the original data
        
        allfields = {**self.tracers,"u":self.u,"v":self.v} ## Combine all fields into one flattened dictionary to iterate over as we fix metadata


        for var in allfields: ## Replace with more generic list of tracer variables that might be included?
            v = f"{var}_{self.seg_name}"
            ## Rename each variable in dataset
            segment_out = segment_out.rename({allfields[var]: v})

            ## Rename vertical coordinate for this variable
            segment_out[f"{var}_{self.seg_name}"] = segment_out[f"{var}_{self.seg_name}"].rename({self.z: f"nz_{self.seg_name}_{var}"})

            ## Replace the old depth coordinates with incremental integers 
            segment_out[f"nz_{self.seg_name}_{var}"] = np.arange(segment_out[f"nz_{self.seg_name}_{var}"].size)

            ## Re-add the secondary dimension (even though it represents one value..)
            segment_out[v] = segment_out[v].expand_dims(
                f"{perpendicular}_{self.seg_name}",axis = axis2
            )


            ## Add the layer thicknesses
            segment_out[f"dz_{v}"] = (
                ["time", f"nz_{v}", f"ny_{self.seg_name}", f"nx_{self.seg_name}"],
                da.broadcast_to(
                    dz.data[None, :, None, None],
                    segment_out[v].shape,
                    chunks=(1, None, None, None), ## Chunk in each time, and every 5 vertical layers
                ),
            )


            encoding_dict[v] = {
                "_FillValue": netCDF4.default_fillvals["f8"],
                "zlib": True,
                # "chunksizes": tuple(s),
            }
            encoding_dict[f"dz_{v}"] = {
                "_FillValue": netCDF4.default_fillvals["f8"],
                "zlib": True,
                # "chunksizes": tuple(s),
            }

            ## appears to be another variable just with integers??
            encoding_dict[f"nz_{self.seg_name}_{var}"] = {
                "dtype": "int32"
            }

        ## Treat eta separately since it has no vertical coordinate. Do the same things as for the surface variables above
        segment_out = segment_out.rename({self.eta: f"eta_{self.seg_name}"})
        encoding_dict[ f"eta_{self.seg_name}"] = {
            "_FillValue": netCDF4.default_fillvals["f8"],
        }
        segment_out[f"eta_{self.seg_name}"] = segment_out[f"eta_{self.seg_name}"].expand_dims(
            f"{perpendicular}_{self.seg_name}",axis = axis2 - 1
        )

        # Overwrite the actual lat/lon values in the dimensions, replace with incrementing integers
        segment_out[f"{parallel}_{self.seg_name}"] = np.arange(segment_out[f"{parallel}_{self.seg_name}"].size)
        segment_out[f"{perpendicular}_{self.seg_name}"] = [0]

        # Store actual lat/lon values here as variables rather than coordinates
        segment_out[f"lon_{self.seg_name}"] = ([f"ny_{self.seg_name}", f"nx_{self.seg_name}"], hgrid_seg.x.data)
        segment_out[f"lat_{self.seg_name}"] = ([f"ny_{self.seg_name}", f"nx_{self.seg_name}"], hgrid_seg.y.data)

        # del segment_out["depth"]


        with ProgressBar():
            segment_out["time"] = segment_out["time"].assign_attrs({"modulo":" "}) ## Add modulo attribute for MOM6 to treat as repeat forcing
            segment_out.load().to_netcdf(
                self.outfolder + f"forcing/forcing_obc_{self.seg_name}.nc", encoding=encoding_dict, unlimited_dims="time"
                )

        return segment_out , encoding_dict
    


    
