import numpy as np
from itertools import cycle
from pathlib import Path
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
import shutil
import os
from .utils import vecdot

warnings.filterwarnings("ignore")

__all__ = [
    "nicer_slicer",
    "motu_requests",
    "dz",
    "angle_between",
    "latlon_to_cartesian",
    "quadrilateral_area",
    "quadrilateral_areas",
    "rectangular_hgrid",
    "experiment",
    "segment",
]


def nicer_slicer(data, xextent, xcoords, buffer=2):
    """Slices longitudes, handling periodicity and 'seams' where the
    data wraps around (commonly either at -180 -> 180 or -270 -> 90)

    The algorithm works in five steps:

    - Determine whether we need to add or subtract 360 to get the
      middle of the xextent to lie within data's lonitude range
      (hereby oldx)

    - Shift the dataset so that its midpoint matches the midpoint of
      xextent (up to a muptiple of 360). Now, the modified oldx
      doesn't increase monotonically W to E since the 'seam' has
      moved.

    - Fix oldx to make it monotonically increasing again. This uses
      the information we have about the way the dataset was
      shifted/rolled

    - Slice the data index-wise. We know that ``|xextent| / 360``
      multiplied by the number of discrete longitude points will give
      the total width of our slice, and we've already set the midpoint
      to be the middle of the target domain. Here we add a buffer
      region on either side if we need it for interpolation.

    - Finally re-add the right multiple of 360 so the whole domain matches the target.

    Args:
        data (xarray.Dataset): The global data you want to slice in longitude
        xextent (Tuple[float, float]): The target longitudes you want to slice
            to. This must be either start negative and progress to
            positive, or be entirely positive
        xcoords (Union[str, List[str]): The name of the longitude
            dimension in your xarray or list of names

    Returns:
        xarray.Dataset: The data after the slicing has been performed.

    """

    if isinstance(xcoords, str):
        xcoords = [xcoords]

    for x in xcoords:
        mp_target = np.mean(xextent)  ## Midpoint of target domain

        ## Find a corresponding value for the intended domain midpint in our data. Assuming here that data has equally spaced longitude values spanning 360deg
        for i in range(-1, 2, 1):
            if data[x][0] <= mp_target + 360 * i <= data[x][-1]:
                _mp_target = (
                    mp_target + 360 * i
                )  ## Shifted version of target midpoint. eg, could be -90 vs 270. i keeps track of what multiple of 360 we need to shift entire grid by to match midpoint

                mp_data = data[x][data[x].shape[0] // 2].values  ## Midpoint of the data

                shift = -1 * (data[x].shape[0] * (_mp_target - mp_data)) // 360
                shift = int(
                    shift
                )  ## This is the number of indices between the data midpoint, and the target midpoint. Sign indicates direction needed to shift

                new_data = data.roll(
                    {x: 1 * shift}, roll_coords=True
                )  ## Shifts data so that the midpoint of the target domain is the middle of the data for easy slicing

                new_x = new_data[
                    x
                ].values  ## Create a new longitude coordinate. We'll modify this to remove any seams (jumps like -270 -> 90)

                ## Take the 'seam' of the data, and either backfill or forward fill based on whether the data was shifted F or west
                if shift > 0:
                    new_seam_index = shift

                    new_x[0:new_seam_index] -= 360

                if shift < 0:
                    new_seam_index = data[x].shape[0] + shift

                    new_x[new_seam_index:] += 360

                new_x -= (
                    i * 360
                )  ## Use this to recentre the midpoint to match that of target domain

                new_data = new_data.assign_coords({x: new_x})

                ## Choose the number of x points to take from the middle, including a buffer. Use this to index the new global dataset

                num_xpoints = (
                    int(data[x].shape[0] * (mp_target - xextent[0])) // 360 + buffer * 2
                )

        data = new_data.isel(
            {
                x: slice(
                    data[x].shape[0] // 2 - num_xpoints,
                    data[x].shape[0] // 2 + num_xpoints,
                )
            }
        )

    return data


def motu_requests(
    xextent,
    yextent,
    daterange,
    outfolder,
    usr,
    pwd,
    segs,
    url="https://my.cmems-du.eu/motu-web/Motu",
    serviceid="GLOBAL_MULTIYEAR_PHY_001_030-TDS",
    productid="cmems_mod_glo_phy_my_0.083_P1D-m",
    buffer=0.3,
):
    """Generates MOTU data request for each specified boundary, as
    well as for the initial condition. By default pulls the GLORYS
    reanalysis dataset.

    Args:
        xextent (List[float]): Extreme values of longitude coordinates for rectangular domain
        yextent (List[float]): Extreme values of latitude coordinates for rectangular domain
        daterange (Tuple[str]): Start and end dates of boundary forcing window. Format: ``%Y-%m-%d %H:%M:%S``
        outfolder (str): Directory in which to receive the downloaded files
        usr (str): MOTU authentication username
        pwd (str): MOTU authentication password
        segs (List[str]): List of the cardinal directions for your boundary forcing
        url (Optional[str]): MOTU server for the request (defaults to CMEMS)
        serviceid (Optional[str]): Service containing the desired dataset
        productid (Optional[str]): Data product within the chosen service.

    Returns:
        str: A bash script which will call ``motuclient`` to invoke the data requests.

    """

    if type(segs) == str:
        return f"\nprintf 'processing {segs} segment' \npython -m motuclient --motu {url} --service-id {serviceid} --product-id {productid} --longitude-min {xextent[0]} --longitude-max {xextent[1]} --latitude-min {yextent[0]} --latitude-max {yextent[1]} --date-min {daterange[0]} --date-max {daterange[1]} --depth-min 0.49 --depth-max 6000 --variable so --variable thetao --variable vo --variable zos --variable uo --out-dir {outfolder} --out-name {segs}_unprocessed --user '{usr}' --pwd '{pwd}'\n"

    ## Buffer pads out our boundaries a small amount to allow for interpolation
    xextent, yextent = np.array(xextent), np.array(yextent)
    script = "#!/bin/bash\n\n"
    for seg in segs:
        if seg == "east":
            script += motu_requests(
                [xextent[1] - buffer, xextent[1] + buffer],
                yextent,
                daterange,
                outfolder,
                usr,
                pwd,
                seg,
                url=url,
                serviceid=serviceid,
                productid=productid,
            )
        if seg == "west":
            script += motu_requests(
                [xextent[0] - buffer, xextent[0] + buffer],
                yextent,
                daterange,
                outfolder,
                usr,
                pwd,
                seg,
                url=url,
                serviceid=serviceid,
                productid=productid,
            )
        if seg == "north":
            script += motu_requests(
                xextent,
                [yextent[1] - buffer, yextent[1] + buffer],
                daterange,
                outfolder,
                usr,
                pwd,
                seg,
                url=url,
                serviceid=serviceid,
                productid=productid,
            )

        if seg == "south":
            script += motu_requests(
                xextent,
                [yextent[0] - buffer, yextent[0] + buffer],
                daterange,
                outfolder,
                usr,
                pwd,
                seg,
                url=url,
                serviceid=serviceid,
                productid=productid,
            )
    ## Now handle the initial condition
    script += motu_requests(
        xextent + np.array([-1 * buffer, buffer]),
        yextent + np.array([-1 * buffer, buffer]),
        [
            daterange[0],
            dt.datetime.strptime(daterange[0], "%Y-%m-%d %H:%M:%S")
            + dt.timedelta(hours=1),
        ],  ## For initial condition just take one day
        outfolder,
        usr,
        pwd,
        "ic",
        url=url,
        serviceid=serviceid,
        productid=productid,
    )
    return script


def dz(npoints, ratio, target_depth, min_dz=0.0001, tolerance=1):
    """Generate a hyperbolic tangent thickness profile for the
    experiment.  Iterates to find the mininum depth value which gives
    the target depth within some tolerance

    Args:
        npoints (int): Number of vertical points
        ratio (float): Ratio of largest to smallest layer
            thickness. Negative values mean higher resolution is at
            bottom rather than top of the column.
        target_depth (float): Maximum depth of a layer
        min_dz (float): Starting layer thickness for iteration
        tolerance (float): Tolerance to the target depth.

    Returns:
        numpy.array: An array containing the thickness profile.
    """

    profile = min_dz + 0.5 * (np.abs(ratio) * min_dz - min_dz) * (
        1 + np.tanh(2 * np.pi * (np.arange(npoints) - npoints // 2) / npoints)
    )
    tot = np.sum(profile)
    if np.abs(tot - target_depth) < tolerance:
        if ratio > 0:
            return profile

        return profile[::-1]

    err_ratio = target_depth / tot

    return dz(npoints, ratio, target_depth, min_dz * err_ratio)


def angle_between(v1, v2, v3):
    """Returns the angle v2-v1-v3 (in radians). That is the angle between vectors v1-v2 and v1-v3."""

    v1xv2 = np.cross(v1, v2)
    v1xv3 = np.cross(v1, v3)

    cosangle = vecdot(v1xv2, v1xv3) / np.sqrt(
        vecdot(v1xv2, v1xv2) * vecdot(v1xv3, v1xv3)
    )

    return np.arccos(cosangle)


def quadrilateral_area(v1, v2, v3, v4):
    """Returns area of a spherical quadrilateral on the unit sphere that
    has vertices on 3-vectors `v1`, `v2`, `v3`, `v4` (counter-clockwise
    orientation is implied). The area is computed via the excess of the
    sum of the spherical angles of the quadrilateral from 2π."""

    if not (
        np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v3, v3)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v4, v4)))
    ):
        raise ValueError("vectors provided must have the same length")

    R = np.sqrt(vecdot(v1, v1))

    a1 = angle_between(v1, v2, v4)
    a2 = angle_between(v2, v3, v1)
    a3 = angle_between(v3, v4, v2)
    a4 = angle_between(v4, v1, v3)

    return (a1 + a2 + a3 + a4 - 2 * np.pi) * R**2


def latlon_to_cartesian(lat, lon, R=1):
    """Convert latitude-longitude (in degrees) to Cartesian coordinates on a sphere of radius `R`.
    By default `R = 1`."""

    x = R * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = R * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = R * np.sin(np.deg2rad(lat))

    return x, y, z


def quadrilateral_areas(lat, lon, R=1):
    """Returns area of spherical quadrilaterals on a sphere of radius `R`. By default, `R = 1`.
    The quadrilaterals are formed by constant latitude and longitude lines on the `lat`-`lon` grid provided.

    Args:
        lat (array): Array of latitude points (in degrees)
        lon (array): Array of longitude points (in degrees)

    Returns:
        areas (array): Array with the areas of the quadrilaterals defined by the
                       `lat`-`lon` grid provided. If the `lat`-`lon` are `m x n`
                       then `areas` is `(m-1) x (n-1)`.
    """

    coords = np.dstack(latlon_to_cartesian(lat, lon, R))

    return quadrilateral_area(
        coords[:-1, :-1, :], coords[:-1, 1:, :], coords[1:, 1:, :], coords[1:, :-1, :]
    )


def rectangular_hgrid(λ, φ):
    """
    Construct a horizontal grid with all the metadata given an array of
    latitudes (`φ`) and longitudes (`λ`).

    Caution:
        Here, it is assumed the grid's boundaries are lines of constant latitude and
        longitude. Rotated grids need to be handled in a different manner.
        Also we assume here that the longitude array values are uniformly spaced.

        Make sure both `λ` and `φ` *must* be monotonically increasing.

    Args:
        λ (numpy.array): All longitude points on the supergrid.
        φ (numpy.array): All latitude points on the supergrid.

    Returns:
        xarray.Dataset: A FMS-compatible *hgrid*, including the required attributes.
    """

    R = 6371e3  # mean radius of the Earth

    dλ = λ[1] - λ[0]  # assuming that longitude is uniformly spaced

    # dx = R * cos(φ) * np.deg2rad(dλ) / 2
    # Note: division by 2 because we're on the supergrid
    dx = np.broadcast_to(
        R * np.cos(np.deg2rad(φ)) * np.deg2rad(dλ) / 2,
        (λ.shape[0] - 1, φ.shape[0]),
    ).T

    # dy = R * np.deg2rad(dφ) / 2
    # Note: division by 2 because we're on the supergrid
    dy = np.broadcast_to(R * np.deg2rad(np.diff(φ)) / 2, (λ.shape[0], φ.shape[0] - 1)).T

    lon, lat = np.meshgrid(λ, φ)

    area = quadrilateral_areas(lat, lon, R)

    attrs = {
        "tile": {
            "standard_name": "grid_tile_spec",
            "geometry": "spherical",
            "north_pole": "0.0 90.0",
            "discretization": "logically_rectangular",
            "conformal": "true",
        },
        "x": {"standard_name": "geographic_longitude", "units": "degree_east"},
        "y": {"standard_name": "geographic_latitude", "units": "degree_north"},
        "dx": {
            "standard_name": "grid_edge_x_distance",
            "units": "metres",
        },
        "dy": {
            "standard_name": "grid_edge_y_distance",
            "units": "metres",
        },
        "area": {
            "standard_name": "grid_cell_area",
            "units": "m2",
        },
        "angle_dx": {
            "standard_name": "grid_vertex_x_angle_WRT_geographic_east",
            "units": "degrees_east",
        },
        "arcx": {
            "standard_name": "grid_edge_x_arc_type",
            "north_pole": "0.0 90.0",
        },
    }

    return xr.Dataset(
        {
            "tile": ((), np.array(b"tile1", dtype="|S255"), attrs["tile"]),
            "x": (["nyp", "nxp"], lon, attrs["x"]),
            "y": (["nyp", "nxp"], lat, attrs["y"]),
            "dx": (["nyp", "nx"], dx, attrs["dx"]),
            "dy": (["ny", "nxp"], dy, attrs["dy"]),
            "area": (["ny", "nx"], area, attrs["area"]),
            "angle_dx": (["nyp", "nxp"], lon * 0, attrs["angle_dx"]),
            "arcx": ((), np.array(b"small_circle", dtype="|S255"), attrs["arcx"]),
        }
    )


class experiment:
    """The main class for setting up a regional experiment.

    Knows everything about your regional experiment! Methods in this
    class will generate the various input files you need to generate a
    MOM6 experiment forced with open boundary conditions (OBCs). It's
    written agnostic to your choice of boundary forcing, topography
    and surface forcing - you need to tell it what your variables are
    all called via mapping dictionaries from MOM6 variable/coordinate
    name to the name in the input dataset.

    Args:
        xextent (Tuple[float]): Extent of the region in longitude.
        yextent (Tuple[float]): Extent of the region in latitude.
        daterange (Tuple[str]): Start and end dates of the boundary forcing window.
        resolution (float): Lateral resolution of the domain, in degrees.
        vlayers (int): Number of vertical layers.
        dz_ratio (float): Ratio of largest to smallest layer thickness, used in :func:`~dz`.
        depth (float): Depth of the domain.
        mom_run_dir (str): Path of the MOM6 control directory.
        mom_input_dir (str): Path of the MOM6 input directory, to receive the forcing files.
        toolpath (str): Path of FREtools binaries.
        gridtype (Optional[str]): Type of grid to generate, only ``even_spacing`` is supported.

    """

    def __init__(
        self,
        xextent,
        yextent,
        daterange,
        resolution,
        vlayers,
        dz_ratio,
        depth,
        mom_run_dir,
        mom_input_dir,
        toolpath,
        gridtype="even_spacing",
    ):
        self.mom_run_dir = Path(mom_run_dir)
        self.mom_input_dir = Path(mom_input_dir)

        self.mom_run_dir.mkdir(exist_ok=True)
        self.mom_input_dir.mkdir(exist_ok=True)

        self.xextent = xextent
        self.yextent = yextent
        self.daterange = [
            dt.datetime.strptime(daterange[0], "%Y-%m-%d %H:%M:%S"),
            dt.datetime.strptime(daterange[1], "%Y-%m-%d %H:%M:%S"),
        ]
        self.res = resolution
        self.vlayers = vlayers
        self.dz_ratio = dz_ratio
        self.depth = depth
        self.toolpath = toolpath
        self.hgrid = self._make_hgrid(gridtype)
        self.vgrid = self._make_vgrid()
        self.gridtype = gridtype

        # create additional directories and links
        (self.mom_input_dir / "weights").mkdir(exist_ok=True)
        (self.mom_input_dir / "forcing").mkdir(exist_ok=True)

        run_inputdir = self.mom_run_dir / "inputdir"
        if not run_inputdir.exists():
            run_inputdir.symlink_to(self.mom_input_dir.resolve())
        input_rundir = self.mom_input_dir / "rundir"
        if not input_rundir.exists():
            input_rundir.symlink_to(self.mom_run_dir.resolve())

    def _make_hgrid(self, gridtype):
        """Sets up hgrid based on users specification of
        domain. Default behaviour leaves latitude and longitude evenly
        spaced.

        If user specifies a resolution of 0.1 degrees, longitude is
        spaced this way and latitude spaced with 0.1
        cos(mean_latitude). This way, grids in the centre of the
        domain are perfectly square, but decrease monatonically in
        area away from the equator

        """

        if gridtype == "even_spacing":
            # longitudes will just be evenly spaced, based only on resolution and bounds
            nx = int((self.xextent[1] - self.xextent[0]) / (self.res / 2))
            if nx % 2 != 1:
                nx += 1

            x = np.linspace(self.xextent[0], self.xextent[1], nx)

            # Latitudes evenly spaced by dx * cos(mean_lat)
            res_y = self.res * np.cos(np.mean(self.yextent) * np.pi / 180)

            ny = int((self.yextent[1] - self.yextent[0]) / (res_y / 2)) + 1
            if ny % 2 != 1:
                ny += 1

            y = np.linspace(self.yextent[0], self.yextent[1], ny)
            hgrid = rectangular_hgrid(x, y)
            hgrid.to_netcdf(self.mom_input_dir / "hgrid.nc")

            return hgrid

    def _make_vgrid(self):
        """Generates a vertical grid based on the number of layers
        and vertical ratio specified at the class level.

        """

        thickness = dz(self.vlayers + 1, self.dz_ratio, self.depth)
        vcoord = xr.Dataset(
            {
                "zi": ("zi", np.cumsum(thickness)),
                "zl": ("zl", (np.cumsum(thickness) + 0.5 * thickness)[0:-1]),
            }  ## THIS MIGHT BE WRONG REVISIT
        )
        vcoord["zi"].attrs = {"units": "meters"}
        vcoord.to_netcdf(self.mom_input_dir / "vcoord.nc")

        return vcoord

    def ocean_forcing(
        self, path, varnames, boundaries=None, gridtype="A", vcoord_type="height"
    ):
        """Reads in the forcing files that force the ocean at
        boundaries (if specified) and for initial condition

        Args:
            path (Union[str, Path]): Path to directory containing the forcing
                files. Files should be named
                ``north_segment_unprocessed`` for each boundary (for
                the cardinal directions) and ``ic_unprocessed`` for the
                initial conditions.
            varnames (Dict[str, str]): Mapping from MOM6
                variable/coordinate names to the name in the input
                dataset.
            boundaries (List[str]): Cardinal directions of included boundaries, in anticlockwise order
            gridtype (Optional[str]): Arakawa grid staggering of input, one of ``A``, ``B`` or ``C``
            vcoord_type (Optional[str]): The type of vertical
                coordinate used in the forcing files. Either
                ``height`` or ``thickness``.

        """

        path = Path(path)

        ## Do initial condition

        ## pull out the initial velocity on MOM5's Bgrid
        ic_raw = xr.open_dataset(path / "ic_unprocessed")

        if varnames["time"] in ic_raw.dims:
            ic_raw = ic_raw.isel({varnames["time"]: 0})

        ## Separate out tracers from two velocity fields of IC
        try:
            ic_raw_tracers = ic_raw[
                [varnames["tracers"][i] for i in varnames["tracers"]]
            ]
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
            ic_raw_tracers = ic_raw_tracers.rename(
                {varnames["x"]: "lon", varnames["y"]: "lat"}
            )
            ic_raw_u = ic_raw_u.rename({varnames["x"]: "lon", varnames["y"]: "lat"})
            ic_raw_v = ic_raw_v.rename({varnames["x"]: "lon", varnames["y"]: "lat"})
            ic_raw_eta = ic_raw_eta.rename({varnames["x"]: "lon", varnames["y"]: "lat"})

        if gridtype == "B":
            ic_raw_tracers = ic_raw_tracers.rename(
                {varnames["xh"]: "lon", varnames["yh"]: "lat"}
            )
            ic_raw_eta = ic_raw_eta.rename(
                {varnames["xh"]: "lon", varnames["yh"]: "lat"}
            )
            ic_raw_u = ic_raw_u.rename({varnames["xq"]: "lon", varnames["yq"]: "lat"})
            ic_raw_v = ic_raw_v.rename({varnames["xq"]: "lon", varnames["yq"]: "lat"})

        if gridtype == "C":
            ic_raw_tracers = ic_raw_tracers.rename(
                {varnames["xh"]: "lon", varnames["yh"]: "lat"}
            )
            ic_raw_eta = ic_raw_eta.rename(
                {varnames["xh"]: "lon", varnames["yh"]: "lat"}
            )
            ic_raw_u = ic_raw_u.rename({varnames["xq"]: "lon", varnames["yh"]: "lat"})
            ic_raw_v = ic_raw_v.rename({varnames["xh"]: "lon", varnames["yq"]: "lat"})

        ## Construct the xq,yh and xh yq grids
        ugrid = (
            self.hgrid[["x", "y"]]
            .isel(nxp=slice(None, None, 2), nyp=slice(1, None, 2))
            .rename({"x": "lon", "y": "lat"})
            .set_coords(["lat", "lon"])
        )
        vgrid = (
            self.hgrid[["x", "y"]]
            .isel(nxp=slice(1, None, 2), nyp=slice(None, None, 2))
            .rename({"x": "lon", "y": "lat"})
            .set_coords(["lat", "lon"])
        )

        ## Construct the cell centre grid for tracers (xh,yh).
        tgrid = xr.Dataset(
            {
                "lon": (
                    ["lon"],
                    self.hgrid.x.isel(nxp=slice(1, None, 2), nyp=1).values,
                ),
                "lat": (
                    ["lat"],
                    self.hgrid.y.isel(nxp=1, nyp=slice(1, None, 2)).values,
                ),
            }
        )

        ### Drop NaNs to be re-added later
        # NaNs are from the land mask. When we interpolate onto a new grid, need to put in the new land mask. If NaNs left in, land mask stays the same
        ic_raw_tracers = (
            ic_raw_tracers.interpolate_na("lon", method="linear")
            .ffill("lon")
            .bfill("lon")
            .ffill("lat")
            .bfill("lat")
            .ffill(varnames["zl"])
        )

        ic_raw_u = (
            ic_raw_u.interpolate_na("lon", method="linear")
            .ffill("lon")
            .bfill("lon")
            .ffill("lat")
            .bfill("lat")
            .ffill(varnames["zl"])
        )

        ic_raw_v = (
            ic_raw_v.interpolate_na("lon", method="linear")
            .ffill("lon")
            .bfill("lon")
            .ffill("lat")
            .bfill("lat")
            .ffill(varnames["zl"])
        )

        ic_raw_eta = (
            ic_raw_eta.interpolate_na("lon", method="linear")
            .ffill("lon")
            .bfill("lon")
            .ffill("lat")
            .bfill("lat")
        )

        ## Make our three horizontal regrideers
        regridder_u = xe.Regridder(
            ic_raw_u,
            ugrid,
            "bilinear",
        )
        regridder_v = xe.Regridder(
            ic_raw_v,
            vgrid,
            "bilinear",
        )

        regridder_t = xe.Regridder(
            ic_raw_tracers,
            tgrid,
            "bilinear",
        )

        print("INITIAL CONDITIONS")
        ## Regrid all fields horizontally.
        print("Regridding Velocities...", end="")
        vel_out = xr.merge(
            [
                regridder_u(ic_raw_u)
                .rename({"lon": "xq", "lat": "yh", "nyp": "ny", varnames["zl"]: "zl"})
                .rename("u"),
                regridder_v(ic_raw_v)
                .rename({"lon": "xh", "lat": "yq", "nxp": "nx", varnames["zl"]: "zl"})
                .rename("v"),
            ]
        )
        print("Done.\nRegridding Tracers...")
        tracers_out = xr.merge(
            [
                regridder_t(ic_raw_tracers[varnames["tracers"][i]]).rename(i)
                for i in varnames["tracers"]
            ]
        ).rename({"lon": "xh", "lat": "yh", varnames["zl"]: "zl"})
        print("Done.\nRegridding Free surface...")

        eta_out = (
            regridder_t(ic_raw_eta).rename({"lon": "xh", "lat": "yh"}).rename("eta_t")
        )  ## eta_t is the name set in MOM_input by default

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

        if np.min(tracers_out["temp"].isel({"zl": 0})) > 100:
            tracers_out["temp"] -= 273.15

        ## Regrid the fields vertically

        ### NEED TO FIX THE HANDLING OF THICKNESS INPUT. will result in smaller number of vertical layers

        if vcoord_type == "thickness":
            tracers_out["zl"] = tracers_out["zl"].diff("zl")
            dz = tracers_out[self.z].diff(self.z)
            dz.name = "dz"
            dz = xr.concat([dz, dz[-1]], dim=self.z)

        tracers_out = tracers_out.interp({"zl": self.vgrid.zl.values})
        vel_out = vel_out.interp({"zl": self.vgrid.zl.values})

        print("Saving outputs... ", end="")

        ## Remove time IF it exists. Users may already have done so for us
        if "time" in vel_out.dims:
            vel_out = vel_out.isel(time=0).drop("time")
        if "time" in tracers_out.dims:
            tracers_out = tracers_out.isel(time=0).drop("time")
        if "time" in eta_out.dims:
            eta_out = eta_out.isel(time=0).drop("time")

        vel_out.fillna(0).to_netcdf(
            self.mom_input_dir / "forcing/init_vel.nc",
            mode="w",
            encoding={
                "u": {"_FillValue": netCDF4.default_fillvals["f4"]},
                "v": {"_FillValue": netCDF4.default_fillvals["f4"]},
            },
        )

        tracers_out.to_netcdf(
            self.mom_input_dir / "forcing/init_tracers.nc",
            mode="w",
            encoding={
                "xh": {"_FillValue": None},
                "yh": {"_FillValue": None},
                "zl": {"_FillValue": None},
                "temp": {"_FillValue": -1e20, "missing_value": -1e20},
                "salt": {"_FillValue": -1e20, "missing_value": -1e20},
            },
        )
        eta_out.to_netcdf(
            self.mom_input_dir / "forcing/init_eta.nc",
            mode="w",
            encoding={
                "xh": {"_FillValue": None},
                "yh": {"_FillValue": None},
                "eta_t": {"_FillValue": None},
            },
        )
        print("done.")

        self.ic_eta = eta_out
        self.ic_tracers = tracers_out
        self.ic_vels = vel_out

        if boundaries is None:
            return

        print("BRUSHCUT BOUNDARIES")

        ## Generate a rectangular OBC domain. This is the default
        ## configuration. For fancier domains, need to use the segment
        ## class manually
        for i, o in enumerate(boundaries):
            print(f"Processing {o}...", end="")
            seg = segment(
                self.hgrid,
                path / f"{o.lower()}_unprocessed",  # location of raw boundary
                self.mom_input_dir,
                varnames,
                "segment_{:03d}".format(i + 1),
                o.lower(),  # orienataion
                self.daterange[0],
                gridtype,
                vcoord_type,
            )

            seg.brushcut()
            print("Done.")

    def bathymetry(
        self,
        bathy_path,
        varnames,
        fill_channels=False,
        minimum_layers=3,
        maketopog=True,
        positivedown=False,
        chunks="auto",
    ):
        """Cuts out and interpolates chosen bathymetry, then fills
        inland lakes.

        It's also possible to optionally fill narrow channels, although this
        is less of an issue for models on a C-grid, like MOM6. Output
        saved to the input folder for your experiment.

        Args:
            bathy_path (str): Path to chosen bathymetry file netCDF file
            varnames (Dict[str, str]): Mapping of coordinate and
                variable names between the input and output.
            fill_channels (Optional[bool]): Whether or not to fill in
                diagonal channels. This removes more narrow inlets,
                but can also connect extra islands to land.
            minimum layers (Optional[int]): The minimum depth allowed
                as an integer number of layers. The default of 3
                layers means that anything shallower than the 3rd
                layer is deemed land.
            maketopog (Optional[bool]): Whether to use FREtools to
                make topography (if true), or read an existing file.
            positivedown (Optional[bool]): If true, assumes that
                bathymetry vertical coordinate is positive down.
            chunks (Optional Dict[str, str]): Chunking scheme for bathymetry, eg {"lon": 100, "lat": 100}. Use lat/lon rather than the coordinate names in the input file.


        """

        if maketopog == True:
            if chunks != "auto":
                chunks = {varnames["xh"]: chunks["lon"], varnames["yh"]: chunks["lat"]}

            bathy = xr.open_dataset(bathy_path, chunks=chunks)[varnames["elevation"]]

            bathy = bathy.sel(
                {
                    varnames["yh"]: slice(self.yextent[0] - 1, self.yextent[1] + 1)
                }  #! Hardcoded 1 degree buffer around bathymetry selection. TODO: automatically select buffer
            ).astype("float")

            ## Here need to make a decision as to whether to slice 'normally' or with nicer slicer for 360 degree domain.

            horizontal_resolution = bathy[varnames["xh"]][1] - bathy[varnames["xh"]][0]
            horizontal_extent = (
                bathy[varnames["xh"]][-1]
                - bathy[varnames["xh"]][0]
                + horizontal_resolution
            )

            if np.isclose(horizontal_extent, 360):
                ## Assume that we're dealing with a global grid, in which case we use nicer slicer
                bathy = nicer_slicer(
                    bathy,
                    np.array(self.xextent)
                    + np.array(
                        [-0.1, 0.1]
                    ),  #! Hardcoded 0.1 degree buffer around bathymetry selection. TODO: automatically select buffer
                    varnames["xh"],
                )
            else:
                ## Otherwise just slice normally
                bathy = bathy.sel(
                    {
                        varnames["xh"]: slice(self.xextent[0] - 1, self.xextent[1] + 1)
                    }  #! Hardcoded 1 degree buffer around bathymetry selection. TODO: automatically select buffer
                )

            bathy.attrs[
                "missing_value"
            ] = -1e20  # This is what FRE tools expects I guess?
            bathyout = xr.Dataset({"elevation": bathy})
            bathy.close()

            bathyout = bathyout.rename({varnames["xh"]: "lon", varnames["yh"]: "lat"})
            bathyout.lon.attrs["units"] = "degrees_east"
            bathyout.lat.attrs["units"] = "degrees_north"
            bathyout.elevation.attrs["_FillValue"] = -1e20
            bathyout.elevation.attrs["units"] = "m"
            bathyout.elevation.attrs[
                "standard_name"
            ] = "height_above_reference_ellipsoid"
            bathyout.elevation.attrs["long_name"] = "Elevation relative to sea level"
            bathyout.elevation.attrs["coordinates"] = "lon lat"
            bathyout.to_netcdf(
                self.mom_input_dir / "bathy_original.nc", mode="w", engine="netcdf4"
            )

            tgrid = xr.Dataset(
                {
                    "lon": (
                        ["lon"],
                        self.hgrid.x.isel(nxp=slice(1, None, 2), nyp=1).values,
                    ),
                    "lat": (
                        ["lat"],
                        self.hgrid.y.isel(nxp=1, nyp=slice(1, None, 2)).values,
                    ),
                }
            )
            tgrid = xr.Dataset(
                data_vars={
                    "elevation": (
                        ["lat", "lon"],
                        np.zeros(
                            self.hgrid.x.isel(
                                nxp=slice(1, None, 2), nyp=slice(1, None, 2)
                            ).shape
                        ),
                    )
                },
                coords={
                    "lon": (
                        ["lon"],
                        self.hgrid.x.isel(nxp=slice(1, None, 2), nyp=1).values,
                    ),
                    "lat": (
                        ["lat"],
                        self.hgrid.y.isel(nxp=1, nyp=slice(1, None, 2)).values,
                    ),
                },
            )

            # rewrite chunks to use lat/lon now for use with xesmf
            if chunks != "auto":
                chunks = {"lon": chunks[varnames["xh"]], "lat": chunks[varnames["yh"]]}

            tgrid = tgrid.chunk(chunks)
            tgrid.lon.attrs["units"] = "degrees_east"
            tgrid.lon.attrs["_FillValue"] = 1e20
            tgrid.lat.attrs["units"] = "degrees_north"
            tgrid.to_netcdf(
                self.mom_input_dir / "topog_raw.nc", mode="w", engine="netcdf4"
            )
            tgrid.close()

            ## Replace subprocess run with regular regridder
            print(
                "Starting to regrid bathymetry. If this process hangs you might be better off calling ESMF directly from a terminal with appropriate computational resources using \n\n mpirun ESMF_Regrid -s bathy_original.nc -d topog_raw.nc -m bilinear --src_var elevation --dst_var elevation --netcdf4 --src_regional --dst_regional\n\nThis is better for larger domains.\n\n"
            )

            # If we have a domain large enough for chunks, we'll run regridder with parallel=True
            parallel = True
            if len(tgrid.chunks) != 2:
                parallel = False
            regridder = xe.Regridder(bathyout, tgrid, "bilinear", parallel=parallel)

            topog = regridder(bathyout)
            topog.to_netcdf(
                self.mom_input_dir / "topog_raw.nc", mode="w", engine="netcdf4"
            )

        ## reopen topography to modify
        print("Reading in regridded bathymetry to fix up metadata...", end="")
        topog = xr.open_dataset(self.mom_input_dir / "topog_raw.nc", engine="netcdf4")

        ## Ensure correct encoding
        topog = xr.Dataset({"depth": (["ny", "nx"], topog["elevation"].values)})
        topog.attrs["depth"] = "meters"
        topog.attrs["standard_name"] = "topographic depth at T-cell centers"
        topog.attrs["coordinates"] = "zi"

        topog.expand_dims("tiles", 0)

        if not positivedown:
            ## Ensure that coordinate is positive down!
            topog["depth"] *= -1

        ## REMOVE INLAND LAKES

        min_depth = self.vgrid.zi[minimum_layers]

        ocean_mask = topog.copy(deep=True).depth.where(topog.depth <= min_depth, 1)
        land_mask = np.abs(ocean_mask - 1)
        changed = True  ## keeps track of whether solution has converged or not

        forward = True  ## only useful for iterating through diagonal channel removal. Means iteration goes SW -> NE

        while changed == True:
            ## First fill in all lakes. This uses a scipy function where it fills holes made of 0's within a field of 1's
            land_mask[:, :] = binary_fill_holes(land_mask.data)
            ## Get the ocean mask instead of land- easier to remove channels this way
            ocean_mask = np.abs(land_mask - 1)

            ## Now fill in all one-cell-wide channels
            newmask = xr.where(
                ocean_mask * (land_mask.shift(nx=1) + land_mask.shift(nx=-1)) == 2, 1, 0
            )
            newmask += xr.where(
                ocean_mask * (land_mask.shift(ny=1) + land_mask.shift(ny=-1)) == 2, 1, 0
            )

            if fill_channels == True:
                ## fill in all one-cell-wide horizontal channels
                newmask = xr.where(
                    ocean_mask * (land_mask.shift(nx=1) + land_mask.shift(nx=-1)) == 2,
                    1,
                    0,
                )
                newmask += xr.where(
                    ocean_mask * (land_mask.shift(ny=1) + land_mask.shift(ny=-1)) == 2,
                    1,
                    0,
                )
                ## Diagonal channels
                if forward == True:
                    ## horizontal channels
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(nx=1))
                        * (
                            land_mask.shift({"nx": 1, "ny": 1})
                            + land_mask.shift({"ny": -1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## up right & below
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(nx=1))
                        * (
                            land_mask.shift({"nx": 1, "ny": -1})
                            + land_mask.shift({"ny": 1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## down right & above
                    ## Vertical channels
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(ny=1))
                        * (
                            land_mask.shift({"nx": 1, "ny": 1})
                            + land_mask.shift({"nx": -1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## up right & left
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(ny=1))
                        * (
                            land_mask.shift({"nx": -1, "ny": 1})
                            + land_mask.shift({"nx": 1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## up left & right

                    forward = False

                if forward == False:
                    ## Horizontal channels
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(nx=-1))
                        * (
                            land_mask.shift({"nx": -1, "ny": 1})
                            + land_mask.shift({"ny": -1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## up left & below
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(nx=-1))
                        * (
                            land_mask.shift({"nx": -1, "ny": -1})
                            + land_mask.shift({"ny": 1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## down left & above
                    ## Vertical channels
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(ny=-1))
                        * (
                            land_mask.shift({"nx": 1, "ny": -1})
                            + land_mask.shift({"nx": -1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## down right & left
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(ny=-1))
                        * (
                            land_mask.shift({"nx": -1, "ny": -1})
                            + land_mask.shift({"nx": 1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## down left & right

                    forward = True

            newmask = xr.where(newmask > 0, 1, 0)
            changed = np.max(newmask) == 1
            land_mask += newmask

        ocean_mask = np.abs(land_mask - 1)

        topog["depth"] *= ocean_mask

        topog["depth"] = topog["depth"].where(topog["depth"] != 0, np.nan)

        topog.expand_dims({"ntiles": 1}).to_netcdf(
            self.mom_input_dir / "topog_deseas.nc",
            mode="w",
            encoding={"depth": {"_FillValue": None}},
        )

        (self.mom_input_dir / "topog_deseas.nc").rename(self.mom_input_dir / "topog.nc")
        print("done.")
        self.topog = topog

    def FRE_tools(self, layout):
        """
        Just a wrapper for FRE Tools check_mask, make_solo_mosaic and make_quick_mosaic. User provides processor layout tuple of processing units.
        """

        if not (self.mom_input_dir / "topog.nc").exists():
            print("No topography file! Need to run make_bathymetry first")
            return

        for p in self.mom_input_dir.glob("mask_table*"):
            p.unlink()

        print(
            "MAKE SOLO MOSAIC",
            subprocess.run(
                self.toolpath
                + "make_solo_mosaic/make_solo_mosaic --num_tiles 1 --dir . --mosaic_name ocean_mosaic --tile_file hgrid.nc",
                shell=True,
                cwd=self.mom_input_dir,
            ),
            sep="\n\n",
        )

        print(
            "QUICK MOSAIC",
            subprocess.run(
                self.toolpath
                + "make_quick_mosaic/make_quick_mosaic --input_mosaic ocean_mosaic.nc --mosaic_name grid_spec --ocean_topog topog.nc",
                shell=True,
                cwd=self.mom_input_dir,
            ),
            sep="\n\n",
        )

        print(
            "CHECK MASK",
            subprocess.run(
                self.toolpath
                + f"check_mask/check_mask --grid_file ocean_mosaic.nc --ocean_topog topog.nc --layout {layout[0]},{layout[1]} --halo 4",
                shell=True,
                cwd=self.mom_input_dir,
            ),
        )
        self.layout = layout

    def setup_run_directory(self,rmom6_path,surface_forcing = "era5",using_payu = False):
        """Sets up the run directory for MOM6. Creates a symbolic link
        to the input directory, and creates a payu configuration file
        if payu is being used.

        Args:
            rmom6_path [str]:   The path to where the regional_mom6 package is installed. This is needed to find the default run directory that this function builds on
            surface_forcing (Optional[str]): The surface forcing to use. One of ``era5`` or ``jra``.
            using_payu (Optional[bool]): Whether or not to use payu to run the model. If True, a payu configuration file will be created.

        """

        ## Copy the default directory to the run directory

        subprocess.run(f"cp {str(Path(rmom6_path) / 'regional_mom6' / 'default_rundir' / surface_forcing)}_surface/* {str(self.mom_run_dir)}",shell=True)
        ## Make symlinks between run and input directories
        if not (self.mom_run_dir / "inputdir").exists():
            os.symlink(str(self.mom_input_dir), str(self.mom_run_dir / "inputdir"))
        if not (self.mom_input_dir / "rundir").exists():
            os.symlink(str(self.mom_run_dir), str(self.mom_input_dir / "rundir"))


        ## Get mask table information
        ncpus = 10
        mask_table = None
        for i in os.listdir(f"{self.mom_input_dir}"):
            if "mask_table" in i:
                mask_table = i
                a = mask_table.split(".")[1]
                b = mask_table.split(".")[2].split("x")
                ncpus = int(b[0]) * int(b[1]) - int(a)
        if mask_table == None:
            print("No mask table found! Run FRE_tools first. Terminating")
            raise ValueError

        print("Number of CPUs required: ", ncpus)

        ## Modify MOM_input
        inputfile = open(f"{self.mom_run_dir}/MOM_input",'r')
        lines = inputfile.readlines()
        inputfile.close()
        for i in range(len(lines)):
            if "MASKTABLE" in lines[i]:
                if mask_table != None:
                    lines[i] = f'MASKTABLE = "{mask_table}"\n'
                else:
                    lines[i] = "# MASKTABLE = no mask table"
            if "LAYOUT =" in lines[i] and "IO" not in lines[i]:
                lines[i] = f'LAYOUT = {self.layout[1]},{self.layout[0]}\n'

            if "NIGLOBAL" in lines[i]: 
                # lines[i] = f"NIGLOBAL = {str(x_indices_centre[1] - x_indices_centre[0])}\n"
                lines[i] = f"NIGLOBAL = {self.hgrid.nx.shape[0]//2}\n"

            if "NJGLOBAL" in lines[i]:
                # lines[i] = f"NJGLOBAL = {str(y_indices_centre[1] - y_indices_centre[0])}\n"
                lines[i] = f"NJGLOBAL = {self.hgrid.ny.shape[0]//2}\n"

                
        inputfile = open(f"{self.mom_run_dir}/MOM_input",'w')

        inputfile.writelines(lines)
        inputfile.close()

        ## Modify SIS_input
        inputfile = open(f"{self.mom_run_dir}/SIS_input",'r')
        lines = inputfile.readlines()
        inputfile.close()
        for i in range(len(lines)):
            if "MASKTABLE" in lines[i]:
                lines[i] = f'MASKTABLE = "{mask_table}"\n'
            if "NIGLOBAL" in lines[i]:
                # lines[i] = f"NIGLOBAL = {str(x_indices_centre[1] - x_indices_centre[0])}\n"
                lines[i] = f"NIGLOBAL = {self.hgrid.nx.shape[0]//2}\n"
            if "LAYOUT =" in lines[i] and "IO" not in lines[i]:
                lines[i] = f'LAYOUT = {self.layout[1]},{self.layout[0]}\n'
            if "NJGLOBAL" in lines[i]:
                # lines[i] = f"NJGLOBAL = {str(y_indices_centre[1] - y_indices_centre[0])}\n"
                lines[i] = f"NJGLOBAL = {self.hgrid.ny.shape[0]//2}\n"
                
        inputfile = open(f"{self.mom_run_dir}/SIS_input",'w')
        inputfile.writelines(lines)
        inputfile.close()


        ## If using payu to run the model, create a payu configuration file
        if not using_payu:
            os.remove(f"{self.mom_run_dir}/config.yaml")

        else:
        ## Modify config.yaml 
            inputfile = open(f"{self.mom_run_dir}/config.yaml",'r')
            lines = inputfile.readlines()
            inputfile.close()
            for i in range(len(lines)):
                if "ncpus" in lines[i]:
                    lines[i] = f'ncpus: {str(ncpus)}\n'
                    
                if "input:" in lines[i]:
                    lines[i + 1] = f"    - {self.mom_input_dir}\n"

            inputfile = open(f"{self.mom_run_dir}/config.yaml",'w')
            inputfile.writelines(lines)
            inputfile.close()


            # Modify input.nml 
            inputfile = open(f"{self.mom_run_dir}/input.nml",'r')
            lines = inputfile.readlines()
            inputfile.close()
            for i in range(len(lines)):
                if "current_date" in lines[i]:
                    tmp = self.daterange[0]
                    lines[i] = f"{lines[i].split(' = ')[0]} = {int(tmp.year)},{int(tmp.month)},{int(tmp.day)},0,0,0,\n"

            
            inputfile = open(f"{self.mom_run_dir}/input.nml",'w')
            inputfile.writelines(lines)
            inputfile.close()

    def setup_era5(self,era5_path):
        """
        Sets up the ERA5 forcing files for your experiment. This assumes that you'd downloaded all of the ERA5 data in your daterange.
        You'll need the following fields:
        2t, 10u, 10v, sp, 2d


        Args:
            era5_path (str): Path to the ERA5 forcing files

        """


        ## Firstly just open all raw data
        rawdata = {}
        for fname , vname in zip(["2t","10u","10v","sp","2d"] , ["t2m","u10","v10","sp","d2m"]):

            ## Cut out this variable to our domain size
            rawdata[fname] = nicer_slicer(
                xr.open_mfdataset(f"{era5_path}/{fname}/{self.daterange[0].year}/{fname}*",decode_times = False,chunks = {"longitude":100,"latitude":100}),
                self.xextent,
                "longitude"
            ).sel(
                latitude = slice(self.yextent[1],self.yextent[0]) ## This is because ERA5 has latitude in decreasing order (??)
            )

            ## Now fix up the latitude and time dimensions

            rawdata[fname] = rawdata[fname].isel(
                latitude = slice(None,None,-1) ## Flip latitude        
                ).assign_coords(
                time = np.arange(0,rawdata[fname].time.shape[0],dtype=float) ## Set the zero date of forcing to start of run
                )
            

            rawdata[fname].time.attrs = {"calendar":"julian","units":f"hours since {self.daterange[0].strftime('%Y-%m-%d %H:%M:%S')}"} ## Fix up calendar to match

            if fname == "2d":
                ## Calculate specific humidity from dewpoint temperature 
                q = xr.Dataset(
                    data_vars= {
                        "q": (0.622 / rawdata["sp"]["sp"]) * (10**(8.07131 - 1730.63 / (233.426 + rawdata["2d"]["d2m"] - 273.15) )) * 101325 / 760
                        }

                )
                q.q.attrs = {"long_name":"Specific Humidity","units": "kg/kg"}
                q.to_netcdf(f"{self.mom_input_dir}/forcing/q_ERA5",unlimited_dims = "time",encoding = {"q":{"dtype":"double"}})
            else:
                rawdata[fname].to_netcdf(f"{self.mom_input_dir}/forcing/{fname}_ERA5",unlimited_dims = "time",encoding = {vname:{"dtype":"double"}})




class segment:
    """Class to turn raw boundary segment data into MOM6 boundary
    segments.

    Boundary segments should only contain the necessary data for that
    segment. No horizontal chunking is done here, so big fat segments
    will process slowly.

    Data should be at daily temporal resolution, iterating upwards
    from the provided startdate. Function ignores the time metadata
    and puts it on Julian calendar.


    Args:
        hgrid (xarray.Dataset): The horizontal grid used for domain
        infile (Union[str, Path]): Path to the raw, unprocessed boundary segment
        outfolder (Union[str, Path]): Path to folder where the model inputs will be stored
        varnames (Dict[str, str]): Mapping between the
            variable/dimension names and standard naming convension of
            this pipeline, e.g. ``{"xq":"longitude, "yh":"latitude",
            "salt":"salinity...}``. Key "tracers" points to nested
            dictionary of tracers to include in boundary
        seg_name (str): Name of the segment. Something like ``segment_001``
        orientation (str): Cardinal direction (lowercase) of the boundary segment
        startdate (str): The starting date to use in the segment calendar
        gridtype (Optional[str]): Arakawa staggering of input grid, one of ``A``, ``B`` or ``C``
        vcoord_type (Optional[str]): Vertical coordinate, either
            interfacial ``height`` or layer ``thickness``
        time_units (str): The units used by raw forcing file,
            e.g. ``hours``, ``days`` (default)

    """

    def __init__(
        self,
        hgrid,
        infile,
        outfolder,
        varnames,
        seg_name,
        orientation,
        startdate,
        gridtype="A",
        vcoord_type="height",
        time_units="days",
    ):
        ## Store coordinate names
        if gridtype == "A":
            self.x = varnames["x"]
            self.y = varnames["y"]

        elif gridtype in ("B", "C"):
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
        self.orientation = orientation.lower()  ## might not be needed? NSEW
        self.grid = gridtype
        self.hgrid = hgrid
        self.seg_name = seg_name
        self.vcoord_type = vcoord_type

    def brushcut(self, ryf=False):
        ### Implement brushcutter scheme on single segment ###
        rawseg = xr.open_dataset(self.infile, decode_times=False, engine="netcdf4")

        ## Depending on the orientation of the segment, cut out the right bit of the hgrid
        ## and define which coordinate is along or into the segment
        if self.orientation == "north":
            hgrid_seg = self.hgrid.isel(nyp=[-1])
            perpendicular = "ny"
            parallel = "nx"

        if self.orientation == "south":
            hgrid_seg = self.hgrid.isel(nyp=[0])
            perpendicular = "ny"
            parallel = "nx"

        if self.orientation == "east":
            hgrid_seg = self.hgrid.isel(nxp=[-1])
            perpendicular = "nx"
            parallel = "ny"

        if self.orientation == "west":
            hgrid_seg = self.hgrid.isel(nxp=[0])
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
        ).set_coords(["lat", "lon"])

        if self.grid == "A":
            rawseg = rawseg.rename({self.x: "lon", self.y: "lat"})
            ## In this case velocities and tracers all on same points
            regridder = xe.Regridder(
                rawseg[self.u],
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_velocity_weights_{self.orientation}.nc",
            )

            segment_out = xr.merge(
                [
                    regridder(
                        rawseg[
                            [self.u, self.v, self.eta]
                            + [self.tracers[i] for i in self.tracers]
                        ]
                    )
                ]
            )

        if self.grid == "B":
            ## All tracers on one grid, all velocities on another
            regridder_velocity = xe.Regridder(
                rawseg[self.u].rename({self.xq: "lon", self.yq: "lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_velocity_weights_{self.orientation}.nc",
            )

            regridder_tracer = xe.Regridder(
                rawseg[self.tracers["salt"]].rename({self.xh: "lon", self.yh: "lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_tracer_weights_{self.orientation}.nc",
            )

            segment_out = xr.merge(
                [
                    regridder_velocity(
                        rawseg[[self.u, self.v]].rename(
                            {self.xq: "lon", self.yq: "lat"}
                        )
                    ),
                    regridder_tracer(
                        rawseg[
                            [self.eta] + [self.tracers[i] for i in self.tracers]
                        ].rename({self.xh: "lon", self.yh: "lat"})
                    ),
                ]
            )

        if self.grid == "C":
            ## All tracers on one grid, all velocities on another
            regridder_uvelocity = xe.Regridder(
                rawseg[self.u].rename({self.xq: "lon", self.yh: "lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_uvelocity_weights_{self.orientation}.nc",
            )

            regridder_vvelocity = xe.Regridder(
                rawseg[self.v].rename({self.xh: "lon", self.yq: "lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_vvelocity_weights_{self.orientation}.nc",
            )

            regridder_tracer = xe.Regridder(
                rawseg[self.tracers["salt"]].rename({self.xh: "lon", self.yh: "lat"}),
                interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_tracer_weights_{self.orientation}.nc",
            )

            segment_out = xr.merge(
                [
                    regridder_vvelocity(rawseg[[self.v]]),
                    regridder_uvelocity(rawseg[[self.u]]),
                    regridder_tracer(
                        rawseg[[self.eta] + [self.tracers[i] for i in self.tracers]]
                    ),
                ]
            )

        ## segment out now contains our interpolated boundary.
        ## Now, we need to fix up all the metadata and save

        del segment_out["lon"]
        del segment_out["lat"]
        ## Convert temperatures to celsius # use pint
        if (
            np.min(segment_out[self.tracers["temp"]].isel({self.time: 0, self.z: 0}))
            > 100
        ):
            segment_out[self.tracers["temp"]] -= 273.15

        # fill in NaNs
        segment_out = (
            segment_out.ffill(self.z)
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
            0,  #! Indexing everything from start of experiment = simple but maybe counterintutive?
            segment_out[self.time].shape[
                0
            ],  ## Time is indexed from start date of window
            dtype=float,
        )

        segment_out = segment_out.assign_coords({"time": time})

        segment_out.time.attrs = {
            "calendar": "julian",
            "units": f"{self.time_units} since {self.startdate}",
            "modulo": " ",
        }
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
            dz = xr.concat([dz, dz[-1]], dim=self.z)

        else:
            dz = segment_out[self.z]
            dz.name = "dz"
        del segment_out[self.z]

        # Here, keep in mind that 'var' keeps track of the mom6 variable names we want, and self.tracers[var] will return the name of the variable from the original data

        allfields = {
            **self.tracers,
            "u": self.u,
            "v": self.v,
        }  ## Combine all fields into one flattened dictionary to iterate over as we fix metadata

        for (
            var
        ) in (
            allfields
        ):  ## Replace with more generic list of tracer variables that might be included?
            v = f"{var}_{self.seg_name}"
            ## Rename each variable in dataset
            segment_out = segment_out.rename({allfields[var]: v})

            ## Rename vertical coordinate for this variable
            segment_out[f"{var}_{self.seg_name}"] = segment_out[
                f"{var}_{self.seg_name}"
            ].rename({self.z: f"nz_{self.seg_name}_{var}"})

            ## Replace the old depth coordinates with incremental integers
            segment_out[f"nz_{self.seg_name}_{var}"] = np.arange(
                segment_out[f"nz_{self.seg_name}_{var}"].size
            )

            ## Re-add the secondary dimension (even though it represents one value..)
            segment_out[v] = segment_out[v].expand_dims(
                f"{perpendicular}_{self.seg_name}", axis=axis2
            )

            ## Add the layer thicknesses
            segment_out[f"dz_{v}"] = (
                ["time", f"nz_{v}", f"ny_{self.seg_name}", f"nx_{self.seg_name}"],
                da.broadcast_to(
                    dz.data[None, :, None, None],
                    segment_out[v].shape,
                    chunks=(
                        1,
                        None,
                        None,
                        None,
                    ),  ## Chunk in each time, and every 5 vertical layers
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
            encoding_dict[f"nz_{self.seg_name}_{var}"] = {"dtype": "int32"}

        ## Treat eta separately since it has no vertical coordinate. Do the same things as for the surface variables above
        segment_out = segment_out.rename({self.eta: f"eta_{self.seg_name}"})
        encoding_dict[f"eta_{self.seg_name}"] = {
            "_FillValue": netCDF4.default_fillvals["f8"],
        }
        segment_out[f"eta_{self.seg_name}"] = segment_out[
            f"eta_{self.seg_name}"
        ].expand_dims(f"{perpendicular}_{self.seg_name}", axis=axis2 - 1)

        # Overwrite the actual lat/lon values in the dimensions, replace with incrementing integers
        segment_out[f"{parallel}_{self.seg_name}"] = np.arange(
            segment_out[f"{parallel}_{self.seg_name}"].size
        )
        segment_out[f"{perpendicular}_{self.seg_name}"] = [0]

        # Store actual lat/lon values here as variables rather than coordinates
        segment_out[f"lon_{self.seg_name}"] = (
            [f"ny_{self.seg_name}", f"nx_{self.seg_name}"],
            hgrid_seg.x.data,
        )
        segment_out[f"lat_{self.seg_name}"] = (
            [f"ny_{self.seg_name}", f"nx_{self.seg_name}"],
            hgrid_seg.y.data,
        )

        with ProgressBar():
            segment_out["time"] = segment_out["time"].assign_attrs(
                {"modulo": " "}
            )  ## Add modulo attribute for MOM6 to treat as repeat forcing
            segment_out.load().to_netcdf(
                self.outfolder / f"forcing/forcing_obc_{self.seg_name}.nc",
                encoding=encoding_dict,
                unlimited_dims="time",
            )

        return segment_out, encoding_dict
