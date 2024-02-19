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
import f90nml
import datetime as dt
import warnings
import shutil
import os
from .utils import vecdot

warnings.filterwarnings("ignore")

__all__ = [
    "nicer_slicer",
    "motu_requests",
    "dz_hyperbolictan",
    "angle_between",
    "latlon_to_cartesian",
    "quadrilateral_area",
    "quadrilateral_areas",
    "rectangular_hgrid",
    "experiment",
    "segment",
]

# Borrowed functions for Tides


def ap2ep(uc, vc):
    """
    Convert complex tidal ``u`` and ``v`` to tidal ellipse.

    Adapted from ap2ep.m for matlab
    Original copyright notice:

    %Authorship Copyright:
    %
    %    The author retains the copyright of this program, while  you are welcome
    % to use and distribute it as long as you credit the author properly and respect
    % the program name itself. Particularly, you are expected to retain the original
    % author's name in this original version or any of its modified version that
    % you might make. You are also expected not to essentially change the name of
    % the programs except for adding possible extension for your own version you
    % might create, e.g. ap2ep_xx is acceptable.  Any suggestions are welcome and
    % enjoy my program(s)!
    %
    %
    %Author Info:
    %_______________________________________________________________________
    %  Zhigang Xu, Ph.D.
    %  (pronounced as Tsi Gahng Hsu)
    %  Research Scientist
    %  Coastal Circulation
    %  Bedford Institute of Oceanography
    %  1 Challenge Dr.
    %  P.O. Box 1006                    Phone  (902) 426-2307 (o)
    %  Dartmouth, Nova Scotia           Fax    (902) 426-7827
    %  CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca
    %_______________________________________________________________________
    %
    % Release Date: Nov. 2000, Revised on May. 2002 to adopt Foreman's northern semi
    % major axis convention.

    Args:
        uc: complex tidal u velocity
        vc: complex tidal v velocity

    Return:
        (semi-major axis, eccentricity, inclination [radians], phase [radians])
    """
    wp = (uc + 1j * vc) / 2.0
    wm = np.conj(uc - 1j * vc) / 2.0

    Wp = np.abs(wp)
    Wm = np.abs(wm)
    THETAp = np.angle(wp)
    THETAm = np.angle(wm)

    SEMA = Wp + Wm
    SEMI = Wp - Wm
    ECC = SEMI / SEMA
    PHA = (THETAm - THETAp) / 2.0
    INC = (THETAm + THETAp) / 2.0

    return SEMA, ECC, INC, PHA


def ep2ap(SEMA, ECC, INC, PHA):
    """
    Convert tidal ellipse to real u and v amplitude and phase.

    Adapted from ep2ap.m for matlab.
    Original copyright notice:

    %Authorship Copyright:
    %
    %    The author of this program retains the copyright of this program, while
    % you are welcome to use and distribute this program as long as you credit
    % the author properly and respect the program name itself. Particularly,
    % you are expected to retain the original author's name in this original
    % version of the program or any of its modified version that you might make.
    % You are also expected not to essentially change the name of the programs
    % except for adding possible extension for your own version you might create,
    % e.g. app2ep_xx is acceptable.  Any suggestions are welcome and enjoy my
    % program(s)!
    %
    %
    %Author Info:
    %_______________________________________________________________________
    %  Zhigang Xu, Ph.D.
    %  (pronounced as Tsi Gahng Hsu)
    %  Research Scientist
    %  Coastal Circulation
    %  Bedford Institute of Oceanography
    %  1 Challenge Dr.
    %  P.O. Box 1006                    Phone  (902) 426-2307 (o)
    %  Dartmouth, Nova Scotia           Fax    (902) 426-7827
    %  CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca
    %_______________________________________________________________________
    %
    %Release Date: Nov. 2000

    Args:
        SEMA: semi-major axis
        ECC: eccentricity
        INC: inclination [radians]
        PHA: phase [radians]

    Return:
        (u amplitude, u phase [radians], v amplitude, v phase [radians])
    """
    Wp = (1 + ECC) / 2.0 * SEMA
    Wm = (1 - ECC) / 2.0 * SEMA
    THETAp = INC - PHA
    THETAm = INC + PHA

    wp = Wp * np.exp(1j * THETAp)
    wm = Wm * np.exp(1j * THETAm)

    cu = wp + np.conj(wm)
    cv = -1j * (wp - np.conj(wm))

    ua = np.abs(cu)
    va = np.abs(cv)
    up = -np.angle(cu)
    vp = -np.angle(cv)

    return ua, va, up, vp


## Auxiliary functions


def nicer_slicer(data, xextent, xcoords, buffer=2):
    """
    Slice longitudes, handling periodicity and 'seams' where the
    data wraps around (commonly either in domain [-180, 180] or in [-270, 90]).

    The algorithm works in five steps:

    - Determine whether we need to add or subtract 360 to get the
      middle of the ``xextent`` to lie within ``data``'s longitude range
      (hereby ``oldx``).

    - Shift the dataset so that its midpoint matches the midpoint of
      ``xextent`` (up to a muptiple of 360). Now, the modified ``oldx``
      does not increase monotonically from West to East  since the
      'seam' has moved.

    - Fix ``oldx`` to make it monotonically increasing again. This uses
      the information we have about the way the dataset was
      shifted/rolled.

    - Slice the ``data`` index-wise. We know that ``|xextent| / 360``
      multiplied by the number of discrete longitude points will give
      the total width of our slice, and we've already set the midpoint
      to be the middle of the target domain. Here we add a ``buffer``
      region on either side if we need it for interpolation.

    - Finally re-add the right multiple of 360 so the whole domain matches
      the target.

    Args:
        data (xarray.Dataset): The global data you want to slice in longitude.
        xextent (Tuple[float, float]): The target longitudes (in degrees) we would
            like to slice to. Must be in increasing order.
        xcoords (Union[str, List[str]): The name or list of names of the longitude
            dimension in ``data``.

    Return:
        xarray.Dataset: The sliced ``data``.
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
    """
    Generate MOTU data request for each specified boundary, as well as for
    the initial condition. By default pulls the GLORYS reanalysis dataset.

    Args:
        xextent (List[float]): Extreme values of longitude coordinates for
            rectangular domain (in degrees).
        yextent (List[float]): Extreme values of latitude coordinates for
            rectangular domain (in degrees).
        daterange (Tuple[str]): Start and end dates of boundary forcing window.
            Format: ``%Y-%m-%d %H:%M:%S``.
        outfolder (str): Directory the files are downloaded.
        usr (str): MOTU authentication username
        pwd (str): MOTU authentication password
        segs (List[str]): List of the cardinal directions for your boundary forcing
        url (Optional[str]): MOTU server for the request. Defaults to the CMEMS,
            i.e. ``"https://my.cmems-du.eu/motu-web/Motu"``.
        serviceid (Optional[str]): Service containing the desired dataset. Default:
            ``"GLOBAL_MULTIYEAR_PHY_001_030-TDS"``.
        productid (Optional[str]): Data product within the chosen service. Default:
            ``"cmems_mod_glo_phy_my_0.083_P1D-m"``.

    Return:
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


def dz_hyperbolictan(npoints, ratio, target_depth, min_dz=0.0001, tolerance=1):
    """Generate a hyperbolic tangent thickness profile for the
    experiment.  Iterates to find the mininum depth value which gives
    the target depth within some tolerance.
    Thickness of layers monatonically increases (or decreases if ratio is negative) from the surface to the bottom of the domain.
    Set the ratio to 1 for a uniformly spaced grid.

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

    return dz_hyperbolictan(npoints, ratio, target_depth, min_dz * err_ratio)


def angle_between(v1, v2, v3):
    """
    Return the angle ``v2``-``v1``-``v3`` (in radians), where
    ``v1``, ``v2``, ``v3`` are 3-vectors. That is, the angle that
    is formed between vectors ``v2 - v1`` and vector ``v3 - v1``.
    """

    v1xv2 = np.cross(v1, v2)
    v1xv3 = np.cross(v1, v3)

    cosangle = vecdot(v1xv2, v1xv3) / np.sqrt(
        vecdot(v1xv2, v1xv2) * vecdot(v1xv3, v1xv3)
    )

    return np.arccos(cosangle)


def quadrilateral_area(v1, v2, v3, v4):
    """
    Return the area of a spherical quadrilateral on the unit sphere that
    has vertices on the 3-vectors ``v1``, ``v2``, ``v3``, ``v4``
    (counter-clockwise orientation is implied). The area is computed via
    the excess of the sum of the spherical angles of the quadrilateral from 2π.
    """

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
    """
    Convert latitude-longitude (in degrees) to Cartesian coordinates on
    a sphere of radius ``R``. By default ``R = 1``.

    Args:
        lat (float): Latitude (in degrees).
        lon (float): Longitude (in degrees).
        R (float): The radius of the sphere; default: 1.

    Return:
        tuple: Tuple with the Cartesian coordinates ``x, y, z``
    """

    x = R * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = R * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = R * np.sin(np.deg2rad(lat))

    return x, y, z


def quadrilateral_areas(lat, lon, R=1):
    """
    Return the area of spherical quadrilaterals on a sphere of radius ``R``.
    By default, ``R = 1``. The quadrilaterals are formed by constant latitude and longitude
    lines on the ``lat``-``lon`` grid provided.

    Args:
        lat (numpy.array): Array of latitude points (in degrees).
        lon (numpy.array): Array of longitude points (in degrees).
        R (float): The radius of the sphere; default: 1.

    Return:
        numpy.array: Array with the areas of the quadrilaterals defined by the ``lat``-``lon`` grid
        provided. If the provided ``lat``, ``lon`` arrays are of dimension *m* :math:`\\times` *n*
        then return areas array is of dimension (*m-1*) :math:`\\times` (*n-1*).
    """

    coords = np.dstack(latlon_to_cartesian(lat, lon, R))

    return quadrilateral_area(
        coords[:-1, :-1, :], coords[:-1, 1:, :], coords[1:, 1:, :], coords[1:, :-1, :]
    )


def rectangular_hgrid(λ, φ):
    """
    Construct a horizontal grid with all the metadata given an array of
    longitudes (``λ``) and latitudes (``φ``) on the supergrid. Here, 'supergrid'
    refers to both cell edges and centres, meaning that there are twice as many
    points along each axis than for any individual field.

    Caution:
        It is assumed the grid's boundaries are lines of constant latitude and
        longitude. Rotated grids need to be handled differently.
        It is also assumed here that the longitude array values are uniformly spaced.

        Ensure both ``λ`` and ``φ`` are monotonically increasing.

    Args:
        λ (numpy.array): All longitude points on the supergrid. Must be uniformly spaced!
        φ (numpy.array): All latitude points on the supergrid.

    Return:
        xarray.Dataset: An FMS-compatible ``hgrid`` that includes all required attributes.
    """

    R = 6371e3  # mean radius of the Earth; https://en.wikipedia.org/wiki/Earth_radius

    # compute longitude spacing and ensure that longitudes are uniformly spaced
    dλ = λ[1] - λ[0]

    assert (
        np.max(np.diff(λ) - dλ) < 1e-14
    ), "provided array of longitudes must be uniformly spaced"

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

    Everything about your regional experiment.

    Methods in this class will generate the various input files needed
    to generate a MOM6 experiment forced with open boundary conditions
    (OBCs). The code is agnostic to your choice of boundary forcing,
    topography and surface forcing - you need to tell it what your variables
    are all called via mapping dictionaries from MOM6 variable/coordinate
    name to the name in the input dataset.

    This can be used to generate the grids for a new experiment, or to read in
    an existing one by setting read-existing to True. In either case,
    the ``xextent``, ``yextent``, ``daterange``, and ``resolution`` must be specified.

    Args:
        xextent (Tuple[float]): Extent of the region in longitude in degrees.
        yextent (Tuple[float]): Extent of the region in latitude in degrees.
        daterange (Tuple[str]): Start and end dates of the boundary forcing window.
        resolution (float): Lateral resolution of the domain, in degrees.
        vlayers (int): Number of vertical layers.
        dz_ratio (float): Ratio of largest to smallest layer thickness, used
            as input in :func:`~dz_hyperbolictan`.
        depth (float): Depth of the domain.
        mom_run_dir (str): Path of the MOM6 control directory.
        mom_input_dir (str): Path of the MOM6 input directory, to receive the forcing files.
        toolpath (str): Path of GFDL's FRE tools (https://github.com/NOAA-GFDL/FRE-NCtools) binaries.
        gridtype (Optional[str]): Type of horizontal grid to generate.
            Currently, only ``even_spacing`` is supported.
        ryf (Optional[bool]): When ``True`` the experiment runs with 'repeat-year forcing'.
            When ``False`` (default) then inter-annual forcing is used.
        read_existing_grids (Optional[Bool]): When ``True``, instead of generating the grids,
            reads the grids and ocean mask from the 'inputdir' and 'rundir'. Useful for modifying or
            troubleshooting experiments. Default: ``False``.
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
        ryf=False,
        read_existing_grids=False,
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
        self.resolution = resolution
        self.vlayers = vlayers
        self.dz_ratio = dz_ratio
        self.depth = depth
        self.toolpath = Path(toolpath)
        self.ocean_mask = None
        if read_existing_grids:
            try:
                self.hgrid = xr.open_dataset(self.mom_input_dir / "hgrid.nc")
                self.vgrid = xr.open_dataset(self.mom_input_dir / "vcoord.nc")
            except:
                print(
                    "Error in reading in existing grids. Make sure you've got files called `hgrid.nc` and `vcoord.nc` in {self.mom_input_dir}"
                )
                raise ValueError
        else:
            self.hgrid = self._make_hgrid(gridtype)
            self.vgrid = self._make_vgrid()
        self.gridtype = gridtype
        self.ryf = ryf
        # create additional directories and links
        (self.mom_input_dir / "weights").mkdir(exist_ok=True)
        (self.mom_input_dir / "forcing").mkdir(exist_ok=True)

        run_inputdir = self.mom_run_dir / "inputdir"
        if not run_inputdir.exists():
            run_inputdir.symlink_to(self.mom_input_dir.resolve())
        input_rundir = self.mom_input_dir / "rundir"
        if not input_rundir.exists():
            input_rundir.symlink_to(self.mom_run_dir.resolve())

    def __getattr__(self, name):
        available_methods = [
            method for method in dir(self) if not method.startswith("__")
        ]
        error_message = (
            f"'{name}' method not found. Available methods are: {available_methods}"
        )
        raise AttributeError(error_message)

    def _make_hgrid(self, gridtype):
        """
        Set up hgrid based on users specification of domain. Default behaviour
        leaves latitude and longitude evenly spaced.
        This is very simple but suitable for small domains.

        Note:
            The intention is for the horizontal grid (``hgrid``) generation to be very flexible.
            For now, there is only one implemented horizontal grid included in the package,
            but you can customise it by simply overwriting the ``hgrid.nc`` file in your ``rundir``
            after initialising an ``experiment``. To conserve the metadata, it might be easiest
            to read the file in, then modify the fields before re-saving.
        """

        if gridtype == "even_spacing":
            # longitudes are evenly spaced based on resolution and bounds
            nx = int((self.xextent[1] - self.xextent[0]) / (self.resolution / 2))
            if nx % 2 != 1:
                nx += 1

            x = np.linspace(
                self.xextent[0], self.xextent[1], nx
            )  # longitudes in degrees

            # Latitudes evenly spaced by dx * cos(central_latitude)
            central_latitude = np.mean(self.yextent)  # degrees
            latitudinal_resolution = self.resolution * np.cos(
                np.deg2rad(central_latitude)
            )

            ny = (
                int((self.yextent[1] - self.yextent[0]) / (latitudinal_resolution / 2))
                + 1
            )
            if ny % 2 != 1:
                ny += 1

            y = np.linspace(
                self.yextent[0], self.yextent[1], ny
            )  # latitudes in degrees

            hgrid = rectangular_hgrid(x, y)
            hgrid.to_netcdf(self.mom_input_dir / "hgrid.nc")

            return hgrid

    def _make_vgrid(self):
        """
        Generate a vertical grid based on the number of layers and vertical ratio
        specified at the class level.
        """

        thickness = dz_hyperbolictan(self.vlayers + 1, self.dz_ratio, self.depth)
        vcoord = xr.Dataset(
            {
                "zi": ("zi", np.cumsum(thickness)),
                "zl": ("zl", (np.cumsum(thickness) + 0.5 * thickness)[0:-1]),
            }
        )
        vcoord["zi"].attrs = {"units": "meters"}
        vcoord.to_netcdf(self.mom_input_dir / "vcoord.nc")

        return vcoord

    def initial_condition(self, ic_path, varnames, gridtype="A", vcoord_type="height"):
        """
        Read the initial condition files, interpolates to the model grid fixes
        up metadata and saves to the input directory.

        Args:
            ic_path (Union[str, Path]): Path to initial condition file.
            varnames (Dict[str, str]): Mapping from MOM6 variable/coordinate names to the names
                in the input dataset. For example, ``{'xq': 'lonq', 'yh': 'lath', 'salt': 'so', ...}``.
            gridtype (Optional[str]): Arakawa grid staggering of input; either ``'A'``, ``'B'``,
                or ``'C'``.
            vcoord_type (Optional[str]): The type of vertical coordinate used in yhe forcing files.
                Either ``height`` or ``thickness``.
        """

        # Remove time dimension if present in the IC. Assume that the first time dim is the intended on if more than one is present

        ic_raw = xr.open_dataset(ic_path)
        if varnames["time"] in ic_raw.dims:
            ic_raw = ic_raw.isel({varnames["time"]: 0})
        if varnames["time"] in ic_raw.coords:
            ic_raw = ic_raw.drop(varnames["time"])

        # Separate out tracers from two velocity fields of IC
        try:
            ic_raw_tracers = ic_raw[
                [varnames["tracers"][i] for i in varnames["tracers"]]
            ]
        except:
            raise ValueError(
                "Error in reading in initial condition tracers. Terminating"
            )
        try:
            ic_raw_u = ic_raw[varnames["u"]]
            ic_raw_v = ic_raw[varnames["v"]]
        except:
            raise ValueError(
                "Error in reading in initial condition tracers. Terminating"
            )

        try:
            ic_raw_eta = ic_raw[varnames["eta"]]
        except:
            raise ValueError(
                "Error in reading in initial condition tracers. Terminating"
            )

        # Rename all coordinates to have 'lon' and 'lat' to work with the xesmf regridder
        if gridtype == "A":
            if (
                "xh" in varnames.keys() and "yh" in varnames.keys()
            ):  ## Handle case where user has provided xh and yh rather than x & y
                # Rename xh with x in dictionary
                varnames["x"] = varnames["xh"]
                varnames["y"] = varnames["yh"]

            if "x" in varnames.keys() and "y" in varnames.keys():
                ic_raw_tracers = ic_raw_tracers.rename(
                    {varnames["x"]: "lon", varnames["y"]: "lat"}
                )
                ic_raw_u = ic_raw_u.rename({varnames["x"]: "lon", varnames["y"]: "lat"})
                ic_raw_v = ic_raw_v.rename({varnames["x"]: "lon", varnames["y"]: "lat"})
                ic_raw_eta = ic_raw_eta.rename(
                    {varnames["x"]: "lon", varnames["y"]: "lat"}
                )
            else:
                raise ValueError(
                    "Can't find required coordinates in initial condition. Given that gridtype is 'A' the 'x' and 'y' coordinates should be provided in the varnames dictionary. E.g {'x':'lon','y':'lat' }. Terminating"
                )
        if gridtype == "B":
            if (
                "xq" in varnames.keys()
                and "yq" in varnames.keys()
                and "xh" in varnames.keys()
                and "yh" in varnames.keys()
            ):
                ic_raw_tracers = ic_raw_tracers.rename(
                    {varnames["xh"]: "lon", varnames["yh"]: "lat"}
                )
                ic_raw_eta = ic_raw_eta.rename(
                    {varnames["xh"]: "lon", varnames["yh"]: "lat"}
                )
                ic_raw_u = ic_raw_u.rename(
                    {varnames["xq"]: "lon", varnames["yq"]: "lat"}
                )
                ic_raw_v = ic_raw_v.rename(
                    {varnames["xq"]: "lon", varnames["yq"]: "lat"}
                )
            else:
                raise ValueError(
                    "Can't find coordinates in initial condition. Given that gridtype is 'B' the names of the cell centre ('xh' & 'yh') as well as the cell edge ('xq' & 'yq') coordinates should be provided in the varnames dictionary. E.g {'xh':'lonh','yh':'lath' etc }. Terminating"
                )
        if gridtype == "C":
            if (
                "xq" in varnames.keys()
                and "yq" in varnames.keys()
                and "xh" in varnames.keys()
                and "yh" in varnames.keys()
            ):
                ic_raw_tracers = ic_raw_tracers.rename(
                    {varnames["xh"]: "lon", varnames["yh"]: "lat"}
                )
                ic_raw_eta = ic_raw_eta.rename(
                    {varnames["xh"]: "lon", varnames["yh"]: "lat"}
                )
                ic_raw_u = ic_raw_u.rename(
                    {varnames["xq"]: "lon", varnames["yh"]: "lat"}
                )
                ic_raw_v = ic_raw_v.rename(
                    {varnames["xh"]: "lon", varnames["yq"]: "lat"}
                )
            else:
                raise ValueError(
                    "Can't find coordinates in initial condition. Given that gridtype is 'C' the names of the cell centre ('xh' & 'yh') as well as the cell edge ('xq' & 'yq') coordinates should be provided in the varnames dictionary. E.g {'xh':'lonh','yh':'lath' etc }. Terminating"
                )
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
        # NaNs might be here from the land mask of the model that the IC has come from. If they're not removed then the coastlines from this other grid will be retained!
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

        if (
            vcoord_type == "thickness"
        ):  ## In this case construct the vertical profile by summing thickness
            tracers_out["zl"] = tracers_out["zl"].diff("zl")
            dz = tracers_out[self.z].diff(self.z)
            dz.name = "dz"
            dz = xr.concat([dz, dz[-1]], dim=self.z)

        tracers_out = tracers_out.interp({"zl": self.vgrid.zl.values})
        vel_out = vel_out.interp({"zl": self.vgrid.zl.values})

        print("Saving outputs... ", end="")

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
        print("done setting up initial condition.")

        self.ic_eta = eta_out
        self.ic_tracers = tracers_out
        self.ic_vels = vel_out
        return

    def rectangular_boundary(
        self, path_to_bc, varnames, orientation, segment_number, gridtype="A"
    ):
        """
        Setup a boundary forcing file for a given orientation. Here the term 'rectangular'
        implies boundaries along lines of constant latitude or longitude.

        Args:
            path_to_bc (str): Path to boundary forcing file. Ideally this should be a pre cut-out
                netCDF file containing only the boundary region and 3 extra boundary points on either
                side. Users can also provide a large dataset containing their entire domain but this
                will be slower.
            varnames (Dict[str, str]): Mapping from MOM6 variable/coordinate names to the name in the
                input dataset.
            orientation (str): Orientation of boundary forcing file, i.e., ``'east'``, ``'west'``,
                ``'north'``, or ``'south'``.
            segment_number (int): Number the segments according to how they'll be specified in
                the ``MOM_input``.
            gridtype (Optional[str]): Arakawa grid staggering of input; either ``'A'``, ``'B'``,
                or ``'C'``.
            ryf (Optional[bool]): When ``True`` the experiment runs with  'repeat-year forcing';
                when ``False``, inter-annual forcing is used.
        """

        print("Processing {} boundary...".format(orientation), end="")

        seg = segment(
            self.hgrid,
            path_to_bc,  # location of raw boundary
            self.mom_input_dir,
            varnames,
            "segment_{:03d}".format(segment_number),
            orientation,  # orienataion
            self.daterange[0],
            gridtype=gridtype,
            ryf=self.ryf,
        )

        seg.rectangular_brushcut()
        print("Done.")
        return

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
        """
        Cut out and interpolate the chosen bathymetry, then fill inland lakes.

        It's also possible to optionally fill narrow channels (see ``fill_channels``
        below), although this is less of an issue for models on a C-grid, like MOM6.

        Output is saved to the input folder for your experiment.

        Args:
            bathy_path (str): Path to chosen bathymetry file netCDF file.
            varnames (Dict[str, str]): Mapping of coordinate and
                variable names between the input and output.
            fill_channels (Optional[bool]): Whether or not to fill in
                diagonal channels. This removes more narrow inlets,
                but can also connect extra islands to land. Default: ``False``.
            minimum_layers (Optional[int]): The minimum depth allowed
                as an integer number of layers. The default value of ``3``
                layers means that anything shallower than the 3rd
                layer (as specified by the ``vcoord``) is deemed land.
            positivedown (Optional[bool]): If ``True``, it assumes that
                bathymetry vertical coordinate is positive down. Default: ``False``.
            chunks (Optional Dict[str, str]): Chunking scheme for bathymetry, e.g.,
                ``{"lon": 100, "lat": 100}``. Use lat/lon rather than the coordinate
                names in the input file.
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

            bathy.attrs["missing_value"] = (
                -1e20
            )  # This is what FRE tools expects I guess?
            bathyout = xr.Dataset({"elevation": bathy})
            bathy.close()

            bathyout = bathyout.rename({varnames["xh"]: "lon", varnames["yh"]: "lat"})
            bathyout.lon.attrs["units"] = "degrees_east"
            bathyout.lat.attrs["units"] = "degrees_north"
            bathyout.elevation.attrs["_FillValue"] = -1e20
            bathyout.elevation.attrs["units"] = "m"
            bathyout.elevation.attrs["standard_name"] = (
                "height_above_reference_ellipsoid"
            )
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
            tgrid.lat.attrs["_FillValue"] = 1e20
            tgrid.elevation.attrs["units"] = "m"
            tgrid.elevation.attrs["coordinates"] = "lon lat"
            tgrid.to_netcdf(
                self.mom_input_dir / "topog_raw.nc", mode="w", engine="netcdf4"
            )
            tgrid.close()

            ## Replace subprocess run with regular regridder
            print(
                "Starting to regrid bathymetry. If this process hangs your domain might be too big to handle this way. Try calling ESMF directly from a terminal with appropriate computational resources opened in the input directory using \n\n mpirun ESMF_Regrid -s bathy_original.nc -d topog_raw.nc -m bilinear --src_var elevation --dst_var elevation --netcdf4 --src_regional --dst_regional\n\n For details see https://xesmf.readthedocs.io/en/latest/large_problems_on_HPC.html \n\nAftewards, run this method again but set 'maketopog = False' so that python skips the computationally expensive step and just fixes up the metadata.\n\n"
            )

            # If we have a domain large enough for chunks, we'll run regridder with parallel=True
            parallel = True
            if len(tgrid.chunks) != 2:
                parallel = False
            print(f"Regridding in parallel: {parallel}")
            bathyout = bathyout.chunk(chunks)
            # return
            regridder = xe.Regridder(bathyout, tgrid, "bilinear", parallel=parallel)

            topog = regridder(bathyout)
            topog.to_netcdf(
                self.mom_input_dir / "topog_raw.nc", mode="w", engine="netcdf4"
            )
            print(
                "Regridding finished. Now excavating inland lakes and fixing up metadata..."
            )
            self.tidy_bathymetry(fill_channels, minimum_layers, positivedown)

    def tidy_bathymetry(
        self, fill_channels=False, minimum_layers=3, positivedown=False
    ):
        """
        An auxillary function for bathymetry. It's used to fix up the metadata and
        remove inland lakes after regridding the bathymetry. The functions are split
        to allow for the regridding to be done as a separate job, since regridding can
        be really expensive for large domains.

        If you've already regridded the bathymetry and just want to fix up the metadata,
        you can call this function directly to read in the existing ``topog_raw.nc`` file
        that should be in the input directory.
        """

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

        self.ocean_mask = np.abs(land_mask - 1)

        topog["depth"] *= self.ocean_mask

        topog["depth"] = topog["depth"].where(topog["depth"] != 0, np.nan)

        topog.expand_dims({"ntiles": 1}).to_netcdf(
            self.mom_input_dir / "topog_deseas.nc",
            mode="w",
            encoding={"depth": {"_FillValue": None}},
        )

        (self.mom_input_dir / "topog_deseas.nc").rename(self.mom_input_dir / "topog.nc")
        print("done.")
        self.topog = topog

    def FRE_tools(self, layout=None):
        """A wrapper for FRE Tools ``check_mask``, ``make_solo_mosaic``, and ``make_quick_mosaic``.
        User provides processor ``layout`` tuple of processing units.
        """

        print(
            "Running GFDL's FRE Tools. The following information is all printed by the FRE tools themselves"
        )
        if not (self.mom_input_dir / "topog.nc").exists():
            print("No topography file! Need to run make_bathymetry first")
            return

        for p in self.mom_input_dir.glob("mask_table*"):
            p.unlink()

        print(
            "OUTPUT FROM MAKE SOLO MOSAIC:",
            subprocess.run(
                str(self.toolpath / "make_solo_mosaic/make_solo_mosaic")
                + " --num_tiles 1 --dir . --mosaic_name ocean_mosaic --tile_file hgrid.nc",
                shell=True,
                cwd=self.mom_input_dir,
            ),
            sep="\n\n",
        )

        print(
            "OUTPUT FROM QUICK QUICK MOSAIC:",
            subprocess.run(
                str(self.toolpath / "make_quick_mosaic/make_quick_mosaic")
                + " --input_mosaic ocean_mosaic.nc --mosaic_name grid_spec --ocean_topog topog.nc",
                shell=True,
                cwd=self.mom_input_dir,
            ),
            sep="\n\n",
        )

        if layout != None:
            self.cpu_layout(layout)

    def cpu_layout(self, layout):
        """
        Wrapper for the ``check_mask`` function of GFDL's FRE Tools. User provides processor
        ``layout`` tuple of processing units.
        """

        print(
            "OUTPUT FROM CHECK MASK:\n\n",
            subprocess.run(
                str(self.toolpath / "check_mask/check_mask")
                + f" --grid_file ocean_mosaic.nc --ocean_topog topog.nc --layout {layout[0]},{layout[1]} --halo 4",
                shell=True,
                cwd=self.mom_input_dir,
            ),
        )
        self.layout = layout
        return

    def setup_run_directory(
        self,
        regional_mom6_path=".",
        surface_forcing=False,
        using_payu=False,
        overwrite=False,
    ):
        """
        Setup the run directory for MOM6. Either copy a pre-made set of files, or modify
        existing files in the 'rundir' directory for the experiment.

        Args:
            regional_mom6_path (str): Path to the regional MOM6 source code that was cloned
                from GitHub. Default is current path, ``'.'``.
            surface_forcing (Optional[str, bool]): Specify the choice of surface forcing, one
                of: ``'jra'`` or ``'era5'``. If left blank, constant fluxes will be used.
            using_payu (Optional[bool]): Whether or not to use payu (https://github.com/payu-org/payu)
                to run the model. If ``True``, a payu configuration file will be created.
                Default: ``False``.
            overwrite (Optional[bool]): Whether or not to overwrite existing files in the
                run directory. If ``False`` (default), will only modify the ``MOM_layout`` file and will
                not re-copy across the rest of the default files.
        """

        # Define the locations of the directories we'll copy files across from. Base contains most of the files, and overwrite replaces files in the base directory.
        base_run_dir = (
            Path(regional_mom6_path)  ## Path to where the demos are stored
            / "demos"
            / "premade_run_directories"
            / "common_files"
        )
        if surface_forcing != False:
            overwrite_run_dir = (
                Path(regional_mom6_path)
                / "demos"
                / "premade_run_directories"
                / f"{surface_forcing}_surface"
            )
            print(overwrite_run_dir)
            if not overwrite_run_dir.exists():
                raise ValueError(
                    f"Surface forcing {surface_forcing} not available. Please choose from {str(os.listdir(base_run_dir.parent))}."  ##Here print all available run directories
                )
        else:
            overwrite_run_dir = False

        # 3 different cases to handle:
        #   1. User is creating a new run directory from scratch. Here we copy across all files and modify.
        #   2. User has already created a run directory, and wants to modify it. Here we only modify the MOM_layout file.
        #   3. User has already created a run directory, and wants to overwrite it. Here we copy across all files and modify. This requires overwrite = True

        if not overwrite:
            for file in base_run_dir.glob(
                "*"
            ):  ## copy each file individually if it doesn't already exist OR overwrite = True
                if not os.path.exists(self.mom_run_dir / file.name):
                    ## Check whether this file exists in an override directory or not
                    if (
                        overwrite_run_dir != False
                        and (overwrite_run_dir / file.name).exists()
                    ):
                        shutil.copy(overwrite_run_dir / file.name, self.mom_run_dir)
                    else:
                        shutil.copy(base_run_dir / file.name, self.mom_run_dir)
        else:
            shutil.copytree(base_run_dir, self.mom_run_dir, dirs_exist_ok=True)
            if overwrite_run_dir != False:
                shutil.copy(base_run_dir / file, self.mom_run_dir)

        ## Make symlinks between run and input directories
        inputdir_in_rundir = self.mom_run_dir / "inputdir"
        rundir_in_inputdir = self.mom_input_dir / "rundir"

        inputdir_in_rundir.unlink(missing_ok=True)
        inputdir_in_rundir.symlink_to(self.mom_input_dir)

        rundir_in_inputdir.unlink(missing_ok=True)
        rundir_in_inputdir.symlink_to(self.mom_run_dir)

        ## Get mask table information
        mask_table = None
        for p in self.mom_input_dir.glob("mask_table.*"):
            if mask_table != None:
                print(
                    f"WARNING: Multiple mask tables found. Defaulting to {mask_table}. If this is not what you want, remove it from the run directory and try again."
                )
                break

            _, masked, layout = p.name.split(".")
            mask_table = p.name
            x, y = (int(v) for v in layout.split("x"))
            ncpus = (x * y) - int(masked)
        if mask_table == None:
            print(
                "No mask table found! This suggests your domain is mostly water, so there are no `non compute` cells that are entirely land. If this doesn't seem right, ensure you've already run .FRE_tools()."
            )
            if not hasattr(self, "layout"):
                raise AttributeError(
                    "No layout information found. This suggests you haven't run .FRE_tools() yet. Please do so first so I know how many processors you'd like to use."
                )
            ncpus = self.layout[0] * self.layout[1]
        print("Number of CPUs required: ", ncpus)

        ## Modify the input namelists to give the correct layouts
        # TODO Re-implement with package that works for this file type? or at least tidy up code
        with open(self.mom_run_dir / "MOM_layout", "r") as file:
            lines = file.readlines()
            for jj in range(len(lines)):
                if "MASKTABLE" in lines[jj]:
                    if mask_table != None:
                        lines[jj] = f'MASKTABLE = "{mask_table}"\n'
                    else:
                        lines[jj] = "# MASKTABLE = no mask table"
                if "LAYOUT =" in lines[jj] and "IO" not in lines[jj]:
                    lines[jj] = f"LAYOUT = {self.layout[1]},{self.layout[0]}\n"

                if "NIGLOBAL" in lines[jj]:
                    lines[jj] = f"NIGLOBAL = {self.hgrid.nx.shape[0]//2}\n"

                if "NJGLOBAL" in lines[jj]:
                    lines[jj] = f"NJGLOBAL = {self.hgrid.ny.shape[0]//2}\n"

        with open(self.mom_run_dir / "MOM_layout", "w") as f:
            f.writelines(lines)

        ## If using payu to run the model, create a payu configuration file
        if not using_payu and os.path.exists(f"{self.mom_run_dir}/config.yaml"):
            os.remove(f"{self.mom_run_dir}/config.yaml")

        else:
            with open(f"{self.mom_run_dir}/config.yaml", "r") as file:
                lines = file.readlines()

                inputfile = open(f"{self.mom_run_dir}/config.yaml", "r")
                lines = inputfile.readlines()
                inputfile.close()
                for i in range(len(lines)):
                    if "ncpus" in lines[i]:
                        lines[i] = f"ncpus: {str(ncpus)}\n"
                    if "jobname" in lines[i]:
                        lines[i] = f"jobname: mom6_{self.mom_input_dir.name}\n"

                    if "input:" in lines[i]:
                        lines[i + 1] = f"    - {self.mom_input_dir}\n"

            with open(f"{self.mom_run_dir}/config.yaml", "w") as file:
                file.writelines(lines)

        # Modify input.nml
        nml = f90nml.read(self.mom_run_dir / "input.nml")
        nml["coupler_nml"]["current_date"] = [
            self.daterange[0].year,
            self.daterange[0].month,
            self.daterange[0].day,
            0,
            0,
            0,
        ]
        nml.write(self.mom_run_dir / "input.nml", force=True)

    def setup_era5(self, era5_path):
        """
        Setup the ERA5 forcing files for your experiment. This assumes that
        all of the ERA5 data in the prescribed date range are downloaded.
        We need the following fields: "2t", "10u", "10v", "sp", "2d", "msdwswrf",
        "msdwlwrf", "lsrr", and "crr".

        Args:
            era5_path (str): Path to the ERA5 forcing files. Specifically, the single-level
                reanalysis product. For example, ``'SOMEPATH/era5/single-levels/reanalysis'``
        """

        ## Firstly just open all raw data
        rawdata = {}
        for fname, vname in zip(
            ["2t", "10u", "10v", "sp", "2d", "msdwswrf", "msdwlwrf", "lsrr", "crr"],
            ["t2m", "u10", "v10", "sp", "d2m", "msdwswrf", "msdwlwrf", "lsrr", "crr"],
        ):
            ## Load data from all relevant years
            years = [
                i for i in range(self.daterange[0].year, self.daterange[1].year + 1)
            ]
            # Loop through each year and read the corresponding files
            for year in years:
                ds = xr.open_mfdataset(
                    f"{era5_path}/{fname}/{year}/{fname}*",
                    decode_times=False,
                    chunks={"longitude": 100, "latitude": 100},
                )

                ## Cut out this variable to our domain size
                rawdata[fname] = nicer_slicer(
                    ds,
                    self.xextent,
                    "longitude",
                ).sel(
                    latitude=slice(
                        self.yextent[1], self.yextent[0]
                    )  ## This is because ERA5 has latitude in decreasing order (??)
                )

                ## Now fix up the latitude and time dimensions

                rawdata[fname] = (
                    rawdata[fname]
                    .isel(latitude=slice(None, None, -1))  ## Flip latitude
                    .assign_coords(
                        time=np.arange(
                            0, rawdata[fname].time.shape[0], dtype=float
                        )  ## Set the zero date of forcing to start of run
                    )
                )

                rawdata[fname].time.attrs = {
                    "calendar": "julian",
                    "units": f"hours since {self.daterange[0].strftime('%Y-%m-%d %H:%M:%S')}",
                }  ## Fix up calendar to match

                if fname == "2d":
                    ## Calculate specific humidity from dewpoint temperature
                    dewpoint = 8.07131 - 1730.63 / (
                        233.426 + rawdata["2d"]["d2m"] - 273.15
                    )
                    humidity = (
                        (0.622 / rawdata["sp"]["sp"]) * (10**dewpoint) * 101325 / 760
                    )
                    q = xr.Dataset(data_vars={"q": humidity})

                    q.q.attrs = {"long_name": "Specific Humidity", "units": "kg/kg"}
                    q.to_netcdf(
                        f"{self.mom_input_dir}/forcing/q_ERA5.nc",
                        unlimited_dims="time",
                        encoding={"q": {"dtype": "double"}},
                    )
                elif fname == "crr":
                    ## Calculate total rain rate from convective and total
                    trr = xr.Dataset(
                        data_vars={
                            "trr": rawdata["crr"]["crr"] + rawdata["lsrr"]["lsrr"]
                        }
                    )

                    trr.trr.attrs = {
                        "long_name": "Total Rain Rate",
                        "units": "kg m**-2 s**-1",
                    }
                    trr.to_netcdf(
                        f"{self.mom_input_dir}/forcing/trr_ERA5-{year}.nc",
                        unlimited_dims="time",
                        encoding={"trr": {"dtype": "double"}},
                    )

                elif fname == "lsrr":
                    ## This is handled by crr as both are added together to calculate total rain rate.
                    pass
                else:
                    rawdata[fname].to_netcdf(
                        f"{self.mom_input_dir}/forcing/{fname}_ERA5-{year}.nc",
                        unlimited_dims="time",
                        encoding={vname: {"dtype": "double"}},
                    )


class segment:
    """
    Class to turn raw boundary segment data into MOM6 boundary
    segments.

    Boundary segments should only contain the necessary data for that
    segment. No horizontal chunking is done here, so big fat segments
    will process slowly.

    Data should be at daily temporal resolution, iterating upwards
    from the provided startdate. Function ignores the time metadata
    and puts it on Julian calendar.

    Only supports z-star (z*) vertical coordinate!

    Args:
        hgrid (xarray.Dataset): The horizontal grid used for domain.
        infile (Union[str, Path]): Path to the raw, unprocessed boundary segment.
        outfolder (Union[str, Path]): Path to folder where the model inputs will
            be stored.
        varnames (Dict[str, str]): Mapping between the variable/dimension names and
            standard naming convension of this pipeline, e.g., ``{"xq": "longitude,
            "yh": "latitude", "salt": "salinity", ...}``. Key "tracers" points to nested
            dictionary of tracers to include in boundary.
        seg_name (str): Name of the segment, e.g., ``'segment_001'``.
        orientation (str): Cardinal direction (lowercase) of the boundary segment.
        startdate (str): The starting date to use in the segment calendar.
        gridtype (Optional[str]): Arakawa staggering of input grid, one of ``'A'``, ``'B'``,
            or ``'C'``
        time_units (str): The units used by the raw forcing files, e.g., ``hours``,
            ``days`` (default).
        tidal_constituents (Optional[int]): An integer determining the number of tidal
            constituents to be included from the list: *M*\ :sub:`2`, *S*\ :sub:`2`, *N*\ :sub:`2`,
            *K*\ :sub:`2`, *K*\ :sub:`1`, *O*\ :sub:`2`, *P*\ :sub:`1`, *Q*\ :sub:`1`, *Mm*,
            *Mf*, and *M*\ :sub:`4`. For example, specifying ``1`` only includes *M*\ :sub:`2`;
            specifying ``2`` includes *M*\ :sub:`2` and *S*\ :sub:`2`, etc. Default: ``None``.
        ryf (Optional[bool]): When ``True`` the experiment runs with 'repeat-year forcing'.
            When ``False`` (default) then inter-annual forcing is used.
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
        time_units="days",
        tidal_constituents=None,
        ryf=False,
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
        self.tidal_constituents = tidal_constituents
        self.ryf = ryf

    def rectangular_brushcut(self):
        """
        Cut out and interpolates tracers. This method assumes that the boundary
        is a simple Northern, Southern, Eastern, or Western boundary. Cuts out
        and interpolates tracers.
        """
        if self.orientation == "north":
            self.hgrid_seg = self.hgrid.isel(nyp=[-1])
            self.perpendicular = "ny"
            self.parallel = "nx"

        if self.orientation == "south":
            self.hgrid_seg = self.hgrid.isel(nyp=[0])
            self.perpendicular = "ny"
            self.parallel = "nx"

        if self.orientation == "east":
            self.hgrid_seg = self.hgrid.isel(nxp=[-1])
            self.perpendicular = "nx"
            self.parallel = "ny"

        if self.orientation == "west":
            self.hgrid_seg = self.hgrid.isel(nxp=[0])
            self.perpendicular = "nx"
            self.parallel = "ny"

        ## Need to keep track of which axis the 'main' coordinate corresponds to for later on when re-adding the 'secondary' axis
        if self.perpendicular == "ny":
            self.axis_to_expand = 2
        else:
            self.axis_to_expand = 3

        ## Grid for interpolating our fields
        self.interp_grid = xr.Dataset(
            {
                "lat": (
                    [f"{self.parallel}_{self.seg_name}"],
                    self.hgrid_seg.y.squeeze().data,
                ),
                "lon": (
                    [f"{self.parallel}_{self.seg_name}"],
                    self.hgrid_seg.x.squeeze().data,
                ),
            }
        ).set_coords(["lat", "lon"])

        rawseg = xr.open_dataset(self.infile, decode_times=False, engine="netcdf4")

        if self.grid == "A":
            rawseg = rawseg.rename({self.x: "lon", self.y: "lat"})
            ## In this case velocities and tracers all on same points
            regridder = xe.Regridder(
                rawseg[self.u],
                self.interp_grid,
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
                self.interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_velocity_weights_{self.orientation}.nc",
            )

            regridder_tracer = xe.Regridder(
                rawseg[self.tracers["salt"]].rename({self.xh: "lon", self.yh: "lat"}),
                self.interp_grid,
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
                self.interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_uvelocity_weights_{self.orientation}.nc",
            )

            regridder_vvelocity = xe.Regridder(
                rawseg[self.v].rename({self.xh: "lon", self.yq: "lat"}),
                self.interp_grid,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_vvelocity_weights_{self.orientation}.nc",
            )

            regridder_tracer = xe.Regridder(
                rawseg[self.tracers["salt"]].rename({self.xh: "lon", self.yh: "lat"}),
                self.interp_grid,
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
            .interpolate_na(f"{self.parallel}_{self.seg_name}")
            .ffill(f"{self.parallel}_{self.seg_name}")
            .bfill(f"{self.parallel}_{self.seg_name}")
        )

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
        dz = segment_out[self.z].diff(self.z)
        dz.name = "dz"
        dz = xr.concat([dz, dz[-1]], dim=self.z)

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
                f"{self.perpendicular}_{self.seg_name}", axis=self.axis_to_expand
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
        ].expand_dims(
            f"{self.perpendicular}_{self.seg_name}", axis=self.axis_to_expand - 1
        )

        # Overwrite the actual lat/lon values in the dimensions, replace with incrementing integers
        segment_out[f"{self.parallel}_{self.seg_name}"] = np.arange(
            segment_out[f"{self.parallel}_{self.seg_name}"].size
        )
        segment_out[f"{self.perpendicular}_{self.seg_name}"] = [0]

        # Store actual lat/lon values here as variables rather than coordinates
        segment_out[f"lon_{self.seg_name}"] = (
            [f"ny_{self.seg_name}", f"nx_{self.seg_name}"],
            self.hgrid_seg.x.data,
        )
        segment_out[f"lat_{self.seg_name}"] = (
            [f"ny_{self.seg_name}", f"nx_{self.seg_name}"],
            self.hgrid_seg.y.data,
        )

        # Add units to the lat / lon to keep the `categorize_axis_from_units` checker happy
        segment_out[f"lat_{self.seg_name}"].attrs = {
            "units": "degrees_north",
        }
        segment_out[f"lon_{self.seg_name}"].attrs = {
            "units": "degrees_east",
        }

        # If repeat-year forcing, add modulo coordinate
        if self.ryf:
            segment_out["time"] = segment_out["time"].assign_attrs({"modulo": " "})

        with ProgressBar():
            segment_out.load().to_netcdf(
                self.outfolder / f"forcing/forcing_obc_{self.seg_name}.nc",
                encoding=encoding_dict,
                unlimited_dims="time",
            )

        return segment_out, encoding_dict
