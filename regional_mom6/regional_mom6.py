import numpy as np
from pathlib import Path
import dask.array as da
import xarray as xr
import xesmf as xe
import subprocess
from scipy.ndimage import binary_fill_holes
import netCDF4
from dask.diagnostics import ProgressBar
import f90nml
import datetime as dt
import warnings
import shutil
import os
import importlib.resources
import datetime
from .utils import (
    quadrilateral_areas,
    ap2ep,
    ep2ap,
)
import pandas as pd
import re
from pathlib import Path
import glob
from collections import defaultdict
import json

warnings.filterwarnings("ignore")

__all__ = [
    "longitude_slicer",
    "hyperbolictan_thickness_profile",
    "generate_rectangular_hgrid",
    "experiment",
    "segment",
    "load_experiment",
]


## Mapping Functions


def convert_to_tpxo_tidal_constituents(tidal_constituents):
    """
    Convert tidal constituents from strings to integers using a dictionary.

    Parameters:
    tidal_constituents (list of str): List of tidal constituent names as strings.

    Returns:
    list of int: List of tidal constituent indices as integers.
    """
    tidal_constituents_tpxo_dict = {
        "M2": 0,
        "S2": 1,
        "N2": 2,
        "K2": 3,
        "K1": 4,
        "O1": 5,
        "P1": 6,
        "Q1": 7,
        "MM": 8,
        "MF": 9,
        # Only supported tidal bc's
    }

    list_of_ints = []
    for tc in tidal_constituents:
        try:
            list_of_ints.append(tidal_constituents_tpxo_dict[tc])
        except:
            raise ValueError(
                "Invalid Input. Tidal constituent {} is not supported.".format(tc)
            )

    return list_of_ints


def find_MOM6_rectangular_orientation(input):
    """
    Convert between MOM6 boundary and the specific segment number needed, or the inverse
    """
    direction_dir = {
        "south": 1,
        "north": 2,
        "west": 3,
        "east": 4,
    }
    direction_dir_inv = {v: k for k, v in direction_dir.items()}

    if type(input) == str:
        try:
            return direction_dir[input]
        except:
            raise ValueError(
                "Invalid Input. Did you spell the direction wrong, it should be lowercase?"
            )
    elif type(input) == int:
        try:
            return direction_dir_inv[input]
        except:
            raise ValueError("Invalid Input. Did you pick a number 1 through 4?")
    else:
        raise ValueError("Invalid type of Input, can only be string or int.")


## Load Experiment Function


def load_experiment(config_file_path):
    print("Reading from config file....")
    with open(config_file_path, "r") as f:
        config_dict = json.load(f)

    print("Creating Empty Experiment Object....")
    expt = experiment.create_empty()

    print("Setting Default Variables.....")
    expt.expt_name = config_dict["name"]
    try:
        expt.longitude_extent = tuple(config_dict["longitude_extent"])
        expt.latitude_extent = tuple(config_dict["latitude_extent"])
    except:
        expt.longitude_extent = None
        expt.latitude_extent = None
    try:
        expt.date_range = config_dict["date_range"]
        expt.date_range[0] = dt.datetime.strptime(expt.date_range[0], "%Y-%m-%d")
        expt.date_range[1] = dt.datetime.strptime(expt.date_range[1], "%Y-%m-%d")
    except:
        expt.date_range = None
    expt.mom_run_dir = Path(config_dict["run_dir"])
    expt.mom_input_dir = Path(config_dict["input_dir"])
    expt.toolpath_dir = Path(config_dict["toolpath_dir"])
    expt.resolution = config_dict["resolution"]
    expt.number_vertical_layers = config_dict["number_vertical_layers"]
    expt.layer_thickness_ratio = config_dict["layer_thickness_ratio"]
    expt.depth = config_dict["depth"]
    expt.hgrid_type = config_dict["hgrid_type"]
    expt.repeat_year_forcing = config_dict["repeat_year_forcing"]
    expt.ocean_mask = None
    expt.layout = None
    expt.minimum_depth = config_dict["minimum_depth"]
    expt.tidal_constituents = config_dict["tidal_constituents"]

    print("Checking for hgrid and vgrid....")
    if os.path.exists(config_dict["hgrid"]):
        print("Found")
        expt.hgrid = xr.open_dataset(config_dict["hgrid"])
    else:
        print("Hgrid not found, call _make_hgrid when you're ready.")
        expt.hgrid = None
    if os.path.exists(config_dict["vgrid"]):
        print("Found")
        expt.vgrid = xr.open_dataset(config_dict["vgrid"])
    else:
        print("Vgrid not found, call _make_vgrid when ready")
        expt.vgrid = None

    print("Checking for bathymetry...")
    if config_dict["bathymetry"] is not None and os.path.exists(
        config_dict["bathymetry"]
    ):
        print("Found")
        expt.bathymetry = xr.open_dataset(config_dict["bathymetry"])
    else:
        print(
            "Bathymetry not found. Please provide bathymetry, or call setup_bathymetry method to set up bathymetry."
        )

    print("Checking for ocean state files....")
    found = True
    for path in config_dict["ocean_state"]:
        if not os.path.exists(path):
            found = False
            print(
                "At least one ocean state file not found. Please provide ocean state files, or call setup_ocean_state_boundaries method to set up ocean state."
            )
            break
    if found:
        print("Found")
    found = True
    print("Checking for initial condition files....")
    for path in config_dict["initial_conditions"]:
        if not os.path.exists(path):
            found = False
            print(
                "At least one initial condition file not found. Please provide initial condition files, or call setup_initial_condition method to set up initial condition."
            )
            break
    if found:
        print("Found")
    found = True
    print("Checking for tides files....")
    for path in config_dict["tides"]:
        if not os.path.exists(path):
            found = False
            print(
                "At least one tides file not found. If you would like tides, call setup_tides_boundaries method to set up tides"
            )
            break
    if found:
        print("Found")
    found = True

    return expt


## Auxiliary functions


def longitude_slicer(data, longitude_extent, longitude_coords):
    """
    Slice longitudes while handling periodicity and the 'seams', that is the
    longitude values where the data wraps around in a global domain (for example,
    longitudes are defined, usually, within domain [0, 360] or [-180, 180]).

    The algorithm works in five steps:

    - Determine whether we need to add or subtract 360 to get the middle of the
      ``longitude_extent`` to lie within ``data``'s longitude range (hereby ``old_lon``).

    - Shift the dataset so that its midpoint matches the midpoint of
      ``longitude_extent`` (up to a multiple of 360). Now, the modified ``old_lon``
      does not increase monotonically from West to East since the 'seam'
      has moved.

    - Fix ``old_lon`` to make it monotonically increasing again. This uses
      the information we have about the way the dataset was shifted/rolled.

    - Slice the ``data`` index-wise. We know that ``|longitude_extent[1] - longitude_extent[0]| / 360``
      multiplied by the number of discrete longitude points in the global input data gives
      the number of longitude points in our slice, and we've already set the midpoint
      to be the middle of the target domain.

    - Finally re-add the correct multiple of 360 so the whole domain matches
      the target.

    Args:
        data (xarray.Dataset): The global data you want to slice in longitude.
        longitude_extent (Tuple[float, float]): The target longitudes (in degrees)
            we want to slice to. Must be in increasing order.
        longitude_coords (Union[str, list[str]): The name or list of names of the
            longitude coordinates(s) in ``data``.
    Returns:
        xarray.Dataset: The sliced ``data``.
    """

    if isinstance(longitude_coords, str):
        longitude_coords = [longitude_coords]

    for lon in longitude_coords:
        central_longitude = np.mean(longitude_extent)  ## Midpoint of target domain

        ## Find a corresponding value for the intended domain midpoint in our data.
        ## It's assumed that data has equally-spaced longitude values.

        lons = data[lon].data
        dlons = lons[1] - lons[0]

        assert np.allclose(
            np.diff(lons), dlons * np.ones(np.size(lons) - 1)
        ), "provided longitude coordinate must be uniformly spaced"

        for i in range(-1, 2, 1):
            if data[lon][0] <= central_longitude + 360 * i <= data[lon][-1]:

                ## Shifted version of target midpoint; e.g., could be -90 vs 270
                ## integer i keeps track of what how many multiples of 360 we need to shift entire
                ## grid by to match central_longitude
                _central_longitude = central_longitude + 360 * i

                ## Midpoint of the data
                central_data = data[lon][data[lon].shape[0] // 2].values

                ## Number of indices between the data midpoint and the target midpoint.
                ## Sign indicates direction needed to shift.
                shift = int(
                    -(data[lon].shape[0] * (_central_longitude - central_data)) // 360
                )

                ## Shift data so that the midpoint of the target domain is the middle of
                ## the data for easy slicing.
                new_data = data.roll({lon: 1 * shift}, roll_coords=True)

                ## Create a new longitude coordinate.
                ## We'll modify this to remove any seams (i.e., jumps like -270 -> 90)
                new_lon = new_data[lon].values

                ## Take the 'seam' of the data, and either backfill or forward fill based on
                ## whether the data was shifted F or west
                if shift > 0:
                    new_seam_index = shift

                    new_lon[0:new_seam_index] -= 360

                if shift < 0:
                    new_seam_index = data[lon].shape[0] + shift

                    new_lon[new_seam_index:] += 360

                ## new_lon is used to re-centre the midpoint to match that of target domain
                new_lon -= i * 360

                new_data = new_data.assign_coords({lon: new_lon})

                ## Choose the number of lon points to take from the middle, including a buffer.
                ## Use this to index the new global dataset
                num_lonpoints = (
                    int(data[lon].shape[0] * (central_longitude - longitude_extent[0]))
                    // 360
                )

        data = new_data.isel(
            {
                lon: slice(
                    data[lon].shape[0] // 2 - num_lonpoints,
                    data[lon].shape[0] // 2 + num_lonpoints,
                )
            }
        )

    return data


def get_glorys_data(
    longitude_extent,
    latitude_extent,
    timerange,
    segment_name,
    download_path,
    modify_existing=True,
):
    """
    Generates a bash script to download all of the required ocean forcing data.

    Args:
        longitude_extent (tuple of floats): Westward and Eastward extents of the segment
        latitude_extent (tuple of floats): Southward and Northward extents of the segment
        timerange (tule of datetime strings): Start and end of the segment in format %Y-%m-%d %H:%M:%S
        segment_range (str): name of the segment (minus .nc extension, eg east_unprocessed)
        download_path (str): Location of where this script is saved
        modify_existing (bool): Whether to add to an existing script or start a new one
        buffer (float): number of
    """
    buffer = 0.24  # Pads downloads to ensure that interpolation onto desired domain doesn't fail. Default of 0.24 is twice Glorys cell width (12th degree)

    path = Path(download_path)

    if modify_existing:
        file = open(Path(path / "get_glorys_data.sh"), "r")
        lines = file.readlines()
        file.close()

    else:
        lines = ["#!/bin/bash\n"]

    file = open(Path(path / "get_glorys_data.sh"), "w")

    lines.append(
        f"""
copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m --variable so --variable thetao --variable uo --variable vo --variable zos --start-datetime {str(timerange[0]).replace(" ","T")} --end-datetime {str(timerange[1]).replace(" ","T")} --minimum-longitude {longitude_extent[0] - buffer} --maximum-longitude {longitude_extent[1] + buffer} --minimum-latitude {latitude_extent[0] - buffer} --maximum-latitude {latitude_extent[1] + buffer} --minimum-depth 0 --maximum-depth 6000 -o {str(path)} -f {segment_name}.nc --force-download\n
"""
    )
    file.writelines(lines)
    file.close()
    return


def hyperbolictan_thickness_profile(nlayers, ratio, total_depth):
    """Generate a hyperbolic tangent thickness profile with ``nlayers`` vertical
    layers and total depth of ``total_depth`` whose bottom layer is (about) ``ratio``
    times larger than the top layer.

    The thickness profile transitions from the top-layer thickness to
    the bottom-layer thickness via a hyperbolic tangent proportional to
    ``tanh(2π * (k / (nlayers - 1) - 1 / 2))``, where ``k = 0, 1, ..., nlayers - 1``
    is the layer index with ``k = 0`` corresponding to the top-most layer.

    The sum of all layer thicknesses is ``total_depth``.

    Positive parameter ``ratio`` prescribes (approximately) the ratio of the thickness
    of the bottom-most layer to the top-most layer. The final ratio of the bottom-most
    layer to the top-most layer ends up a bit different from the prescribed ``ratio``.
    In particular, the final ratio of the bottom over the top-most layer thickness is
    ``(1 + ratio * exp(2π)) / (ratio + exp(2π))``. This slight departure comes about
    because of the value of the hyperbolic tangent profile at the end-points ``tanh(π)``,
    which is approximately 0.9963 and not 1. Note that because ``exp(2π)`` is much greater
    than 1, the value of the actual ratio is not that different from the prescribed value
    ``ratio``, e.g., for ``ratio`` values between 1/100 and 100 the final ratio of the
    bottom-most layer to the top-most layer only departs from the prescribed ``ratio``
    by ±20%.

    Args:
        nlayers (int): Number of vertical layers.
        ratio (float): The desired value of the ratio of bottom-most to
            the top-most layer thickness. Note that the final value of
            the ratio of bottom-most to the top-most layer thickness
            ends up ``(1 + ratio * exp(2π)) / (ratio + exp(2π))``. Must
            be positive.
        total_depth (float): The total depth of grid, i.e., the sum
            of all thicknesses.

    Returns:
        numpy.array: An array containing the layer thicknesses.

    Examples:

        The spacings for a vertical grid with 20 layers, with maximum depth 1000 meters,
        and for which the top-most layer is about 4 times thinner than the bottom-most
        one.

        >>> from regional_mom6 import hyperbolictan_thickness_profile
        >>> nlayers, total_depth = 20, 1000
        >>> ratio = 4
        >>> dz = hyperbolictan_thickness_profile(nlayers, ratio, total_depth)
        >>> dz
        array([20.11183771, 20.2163053 , 20.41767549, 20.80399084, 21.53839043,
               22.91063751, 25.3939941 , 29.6384327 , 36.23006369, 45.08430684,
               54.91569316, 63.76993631, 70.3615673 , 74.6060059 , 77.08936249,
               78.46160957, 79.19600916, 79.58232451, 79.7836947 , 79.88816229])
        >>> dz.sum()
        1000.0
        >>> dz[-1] / dz[0]
        3.9721960481753706

        If we want the top layer to be thicker then we need to prescribe ``ratio < 1``.

        >>> from regional_mom6 import hyperbolictan_thickness_profile
        >>> nlayers, total_depth = 20, 1000
        >>> ratio = 1/4
        >>> dz = hyperbolictan_thickness_profile(nlayers, ratio, total_depth)
        >>> dz
        array([79.88816229, 79.7836947 , 79.58232451, 79.19600916, 78.46160957,
               77.08936249, 74.6060059 , 70.3615673 , 63.76993631, 54.91569316,
               45.08430684, 36.23006369, 29.6384327 , 25.3939941 , 22.91063751,
               21.53839043, 20.80399084, 20.41767549, 20.2163053 , 20.11183771])
        >>> dz.sum()
        1000.0
        >>> dz[-1] / dz[0]
        0.25174991059652

        Now how about a grid with the same total depth as above but with equally-spaced
        layers.

        >>> from regional_mom6 import hyperbolictan_thickness_profile
        >>> nlayers, total_depth = 20, 1000
        >>> ratio = 1
        >>> dz = hyperbolictan_thickness_profile(nlayers, ratio, total_depth)
        >>> dz
        array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
               50., 50., 50., 50., 50., 50., 50.])
    """

    assert isinstance(nlayers, int), "nlayers must be an integer"

    if nlayers == 1:
        return np.array([float(total_depth)])

    assert ratio > 0, "ratio must be > 0"

    # The hyberbolic tangent profile below implies that the sum of
    # all layer thicknesses is:
    #
    # nlayers * (top_layer_thickness + bottom_layer_thickness) / 2
    #
    # By choosing the top_layer_thickness appropriately we ensure that
    # the sum of all layer thicknesses is the prescribed total_depth.
    top_layer_thickness = 2 * total_depth / (nlayers * (1 + ratio))

    bottom_layer_thickness = ratio * top_layer_thickness

    layer_thicknesses = top_layer_thickness + 0.5 * (
        bottom_layer_thickness - top_layer_thickness
    ) * (1 + np.tanh(2 * np.pi * (np.arange(nlayers) / (nlayers - 1) - 1 / 2)))

    sum_of_thicknesses = np.sum(layer_thicknesses)

    atol = np.finfo(type(sum_of_thicknesses)).eps

    assert np.isclose(total_depth, sum_of_thicknesses, atol=atol)  # just checking ;)

    return layer_thicknesses


def generate_rectangular_hgrid(lons, lats):
    """
    Construct a horizontal grid with all the metadata required by MOM6, based on
    arrays of longitudes (``lons``) and latitudes (``lats``) on the supergrid.
    Here, 'supergrid' refers to both cell edges and centres, meaning that there
    are twice as many points along each axis than for any individual field.

    Caution:
        It is assumed the grid's boundaries are lines of constant latitude and
        longitude. Rotated grids need to be handled differently.

        It is also assumed here that the longitude array values are uniformly spaced.

        Ensure both ``lons`` and ``lats`` are monotonically increasing.

    Args:
        lons (numpy.array): All longitude points on the supergrid. Must be uniformly spaced.
        lats (numpy.array): All latitude points on the supergrid.

    Returns:
        xarray.Dataset: An FMS-compatible horizontal grid (``hgrid``) that includes all required attributes.
    """

    assert np.all(
        np.diff(lons) > 0
    ), "longitudes array lons must be monotonically increasing"
    assert np.all(
        np.diff(lats) > 0
    ), "latitudes array lats must be monotonically increasing"

    R = 6371e3  # mean radius of the Earth; https://en.wikipedia.org/wiki/Earth_radius

    # compute longitude spacing and ensure that longitudes are uniformly spaced
    dlons = lons[1] - lons[0]

    assert np.allclose(
        np.diff(lons), dlons * np.ones(np.size(lons) - 1)
    ), "provided array of longitudes must be uniformly spaced"

    # dx = R * cos(np.deg2rad(lats)) * np.deg2rad(dlons) / 2
    # Note: division by 2 because we're on the supergrid
    dx = np.broadcast_to(
        R * np.cos(np.deg2rad(lats)) * np.deg2rad(dlons) / 2,
        (lons.shape[0] - 1, lats.shape[0]),
    ).T

    # dy = R * np.deg2rad(dlats) / 2
    # Note: division by 2 because we're on the supergrid
    dy = np.broadcast_to(
        R * np.deg2rad(np.diff(lats)) / 2, (lons.shape[0], lats.shape[0] - 1)
    ).T

    lon, lat = np.meshgrid(lons, lats)

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
            "units": "meters",
        },
        "dy": {
            "standard_name": "grid_edge_y_distance",
            "units": "meters",
        },
        "area": {
            "standard_name": "grid_cell_area",
            "units": "m**2",
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

    Everything about the regional experiment.

    Methods in this class generate the various input files needed for a MOM6
    experiment forced with open boundary conditions (OBCs). The code is agnostic
    to the user's choice of boundary forcing, bathymetry, and surface forcing;
    users need to prescribe what variables are all called via mapping dictionaries
    from MOM6 variable/coordinate name to the name in the input dataset.

    The class can be used to generate the grids for a new experiment, or to read in
    an existing one (when ``read_existing_grids=True``; see argument description below).

    Args:
        longitude_extent (Tuple[float]): Extent of the region in longitude (in degrees). For
            example: ``(40.5, 50.0)``.
        latitude_extent (Tuple[float]): Extent of the region in latitude (in degrees). For
            example: ``(-20.0, 30.0)``.
        date_range (Tuple[str]): Start and end dates of the boundary forcing window. For
            example: ``("2003-01-01", "2003-01-31")``.
        resolution (float): Lateral resolution of the domain (in degrees).
        number_vertical_layers (int): Number of vertical layers.
        layer_thickness_ratio (float): Ratio of largest to smallest layer thickness;
            used as input in :func:`~hyperbolictan_thickness_profile`.
        depth (float): Depth of the domain.
        mom_run_dir (str): Path of the MOM6 control directory.
        mom_input_dir (str): Path of the MOM6 input directory, to receive the forcing files.
        toolpath_dir (str): Path of GFDL's FRE tools (https://github.com/NOAA-GFDL/FRE-NCtools)
            binaries.
        hgrid_type (Optional[str]): Type of horizontal grid to generate.
            Currently, only ``'even_spacing'`` is supported.
        repeat_year_forcing (Optional[bool]): When ``True`` the experiment runs with
            repeat-year forcing. When ``False`` (default) then inter-annual forcing is used.
        read_existing_grids (Optional[Bool]): When ``True``, instead of generating the grids,
            the grids and the ocean mask are being read from within the ``mom_input_dir`` and
            ``mom_run_dir`` directories. Useful for modifying or troubleshooting experiments.
            Default: ``False``.
        minimum_depth (Optional[int]): The minimum depth in meters of a grid cell allowed before it is masked out and treated as land.
    """

    @classmethod
    def create_empty(
        self,
        longitude_extent=None,
        latitude_extent=None,
        date_range=None,
        resolution=None,
        number_vertical_layers=None,
        layer_thickness_ratio=None,
        depth=None,
        mom_run_dir=None,
        mom_input_dir=None,
        toolpath_dir=None,
        hgrid_type="even_spacing",
        repeat_year_forcing=False,
        minimum_depth=4,
        tidal_constituents=["M2"],
        name=None,
    ):
        """
        Substitute init method to creates an empty expirement object, with the opportunity to override whatever values wanted.
        """
        expt = self(
            longitude_extent=None,
            latitude_extent=None,
            date_range=None,
            resolution=None,
            number_vertical_layers=None,
            layer_thickness_ratio=None,
            depth=None,
            minimum_depth=None,
            mom_run_dir=None,
            mom_input_dir=None,
            toolpath_dir=None,
            create_empty=True,
            hgrid_type=None,
            repeat_year_forcing=None,
            tidal_constituents=None,
            name=None,
        )

        expt.expt_name = name
        expt.tidal_constituents = tidal_constituents
        expt.repeat_year_forcing = repeat_year_forcing
        expt.hgrid_type = hgrid_type
        expt.toolpath_dir = toolpath_dir
        expt.mom_run_dir = mom_run_dir
        expt.mom_input_dir = mom_input_dir
        expt.minimum_depth = minimum_depth
        expt.depth = depth
        expt.layer_thickness_ratio = layer_thickness_ratio
        expt.number_vertical_layers = number_vertical_layers
        expt.resolution = resolution
        expt.date_range = date_range
        expt.latitude_extent = latitude_extent
        expt.longitude_extent = longitude_extent
        expt.ocean_mask = None
        expt.layout = None
        return expt

    def __init__(
        self,
        *,
        date_range,
        resolution,
        number_vertical_layers,
        layer_thickness_ratio,
        depth,
        mom_run_dir,
        mom_input_dir,
        toolpath_dir,
        longitude_extent=None,
        latitude_extent=None,
        hgrid_type="even_spacing",
        vgrid_type="hyperbolic_tangent",
        repeat_year_forcing=False,
        minimum_depth=4,
        tidal_constituents=["M2"],
        create_empty=False,
        name=None,
    ):

        # Creates empty experiment object for testing and experienced user manipulation.
        # Kinda seems like a logical spinoff of this is to divorce the hgrid/vgrid creation from the experiment object initialization.
        # Probably more of a CS workflow. That way read_existing_grids could be a function on its own, which ties in better with
        # For now, check out the create_empty method for more explanation
        if create_empty:
            return

        # ## Set up the experiment with no config file
        ## in case list was given, convert to tuples
        self.expt_name = name
        self.date_range = tuple(date_range)

        self.mom_run_dir = Path(mom_run_dir)
        self.mom_input_dir = Path(mom_input_dir)
        self.toolpath_dir = Path(toolpath_dir)

        self.mom_run_dir.mkdir(exist_ok=True)
        self.mom_input_dir.mkdir(exist_ok=True)

        self.date_range = [
            dt.datetime.strptime(date_range[0], "%Y-%m-%d %H:%M:%S"),
            dt.datetime.strptime(date_range[1], "%Y-%m-%d %H:%M:%S"),
        ]
        self.resolution = resolution
        self.number_vertical_layers = number_vertical_layers
        self.layer_thickness_ratio = layer_thickness_ratio
        self.depth = depth
        self.hgrid_type = hgrid_type
        self.vgrid_type = vgrid_type
        self.repeat_year_forcing = repeat_year_forcing
        self.ocean_mask = None
        self.layout = None  # This should be a tuple. Leaving in a dummy 'None' makes it easy to remind the user to provide a value later on.
        self.minimum_depth = minimum_depth  # Minimum depth allowed in bathy file
        self.tidal_constituents = tidal_constituents

        if hgrid_type == "from_file":
            try:
                self.hgrid = xr.open_dataset(self.mom_input_dir / "hgrid.nc")
                self.longitude_extent = (
                    float(self.hgrid.x.min()),
                    float(self.hgrid.x.max()),
                )
                self.latitude_extent = (
                    float(self.hgrid.y.min()),
                    float(self.hgrid.y.max()),
                )
            except:
                print(
                    "Error while reading in existing horizontal grid!\n\n"
                    + f"Make sure `hgrid.nc`exists in {self.mom_input_dir} directory."
                )
                raise ValueError
        else:
            self.longitude_extent = tuple(longitude_extent)
            self.latitude_extent = tuple(latitude_extent)
            self.hgrid = self._make_hgrid()

        if vgrid_type == "from_file":
            try:
                self.vgrid = xr.open_dataset(self.mom_input_dir / "vcoord.nc")

            except:
                print(
                    "Error while reading in existing vertical coordinates!\n\n"
                    + f"Make sure `vcoord.nc`exists in {self.mom_input_dir} directory."
                )
                raise ValueError
        else:
            self.vgrid = self._make_vgrid()

        self.segments = (
            {}
        )  # Holds segements for use in setting up the ocean state boundary conditions (GLORYS) and the tidal boundary conditions (TPXO)

        # create additional directories and links
        (self.mom_input_dir / "weights").mkdir(exist_ok=True)
        (self.mom_input_dir / "forcing").mkdir(exist_ok=True)

        run_inputdir = self.mom_run_dir / "inputdir"
        if not run_inputdir.exists():
            run_inputdir.symlink_to(self.mom_input_dir.resolve())
        input_rundir = self.mom_input_dir / "rundir"
        if not input_rundir.exists():
            input_rundir.symlink_to(self.mom_run_dir.resolve())

    def __str__(self) -> str:
        return json.dumps(self.write_config_file(export=False, quiet=True), indent=4)

    def __getattr__(self, name):
        available_methods = [
            method for method in dir(self) if not method.startswith("__")
        ]
        error_message = (
            f"{name} method not found. Available methods are: {available_methods}"
        )
        raise AttributeError(error_message)

    def _make_hgrid(self):
        """
        Set up a horizontal grid based on user's specification of the domain.
        The default behaviour generates a grid evenly spaced both in longitude
        and in latitude.

        The latitudinal resolution is scaled with the cosine of the central
        latitude of the domain, i.e., ``Δlats = cos(lats_central) * Δlons``, where ``Δlons``
        is the longitudinal spacing. This way, for a sufficiently small domain,
        the linear distances between grid points are nearly identical:
        ``Δx = R * cos(lats) * Δlons`` and ``Δy = R * Δlats = R * cos(lats_central) * Δlons``
        (here ``R`` is Earth's radius and ``lats``, ``lats_central``, ``Δlons``, and ``Δlats``
        are all expressed in radians).
        That is, if the domain is small enough that so that ``cos(lats_North_Side)``
        is not much different from ``cos(lats_South_Side)``, then ``Δx`` and ``Δy``
        are similar.

        Note:
            The intention is for the horizontal grid (``hgrid``) generation to be flexible.
            For now, there is only one implemented horizontal grid included in the package,
            but you can customise it by simply overwriting the ``hgrid.nc`` file in the
            ``mom_run_dir`` directory after initialising an ``experiment``. To preserve the
            metadata, it might be easiest to read the file in, then modify the fields before
            re-saving.
        """

        assert (
            self.hgrid_type == "even_spacing"
        ), "only even_spacing grid type is implemented"

        if self.hgrid_type == "even_spacing":

            # longitudes are evenly spaced based on resolution and bounds
            nx = int(
                (self.longitude_extent[1] - self.longitude_extent[0])
                / (self.resolution / 2)
            )
            if nx % 2 != 1:
                nx += 1

            lons = np.linspace(
                self.longitude_extent[0], self.longitude_extent[1], nx
            )  # longitudes in degrees

            # Latitudes evenly spaced by dx * cos(central_latitude)
            central_latitude = np.mean(self.latitude_extent)  # degrees
            latitudinal_resolution = self.resolution * np.cos(
                np.deg2rad(central_latitude)
            )

            ny = (
                int(
                    (self.latitude_extent[1] - self.latitude_extent[0])
                    / (latitudinal_resolution / 2)
                )
                + 1
            )

            if ny % 2 != 1:
                ny += 1

            lats = np.linspace(
                self.latitude_extent[0], self.latitude_extent[1], ny
            )  # latitudes in degrees

            hgrid = generate_rectangular_hgrid(lons, lats)
            hgrid.to_netcdf(self.mom_input_dir / "hgrid.nc")

            return hgrid

    def _make_vgrid(self):
        """
        Generates a vertical grid based on the ``number_vertical_layers``, the ratio
        of largest to smallest layer thickness (``layer_thickness_ratio``) and the
        total ``depth`` parameters.
        (All these parameters are specified at the class level.)
        """

        thicknesses = hyperbolictan_thickness_profile(
            self.number_vertical_layers, self.layer_thickness_ratio, self.depth
        )

        zi = np.cumsum(thicknesses)
        zi = np.insert(zi, 0, 0.0)  # add zi = 0.0 as first interface

        zl = zi[0:-1] + thicknesses / 2  # the mid-points between interfaces zi

        vcoord = xr.Dataset({"zi": ("zi", zi), "zl": ("zl", zl)})

        ## Check whether the minimum depth is less than the first three layers

        if self.minimum_depth < zi[2]:
            print(
                f"Warning: Minimum depth of {self.minimum_depth}m is less than the depth of the third interface ({zi[2]}m)!\n"
                + "This means that some areas may only have one or two layers between the surface and sea floor. \n"
                + "For increased stability, consider increasing the minimum depth, or adjusting the vertical coordinate to add more layers near the surface."
            )

        vcoord["zi"].attrs = {"units": "meters"}
        vcoord["zl"].attrs = {"units": "meters"}

        vcoord.to_netcdf(self.mom_input_dir / "vcoord.nc")

        return vcoord

    @property
    def ocean_state_boundaries(self):
        """
        Read the ocean state files from disk, and print 'em
        """
        ocean_state_path = self.mom_input_dir / "forcing"
        try:
            # Use glob to find all tides files
            patterns = [
                "forcing_*",
                "weights/bi*",
            ]
            all_files = []
            for pattern in patterns:
                all_files.extend(glob.glob(Path(ocean_state_path / pattern)))
                all_files.extend(glob.glob(Path(self.mom_input_dir / pattern)))

            if len(all_files) == 0:
                return "No ocean state files set up yet (or files misplaced from {}). Call `setup_ocean_state_boundaries` method to set up ocean state.".format(
                    ocean_state_path
                )

            # Open the files as xarray datasets
            # datasets = [xr.open_dataset(file) for file in all_files]
            return all_files
        except:
            return "Error retrieving ocean state files"

    @property
    def tides_boundaries(self):
        """
        Read the tides from disk, and print 'em
        """
        tides_path = self.mom_input_dir / "forcing"
        try:
            # Use glob to find all tides files
            patterns = ["regrid*", "tu_*", "tz_*"]
            all_files = []
            for pattern in patterns:
                all_files.extend(glob.glob(Path(tides_path / pattern)))
                all_files.extend(glob.glob(Path(self.mom_input_dir / pattern)))

            if len(all_files) == 0:
                return "No tides files set up yet (or files misplaced from {}). Call `setup_tides_boundaries` method to set up tides.".format(
                    tides_path
                )

            # Open the files as xarray datasets
            # datasets = [xr.open_dataset(file) for file in all_files]
            return all_files
        except:
            return "Error retrieving tides files"

    @property
    def era5(self):
        """
        Read the era5's from disk, and print 'em
        """
        era5_path = self.mom_input_dir / "forcing"
        try:
            # Use glob to find all *_ERA5.nc files
            all_files = glob.glob(Path(era5_path / "*_ERA5.nc"))
            if len(all_files) == 0:
                return "No era5 files set up yet (or files misplaced from {}). Call `setup_era5` method to set up era5.".format(
                    era5_path
                )

            # Open the files as xarray datasets
            # datasets = [xr.open_dataset(file) for file in all_files]
            return all_files
        except:
            return "Error retrieving ERA5 files"

    @property
    def initial_condition(self):
        """
        Read the ic's from disk, and print 'em
        """
        forcing_path = self.mom_input_dir / "forcing"
        try:
            all_files = glob.glob(Path(forcing_path / "init_*.nc"))
            all_files = glob.glob(Path(self.mom_input_dir / "init_*.nc"))
            if len(all_files) == 0:
                return "No initial conditions files set up yet (or files misplaced from {}). Call `setup_initial_condition` method to set up initial conditions.".format(
                    forcing_path
                )

            # Open the files as xarray datasets
            # datasets = [xr.open_dataset(file) for file in all_files]
            # return datasets

            return all_files
        except:
            return "No initial condition set up yet (or files misplaced from {}). Call `setup_initial_condition` method to set up initial conditions.".format(
                self.mom_input_dir / "forcing"
            )

    @property
    def bathymetry_property(self):
        """
        Read the bathymetry from disk, and print 'em
        """

        try:
            bathy = xr.open_dataset(self.mom_input_dir / "bathymetry.nc")
            # return [bathy]
            return str(self.mom_input_dir / "bathymetry.nc")
        except:
            return "No bathymetry set up yet (or files misplaced from {}). Call `setup_bathymetry` method to set up bathymetry.".format(
                self.mom_input_dir
            )

    def write_config_file(self, path=None, export=True, quiet=False):
        """
        Write a configuration file for the experiment. This is a simple json file
        that contains the expirment object information to allow for reproducibility, to pick up where a user left off, and
        to make information about the expirement readable.
        """
        if not quiet:
            print("Writing Config File.....")
        ## check if files exist
        vgrid_path = None
        hgrid_path = None
        if os.path.exists(self.mom_input_dir / "vcoord.nc"):
            vgrid_path = self.mom_input_dir / "vcoord.nc"
        if os.path.exists(self.mom_input_dir / "hgrid.nc"):
            hgrid_path = self.mom_input_dir / "hgrid.nc"

        try:
            date_range = [
                self.date_range[0].strftime("%Y-%m-%d"),
                self.date_range[1].strftime("%Y-%m-%d"),
            ]
        except:
            date_range = None
        config_dict = {
            "name": self.expt_name,
            "date_range": date_range,
            "latitude_extent": self.latitude_extent,
            "longitude_extent": self.longitude_extent,
            "run_dir": str(self.mom_run_dir),
            "input_dir": str(self.mom_input_dir),
            "toolpath_dir": str(self.toolpath_dir),
            "resolution": self.resolution,
            "number_vertical_layers": self.number_vertical_layers,
            "layer_thickness_ratio": self.layer_thickness_ratio,
            "depth": self.depth,
            "grid_type": self.hgrid_type,
            "repeat_year_forcing": self.repeat_year_forcing,
            "ocean_mask": self.ocean_mask,
            "layout": self.layout,
            "minimum_depth": self.minimum_depth,
            "vgrid": str(vgrid_path),
            "hgrid": str(hgrid_path),
            "bathymetry": self.bathymetry_property,
            "ocean_state": self.ocean_state_boundaries,
            "tides": self.tides_boundaries,
            "initial_conditions": self.initial_condition,
            "tidal_constituents": self.tidal_constituents,
        }
        if export:
            if path is not None:
                export_path = path
            else:
                export_path = self.mom_run_dir / "rmom6_config.json"
            with open(export_path, "w") as f:
                json.dump(
                    config_dict,
                    f,
                    indent=4,
                )
        if not quiet:
            print("Done.")
        return config_dict

    def setup_initial_condition(
        self,
        raw_ic_path,
        varnames,
        arakawa_grid="A",
        vcoord_type="height",
    ):
        """
        Reads the initial condition from files in ``ic_path``, interpolates to the
        model grid, fixes up metadata, and saves back to the input directory.

        Args:
            raw_ic_path (Union[str, Path,list of str]): Path(s) to raw initial condition file(s) to read in.
            varnames (Dict[str, str]): Mapping from MOM6 variable/coordinate names to the names
                in the input dataset. For example, ``{'xq': 'lonq', 'yh': 'lath', 'salt': 'so', ...}``.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the initial condition.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
            vcoord_type (Optional[str]): The type of vertical coordinate used in the forcing files.
                Either ``'height'`` or ``'thickness'``.
        """

        # Remove time dimension if present in the IC.
        # Assume that the first time dim is the intended on if more than one is present

        ic_raw = xr.open_mfdataset(raw_ic_path)
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
                "Error in reading in initial condition tracers. Terminating!"
            )
        try:
            ic_raw_u = ic_raw[varnames["u"]]
            ic_raw_v = ic_raw[varnames["v"]]
        except:
            raise ValueError(
                "Error in reading in initial condition tracers. Terminating!"
            )

        try:
            ic_raw_eta = ic_raw[varnames["eta"]]
        except:
            raise ValueError(
                "Error in reading in initial condition tracers. Terminating!"
            )

        ## if min(temperature) > 100 then assume that units must be degrees K
        ## (otherwise we can't be on Earth) and convert to degrees C
        if np.nanmin(ic_raw[varnames["tracers"]["temp"]]) > 100:
            ic_raw[varnames["tracers"]["temp"]] -= 273.15
            ic_raw[varnames["tracers"]["temp"]].attrs["units"] = "degrees Celsius"

        # Rename all coordinates to have 'lon' and 'lat' to work with the xesmf regridder
        if arakawa_grid == "A":
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
                    "Can't find required coordinates in initial condition.\n\n"
                    + "Given that arakawa_grid is 'A' the 'x' and 'y' coordinates should be provided"
                    + "in the varnames dictionary. For example, {'x': 'lon', 'y': 'lat'}.\n\n"
                    + "Terminating!"
                )

        if arakawa_grid == "B":
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
                    "Can't find coordinates in initial condition.\n\n"
                    + "Given that arakawa_grid is 'B' the names of the cell centers ('xh' & 'yh')"
                    + "as well as the cell edges ('xq' & 'yq') coordinates should be provided in "
                    + "the varnames dictionary. For example, {'xh': 'lonh', 'yh': 'lath', ...}.\n\n"
                    + "Terminating!"
                )
        if arakawa_grid == "C":
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
                    "Can't find coordinates in initial condition.\n\n"
                    + "Given that arakawa_grid is 'C' the names of the cell centers ('xh' & 'yh')"
                    + "as well as the cell edges ('xq' & 'yq') coordinates should be provided in "
                    + "in the varnames dictionary. For example, {'xh': 'lonh', 'yh': 'lath', ...}.\n\n"
                    + "Terminating!"
                )

        ## Construct the xq, yh and xh, yq grids
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

        ## Construct the cell centre grid for tracers (xh, yh).
        tgrid = (
            self.hgrid[["x", "y"]]
            .isel(nxp=slice(1, None, 2), nyp=slice(1, None, 2))
            .rename({"x": "lon", "y": "lat", "nxp": "nx", "nyp": "ny"})
            .set_coords(["lat", "lon"])
        )

        # NaNs might be here from the land mask of the model that the IC has come from.
        # If they're not removed then the coastlines from this other grid will be retained!
        # The land mask comes from the bathymetry file, so we don't need NaNs
        # to tell MOM6 where the land is.
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

        ## Make our three horizontal regridders
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

        print("Regridding Velocities... ", end="")

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

        print("Done.\nRegridding Tracers... ", end="")

        tracers_out = (
            xr.merge(
                [
                    regridder_t(ic_raw_tracers[varnames["tracers"][i]]).rename(i)
                    for i in varnames["tracers"]
                ]
            )
            .rename({"lon": "xh", "lat": "yh", varnames["zl"]: "zl"})
            .transpose("zl", "ny", "nx")
        )

        # tracers_out = tracers_out.assign_coords(
        #     {"nx":np.arange(tracers_out.sizes["nx"]).astype(float),
        #      "ny":np.arange(tracers_out.sizes["ny"]).astype(float)})

        tracers_out = tracers_out.assign_coords(
            {
                "nx": np.arange(tracers_out.sizes["nx"]).astype(float),
                "ny": np.arange(tracers_out.sizes["ny"]).astype(float),
            }
        )

        print("Done.\nRegridding Free surface... ", end="")

        eta_out = (
            regridder_t(ic_raw_eta)
            .rename({"lon": "xh", "lat": "yh"})
            .rename("eta_t")
            .transpose("ny", "nx")
        )  ## eta_t is the name set in MOM_input by default
        print("Done.")

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
            self.mom_input_dir / "init_vel.nc",
            mode="w",
            encoding={
                "u": {"_FillValue": netCDF4.default_fillvals["f4"]},
                "v": {"_FillValue": netCDF4.default_fillvals["f4"]},
            },
        )

        tracers_out.to_netcdf(
            self.mom_input_dir / "init_tracers.nc",
            mode="w",
            encoding={
                # "xh": {"_FillValue": None},
                # "yh": {"_FillValue": None},
                # "zl": {"_FillValue": None},
                "temp": {"_FillValue": -1e20, "missing_value": -1e20},
                "salt": {"_FillValue": -1e20, "missing_value": -1e20},
            },
        )
        eta_out.to_netcdf(
            self.mom_input_dir / "init_eta.nc",
            mode="w",
            encoding={
                # "xh": {"_FillValue": None},
                # "yh": {"_FillValue": None},
                "eta_t": {"_FillValue": None},
            },
        )

        self.ic_eta = eta_out
        self.ic_tracers = tracers_out
        self.ic_vels = vel_out

        print("done setting up initial condition.")

        return

    def get_glorys_rectangular(
        self, raw_boundaries_path, boundaries=["south", "north", "west", "east"]
    ):
        """
        This function is a wrapper for `get_glorys_data`, calling this function once for each of the rectangular boundary segments and the initial condition. For more complex boundary shapes, call `get_glorys_data` directly for each of your boundaries that aren't parallel to lines of constant latitude or longitude. For example, for an angled Northern boundary that spans multiple latitudes, you'll need to download a wider rectangle containing the entire boundary.

        args:
            raw_boundaries_path (str): Path to the directory containing the raw boundary forcing files.
            boundaries (List[str]): List of cardinal directions for which to create boundary forcing files.
                Default is `["south", "north", "west", "east"]`.
        """

        # Initial Condition
        get_glorys_data(
            longitude_extent=[float(self.hgrid.x.min()), float(self.hgrid.x.max())],
            latitude_extent=[float(self.hgrid.y.min()), float(self.hgrid.y.max())],
            timerange=[
                self.date_range[0],
                self.date_range[0] + datetime.timedelta(days=1),
            ],
            segment_name="ic_unprocessed",
            download_path=raw_boundaries_path,
            modify_existing=False,  # This is the first line, so start bash script anew
        )
        if "east" in boundaries:
            get_glorys_data(
                longitude_extent=[
                    float(self.hgrid.x.isel(nxp=-1).min()),
                    float(self.hgrid.x.isel(nxp=-1).max()),
                ],  ## Collect from Eastern (x = -1) side
                latitude_extent=[
                    float(self.hgrid.y.isel(nxp=-1).min()),
                    float(self.hgrid.y.isel(nxp=-1).max()),
                ],
                timerange=self.date_range,
                segment_name="east_unprocessed",
                download_path=raw_boundaries_path,
            )
        if "west" in boundaries:
            get_glorys_data(
                longitude_extent=[
                    float(self.hgrid.x.isel(nxp=0).min()),
                    float(self.hgrid.x.isel(nxp=0).max()),
                ],  ## Collect from Western (x = 0) side
                latitude_extent=[
                    float(self.hgrid.y.isel(nxp=0).min()),
                    float(self.hgrid.y.isel(nxp=0).max()),
                ],
                timerange=self.date_range,
                segment_name="west_unprocessed",
                download_path=raw_boundaries_path,
            )
        if "south" in boundaries:
            get_glorys_data(
                longitude_extent=[
                    float(self.hgrid.x.isel(nyp=0).min()),
                    float(self.hgrid.x.isel(nyp=0).max()),
                ],  ## Collect from Southern (y = 0) side
                latitude_extent=[
                    float(self.hgrid.y.isel(nyp=0).min()),
                    float(self.hgrid.y.isel(nyp=0).max()),
                ],
                timerange=self.date_range,
                segment_name="south_unprocessed",
                download_path=raw_boundaries_path,
            )
        if "north" in boundaries:
            get_glorys_data(
                longitude_extent=[
                    float(self.hgrid.x.isel(nyp=-1).min()),
                    float(self.hgrid.x.isel(nyp=-1).max()),
                ],  ## Collect from Southern (y = -1) side
                latitude_extent=[
                    float(self.hgrid.y.isel(nyp=-1).min()),
                    float(self.hgrid.y.isel(nyp=-1).max()),
                ],
                timerange=self.date_range,
                segment_name="north_unprocessed",
                download_path=raw_boundaries_path,
            )

        print(
            f"script `get_glorys_data.sh` has been created at {raw_boundaries_path}.\n Run this script via bash to download the data from a terminal with internet access. \nYou will need to enter your Copernicus Marine username and password.\nIf you don't have an account, make one here:\nhttps://data.marine.copernicus.eu/register"
        )
        return

    def setup_ocean_state_boundaries(
        self,
        raw_boundaries_path,
        varnames,
        boundaries=["south", "north", "west", "east"],
        arakawa_grid="A",
        boundary_type="rectangular",
    ):
        """
        This function is a wrapper for `simple_boundary`. Given a list of up to four cardinal directions,
        it creates a boundary forcing file for each one. Ensure that the raw boundaries are all saved in the same directory,
        and that they are named using the format `east_unprocessed.nc`

        Args:
            raw_boundaries_path (str): Path to the directory containing the raw boundary forcing files.
            varnames (Dict[str, str]): Mapping from MOM6 variable/coordinate names to the name in the
                input dataset.
            boundaries (List[str]): List of cardinal directions for which to create boundary forcing files.
                Default is `["south", "north", "west", "east"]`.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the boundary forcing.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
            boundary_type (Optional[str]): Type of box around region. Currently, only ``'rectangular'`` is supported.
        """
        for i in boundaries:
            if i not in ["south", "north", "west", "east"]:
                raise ValueError(
                    f"Invalid boundary direction: {i}. Must be one of ['south', 'north', 'west', 'east']"
                )

        if len(boundaries) < 4:
            print(
                "NOTE: the 'setup_run_directories' method assumes that you have four boundaries. You'll need to modify the MOM_input file manually to reflect the number of boundaries you have, and their orientations. You should be able to find the relevant section in the MOM_input file by searching for 'segment_'. Ensure that the segment names match those in your inputdir/forcing folder"
            )

        if len(boundaries) > 4:
            raise ValueError(
                "This method only supports up to four boundaries. To set up more complex boundary shapes you can manually call the 'simple_boundary' method for each boundary."
            )
        if boundary_type != "rectangular":
            raise ValueError(
                "Only rectangular boundaries are supported by this method. To set up more complex boundary shapes you can manually call the 'simple_boundary' method for each boundary."
            )
        # Now iterate through our four boundaries
        for orientation in boundaries:
            self.setup_single_boundary(
                Path(
                    os.path.join(
                        (raw_boundaries_path), (orientation + "_unprocessed.nc")
                    )
                ),
                varnames,
                orientation,  # The cardinal direction of the boundary
                find_MOM6_rectangular_orientation(
                    orientation
                ),  # A number to identify the boundary; indexes from 1
                arakawa_grid=arakawa_grid,
            )

    def setup_single_boundary(
        self,
        path_to_bc,
        varnames,
        orientation,
        segment_number,
        arakawa_grid="A",
        boundary_type="simple",
    ):
        """
        Here 'simple' refers to boundaries that are parallel to lines of constant longitude or latitude.
        Set up a boundary forcing file for a given orientation.

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
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the boundary forcing.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
            boundary_type (Optional[str]): Type of boundary. Currently, only ``'simple'`` is supported. Here 'simple' refers to boundaries that are parallel to lines of constant longitude or latitude.
        """

        print("Processing {} boundary...".format(orientation), end="")
        if not path_to_bc.exists():
            raise FileNotFoundError(
                f"Boundary file not found at {path_to_bc}. Please ensure that the files are named in the format `east_unprocessed.nc`."
            )
        if boundary_type != "simple":
            raise ValueError("Only simple boundaries are supported by this method.")
        seg = segment(
            hgrid=self.hgrid,
            infile=path_to_bc,  # location of raw boundary
            outfolder=self.mom_input_dir,
            varnames=varnames,
            segment_name="segment_{:03d}".format(segment_number),
            orientation=orientation,  # orienataion
            startdate=self.date_range[0],
            arakawa_grid=arakawa_grid,
            repeat_year_forcing=self.repeat_year_forcing,
        )

        seg.regrid_velocity_tracers()

        # Save Segment to Experiment
        self.segments[orientation] = seg
        print("Done.")
        return

    def setup_boundary_tides(
        self,
        path_to_td,
        tidal_filename,
        tidal_constituents="read_from_expt_init",
        boundary_type="rectangle",
    ):
        """
        This function:
        We subset our tidal data and generate more boundary files!

        Args:
            path_to_td (str): Path to boundary tidal file.
            tidal_filename: Name of the tpxo product that's used in the tidal_filename. Should be h_{tidal_filename}, u_{tidal_filename}
            tidal_constiuents: List of tidal constituents to include in the regridding. Default is [0] which is the M2 constituent.
            boundary_type (Optional[str]): Type of boundary. Currently, only ``'rectangle'`` is supported. Here 'rectangle' refers to boundaries that are parallel to lines of constant longitude or latitude.
        Returns:
            *.nc files: Regridded tidal velocity and elevation files in 'inputdir/forcing'

        General Description:
        This tidal data functions are sourced from the GFDL NWA25 and changed in the following ways:
         - Converted code for RM6 segment class
         - Implemented Horizontal Subsetting
         - Combined all functions of NWA25 into a four function process (in the style of rm6) (expt.setup_tides_rectangular_boundaries, segment.coords, segment.regrid_tides, segment.encode_tidal_files_and_output)


        Original Code was sourced from:
        Author(s): GFDL, James Simkins, Rob Cermak, etc..
        Year: 2022
        Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
        Version: N/A
        Type: Python Functions, Source Code
        Web Address: https://github.com/jsimkins2/nwa25
        """
        if boundary_type != "rectangle":
            raise ValueError(
                "Only rectangular boundaries are supported by this method."
            )
        if tidal_constituents != "read_from_expt_init":
            self.tidal_constituents = tidal_constituents
        tpxo_h = (
            xr.open_dataset(Path(path_to_td / f"h_{tidal_filename}"))
            .rename({"lon_z": "lon", "lat_z": "lat", "nc": "constituent"})
            .isel(
                constituent=convert_to_tpxo_tidal_constituents(self.tidal_constituents)
            )
        )

        h = tpxo_h["ha"] * np.exp(-1j * np.radians(tpxo_h["hp"]))
        tpxo_h["hRe"] = np.real(h)
        tpxo_h["hIm"] = np.imag(h)
        tpxo_u = (
            xr.open_dataset(Path(path_to_td / f"u_{tidal_filename}"))
            .rename({"lon_u": "lon", "lat_u": "lat", "nc": "constituent"})
            .isel(
                constituent=convert_to_tpxo_tidal_constituents(self.tidal_constituents)
            )
        )
        tpxo_u["ua"] *= 0.01  # convert to m/s
        u = tpxo_u["ua"] * np.exp(-1j * np.radians(tpxo_u["up"]))
        tpxo_u["uRe"] = np.real(u)
        tpxo_u["uIm"] = np.imag(u)
        tpxo_v = (
            xr.open_dataset(Path(path_to_td / f"u_{tidal_filename}"))
            .rename({"lon_v": "lon", "lat_v": "lat", "nc": "constituent"})
            .isel(
                constituent=convert_to_tpxo_tidal_constituents(self.tidal_constituents)
            )
        )
        tpxo_v["va"] *= 0.01  # convert to m/s
        v = tpxo_v["va"] * np.exp(-1j * np.radians(tpxo_v["vp"]))
        tpxo_v["vRe"] = np.real(v)
        tpxo_v["vIm"] = np.imag(v)
        times = xr.DataArray(
            pd.date_range(
                self.date_range[0], periods=1
            ),  # Import pandas for this shouldn't be a big deal b/c it's already required in rm6 dependencies
            dims=["time"],
        )
        boundaries = ["south", "north", "west", "east"]

        # Initialize or find boundary segment
        for b in boundaries:
            print("Processing {} boundary...".format(b), end="")

            # If the GLORYS ocean_state has already created segments, we don't create them again.
            if b not in self.segments:
                seg = segment(
                    hgrid=self.hgrid,
                    infile=None,  # location of raw boundary
                    outfolder=self.mom_input_dir,
                    varnames=None,
                    segment_name="segment_{:03d}".format(
                        find_MOM6_rectangular_orientation(b)
                    ),
                    orientation=b,  # orienataion
                    startdate=self.date_range[0],
                    repeat_year_forcing=self.repeat_year_forcing,
                )
            else:
                seg = self.segments[b]

            # Output and regrid tides
            seg.regrid_tides(tpxo_v, tpxo_u, tpxo_h, times)
            print("Done")

    def setup_bathymetry(
        self,
        *,
        bathymetry_path,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",  # This is to match GEBCO
        fill_channels=False,
        positive_down=False,
    ):
        """
        Cut out and interpolate the chosen bathymetry and then fill inland lakes.

        It's also possible to optionally fill narrow channels (see ``fill_channels``
        below), although narrow channels are less of an issue for models that are
        discretized on an Arakawa C grid, like MOM6.

        Output is saved in the input directory of the experiment.

        Args:
            bathymetry_path (str): Path to the netCDF file with the bathymetry.
            longitude_coordinate_name (Optional[str]): The name of the longitude coordinate in the bathymetry
                dataset at ``bathymetry_path``. For example, for GEBCO bathymetry: ``'lon'`` (default).
            latitude_coordinate_name (Optional[str]): The name of the latitude coordinate in the bathymetry
                dataset at ``bathymetry_path``. For example, for GEBCO bathymetry: ``'lat'`` (default).
            vertical_coordinate_name (Optional[str]): The name of the vertical coordinate in the bathymetry
                dataset at ``bathymetry_path``. For example, for GEBCO bathymetry: ``'elevation'`` (default).
            fill_channels (Optional[bool]): Whether or not to fill in
                diagonal channels. This removes more narrow inlets,
                but can also connect extra islands to land. Default: ``False``.
            positive_down (Optional[bool]): If ``True``, it assumes that
                bathymetry vertical coordinate is positive down. Default: ``False``.
        """

        ## Convert the provided coordinate names into a dictionary mapping to the
        ## coordinate names that MOM6 expects.
        coordinate_names = {
            "xh": longitude_coordinate_name,
            "yh": latitude_coordinate_name,
            "depth": vertical_coordinate_name,
        }

        bathymetry = xr.open_dataset(bathymetry_path, chunks="auto")[
            coordinate_names["depth"]
        ]

        bathymetry = bathymetry.sel(
            {
                coordinate_names["yh"]: slice(
                    self.latitude_extent[0] - 0.5, self.latitude_extent[1] + 0.5
                )
            }  # 0.5 degree latitude buffer (hardcoded) for regridding
        ).astype("float")

        ## Check if the original bathymetry provided has a longitude extent that goes around the globe
        ## to take care of the longitude seam when we slice out the regional domain.

        horizontal_resolution = (
            bathymetry[coordinate_names["xh"]][1]
            - bathymetry[coordinate_names["xh"]][0]
        )

        horizontal_extent = (
            bathymetry[coordinate_names["xh"]][-1]
            - bathymetry[coordinate_names["xh"]][0]
            + horizontal_resolution
        )

        longitude_buffer = 0.5  # 0.5 degree longitude buffer (hardcoded) for regridding

        if np.isclose(horizontal_extent, 360):
            ## longitude extent that goes around the globe -- use longitude_slicer
            bathymetry = longitude_slicer(
                bathymetry,
                np.array(self.longitude_extent)
                + np.array([-longitude_buffer, longitude_buffer]),
                coordinate_names["xh"],
            )
        else:
            ## otherwise, slice normally
            bathymetry = bathymetry.sel(
                {
                    coordinate_names["xh"]: slice(
                        self.longitude_extent[0] - longitude_buffer,
                        self.longitude_extent[1] + longitude_buffer,
                    )
                }
            )

        bathymetry.attrs["missing_value"] = -1e20  # missing value expected by FRE tools
        bathymetry_output = xr.Dataset({"depth": bathymetry})
        bathymetry.close()

        bathymetry_output = bathymetry_output.rename(
            {coordinate_names["xh"]: "lon", coordinate_names["yh"]: "lat"}
        )
        bathymetry_output.lon.attrs["units"] = "degrees_east"
        bathymetry_output.lat.attrs["units"] = "degrees_north"
        bathymetry_output.depth.attrs["_FillValue"] = -1e20
        bathymetry_output.depth.attrs["units"] = "meters"
        bathymetry_output.depth.attrs["standard_name"] = (
            "height_above_reference_ellipsoid"
        )
        bathymetry_output.depth.attrs["long_name"] = "Elevation relative to sea level"
        bathymetry_output.depth.attrs["coordinates"] = "lon lat"
        bathymetry_output.to_netcdf(
            self.mom_input_dir / "bathymetry_original.nc", mode="w", engine="netcdf4"
        )

        tgrid = xr.Dataset(
            data_vars={
                "depth": (
                    ["nx", "ny"],
                    np.zeros(
                        self.hgrid.x.isel(
                            nxp=slice(1, None, 2), nyp=slice(1, None, 2)
                        ).shape
                    ),
                )
            },
            coords={
                "lon": (
                    ["nx", "ny"],
                    self.hgrid.x.isel(
                        nxp=slice(1, None, 2), nyp=slice(1, None, 2)
                    ).values,
                ),
                "lat": (
                    ["nx", "ny"],
                    self.hgrid.y.isel(
                        nxp=slice(1, None, 2), nyp=slice(1, None, 2)
                    ).values,
                ),
            },
        )

        # rewrite chunks to use lat/lon now for use with xesmf
        tgrid.lon.attrs["units"] = "degrees_east"
        tgrid.lon.attrs["_FillValue"] = 1e20
        tgrid.lat.attrs["units"] = "degrees_north"
        tgrid.lat.attrs["_FillValue"] = 1e20
        tgrid.depth.attrs["units"] = "meters"
        tgrid.depth.attrs["coordinates"] = "lon lat"
        tgrid.to_netcdf(
            self.mom_input_dir / "bathymetry_unfinished.nc", mode="w", engine="netcdf4"
        )
        tgrid.close()

        bathymetry_output = bathymetry_output.load()

        print(
            "Begin regridding bathymetry...\n\n"
            + f"Original bathymetry size: {bathymetry_output.nbytes/1e6:.2f} Mb\n"
            + f"Regridded size: {tgrid.nbytes/1e6:.2f} Mb\n"
            + "Automatic regridding may fail if your domain is too big! If this process hangs or crashes,"
            + "open a terminal with appropriate computational and resources try calling ESMF "
            + f"directly in the input directory {self.mom_input_dir} via\n\n"
            + "`mpirun -np NUMBER_OF_CPUS ESMF_Regrid -s bathymetry_original.nc -d bathymetry_unfinished.nc -m bilinear --src_var elevation --dst_var elevation --netcdf4 --src_regional --dst_regional`\n\n"
            + "For details see https://xesmf.readthedocs.io/en/latest/large_problems_on_HPC.html\n\n"
            + "Afterwards, run the 'expt.tidy_bathymetry' method to skip the expensive interpolation step, and finishing metadata, encoding and cleanup.\n\n\n"
        )
        regridder = xe.Regridder(bathymetry_output, tgrid, "bilinear", parallel=False)
        bathymetry = regridder(bathymetry_output)
        bathymetry.to_netcdf(
            self.mom_input_dir / "bathymetry_unfinished.nc", mode="w", engine="netcdf4"
        )
        print(
            "Regridding successful! Now calling `tidy_bathymetry` method for some finishing touches..."
        )

        self.tidy_bathymetry(fill_channels, positive_down)
        print("setup bathymetry has finished successfully.")
        return

    def tidy_bathymetry(
        self, fill_channels=False, positive_down=False, vertical_coordinate_name="depth"
    ):
        """
        An auxiliary function for bathymetry used to fix up the metadata and remove inland
        lakes after regridding the bathymetry. Having `tidy_bathymetry` as a separate
        method from :func:`~setup_bathymetry` allows for the regridding to be done separately,
        since regridding can be really expensive for large domains.

        If the bathymetry is already regridded and what is left to be done is fixing the metadata
        or fill in some channels, then call this function directly to read in the existing
        ``bathymetry_unfinished.nc`` file that should be in the input directory.

        Args:
            fill_channels (Optional[bool]): Whether to fill in
                diagonal channels. This removes more narrow inlets,
                but can also connect extra islands to land. Default: ``False``.
            positive_down (Optional[bool]): If ``False`` (default), assume that
                bathymetry vertical coordinate is positive down, as is the case in GEBCO for example.
        """

        ## reopen bathymetry to modify
        print(
            "Tidy bathymetry: Reading in regridded bathymetry to fix up metadata...",
            end="",
        )
        bathymetry = xr.open_dataset(
            self.mom_input_dir / "bathymetry_unfinished.nc", engine="netcdf4"
        )

        ## Ensure correct encoding
        bathymetry = xr.Dataset(
            {"depth": (["ny", "nx"], bathymetry[vertical_coordinate_name].values)}
        )
        bathymetry.attrs["depth"] = "meters"
        bathymetry.attrs["standard_name"] = "bathymetric depth at T-cell centers"
        bathymetry.attrs["coordinates"] = "zi"

        bathymetry.expand_dims("tiles", 0)

        if not positive_down:
            ## Ensure that coordinate is positive down!
            bathymetry["depth"] *= -1

        ## Make a land mask based on the bathymetry
        ocean_mask = xr.where(bathymetry.copy(deep=True).depth <= 0, 0, 1)
        land_mask = np.abs(ocean_mask - 1)

        ## REMOVE INLAND LAKES
        print("done. Filling in inland lakes and channels... ", end="")

        changed = True  ## keeps track of whether solution has converged or not

        forward = True  ## only useful for iterating through diagonal channel removal. Means iteration goes SW -> NE

        while changed == True:
            ## First fill in all lakes.
            ## scipy.ndimage.binary_fill_holes fills holes made of 0's within a field of 1's
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

        bathymetry["depth"] *= self.ocean_mask

        ## Now, any points in the bathymetry that are shallower than minimum depth are set to minimum depth.
        ## This preserves the true land/ocean mask.
        bathymetry["depth"] = bathymetry["depth"].where(bathymetry["depth"] > 0, np.nan)
        bathymetry["depth"] = bathymetry["depth"].where(
            ~(bathymetry.depth <= self.minimum_depth), self.minimum_depth + 0.1
        )

        bathymetry.expand_dims({"ntiles": 1}).to_netcdf(
            self.mom_input_dir / "bathymetry.nc",
            mode="w",
            encoding={"depth": {"_FillValue": None}},
        )

        print("done.")
        return

    def run_FRE_tools(self, layout=None):
        """A wrapper for FRE Tools ``check_mask``, ``make_solo_mosaic``, and ``make_quick_mosaic``.
        User provides processor ``layout`` tuple of processing units.
        """

        print(
            "Running GFDL's FRE Tools. The following information is all printed by the FRE tools themselves"
        )
        if not (self.mom_input_dir / "bathymetry.nc").exists():
            print("No bathymetry file! Need to run setup_bathymetry method first")
            return

        for p in self.mom_input_dir.glob("mask_table*"):
            p.unlink()

        print(
            "OUTPUT FROM MAKE SOLO MOSAIC:",
            subprocess.run(
                str(self.toolpath_dir / "make_solo_mosaic/make_solo_mosaic")
                + " --num_tiles 1 --dir . --mosaic_name ocean_mosaic --tile_file hgrid.nc",
                shell=True,
                cwd=self.mom_input_dir,
            ),
            sep="\n\n",
        )

        print(
            "OUTPUT FROM QUICK MOSAIC:",
            subprocess.run(
                str(self.toolpath_dir / "make_quick_mosaic/make_quick_mosaic")
                + " --input_mosaic ocean_mosaic.nc --mosaic_name grid_spec --ocean_topog bathymetry.nc",
                shell=True,
                cwd=self.mom_input_dir,
            ),
            sep="\n\n",
        )

        if layout != None:
            self.configure_cpu_layout(layout)

    def configure_cpu_layout(self, layout):
        """
        Wrapper for the ``check_mask`` function of GFDL's FRE Tools. User provides processor
        ``layout`` tuple of processing units.
        """

        print(
            "OUTPUT FROM CHECK MASK:\n\n",
            subprocess.run(
                str(self.toolpath_dir / "check_mask/check_mask")
                + f" --grid_file ocean_mosaic.nc --ocean_topog bathymetry.nc --layout {layout[0]},{layout[1]} --halo 4",
                shell=True,
                cwd=self.mom_input_dir,
            ),
        )
        self.layout = layout
        return

    def setup_run_directory(
        self,
        surface_forcing=None,
        using_payu=False,
        overwrite=False,
        with_tides=False,
        boundaries=["south", "north", "west", "east"],
    ):
        """
        Set up the run directory for MOM6. Either copy a pre-made set of files, or modify
        existing files in the 'rundir' directory for the experiment.

        Args:
            surface_forcing (Optional[str]): Specify the choice of surface forcing, one
                of: ``'jra'`` or ``'era5'``. If not prescribed then constant fluxes are used.
            using_payu (Optional[bool]): Whether or not to use payu (https://github.com/payu-org/payu)
                to run the model. If ``True``, a payu configuration file will be created.
                Default: ``False``.
            overwrite (Optional[bool]): Whether or not to overwrite existing files in the
                run directory. If ``False`` (default), will only modify the ``MOM_layout`` file and will
                not re-copy across the rest of the default files.
        """

        ## Get the path to the regional_mom package on this computer
        premade_rundir_path = Path(
            importlib.resources.files("regional_mom6")
            / "demos"
            / "premade_run_directories"
        )
        if not premade_rundir_path.exists():
            print("Could not find premade run directories at ", premade_rundir_path)
            print(
                "Perhaps the package was imported directly rather than installed with conda. Checking if this is the case... "
            )

            premade_rundir_path = Path(
                importlib.resources.files("regional_mom6").parent
                / "demos"
                / "premade_run_directories"
            )
            if not premade_rundir_path.exists():
                raise ValueError(
                    f"Cannot find the premade run directory files at {premade_rundir_path} either.\n\n"
                    + "There may be an issue with package installation. Check that the `premade_run_directory` folder is present in one of these two locations"
                )
            else:
                print("Found run files. Continuing...")

        # Define the locations of the directories we'll copy files across from. Base contains most of the files, and overwrite replaces files in the base directory.
        base_run_dir = Path(premade_rundir_path / "common_files")
        if not premade_rundir_path.exists():
            raise ValueError(
                f"Cannot find the premade run directory files at {premade_rundir_path}.\n\n"
                + "These files missing might be indicating an error during the package installation!"
            )
        if surface_forcing:
            overwrite_run_dir = Path(premade_rundir_path / f"{surface_forcing}_surface")

            if not overwrite_run_dir.exists():
                available = [x for x in premade_rundir_path.iterdir() if x.is_dir()]
                raise ValueError(
                    f"Surface forcing {surface_forcing} not available. Please choose from {str(available)}"  ##Here print all available run directories
                )
        else:
            ## In case there is additional forcing (e.g., tides) then we need to modify the run dir to include the additional forcing.
            overwrite_run_dir = False

        # Check if we can implement tides
        if with_tides:
            tidal_files_exist = any(
                "tidal" in filename
                for filename in (
                    os.listdir(Path(self.mom_input_dir / "forcing"))
                    + os.listdir(Path(self.mom_input_dir))
                )
            )
            if not tidal_files_exist:
                raise (
                    "No files with 'tidal' in their names found in the forcing or input directory. If you meant to use tides, please run the setup_tides_rectangle_boundaries method first. That does output some tidal files."
                )

        # 3 different cases to handle:
        #   1. User is creating a new run directory from scratch. Here we copy across all files and modify.
        #   2. User has already created a run directory, and wants to modify it. Here we only modify the MOM_layout file.
        #   3. User has already created a run directory, and wants to overwrite it. Here we copy across all files and modify. This requires overwrite = True

        if not overwrite:
            for file in base_run_dir.glob(
                "*"
            ):  ## copy each file individually if it doesn't already exist
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
                shutil.copytree(base_run_dir, self.mom_run_dir, dirs_exist_ok=True)

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
            layout = (
                x,
                y,
            )  # This is a local variable keeping track of the layout as read from the mask table. Not to be confused with self.layout which is unchanged and may differ.

            print(
                f"Mask table {p.name} read. Using this to infer the cpu layout {layout}, total masked out cells {masked}, and total number of CPUs {ncpus}."
            )
        # Case where there's no mask table. Either because user hasn't run FRE tools, or because the domain is mostly water.
        if mask_table == None:
            # Here we define a local copy of the layout just for use within this function.
            # This prevents the layout from being overwritten in the main class in case
            # in case the user accidentally loads in the wrong mask table.
            layout = self.layout
            if layout == None:
                print(
                    "WARNING: No mask table found, and the cpu layout has not been set. \nAt least one of these is requiret to set up the experiment if you're running MOM6 standalone with the FMS coupler. \nIf you're running within CESM, ignore this message."
                )
            else:
                print(
                    f"No mask table found, but the cpu layout has been set to {self.layout} This suggests the domain is mostly water, so there are "
                    + "no `non compute` cells that are entirely land. If this doesn't seem right, "
                    + "ensure you've already run the `FRE_tools` method which sets up the cpu mask table. Keep an eye on any errors that might print while"
                    + "the FRE tools (which run C++ in the background) are running."
                )

                ncpus = layout[0] * layout[1]
                print("Number of CPUs required: ", ncpus)

        ## Modify the MOM_layout file to have correct horizontal dimensions and CPU layout
        # TODO Re-implement with package that works for this file type? or at least tidy up code
        MOM_layout_dict = self.read_MOM_file_as_dict("MOM_layout")
        if "MASKTABLE" in MOM_layout_dict.keys():
            if mask_table != None:
                MOM_layout_dict["MASKTABLE"]["value"] = mask_table
            else:
                MOM_layout_dict["MASKTABLE"]["value"] = "# MASKTABLE = no mask table"
        if (
            "LAYOUT" in MOM_layout_dict.keys()
            and "IO" not in MOM_layout_dict.keys()
            and layout != None
        ):
            MOM_layout_dict["LAYOUT"]["value"] = str(layout[1]) + "," + str(layout[0])
        if "NIGLOBAL" in MOM_layout_dict.keys():
            MOM_layout_dict["NIGLOBAL"]["value"] = self.hgrid.nx.shape[0] // 2
        if "NJGLOBAL" in MOM_layout_dict.keys():
            MOM_layout_dict["NJGLOBAL"]["value"] = self.hgrid.ny.shape[0] // 2
        self.write_MOM_file(MOM_layout_dict)

        MOM_input_dict = self.read_MOM_file_as_dict("MOM_input")
        MOM_override_dict = self.read_MOM_file_as_dict("MOM_override")
        # The number of boundaries is reflected in the number of segments setup in setup_ocean_state_boundary under expt.segments.
        # The setup_tides_boundaries function currently only works with rectangular grids amd sets up 4 segments, but DOESN"T save them to expt.segments.
        # Therefore, we can use expt.segments to determine how many segments we need for MOM_input. We can fill the empty segments with a empty string to make sure it is overriden correctly.

        # Others
        MOM_override_dict["MINIMUM_DEPTH"]["value"] = float(self.minimum_depth)
        MOM_override_dict["NK"]["value"] = len(self.vgrid.zl.values)

        # OBC Adjustments

        # Delete MOM_input OBC stuff that is indexed because we want them only in MOM_override.
        print(
            "Deleting indexed OBC keys from MOM_input_dict in case we have a different number of segments"
        )
        keys_to_delete = [key for key in MOM_input_dict if "_SEGMENT_00" in key]
        for key in keys_to_delete:
            del MOM_input_dict[key]

        # Define number of OBC segments
        MOM_override_dict["OBC_NUMBER_OF_SEGMENTS"]["value"] = len(
            boundaries
        )  # This means that each SEGMENT_00{num} has to be configured to point to the right file, which based on our other functions needs to be specified.

        # More OBC Consts
        MOM_override_dict["OBC_FREESLIP_VORTICITY"]["value"] = "False"
        MOM_override_dict["OBC_FREESLIP_STRAIN"]["value"] = "False"
        MOM_override_dict["OBC_COMPUTED_VORTICITY"]["value"] = "True"
        MOM_override_dict["OBC_COMPUTED_STRAIN"]["value"] = "True"
        MOM_override_dict["OBC_ZERO_BIHARMONIC"]["value"] = "True"
        MOM_override_dict["OBC_TRACER_RESERVOIR_LENGTH_SCALE_OUT"]["value"] = "3.0E+04"
        MOM_override_dict["OBC_TRACER_RESERVOIR_LENGTH_SCALE_IN"]["value"] = "3000.0"
        MOM_override_dict["BRUSHCUTTER_MODE"]["value"] = "True"

        # Define Specific Segments
        for ind, seg in enumerate(boundaries):
            ind_seg = ind + 1
            key_start = "OBC_SEGMENT_00" + str(ind_seg)
            ## Position and Config
            key_POSITION = key_start
            if find_MOM6_rectangular_orientation(seg) == 1:
                index_str = '"J=0,I=0:N'
            elif find_MOM6_rectangular_orientation(seg) == 2:
                index_str = '"J=N,I=N:0'
            elif find_MOM6_rectangular_orientation(seg) == 3:
                index_str = '"I=0,J=N:0'
            elif find_MOM6_rectangular_orientation(seg) == 4:
                index_str = '"I=N,J=0:N'
            MOM_override_dict[key_POSITION]["value"] = (
                index_str + ',FLATHER,ORLANSKI,NUDGED,ORLANSKI_TAN,NUDGED_TAN"'
            )

            # Nudging Key
            key_NUDGING = key_start + "_VELOCITY_NUDGING_TIMESCALES"
            MOM_override_dict[key_NUDGING]["value"] = "0.3, 360.0"

            # Data Key
            key_DATA = key_start + "_DATA"
            file_num_obc = str(
                find_MOM6_rectangular_orientation(seg)
            )  # 1,2,3,4 for rectangular boundaries, BUT if we have less than 4 segments we use the index to specific the number, but keep filenames as if we had four boundaries
            MOM_override_dict[key_DATA][
                "value"
            ] = f'"U=file:forcing_obc_segment_00{file_num_obc}.nc(u),V=file:forcing_obc_segment_00{file_num_obc}.nc(v),SSH=file:forcing_obc_segment_00{file_num_obc}.nc(eta),TEMP=file:forcing_obc_segment_00{file_num_obc}.nc(temp),SALT=file:forcing_obc_segment_00{file_num_obc}.nc(salt)'
            if with_tides:
                MOM_override_dict[key_DATA]["value"] = (
                    MOM_override_dict[key_DATA]["value"]
                    + f',Uamp=file:tu_segment_00{file_num_obc}.nc(uamp),Uphase=file:tu_segment_00{file_num_obc}.nc(uphase),Vamp=file:tu_segment_00{file_num_obc}.nc(vamp),Vphase=file:tu_segment_00{file_num_obc}.nc(vphase),SSHamp=file:tz_segment_00{file_num_obc}.nc(zamp),SSHphase=file:tz_segment_00{file_num_obc}.nc(zphase)"'
                )
            else:
                MOM_override_dict[key_DATA]["value"] = (
                    MOM_override_dict[key_DATA]["value"] + '"'
                )

        # Tides OBC adjustments
        if with_tides:

            # Include internal tide forcing
            MOM_override_dict["TIDES"]["value"] = "True"

            # OBC tides
            MOM_override_dict["OBC_TIDE_ADD_EQ_PHASE"]["value"] = "True"
            MOM_override_dict["OBC_TIDE_N_CONSTITUENTS"]["value"] = len(
                self.tidal_constituents
            )
            MOM_override_dict["OBC_TIDE_CONSTITUENTS"]["value"] = (
                '"' + ", ".join(self.tidal_constituents) + '"'
            )
            MOM_override_dict["OBC_TIDE_REF_DATE"]["value"] = (
                str(self.date_range[0].year)
                + ", "
                + str(self.date_range[0].month)
                + ", "
                + str(self.date_range[0].day)
            )

        for key in MOM_override_dict.keys():
            if type(MOM_override_dict[key]) == dict:
                MOM_override_dict[key]["override"] = True
        self.write_MOM_file(MOM_input_dict)
        self.write_MOM_file(MOM_override_dict)

        ## If using payu to run the model, create a payu configuration file
        if not using_payu and os.path.exists(f"{self.mom_run_dir}/config.yaml"):
            os.remove(f"{self.mom_run_dir}/config.yaml")
        elif ncpus == None:
            print(
                "WARNING: Layout has not been set! Cannot create payu configuration file. Run the FRE_tools first."
            )
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
            self.date_range[0].year,
            self.date_range[0].month,
            self.date_range[0].day,
            0,
            0,
            0,
        ]
        nml.write(self.mom_run_dir / "input.nml", force=True)
        return

    def change_MOM_parameter(
        self, param_name, param_value=None, comment=None, delete=False
    ):
        """
        *Requires already copied MOM parameter files in the run directory*
        Change a parameter in the MOM_input or MOM_override file. Returns original value if there was one.
        If delete is specified, ONLY MOM_override version will be deleted. Deleting from MOM_input is not safe.
        If the parameter does not exist, it will be added to the file. if delete is set to True, the parameter will be removed.
        Args:
            param_name (str):
                Parameter name we are working with
            param_value (Optional[str]):
                New Assigned Value
            comment (Optional[str]):
                Any comment to add
            delete (Optional[bool]):
                Whether to delete the specified param_name

        """
        if not delete and param_value is None:
            raise ValueError(
                "If not deleting a parameter, you must specify a new value for it."
            )

        MOM_input_dict = self.read_MOM_file_as_dict("MOM_input")
        MOM_override_dict = self.read_MOM_file_as_dict("MOM_override")
        original_val = "No original val"
        if not delete:
            # We don't want to keep any parameters in MOM_input that we change. We want to clearly list them in MOM_override.
            if param_name in MOM_input_dict.keys():
                original_val = MOM_override_dict[param_name]["value"]
                print("Removing original value {} from MOM_input".format(original_val))
                del MOM_input_dict[param_name]
            if param_name in MOM_override_dict.keys():
                original_val = MOM_override_dict[param_name]["value"]
                print(
                    "This parameter {} is begin replaced from {} to {} in MOM_override".format(
                        param_name, original_val, param_value
                    )
                )

            MOM_override_dict[param_name]["value"] = param_value
            MOM_override_dict[param_name]["comment"] = comment
        else:
            if param_name in MOM_override_dict.keys():
                original_val = MOM_override_dict[param_name]["value"]
                print("Deleting parameter {} from MOM_override".format(param_name))
                del MOM_override_dict[param_name]
            else:
                print(
                    "Key to be deleted {} was not in MOM_override to begin with.".format(
                        param_name
                    )
                )
        self.write_MOM_file(MOM_input_dict)
        self.write_MOM_file(MOM_override_dict)
        return original_val

    def read_MOM_file_as_dict(self, filename):
        """
        Read the MOM_input file and return a dictionary of the variables and their values.
        """

        # Default information for each parameter
        default_layout = {"value": None, "override": False, "comment": None}

        if not os.path.exists(Path(self.mom_run_dir / filename)):
            raise ValueError(
                f"File {filename} does not exist in the run directory {self.mom_run_dir}"
            )
        with open(Path(self.mom_run_dir / filename), "r") as file:
            lines = file.readlines()

            # Set the default initialization for a new key
            MOM_file_dict = defaultdict(lambda: default_layout.copy())
            MOM_file_dict["filename"] = filename
            dlc = default_layout.copy()
            for jj in range(len(lines)):
                if "=" in lines[jj] and not "===" in lines[jj]:
                    split = lines[jj].split("=", 1)
                    var = split[0]
                    value = split[1]
                    if "#override" in var:
                        var = var.split("#override")[1].strip()
                        dlc["override"] = True
                    else:
                        dlc["override"] = False
                    if "!" in value:
                        dlc["comment"] = value.split("!")[1]
                        value = value.split("!")[0].strip()  # Remove Comments
                        dlc["value"] = str(value)
                    else:
                        dlc["value"] = str(value.strip())
                        dlc["comment"] = None

                    MOM_file_dict[var.strip()] = dlc.copy()

            # Save a copy of the original dictionary
            MOM_file_dict["original"] = MOM_file_dict.copy()
        return MOM_file_dict

    def write_MOM_file(self, MOM_file_dict):
        """
        Write the MOM_input file from a dictionary of variables and their values. Does not support removing fields.
        """
        # Replace specific variable values
        original_MOM_file_dict = MOM_file_dict.pop("original")
        with open(Path(self.mom_run_dir / MOM_file_dict["filename"]), "r") as file:
            lines = file.readlines()
            for jj in range(len(lines)):
                if "=" in lines[jj] and not "===" in lines[jj]:
                    var = lines[jj].split("=", 1)[0].strip()
                    if var in MOM_file_dict.keys() and (
                        str(MOM_file_dict[var]["value"])
                    ) != str(original_MOM_file_dict[var]["value"]):
                        lines[jj] = lines[jj].replace(
                            str(original_MOM_file_dict[var]["value"]),
                            str(MOM_file_dict[var]["value"]),
                        )
                        lines[jj] = lines[jj].replace(
                            original_MOM_file_dict[var]["comment"],
                            str(MOM_file_dict[var]["comment"]),
                        )
                        print(
                            "Changed",
                            var,
                            "from",
                            original_MOM_file_dict[var],
                            "to",
                            MOM_file_dict[var],
                            "in {}!".format(MOM_file_dict["filename"]),
                        )

        # Add new fields
        lines.append("! === Added with RM6 ===\n")
        for key in MOM_file_dict.keys():
            if key not in original_MOM_file_dict.keys():
                if MOM_file_dict[key]["override"]:
                    lines.append(
                        f"#override {key} = {MOM_file_dict[key]['value']} !{MOM_file_dict[key]['comment']}\n"
                    )
                else:
                    lines.append(
                        f"{key} = {MOM_file_dict[key]['value']} !{MOM_file_dict[key]['comment']}\n"
                    )
                print(
                    "Added",
                    key,
                    "to",
                    MOM_file_dict["filename"],
                    "with value",
                    MOM_file_dict[key],
                )

        # Check any fields removed
        for key in original_MOM_file_dict.keys():
            if key not in MOM_file_dict.keys():
                search_words = [
                    key,
                    original_MOM_file_dict[key]["value"],
                    original_MOM_file_dict[key]["comment"],
                ]
                lines = [
                    line
                    for line in lines
                    if not all(word in line for word in search_words)
                ]
                print(
                    "Removed",
                    key,
                    "in",
                    MOM_file_dict["filename"],
                    "with value",
                    original_MOM_file_dict[key],
                )

        with open(Path(self.mom_run_dir / MOM_file_dict["filename"]), "w") as f:
            f.writelines(lines)

    def setup_era5(self, era5_path):
        """
        Setup the ERA5 forcing files for the experiment. This assumes that
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
                i for i in range(self.date_range[0].year, self.date_range[1].year + 1)
            ]
            # construct a list of all paths for all years to use for open_mfdataset
            paths_per_year = [Path(era5_path / fname / year) for year in years]
            all_files = []
            for path in paths_per_year:
                # Use glob to find all files that match the pattern
                files = list(path.glob(f"{fname}*.nc"))
                # Add the files to the all_files list
                all_files.extend(files)

            ds = xr.open_mfdataset(
                all_files,
                decode_times=False,
                chunks={"longitude": 100, "latitude": 100},
            )

            ## Cut out this variable to our domain size
            rawdata[fname] = longitude_slicer(
                ds,
                self.longitude_extent,
                "longitude",
            ).sel(
                latitude=slice(
                    self.latitude_extent[1], self.latitude_extent[0]
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
                "units": f"hours since {self.date_range[0].strftime('%Y-%m-%d %H:%M:%S')}",
            }  ## Fix up calendar to match

            if fname == "2d":
                ## Calculate specific humidity from dewpoint temperature
                dewpoint = 8.07131 - 1730.63 / (233.426 + rawdata["2d"]["d2m"] - 273.15)
                humidity = (0.622 / rawdata["sp"]["sp"]) * (10**dewpoint) * 101325 / 760
                q = xr.Dataset(data_vars={"q": humidity})

                q.q.attrs = {"long_name": "Specific Humidity", "units": "kg/kg"}
                q.to_netcdf(
                    f"{self.mom_input_dir}/q_ERA5.nc",
                    unlimited_dims="time",
                    encoding={"q": {"dtype": "double"}},
                )
            elif fname == "crr":
                ## Calculate total rain rate from convective and total
                trr = xr.Dataset(
                    data_vars={"trr": rawdata["crr"]["crr"] + rawdata["lsrr"]["lsrr"]}
                )

                trr.trr.attrs = {
                    "long_name": "Total Rain Rate",
                    "units": "kg m**-2 s**-1",
                }
                trr.to_netcdf(
                    f"{self.mom_input_dir}/trr_ERA5.nc",
                    unlimited_dims="time",
                    encoding={"trr": {"dtype": "double"}},
                )

            elif fname == "lsrr":
                ## This is handled by crr as both are added together to calculate total rain rate.
                pass
            else:
                rawdata[fname].to_netcdf(
                    f"{self.mom_input_dir}/{fname}_ERA5.nc",
                    unlimited_dims="time",
                    encoding={vname: {"dtype": "double"}},
                )


class segment:
    """
    Class to turn raw boundary and tidal segment data into MOM6 boundary
    and tidal segments.

    Boundary segments should only contain the necessary data for that
    segment. No horizontal chunking is done here, so big fat segments
    will process slowly.

    Data should be at daily temporal resolution, iterating upwards
    from the provided startdate. Function ignores the time metadata
    and puts it on Julian calendar.

    Note:
        Only supports z-star (z*) vertical coordinate.

    Args:
        hgrid (xarray.Dataset): The horizontal grid used for domain.
        infile (Union[str, Path]): Path to the raw, unprocessed boundary segment.
        outfolder (Union[str, Path]): Path to folder where the model inputs will
            be stored.
        varnames (Dict[str, str]): Mapping between the variable/dimension names and
            standard naming convention of this pipeline, e.g., ``{"xq": "longitude,
            "yh": "latitude", "salt": "salinity", ...}``. Key "tracers" points to nested
            dictionary of tracers to include in boundary.
        segment_name (str): Name of the segment, e.g., ``'segment_001'``.
        orientation (str): Cardinal direction (lowercase) of the boundary segment,
            i.e., ``'east'``, ``'west'``, ``'north'``, or ``'south'``.
        startdate (str): The starting date to use in the segment calendar.
        arakawa_grid (Optional[str]): Arakawa grid staggering type of the boundary forcing.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
        time_units (str): The units used by the raw forcing files, e.g., ``hours``,
            ``days`` (default).
        repeat_year_forcing (Optional[bool]): When ``True`` the experiment runs with repeat-year
            forcing. When ``False`` (default) then inter-annual forcing is used.
    """

    def __init__(
        self,
        *,
        hgrid,
        infile,
        outfolder,
        varnames,
        segment_name,
        orientation,
        startdate,
        arakawa_grid="A",
        time_units="days",
        repeat_year_forcing=False,
    ):
        ## Store coordinate names
        if arakawa_grid == "A" and infile is not None:
            try:
                self.x = varnames["x"]
                self.y = varnames["y"]
            ## In case user continues using T point names for A grid
            except:
                self.x = varnames["xh"]
                self.y = varnames["yh"]

        elif arakawa_grid in ("B", "C"):
            self.xq = varnames["xq"]
            self.xh = varnames["xh"]
            self.yq = varnames["yq"]
            self.yh = varnames["yh"]

        ## Store velocity names
        if infile is not None:
            self.u = varnames["u"]
            self.v = varnames["v"]
            self.z = varnames["zl"]
            self.eta = varnames["eta"]
            self.time = varnames["time"]
        self.startdate = startdate

        ## Store tracer names
        if infile is not None:
            self.tracers = varnames["tracers"]
        self.time_units = time_units

        ## Store other data
        orientation = orientation.lower()
        if orientation not in ("north", "south", "east", "west"):
            raise ValueError(
                "orientation must be one of: 'north', 'south', 'east', or 'west'"
            )
        self.orientation = orientation

        if arakawa_grid not in ("A", "B", "C"):
            raise ValueError("arakawa_grid must be one of: 'A', 'B', or 'C'")
        self.arakawa_grid = arakawa_grid

        self.infile = infile
        self.outfolder = outfolder
        self.hgrid = hgrid
        self.segment_name = segment_name
        self.repeat_year_forcing = repeat_year_forcing

    @property
    def coords(self):
        """


        This function:
        Allows us to call the self.coords for use in the xesmf.Regridder in the regrid_tides function. self.coords gives us the subset of the hgrid based on the orientation.

        Args:
            None
        Returns:
            xr.Dataset: The correct coordinate space for the orientation

        General Description:
        This tidal data functions are sourced from the GFDL NWA25 and changed in the following ways:
         - Converted code for RM6 segment class
         - Implemented Horizontal Subsetting
         - Combined all functions of NWA25 into a four function process (in the style of rm6) (expt.setup_tides_rectangular_boundaries, segment.coords, segment.regrid_tides, segment.encode_tidal_files_and_output)


        Code adapted from:
        Author(s): GFDL, James Simkins, Rob Cermak, etc..
        Year: 2022
        Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
        Version: N/A
        Type: Python Functions, Source Code
        Web Address: https://github.com/jsimkins2/nwa25

        """
        # Rename nxp and nyp to locations
        if self.orientation == "south":
            rcoord = xr.Dataset(
                {
                    "lon": self.hgrid["x"].isel(nyp=0),
                    "lat": self.hgrid["y"].isel(nyp=0),
                    "angle": self.hgrid["angle_dx"].isel(nyp=0),
                }
            )
            rcoord = rcoord.rename_dims({"nxp": f"nx_{self.segment_name}"})
            rcoord.attrs["perpendicular"] = "ny"
            rcoord.attrs["parallel"] = "nx"
            rcoord.attrs["axis_to_expand"] = (
                2  ## Need to keep track of which axis the 'main' coordinate corresponds to when re-adding the 'secondary' axis
            )
            rcoord.attrs["locations_name"] = (
                f"nx_{self.segment_name}"  # Legacy name of nx_... was locations. This provides a clear transform in regrid_tides
            )
        elif self.orientation == "north":
            rcoord = xr.Dataset(
                {
                    "lon": self.hgrid["x"].isel(nyp=-1),
                    "lat": self.hgrid["y"].isel(nyp=-1),
                    "angle": self.hgrid["angle_dx"].isel(nyp=-1),
                }
            )
            rcoord = rcoord.rename_dims({"nxp": f"nx_{self.segment_name}"})
            rcoord.attrs["perpendicular"] = "ny"
            rcoord.attrs["parallel"] = "nx"
            rcoord.attrs["axis_to_expand"] = 2
            rcoord.attrs["locations_name"] = f"nx_{self.segment_name}"
        elif self.orientation == "west":
            rcoord = xr.Dataset(
                {
                    "lon": self.hgrid["x"].isel(nxp=0),
                    "lat": self.hgrid["y"].isel(nxp=0),
                    "angle": self.hgrid["angle_dx"].isel(nxp=0),
                }
            )
            rcoord = rcoord.rename_dims({"nyp": f"ny_{self.segment_name}"})
            rcoord.attrs["perpendicular"] = "nx"
            rcoord.attrs["parallel"] = "ny"
            rcoord.attrs["axis_to_expand"] = 3
            rcoord.attrs["locations_name"] = f"ny_{self.segment_name}"
        elif self.orientation == "east":
            rcoord = xr.Dataset(
                {
                    "lon": self.hgrid["x"].isel(nxp=-1),
                    "lat": self.hgrid["y"].isel(nxp=-1),
                    "angle": self.hgrid["angle_dx"].isel(nxp=-1),
                }
            )
            rcoord = rcoord.rename_dims({"nyp": f"ny_{self.segment_name}"})
            rcoord.attrs["perpendicular"] = "nx"
            rcoord.attrs["parallel"] = "ny"
            rcoord.attrs["axis_to_expand"] = 3
            rcoord.attrs["locations_name"] = f"ny_{self.segment_name}"

        # Make lat and lon coordinates
        rcoord = rcoord.assign_coords(lat=rcoord["lat"], lon=rcoord["lon"])

        return rcoord

    def rotate(self, u, v):
        # Make docstring

        """
        Rotate the velocities to the grid orientation.

        Args:
            u (xarray.DataArray): The u-component of the velocity.
            v (xarray.DataArray): The v-component of the velocity.

        Returns:
            Tuple[xarray.DataArray, xarray.DataArray]: The rotated u and v components of the velocity.
        """

        angle = self.coords.angle.values * np.pi / 180
        u_rot = u * np.cos(angle) - v * np.sin(angle)
        v_rot = u * np.sin(angle) + v * np.cos(angle)
        return u_rot, v_rot

    def regrid_velocity_tracers(self):
        """
        Cut out and interpolate the velocities and tracers
        """

        rawseg = xr.open_dataset(self.infile, decode_times=False, engine="netcdf4")

        if self.arakawa_grid == "A":
            rawseg = rawseg.rename({self.x: "lon", self.y: "lat"})
            ## In this case velocities and tracers all on same points
            regridder = xe.Regridder(
                rawseg[self.u],
                self.coords,
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

        if self.arakawa_grid == "B":
            ## All tracers on one grid, all velocities on another
            regridder_velocity = xe.Regridder(
                rawseg[self.u].rename({self.xq: "lon", self.yq: "lat"}),
                self.coords,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_velocity_weights_{self.orientation}.nc",
            )

            regridder_tracer = xe.Regridder(
                rawseg[self.tracers["salt"]].rename({self.xh: "lon", self.yh: "lat"}),
                self.coords,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_tracer_weights_{self.orientation}.nc",
            )

            velocities_out = regridder_velocity(
                rawseg[[self.u, self.v]].rename({self.xq: "lon", self.yq: "lat"})
            )

            velocities_out["u"], velocities_out["v"] = self.rotate(
                velocities_out["u"], velocities_out["v"]
            )

            segment_out = xr.merge(
                [
                    velocities_out,
                    regridder_tracer(
                        rawseg[
                            [self.eta] + [self.tracers[i] for i in self.tracers]
                        ].rename({self.xh: "lon", self.yh: "lat"})
                    ),
                ]
            )

        if self.arakawa_grid == "C":
            ## All tracers on one grid, all velocities on another
            regridder_uvelocity = xe.Regridder(
                rawseg[self.u].rename({self.xq: "lon", self.yh: "lat"}),
                self.coords,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_uvelocity_weights_{self.orientation}.nc",
            )

            regridder_vvelocity = xe.Regridder(
                rawseg[self.v].rename({self.xh: "lon", self.yq: "lat"}),
                self.coords,
                "bilinear",
                locstream_out=True,
                reuse_weights=False,
                filename=self.outfolder
                / f"weights/bilinear_vvelocity_weights_{self.orientation}.nc",
            )

            regridder_tracer = xe.Regridder(
                rawseg[self.tracers["salt"]].rename({self.xh: "lon", self.yh: "lat"}),
                self.coords,
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
            np.nanmin(segment_out[self.tracers["temp"]].isel({self.time: 0, self.z: 0}))
            > 100
        ):
            segment_out[self.tracers["temp"]] -= 273.15
            segment_out[self.tracers["temp"]].attrs["units"] = "degrees Celsius"

        # fill in NaNs
        segment_out = (
            segment_out.ffill(self.z)
            .interpolate_na(f"{self.coords.attrs['parallel']}_{self.segment_name}")
            .ffill(f"{self.coords.attrs['parallel']}_{self.segment_name}")
            .bfill(f"{self.coords.attrs['parallel']}_{self.segment_name}")
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
            f"nx_{self.segment_name}": {
                "dtype": "int32",
            },
            f"ny_{self.segment_name}": {
                "dtype": "int32",
            },
        }

        ### Generate the dz variable; needs to be in layer thicknesses
        dz = segment_out[self.z].diff(self.z)
        dz.name = "dz"
        dz = xr.concat([dz, dz[-1]], dim=self.z)

        # Here, keep in mind that 'var' keeps track of the mom6 variable names we want, and self.tracers[var]
        # will return the name of the variable from the original data

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
            v = f"{var}_{self.segment_name}"
            ## Rename each variable in dataset
            segment_out = segment_out.rename({allfields[var]: v})

            ## Rename vertical coordinate for this variable
            segment_out[f"{var}_{self.segment_name}"] = segment_out[
                f"{var}_{self.segment_name}"
            ].rename({self.z: f"nz_{self.segment_name}_{var}"})

            ## Replace the old depth coordinates with incremental integers
            segment_out[f"nz_{self.segment_name}_{var}"] = np.arange(
                segment_out[f"nz_{self.segment_name}_{var}"].size
            )

            ## Re-add the secondary dimension (even though it represents one value..)
            segment_out[v] = segment_out[v].expand_dims(
                f"{self.coords.attrs['perpendicular']}_{self.segment_name}",
                axis=self.coords.attrs["axis_to_expand"],
            )

            ## Add the layer thicknesses
            segment_out[f"dz_{v}"] = (
                [
                    "time",
                    f"nz_{v}",
                    f"ny_{self.segment_name}",
                    f"nx_{self.segment_name}",
                ],
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
            encoding_dict[f"nz_{self.segment_name}_{var}"] = {"dtype": "int32"}

        ## Treat eta separately since it has no vertical coordinate. Do the same things as for the surface variables above
        segment_out = segment_out.rename({self.eta: f"eta_{self.segment_name}"})
        encoding_dict[f"eta_{self.segment_name}"] = {
            "_FillValue": netCDF4.default_fillvals["f8"],
        }
        segment_out[f"eta_{self.segment_name}"] = segment_out[
            f"eta_{self.segment_name}"
        ].expand_dims(
            f"{self.coords.attrs['perpendicular']}_{self.segment_name}",
            axis=self.coords.attrs["axis_to_expand"] - 1,
        )

        # Overwrite the actual lat/lon values in the dimensions, replace with incrementing integers
        segment_out[f"{self.coords.attrs['parallel']}_{self.segment_name}"] = np.arange(
            segment_out[f"{self.coords.attrs['parallel']}_{self.segment_name}"].size
        )
        segment_out[f"{self.coords.attrs['perpendicular']}_{self.segment_name}"] = [0]
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

        # Store actual lat/lon values here as variables rather than coordinates
        segment_out[f"lon_{self.segment_name}"] = (
            [f"ny_{self.segment_name}", f"nx_{self.segment_name}"],
            self.coords.lon.expand_dims(
                dim="blank", axis=self.coords.attrs["axis_to_expand"] - 2
            ).data,
        )
        segment_out[f"lat_{self.segment_name}"] = (
            [f"ny_{self.segment_name}", f"nx_{self.segment_name}"],
            self.coords.lon.expand_dims(
                dim="blank", axis=self.coords.attrs["axis_to_expand"] - 2
            ).data,
        )

        # Add units to the lat / lon to keep the `categorize_axis_from_units` checker from throwing warnings
        segment_out[f"lat_{self.segment_name}"].attrs = {
            "units": "degrees_north",
        }
        segment_out[f"lon_{self.segment_name}"].attrs = {
            "units": "degrees_east",
        }
        segment_out[f"ny_{self.segment_name}"].attrs = {
            "units": "degrees_north",
        }
        segment_out[f"nx_{self.segment_name}"].attrs = {
            "units": "degrees_east",
        }
        # If repeat-year forcing, add modulo coordinate
        if self.repeat_year_forcing:
            segment_out["time"] = segment_out["time"].assign_attrs({"modulo": " "})

        with ProgressBar():
            segment_out.load().to_netcdf(
                self.outfolder / f"forcing_obc_{self.segment_name}.nc",
                encoding=encoding_dict,
                unlimited_dims="time",
            )

        return segment_out, encoding_dict

    def regrid_tides(
        self, tpxo_v, tpxo_u, tpxo_h, times, method="nearest_s2d", periodic=False
    ):
        """
        This function:
        Regrids and interpolates the tidal data for MOM6, originally inspired by GFDL NWA25 repo code & edited by Ashley.
        - Read in raw tidal data (all constituents)
        - Perform minor transformations/conversions
        - Regridded the tidal elevation, and tidal velocity
        - Encoding the output

        Args:
            infile_td (str): Raw Tidal File/Dir
            tpxo_v, tpxo_u, tpxo_h (xarray.Dataset): Specific adjusted for MOM6 tpxo datasets (Adjusted with setup_tides)
            times (pd.DateRange): The start date of our model period
        Returns:
            *.nc files: Regridded tidal velocity and elevation files in 'inputdir/forcing'

        General Description:
        This tidal data functions are sourced from the GFDL NWA25 and changed in the following ways:
         - Converted code for RM6 segment class
         - Implemented Horizontal Subsetting
         - Combined all functions of NWA25 into a four function process (in the style of rm6) (expt.setup_tides_rectangular_boundaries, segment.coords, segment.regrid_tides, segment.encode_tidal_files_and_output)


        Original Code was sourced from:
        Author(s): GFDL, James Simkins, Rob Cermak, etc..
        Year: 2022
        Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
        Version: N/A
        Type: Python Functions, Source Code
        Web Address: https://github.com/jsimkins2/nwa25
        """

        ########## Tidal Elevation: Horizontally interpolate elevation components ############
        regrid = xe.Regridder(
            tpxo_h[["lon", "lat", "hRe"]],
            self.coords,
            method="nearest_s2d",
            locstream_out=True,
            periodic=False,
            filename=Path(
                self.outfolder / "forcing" / f"regrid_{self.segment_name}_tidal_elev.nc"
            ),
            reuse_weights=False,
        )
        redest = regrid(tpxo_h[["lon", "lat", "hRe"]])
        imdest = regrid(tpxo_h[["lon", "lat", "hIm"]])

        # Fill missing data.
        # Need to do this first because complex would get converted to real
        redest = redest.ffill(dim=self.coords.attrs["locations_name"], limit=None)[
            "hRe"
        ]
        imdest = imdest.ffill(dim=self.coords.attrs["locations_name"], limit=None)[
            "hIm"
        ]

        # Convert complex
        cplex = redest + 1j * imdest

        # Convert to real amplitude and phase.
        ds_ap = xr.Dataset({f"zamp_{self.segment_name}": np.abs(cplex)})
        # np.angle doesn't return dataarray
        ds_ap[f"zphase_{self.segment_name}"] = (
            ("constituent", self.coords.attrs["locations_name"]),
            -1 * np.angle(cplex),
        )  # radians

        # Add time coordinate and transpose so that time is first,
        # so that it can be the unlimited dimension
        ds_ap, _ = xr.broadcast(ds_ap, times)
        ds_ap = ds_ap.transpose(
            "time", "constituent", self.coords.attrs["locations_name"]
        )

        self.encode_tidal_files_and_output(ds_ap, "tz")

        ########### Regrid Tidal Velocity ######################
        regrid_u = xe.Regridder(
            tpxo_u[["lon", "lat", "uRe"]],
            self.coords,
            method=method,
            locstream_out=True,
            periodic=periodic,
            reuse_weights=False,
        )

        regrid_v = xe.Regridder(
            tpxo_v[["lon", "lat", "vRe"]],
            self.coords,
            method=method,
            locstream_out=True,
            periodic=periodic,
            reuse_weights=False,
        )

        # Interpolate each real and imaginary parts to segment.
        uredest = regrid_u(tpxo_u[["lon", "lat", "uRe"]])["uRe"]
        uimdest = regrid_u(tpxo_u[["lon", "lat", "uIm"]])["uIm"]
        vredest = regrid_v(tpxo_v[["lon", "lat", "vRe"]])["vRe"]
        vimdest = regrid_v(tpxo_v[["lon", "lat", "vIm"]])["vIm"]

        # Fill missing data.
        # Need to do this first because complex would get converted to real
        uredest = uredest.ffill(dim=self.coords.attrs["locations_name"], limit=None)
        uimdest = uimdest.ffill(dim=self.coords.attrs["locations_name"], limit=None)
        vredest = vredest.ffill(dim=self.coords.attrs["locations_name"], limit=None)
        vimdest = vimdest.ffill(dim=self.coords.attrs["locations_name"], limit=None)

        # Convert to complex, remaining separate for u and v.
        ucplex = uredest + 1j * uimdest
        vcplex = vredest + 1j * vimdest

        # Convert complex u and v to ellipse,
        # rotate ellipse from earth-relative to model-relative,
        # and convert ellipse back to amplitude and phase.
        SEMA, ECC, INC, PHA = ap2ep(ucplex, vcplex)

        # Rotate to the model grid by adjusting the inclination.
        # Requries that angle is in radians.

        ua, va, up, vp = ep2ap(SEMA, ECC, INC, PHA)

        ds_ap = xr.Dataset(
            {f"uamp_{self.segment_name}": ua, f"vamp_{self.segment_name}": va}
        )
        # up, vp aren't dataarrays
        ds_ap[f"uphase_{self.segment_name}"] = (
            ("constituent", self.coords.attrs["locations_name"]),
            up,
        )  # radians
        ds_ap[f"vphase_{self.segment_name}"] = (
            ("constituent", self.coords.attrs["locations_name"]),
            vp,
        )  # radians

        ds_ap, _ = xr.broadcast(ds_ap, times)

        # Need to transpose so that time is first,
        # so that it can be the unlimited dimension
        ds_ap = ds_ap.transpose(
            "time", "constituent", self.coords.attrs["locations_name"]
        )

        # Some things may have become missing during the transformation
        ds_ap = ds_ap.ffill(dim=self.coords.attrs["locations_name"], limit=None)

        self.encode_tidal_files_and_output(ds_ap, "tu")

        return

    def encode_tidal_files_and_output(self, ds, filename):
        """
        This function:
         - Expands the dimensions (with the segment name)
         - Renames some dimensions to be more specific to the segment
         - Provides an output file encoding
         - Exports the files.

        Args:
            self.outfolder (str/path): The output folder to save the tidal files into
            dataset (xarray.Dataset): The processed tidal dataset
            filename (str): The output file name
        Returns:
            *.nc files: Regridded [FILENAME] files in 'self.outfolder/[filename]_[segmentname].nc'

        General Description:
        This tidal data functions are sourced from the GFDL NWA25 and changed in the following ways:
         - Converted code for RM6 segment class
         - Implemented Horizontal Subsetting
         - Combined all functions of NWA25 into a four function process (in the style of rm6) (expt.setup_tides_rectangular_boundaries, segment.coords, segment.regrid_tides, segment.encode_tidal_files_and_output)


        Original Code was sourced from:
        Author(s): GFDL, James Simkins, Rob Cermak, etc..
        Year: 2022
        Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
        Version: N/A
        Type: Python Functions, Source Code
        Web Address: https://github.com/jsimkins2/nwa25


        """

        ## Expand Tidal Dimensions ##
        if "z" in ds.coords or "constituent" in ds.dims:
            offset = 0
        else:
            offset = 1
        if self.orientation in ["south", "north"]:
            ds = ds.expand_dims(f"ny_{self.segment_name}", 2 - offset)
        elif self.orientation in ["west", "east"]:
            ds = ds.expand_dims(f"nx_{self.segment_name}", 3 - offset)

        ## Rename Tidal Dimensions ##
        ds = ds.rename(
            {"lon": f"lon_{self.segment_name}", "lat": f"lat_{self.segment_name}"}
        )
        if "z" in ds.coords:
            ds = ds.rename({"z": f"nz_{self.segment_name}"})
        if self.orientation in ["south", "north"]:
            ds = ds.rename(
                {self.coords.attrs["locations_name"]: f"nx_{self.segment_name}"}
            )
        elif self.orientation in ["west", "east"]:
            ds = ds.rename(
                {self.coords.attrs["locations_name"]: f"ny_{self.segment_name}"}
            )

        ## Perform Encoding ##
        for v in ds:
            ds[v].encoding["_FillValue"] = 1.0e20
        fname = f"{filename}_{self.segment_name}.nc"
        # Set format and attributes for coordinates, including time if it does not already have calendar attribute
        # (may change this to detect whether time is a time type or a float).
        # Need to include the fillvalue or it will be back to nan
        encoding = {
            "time": dict(_FillValue=1.0e20),
            f"lon_{self.segment_name}": dict(dtype="float64", _FillValue=1.0e20),
            f"lat_{self.segment_name}": dict(dtype="float64", _FillValue=1.0e20),
        }
        if "calendar" not in ds["time"].attrs and "modulo" not in ds["time"].attrs:
            encoding.update(
                {"time": dict(dtype="float64", calendar="gregorian", _FillValue=1.0e20)}
            )

        ## Export Files ##
        ds.to_netcdf(
            Path(self.outfolder / "forcing" / fname),
            engine="netcdf4",
            encoding=encoding,
            unlimited_dims="time",
        )
        return
