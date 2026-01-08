import numpy as np
import dask.array as da
import xarray as xr
import xesmf as xe
import subprocess
from scipy.ndimage import binary_fill_holes
import netCDF4
import f90nml
import datetime as dt
import warnings
import shutil
import os
import importlib.resources
import datetime
import pandas as pd
from pathlib import Path
import json
from regional_mom6 import MOM_parameter_tools as mpt
from regional_mom6 import regridding as rgd
from regional_mom6 import rotation as rot
from regional_mom6.config import Config
from regional_mom6.utils import (
    quadrilateral_areas,
    ap2ep,
    ep2ap,
    rotate,
    find_files_by_pattern,
)


warnings.filterwarnings("ignore")

__all__ = [
    "longitude_slicer",
    "hyperbolictan_thickness_profile",
    "generate_rectangular_hgrid",
    "experiment",
    "segment",
    "get_glorys_data",
]


## Mapping Functions


def convert_to_tpxo_tidal_constituents(tidal_constituents):
    """
    Convert tidal constituents from strings to integers using a dictionary.

    Arguments:
        tidal_constituents (list of str): List of tidal constituent names as strings.

    Returns:
        list of int: List of tidal constituent indices as integers.
    """
    tpxo_tidal_constituent_map = {
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

    try:
        constituent_indices = [
            tpxo_tidal_constituent_map[tc] for tc in tidal_constituents
        ]
    except KeyError as e:
        raise ValueError(f"Invalid tidal constituent: {e.args[0]}")

    return constituent_indices


## Auxiliary functions


def longitude_slicer(data, longitude_extent, longitude_coords):
    """
    Slices longitudes while handling periodicity and the 'seams', that is the
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

    - Add back the correct multiple of 360 so the whole domain matches the target.

    Arguments:
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

    Arguments:
        longitude_extent (tuple of floats): Westward and Eastward extents of the segment
        latitude_extent (tuple of floats): Southward and Northward extents of the segment
        timerange (tuple of datetime strings): Start and end of the segment, each in format %Y-%m-%d %H:%M:%S
        segment_range (str): name of the segment (without the ``.nc`` extension, e.g., ``east_unprocessed``)
        download_path (str): Location of where the script is saved
        modify_existing (bool): Whether to add to an existing script or start a new one
    Returns:
        file path
    """

    buffer = 0.24  # Pads download regions to ensure that interpolation onto desired domain doesn't fail.
    # Default is 0.24 degrees; just under three times the Glorys cell width (3 x 1/12 = 0.25).

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
copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m --variable so --variable thetao --variable uo --variable vo --variable zos --start-datetime {str(timerange[0]).replace(" ","T")} --end-datetime {str(timerange[1]).replace(" ","T")} --minimum-longitude {longitude_extent[0] - buffer} --maximum-longitude {longitude_extent[1] + buffer} --minimum-latitude {latitude_extent[0] - buffer} --maximum-latitude {latitude_extent[1] + buffer} --minimum-depth 0 --maximum-depth 6000 -o {str(path)} -f {segment_name}.nc\n
"""
    )
    file.writelines(lines)
    file.close()
    return Path(path / "get_glorys_data.sh")


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

    Arguments:
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

    Arguments:
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
    an existing one (see argument description below).

    Arguments:
        date_range (Tuple[str]): Start and end dates of the boundary forcing window. For
            example: ``("2003-01-01", "2003-01-31")``.
        resolution (float): Lateral resolution of the domain (in degrees).
        number_vertical_layers (int): Number of vertical layers.
        layer_thickness_ratio (float): Ratio of largest to smallest layer thickness;
            used as input in :func:`~hyperbolictan_thickness_profile`.
        depth (float): Depth of the domain.
        mom_run_dir (str): Path of the MOM6 control directory.
        mom_input_dir (str): Path of the MOM6 input directory, to receive the forcing files.
        fre_tools_dir (str): Path of GFDL's FRE tools (https://github.com/NOAA-GFDL/FRE-NCtools)
            binaries.
        longitude_extent (Tuple[float]): Extent of the region in longitude (in degrees). For
            example: ``(40.5, 50.0)``.
        latitude_extent (Tuple[float]): Extent of the region in latitude (in degrees). For
            example: ``(-20.0, 30.0)``.
        hgrid_type (str): Type of horizontal grid to generate. Currently, only ``'even_spacing'`` is supported. Setting this argument to ``'from_file'`` requires the additional hgrid_path argument
        hgrid_path (str): Path to the horizontal grid file if the hgrid_type is ``'from_file'``.
        vgrid_type (str): Type of vertical grid to generate.
            Currently, only ``'hyperbolic_tangent'`` is supported. Setting this argument to ``'from_file'`` requires the additional vgrid_path argument
        vgrid_path (str): Path to the vertical grid file if the vgrid_type is ``'from_file'``.
        repeat_year_forcing (bool): When ``True`` the experiment runs with
            repeat-year forcing. When ``False`` (default) then inter-annual forcing is used.
        minimum_depth (int): The minimum depth in meters of a grid cell allowed before it is masked out and treated as land.
        tidal_constituents (List[str]): List of tidal constituents to be used in the experiment. Default is ``["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MM", "MF"]``.
        create_empty (bool): If ``True``, the experiment object is initialized empty. This is used for testing and experienced user manipulation.
        expt_name (str): The name of the experiment (for config file use)
        boundaries (List[str]): List of (rectangular) boundaries to be set. Default is ``["south", "north", "west", "east"]``. The boundaries are set as (list index + 1) in MOM_override in the order of the list, and less than 4 boundaries can be set.
        regridding_method (str): regridding method to use throughout the entire experiment. Default is ``'bilinear'``. Any other xesmf regridding method can be used.
        fill_method (Function): The fill function to be used after regridding datasets. it takes a xarray DataArray and returns a filled DataArray. Default is ``rgd.fill_missing_data``.
    """

    @classmethod
    def create_empty(
        cls,
        longitude_extent=None,
        latitude_extent=None,
        date_range=None,
        resolution=None,
        number_vertical_layers=None,
        layer_thickness_ratio=None,
        depth=None,
        mom_run_dir=None,
        mom_input_dir=None,
        fre_tools_dir=None,
        hgrid_type="even_spacing",
        repeat_year_forcing=False,
        minimum_depth=4,
        tidal_constituents=["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MM", "MF"],
        expt_name=None,
        boundaries=["south", "north", "west", "east"],
        regridding_method="bilinear",
        fill_method=rgd.fill_missing_data,
    ):
        """
        **Note**: This method is unsafe; *only* experience users are urged to use it!

        Alternative to the initialisation method to create an empty expirement object, with the opportunity to override
        whatever values wanted.

        This method allows developers and experienced users to set specific variables for specific function requirements,
        like just regridding the initial condition or subsetting bathymetry, instead of having to set so many other variables
        that aren't needed.
        """
        expt = cls(
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
            fre_tools_dir=None,
            create_empty=True,
            hgrid_type=None,
            repeat_year_forcing=None,
            tidal_constituents=None,
            expt_name=None,
            regridding_method=None,
            fill_method=None,
        )

        expt.expt_name = expt_name
        expt.tidal_constituents = tidal_constituents
        expt.repeat_year_forcing = repeat_year_forcing
        expt.hgrid_type = hgrid_type
        expt.fre_tools_dir = fre_tools_dir
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
        expt.segments = {}
        expt.boundaries = boundaries
        expt.regridding_method = regridding_method
        expt.fill_method = fill_method
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
        fre_tools_dir=None,
        longitude_extent=None,
        latitude_extent=None,
        hgrid_type="even_spacing",
        hgrid_path=None,
        vgrid_type="hyperbolic_tangent",
        vgrid_path=None,
        repeat_year_forcing=False,
        minimum_depth=4,
        tidal_constituents=["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MM", "MF"],
        create_empty=False,
        expt_name=None,
        boundaries=["south", "north", "west", "east"],
        regridding_method="bilinear",
        fill_method=rgd.fill_missing_data,
    ):

        # Creates an empty experiment object for testing and experienced user manipulation.
        if create_empty:
            return

        # ## Set up the experiment with no config file
        ## in case list was given, convert to tuples
        self.expt_name = expt_name
        self.date_range = tuple(date_range)

        self.mom_run_dir = Path(mom_run_dir)
        self.mom_input_dir = Path(mom_input_dir)
        self.fre_tools_dir = Path(fre_tools_dir) if fre_tools_dir is not None else None

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
        self.layout = None  # This should be a tuple. Leaving it as 'None' makes it easy to remind the user to provide a value later.
        self.minimum_depth = minimum_depth  # Minimum depth allowed in the bathymetry
        self.tidal_constituents = tidal_constituents
        self.regridding_method = regridding_method
        self.fill_method = fill_method
        if hgrid_type == "from_file":
            if hgrid_path is None:
                hgrid_path = self.mom_input_dir / "hgrid.nc"
            else:
                hgrid_path = Path(hgrid_path)
            try:
                self.hgrid = xr.open_dataset(hgrid_path)
                self.longitude_extent = (
                    float(self.hgrid.x.min()),
                    float(self.hgrid.x.max()),
                )
                self.latitude_extent = (
                    float(self.hgrid.y.min()),
                    float(self.hgrid.y.max()),
                )
            except FileNotFoundError:
                if hgrid_path is None:
                    raise FileNotFoundError(
                        f"Horizontal grid {self.mom_input_dir}/hgrid.nc not found. Make sure `hgrid.nc`exists in {self.mom_input_dir} directory."
                    )
                else:
                    raise FileNotFoundError(f"Horizontal grid {hgrid_path} not found.")

        else:
            if hgrid_path:
                raise ValueError(
                    "hgrid_path can only be set if hgrid_type is 'from_file'."
                )
            self.longitude_extent = tuple(longitude_extent)
            self.latitude_extent = tuple(latitude_extent)
            self.hgrid = self._make_hgrid()

        if vgrid_type == "from_file":
            if vgrid_path is None:
                vgrid_path = self.mom_input_dir / "vgrid.nc"
            else:
                vgrid_path = Path(vgrid_path)

            try:
                vgrid_from_file = xr.open_dataset(vgrid_path)

            except FileNotFoundError:
                if vgrid_path is None:
                    raise FileNotFoundError(
                        f"Vertical grid {self.mom_input_dir}/vcoord.nc not found. Make sure `vcoord.nc`exists in {self.mom_input_dir} directory."
                    )
                else:
                    raise FileNotFoundError(f"Vertical grid {vgrid_path} not found.")

            self.vgrid = self._make_vgrid(vgrid_from_file.dz.data)
        else:
            if vgrid_path:
                raise ValueError(
                    "vgrid_path can only be set if vgrid_type is 'from_file'."
                )
            self.vgrid = self._make_vgrid()

        self.segments = {}
        self.boundaries = boundaries

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
        return json.dumps(Config.save_to_json(self, export=False), indent=4)

    @property
    def bathymetry(self):
        try:
            return xr.open_dataset(
                self.mom_input_dir / "bathymetry.nc",
                decode_cf=False,
                decode_times=False,
            )
        except Exception as e:
            print(
                f"Error: {e}. Opening bathymetry threw an error! Make sure you've successfully run the setup_bathymetry method, or copied a bathymetry.nc file into {self.mom_input_dir}."
            )
            return None

    @property
    def init_velocities(self):
        try:
            return xr.open_dataset(
                self.mom_input_dir / "init_vel.nc",
                decode_cf=False,
                decode_times=False,
            )
        except Exception as e:
            print(
                f"Error: {e}. Opening init_vel threw an error! Make sure you've successfully run the setup_initial_condition method, or copied an init_vel.nc file into {self.mom_input_dir}."
            )
            return

    @property
    def init_tracers(self):
        try:
            return xr.open_dataset(
                self.mom_input_dir / "init_tracers.nc",
                decode_cf=False,
                decode_times=False,
            )
        except Exception as e:
            print(
                f"Error: {e}. Opening init_tracers threw an error! Make sure you've successfully run the setup_initial_condition method, or copied an init_tracers.nc file into {self.mom_input_dir}."
            )
            return

    @property
    def ocean_state_boundary_paths(self):
        """
        Finds the ocean state files from disk, and prints the file paths
        """
        ocean_state_path = Path(self.mom_input_dir / "forcing")
        patterns = [
            "forcing_*",
            "weights/bi*",
        ]
        return find_files_by_pattern(
            [ocean_state_path, self.mom_input_dir],
            patterns,
            error_message="No ocean state files set up yet (or files misplaced from {}). Call `setup_ocean_state_boundaries` method to set up ocean state.".format(
                ocean_state_path
            ),
        )

    @property
    def tides_boundary_paths(self):
        """
        Finds the tides files from disk, and prints the file paths
        """
        tides_path = self.mom_input_dir / "forcing"
        patterns = ["regrid*", "tu_*", "tz_*"]
        return find_files_by_pattern(
            [tides_path, self.mom_input_dir],
            patterns,
            error_message="No tides files set up yet (or files misplaced from {}). Call `setup_boundary_tides` method to set up tides.".format(
                tides_path
            ),
        )

    @property
    def era5_paths(self):
        """
        Finds the ERA5 files from disk, and prints the file paths
        """
        era5_path = self.mom_input_dir / "forcing"
        # Use glob to find all *_ERA5.nc files
        return find_files_by_pattern(
            [era5_path],
            ["*_ERA5.nc"],
            error_message="No ERA5 files set up yet (or files misplaced from {}). Call `setup_era5` method to set up era5.".format(
                era5_path
            ),
        )

    @property
    def initial_condition_paths(self):
        """
        Finds the initial condition files from disk, and prints the file paths
        """
        forcing_path = self.mom_input_dir / "forcing"
        return find_files_by_pattern(
            [forcing_path, self.mom_input_dir],
            ["init_*.nc"],
            error_message="No initial conditions files set up yet (or files misplaced from {}). Call `setup_initial_condition` method to set up initial conditions.".format(
                forcing_path
            ),
        )

    @property
    def bathymetry_path(self):
        """
        Finds the bathymetry file from disk, and returns the file path.
        """
        if (self.mom_input_dir / "bathymetry.nc").exists():
            return str(self.mom_input_dir / "bathymetry.nc")
        else:
            return "Not Found"

    def __getattr__(self, name):

        ## First, check whether the attribute is an input file
        if "segment" in name:
            try:
                return xr.open_mfdataset(
                    str(self.mom_input_dir / f"*{name}*.nc"),
                    decode_times=False,
                    decode_cf=False,
                )
            except Exception as e:
                print(
                    f"Error: {e}. {name} files threw an error! Make sure you've successfully run the setup_ocean_state_boundaries method, or copied your own segment files into {self.mom_input_dir}."
                )
                return None

        ## If we get here, attribute wasn't found

        available_methods = [
            method for method in dir(self) if not method.startswith("__")
        ]
        error_message = f"{name} not found. Available methods and attributes are: {available_methods}"
        raise AttributeError(error_message)

    def find_MOM6_rectangular_orientation(self, input):
        """
        Convert between MOM6 boundary and the specific segment number needed, or the inverse.
        """

        direction_dir = {}
        counter = 1
        for b in self.boundaries:
            direction_dir[b] = counter
            counter += 1
        direction_dir_inv = {v: k for k, v in direction_dir.items()}
        merged_dict = {**direction_dir, **direction_dir_inv}
        try:
            val = merged_dict[input]
        except KeyError:
            raise ValueError(
                "Invalid direction or segment number for MOM6 rectangular orientation"
            )
        return val

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

    def _make_vgrid(self, thicknesses=None):
        """
        Generates a vertical grid based on the ``number_vertical_layers``, the ratio
        of largest to smallest layer thickness (``layer_thickness_ratio``) and the
        total ``depth`` parameters.
        (All these parameters are specified at the class level.)

        Arguments:
            thicknesses (Optional[np.ndarray]): An array of layer thicknesses. If not provided,
                the layer thicknesses are generated using the :func:`~hyperbolictan_thickness_profile`
                function.
        """

        if thicknesses is None:
            thicknesses = hyperbolictan_thickness_profile(
                self.number_vertical_layers, self.layer_thickness_ratio, self.depth
            )

        if not isinstance(thicknesses, np.ndarray):
            raise ValueError("thicknesses must be a numpy array")

        zi = np.cumsum(thicknesses)
        zi = np.insert(zi, 0, 0.0)  # add zi = 0.0 as first interface

        zl = zi[0:-1] + thicknesses / 2  # the mid-points between interfaces zi

        vcoord = xr.Dataset({"zi": ("zi", zi), "zl": ("zl", zl)})

        ## Check whether the minimum depth is less than the first three layers

        if len(zi) > 2 and self.minimum_depth < zi[2]:
            print(
                f"Warning: Minimum depth of {self.minimum_depth}m is less than the depth of the third interface ({zi[2]}m)!\n"
                + "This means that some areas may only have one or two layers between the surface and sea floor. \n"
                + "For increased stability, consider increasing the minimum depth, or adjusting the vertical coordinate to add more layers near the surface."
            )

        vcoord["zi"].attrs = {"units": "meters"}
        vcoord["zl"].attrs = {"units": "meters"}

        vcoord.to_netcdf(self.mom_input_dir / "vcoord.nc")

        return vcoord

    def setup_initial_condition(
        self,
        raw_ic_path,
        varnames,
        arakawa_grid="A",
        vcoord_type="height",
        rotational_method=rot.RotationMethod.EXPAND_GRID,
        regridding_method=None,
    ):
        """
        Reads the initial condition from files in ``raw_ic_path``, interpolates to the
        model grid, fixes up metadata, and saves back to the input directory.

        Arguments:
            raw_ic_path (Union[str, Path, list[str]]): Path(s) to raw initial condition file(s) to read in.
            varnames (Dict[str, str]): Mapping from MOM6 variable/coordinate names to the names
                in the input dataset. For example, ``{'xq': 'lonq', 'yh': 'lath', 'salt': 'so', ...}``.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the initial condition.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
            vcoord_type (Optional[str]): The type of vertical coordinate used in the forcing files.
                Either ``'height'`` or ``'thickness'``.
            rotational_method (Optional[RotationMethod]): The method used to rotate the velocities.
            regridding_method (Optional[str]): The type of regridding method to use. Defaults to self.regridding_method
        """
        if regridding_method is None:
            regridding_method = self.regridding_method

        reprocessed_var_map = apply_arakawa_grid_mapping(
            var_mapping=varnames, arakawa_grid=arakawa_grid
        )
        ic_raw = xr.open_mfdataset(raw_ic_path)

        # There is a case where MARBL tracers have multiple zdims, this is not supported for initial conditions:
        if type(reprocessed_var_map["depth_coord"]) == list:
            reprocessed_var_map["depth_coord"] = reprocessed_var_map["depth_coord"][0]

        # Remove time dimension if present in the IC.
        # Assume that the first time dim is the intended one if more than one is present

        if reprocessed_var_map["time_var_name"] in ic_raw.dims:
            ic_raw = ic_raw.isel({reprocessed_var_map["time_var_name"]: 0})
        if reprocessed_var_map["time_var_name"] in ic_raw.coords:
            ic_raw = ic_raw.drop(reprocessed_var_map["time_var_name"])

        # Separate out tracers from two velocity fields of IC
        try:
            ic_raw_tracers = ic_raw[
                [
                    reprocessed_var_map["tracer_var_names"][i]
                    for i in reprocessed_var_map["tracer_var_names"]
                ]
            ]
        except:
            raise ValueError(
                "Error in reading in initial condition tracers. Terminating!"
            )
        try:
            ic_raw_u = ic_raw[reprocessed_var_map["u_var_name"]]
            ic_raw_v = ic_raw[reprocessed_var_map["v_var_name"]]
        except:
            raise ValueError(
                "Error in reading in initial condition tracers. Terminating!"
            )

        try:
            ic_raw_eta = ic_raw[reprocessed_var_map["eta_var_name"]]
        except:
            raise ValueError(
                "Error in reading in initial condition tracers. Terminating!"
            )

        ## if min(temperature) > 100 then assume that units must be degrees K
        ## (otherwise we can't be on Earth) and convert to degrees C
        if np.nanmin(ic_raw[reprocessed_var_map["tracer_var_names"]["temp"]]) > 100:
            ic_raw[reprocessed_var_map["tracer_var_names"]["temp"]] -= 273.15
            ic_raw[reprocessed_var_map["tracer_var_names"]["temp"]].attrs[
                "units"
            ] = "degrees Celsius"
        # NaNs might be here from the land mask of the model that the IC has come from.
        # If they're not removed then the coastlines from this other grid will be retained!
        # The land mask comes from the bathymetry file, so we don't need NaNs
        # to tell MOM6 where the land is.
        ic_raw_tracers = (
            ic_raw_tracers.interpolate_na(
                reprocessed_var_map["tracer_x_coord"], method="linear"
            )
            .ffill(reprocessed_var_map["tracer_x_coord"])
            .bfill(reprocessed_var_map["tracer_x_coord"])
            .ffill(reprocessed_var_map["tracer_y_coord"])
            .bfill(reprocessed_var_map["tracer_y_coord"])
            .ffill(reprocessed_var_map["depth_coord"])
        )

        ic_raw_u = (
            ic_raw_u.interpolate_na(reprocessed_var_map["u_x_coord"], method="linear")
            .ffill(reprocessed_var_map["u_x_coord"])
            .bfill(reprocessed_var_map["u_x_coord"])
            .ffill(reprocessed_var_map["u_y_coord"])
            .bfill(reprocessed_var_map["u_y_coord"])
            .ffill(reprocessed_var_map["depth_coord"])
        )

        ic_raw_v = (
            ic_raw_v.interpolate_na(reprocessed_var_map["v_x_coord"], method="linear")
            .ffill(reprocessed_var_map["v_x_coord"])
            .bfill(reprocessed_var_map["v_x_coord"])
            .ffill(reprocessed_var_map["v_y_coord"])
            .bfill(reprocessed_var_map["v_y_coord"])
            .ffill(reprocessed_var_map["depth_coord"])
        )

        ic_raw_eta = (
            ic_raw_eta.interpolate_na(
                reprocessed_var_map["tracer_x_coord"], method="linear"
            )
            .ffill(reprocessed_var_map["tracer_x_coord"])
            .bfill(reprocessed_var_map["tracer_x_coord"])
            .ffill(reprocessed_var_map["tracer_y_coord"])
            .bfill(reprocessed_var_map["tracer_y_coord"])
        )

        # If the input data is on a curvilinear grid, the lat/lon values are a different dimension name then the variable dims (think velocity(depth, time, x,y) and lat(x,y))
        # So use lon/lat coord is specified for u, v, & tracers which is different than an x or y coord in each regridding (because regridding needs the lat/lon)

        ic_raw_u = ic_raw_u.rename(
            {
                reprocessed_var_map["u_lat_coord"]: "lat",
                reprocessed_var_map["u_lon_coord"]: "lon",
            }
        )

        ic_raw_v = ic_raw_v.rename(
            {
                reprocessed_var_map["v_lat_coord"]: "lat",
                reprocessed_var_map["v_lon_coord"]: "lon",
            }
        )

        ic_raw_tracers = ic_raw_tracers.rename(
            {
                reprocessed_var_map["tracer_lat_coord"]: "lat",
                reprocessed_var_map["tracer_lon_coord"]: "lon",
            }
        )

        self.hgrid["lon"] = self.hgrid["x"]
        self.hgrid["lat"] = self.hgrid["y"]
        tgrid = (
            rgd.get_hgrid_arakawa_c_points(self.hgrid, "t")
            .rename({"tlon": "lon", "tlat": "lat", "nxp": "nx", "nyp": "ny"})
            .set_coords(["lat", "lon"])
        )

        ## Make our three horizontal regridders

        regridder_u = rgd.create_regridder(
            ic_raw_u, self.hgrid, locstream_out=False, method=regridding_method
        )
        regridder_v = rgd.create_regridder(
            ic_raw_v, self.hgrid, locstream_out=False, method=regridding_method
        )
        regridder_t = rgd.create_regridder(
            ic_raw_tracers, tgrid, locstream_out=False, method=regridding_method
        )

        ## Construct the cell centre grid for tracers (xh, yh).
        print("Setting up Initial Conditions")

        ## Regrid all fields horizontally.

        print("Regridding Velocities... ", end="")
        regridded_u = regridder_u(ic_raw_u)
        regridded_v = regridder_v(ic_raw_v)
        rotated_u, rotated_v = rotate(
            regridded_u,
            regridded_v,
            radian_angle=np.radians(
                rot.get_rotation_angle(rotational_method, self.hgrid).values
            ),
        )

        # Slice the velocites to the u and v grid.
        u_points = rgd.get_hgrid_arakawa_c_points(self.hgrid, "u")
        v_points = rgd.get_hgrid_arakawa_c_points(self.hgrid, "v")
        rotated_v = rotated_v[:, v_points.v_points_y.values, v_points.v_points_x.values]
        rotated_u = rotated_u[:, u_points.u_points_y.values, u_points.u_points_x.values]
        rotated_u["lon"] = u_points.ulon
        rotated_u["lat"] = u_points.ulat
        rotated_v["lon"] = v_points.vlon
        rotated_v["lat"] = v_points.vlat

        # Merge Vels
        vel_out = xr.merge(
            [
                rotated_u.rename(
                    {
                        "lon": "xq",
                        "lat": "yh",
                        "nyp": "ny",
                        reprocessed_var_map["depth_coord"]: "zl",
                    }
                ).rename("u"),
                rotated_v.rename(
                    {
                        "lon": "xh",
                        "lat": "yq",
                        "nxp": "nx",
                        reprocessed_var_map["depth_coord"]: "zl",
                    }
                ).rename("v"),
            ]
        )

        print("Done.\nRegridding Tracers... ", end="")

        tracers_out = (
            xr.merge(
                [
                    regridder_t(
                        ic_raw_tracers[reprocessed_var_map["tracer_var_names"][i]]
                    ).rename(i)
                    for i in reprocessed_var_map["tracer_var_names"]
                ]
            )
            .rename(
                {"lon": "xh", "lat": "yh", reprocessed_var_map["depth_coord"]: "zl"}
            )
            .transpose("zl", "ny", "nx", ...)
        )

        # tracers_out = tracers_out.assign_coords(
        #     {"nx":np.arange(tracers_out.sizes["nx"]).astype(float),
        #      "ny":np.arange(tracers_out.sizes["ny"]).astype(float)})
        # Add dummy values for the nx and ny dimensions. Otherwise MOM6 complains that it's missing data??
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
        vel_out.zl.attrs = ic_raw_u[reprocessed_var_map["depth_coord"]].attrs

        tracers_out.xh.attrs = ic_raw_tracers.lon.attrs
        tracers_out.yh.attrs = ic_raw_tracers.lat.attrs
        tracers_out.zl.attrs = ic_raw_tracers[reprocessed_var_map["depth_coord"]].attrs
        for i in reprocessed_var_map["tracer_var_names"]:
            tracers_out[i].attrs = ic_raw_tracers[
                reprocessed_var_map["tracer_var_names"][i]
            ].attrs

        eta_out.xh.attrs = ic_raw_tracers.lon.attrs
        eta_out.yh.attrs = ic_raw_tracers.lat.attrs
        eta_out.attrs = ic_raw_eta.attrs

        ## Regrid the fields vertically
        if (
            vcoord_type == "thickness"
        ):  ## In this case construct the vertical profile by summing thickness
            tracers_out["zl"] = tracers_out["zl"].diff("zl")
            dz = rgd.generate_dz(tracers_out, self.z)

        # The extrapolate arg allows the initial condition to fill beyond the range of the input data.
        tracers_out = tracers_out.interp(
            {"zl": self.vgrid.zl.values}, kwargs={"fill_value": "extrapolate"}
        )
        vel_out = vel_out.interp(
            {"zl": self.vgrid.zl.values}, kwargs={"fill_value": "extrapolate"}
        )

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
                "temp": {"_FillValue": -1e20, "missing_value": -1e20},
                "salt": {"_FillValue": -1e20, "missing_value": -1e20},
            },
        )
        eta_out.to_netcdf(
            self.mom_input_dir / "init_eta.nc",
            mode="w",
            encoding={
                "eta_t": {"_FillValue": None},
            },
        )

        self.ic_eta = eta_out
        self.ic_tracers = tracers_out
        self.ic_vels = vel_out

        print("done setting up initial condition.")

        return

    def get_glorys(self, raw_boundaries_path):
        """
        This is a wrapper that calls :func:`~get_glorys_data` once for each of the rectangular boundary segments
        and the initial condition. For more complex boundary shapes, call :func:`~get_glorys_data` directly for
        each of your boundaries that aren't parallel to lines of constant latitude or longitude. For example,
        for an angled Northern boundary that spans multiple latitudes, we need to download a wider rectangle
        containing the entire boundary.

        Arguments:
            raw_boundaries_path (str): Path to the directory containing the raw boundary forcing files.
            boundaries (List[str]): List of cardinal directions for which to create boundary forcing files.
                Default is ``["south", "north", "west", "east"]``.
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
        if "east" in self.boundaries:
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
        if "west" in self.boundaries:
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
        if "south" in self.boundaries:
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
        if "north" in self.boundaries:
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
            f"The script `get_glorys_data.sh` has been generated at:\n  {raw_boundaries_path}.\n"
            f"To download the data, run this script using `bash` in a terminal with internet access.\n\n"
            f"Important instructions:\n"
            f"1. You will need your Copernicus Marine username and password.\n"
            f"   If you do not have an account, you can create one here: \n"
            f"   https://data.marine.copernicus.eu/register\n"
            f"2. You will be prompted to enter your Copernicus Marine credentials multiple times: once for each dataset.\n"
            f"3. Depending on the dataset size, the download process may take significant time and resources.\n"
            f"4. Thus, on certain systems, you may need to run this script as a batch job.\n"
        )
        return

    def setup_ocean_state_boundaries(
        self,
        raw_boundaries_path,
        varnames,
        arakawa_grid="A",
        bathymetry_path=None,
        rotational_method=rot.RotationMethod.EXPAND_GRID,
        regridding_method=None,
        fill_method=None,
    ):
        """
        A wrapper for :func:`~setup_single_boundary`. Given a list of up to four cardinal directions,
        it creates a boundary forcing file for each one. Ensure that the raw boundaries are all saved
        in the same directory, and that they are named using the format ``east_unprocessed.nc``.

        Arguments:
            raw_boundaries_path (str): Path to the directory containing the raw boundary forcing files.
            varnames (Dict[str, str]): Mapping from MOM6 variable/coordinate names to the name in the
                input dataset.
            boundaries (List[str]): List of cardinal directions for which to create boundary forcing files.
                Default is ``["south", "north", "west", "east"]``.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the boundary forcing.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
            bathymetry_path (Optional[str]): Path to the bathymetry file. Default is ``None``, in which case the
                boundary condition is not masked.
            rotational_method (Optional[str]): Method to use for rotating the boundary velocities.
                Default is ``EXPAND_GRID``.
            regridding_method (Optional[str]): The type of regridding method to use. Defaults to self.regridding_method
            fill_method (Function): Fill method to use throughout the function. Default is ``self.fill_method``
        """
        if regridding_method is None:
            regridding_method = self.regridding_method
        if fill_method is None:
            fill_method = self.fill_method
        for i in self.boundaries:
            if i not in ["south", "north", "west", "east"]:
                raise ValueError(
                    f"Invalid boundary direction: {i}. Must be one of ['south', 'north', 'west', 'east']"
                )

        if len(self.boundaries) < 4:
            print(
                "NOTE: the 'setup_run_directories' method does understand the less than four boundaries but be careful. Please check the MOM_input/override file carefully to reflect the number of boundaries you have, and their orientations. You should be able to find the relevant section in the MOM_input/override file by searching for 'segment_'. Ensure that the segment names match those in your inputdir/forcing folder"
            )

        if len(self.boundaries) > 4:
            raise ValueError(
                "This method only supports up to four boundaries. To set up more complex boundary shapes you can manually call the 'simple_boundary' method for each boundary."
            )

        # Now iterate through our four boundaries
        for orientation in self.boundaries:
            self.setup_single_boundary(
                Path(raw_boundaries_path / (orientation + "_unprocessed.nc")),
                varnames,
                orientation,  # The cardinal direction of the boundary
                self.find_MOM6_rectangular_orientation(
                    orientation
                ),  # A number to identify the boundary; indexes from 1
                arakawa_grid=arakawa_grid,
                bathymetry_path=bathymetry_path,
                rotational_method=rotational_method,
                regridding_method=regridding_method,
                fill_method=fill_method,
            )

    def setup_single_boundary(
        self,
        path_to_bc,
        varnames,
        orientation,
        segment_number,
        arakawa_grid="A",
        bathymetry_path=None,
        rotational_method=rot.RotationMethod.EXPAND_GRID,
        regridding_method=None,
        fill_method=None,
    ):
        """
        Set up a boundary forcing file for a given ``orientation``.

        Arguments:
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
            bathymetry_path (str): Path to the bathymetry file. Default is ``None``, in which case
                the boundary condition is not masked.
            rotational_method (Optional[str]): Method to use for rotating the boundary velocities.
                Default is 'EXPAND_GRID'.
            regridding_method (Optional[str]): The type of regridding method to use. Defaults to self.regridding_method
            fill_method (Function): Fill method to use throughout the function. Default is ``rgd.fill_missing_data``

        """
        if regridding_method is None:
            regridding_method = self.regridding_method
        if fill_method is None:
            fill_method = self.fill_method

        print(
            "Processing {} boundary velocity & tracers...".format(orientation), end=""
        )
        if not path_to_bc.exists():
            raise FileNotFoundError(
                f"Boundary file not found at {path_to_bc}. Please ensure that the files are named in the format `east_unprocessed.nc`."
            )
        self.segments[orientation] = segment(
            hgrid=self.hgrid,
            bathymetry_path=bathymetry_path,
            outfolder=self.mom_input_dir,
            segment_name="segment_{:03d}".format(segment_number),
            orientation=orientation,  # orienataion
            startdate=self.date_range[0],
            repeat_year_forcing=self.repeat_year_forcing,
        )

        self.segments[orientation].regrid_velocity_tracers(
            infile=path_to_bc,  # location of raw boundary
            varnames=varnames,
            arakawa_grid=arakawa_grid,
            rotational_method=rotational_method,
            regridding_method=regridding_method,
            fill_method=fill_method,
        )

        print("Done.")
        return

    def setup_boundary_tides(
        self,
        tpxo_elevation_filepath,
        tpxo_velocity_filepath,
        tidal_constituents=None,
        bathymetry_path=None,
        rotational_method=rot.RotationMethod.EXPAND_GRID,
        regridding_method=None,
        fill_method=None,
    ):
        """
        Subset the tidal data and generate more boundary files.

        Arguments:
            path_to_td (str): Path to boundary tidal file.
            tpxo_elevation_filepath: Filepath to the TPXO elevation product. Generally of the form ``h_tidalversion.nc``
            tpxo_velocity_filepath: Filepath to the TPXO velocity product. Generally of the form ``u_tidalversion.nc``
            tidal_constituents: List of tidal constituents to include in the regridding. Default is set in the experiment constructor (See :class:`~Experiment`)
            bathymetry_path (str): Path to the bathymetry file. Default is ``None``, in which case the boundary condition is not masked
            rotational_method (str): Method to use for rotating the tidal velocities. Default is 'EXPAND_GRID'.
            regridding_method (Optional[str]): The type of regridding method to use. Defaults to self.regridding_method
            fill_method (Function): Fill method to use throughout the function. Default is ``self.fill_method``

        Returns:
            netCDF files: Regridded tidal velocity and elevation files in 'inputdir/forcing'

        The tidal data functions are sourced from the GFDL NWA25 and modified so that:

        - Converted code for regional-mom6 :func:`segment` class
        - Implemented horizontal subsetting.
        - Combined all functions of NWA25 into a four function process (in the style of regional-mom6), i.e.,
          :func:`~experiment.setup_boundary_tides`, :func:`~regional_mom6.regridding.coords`, :func:`segment.regrid_tides`, and
          :func:`segment.encode_tidal_files_and_output`.

        Code credit:

        .. code-block:: python

            Author(s): GFDL, James Simkins, Rob Cermak, and contributors
            Year: 2022
            Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
            Version: N/A
            Type: Python Functions, Source Code
            Web Address: https://github.com/jsimkins2/nwa25
        """

        if regridding_method is None:
            regridding_method = self.regridding_method
        if fill_method is None:
            fill_method = self.fill_method
        if tidal_constituents is not None:
            self.tidal_constituents = tidal_constituents
        tpxo_h = (
            xr.open_dataset(Path(tpxo_elevation_filepath))
            .rename({"lon_z": "lon", "lat_z": "lat", "nc": "constituent"})
            .isel(
                constituent=convert_to_tpxo_tidal_constituents(self.tidal_constituents)
            )
        )

        h = tpxo_h["ha"] * np.exp(-1j * np.radians(tpxo_h["hp"]))
        tpxo_h["hRe"] = np.real(h)
        tpxo_h["hIm"] = np.imag(h)
        tpxo_u = (
            xr.open_dataset(Path(tpxo_velocity_filepath))
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
            xr.open_dataset(Path(tpxo_velocity_filepath))
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
            ),  # Import pandas for this shouldn't be a big deal b/c it's already required in regional-mom6 dependencies
            dims=["time"],
        )
        # Initialize or find boundary segment
        for b in self.boundaries:
            print("Processing {} boundary...".format(b), end="")

            # If the GLORYS ocean_state has already created segments, we don't create them again.
            seg = segment(
                hgrid=self.hgrid,
                bathymetry_path=bathymetry_path,
                outfolder=self.mom_input_dir,
                segment_name="segment_{:03d}".format(
                    self.find_MOM6_rectangular_orientation(b)
                ),
                orientation=b,
                startdate=self.date_range[0],
                repeat_year_forcing=self.repeat_year_forcing,
            )

            # Output and regrid tides
            seg.regrid_tides(
                tpxo_v,
                tpxo_u,
                tpxo_h,
                times,
                rotational_method=rotational_method,
                regridding_method=regridding_method,
            )
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
        write_to_file=True,
        regridding_method=None,
    ):
        """
        Cut out and interpolate the chosen bathymetry and then fill inland lakes.

        Users can optionally fill narrow channels (see ``fill_channels`` keyword argument
        below). Note, however, that narrow channels are less of an issue for models that
        are discretized on an Arakawa C grid, like MOM6.

        Output is saved in the input directory of the experiment.

        Arguments:
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
            positive_down (Optional[bool]): If ``True``, it assumes that the
                bathymetry vertical coordinate is positive downwards. Default: ``False``.
            write_to_file (Optional[bool]): Whether to write the bathymetry to a file. Default: ``True``.
            regridding_method (Optional[str]): The type of regridding method to use. Defaults to self.regridding_method
        """
        if regridding_method is None:
            regridding_method = self.regridding_method

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

        bathymetry_output.depth.attrs["_FillValue"] = -1e20
        bathymetry_output.depth.attrs["units"] = "meters"
        bathymetry_output.depth.attrs["standard_name"] = (
            "height_above_reference_ellipsoid"
        )
        bathymetry_output.depth.attrs["long_name"] = "Elevation relative to sea level"
        bathymetry_output.depth.attrs["coordinates"] = "lon lat"
        if write_to_file:
            bathymetry_output.to_netcdf(
                self.mom_input_dir / "bathymetry_original.nc",
                mode="w",
                engine="netcdf4",
            )

        empty_bathy = rgd.get_hgrid_arakawa_c_points(self.hgrid, "t")
        empty_bathy = empty_bathy.rename(
            {"tlon": "lon", "tlat": "lat", "nyp": "ny", "nxp": "nx"}
        )
        empty_bathy = empty_bathy.set_coords(("lon", "lat"))
        empty_bathy["depth"] = xr.zeros_like(empty_bathy["lon"])
        empty_bathy.lon.attrs["units"] = "degrees_east"
        empty_bathy.lon.attrs["_FillValue"] = 1e20
        empty_bathy.lat.attrs["units"] = "degrees_north"
        empty_bathy.lat.attrs["_FillValue"] = 1e20
        empty_bathy.depth.attrs["units"] = "meters"
        empty_bathy.depth.attrs["coordinates"] = "lon lat"
        if write_to_file:
            empty_bathy.to_netcdf(
                self.mom_input_dir / "bathymetry_unfinished.nc",
                mode="w",
                engine="netcdf4",
            )
            empty_bathy.close()

        bathymetry_output = bathymetry_output.load()
        print(
            "Begin regridding bathymetry...\n\n"
            + f"Original bathymetry size: {bathymetry_output.nbytes/1e6:.2f} Mb\n"
            + f"Regridded size: {empty_bathy.nbytes/1e6:.2f} Mb\n"
            + "Automatic regridding may fail if your domain is too big! If this process hangs or crashes,"
            + "make sure function argument write_to_file = True and,"
            + "open a terminal with appropriate computational and resources try calling ESMF "
            + f"directly in the input directory {self.mom_input_dir} via\n\n"
            + "`mpirun -np NUMBER_OF_CPUS ESMF_Regrid -s bathymetry_original.nc -d bathymetry_unfinished.nc -m bilinear --src_var depth --dst_var depth --netcdf4 --src_regional --dst_regional`\n\n"
            + "For details see https://xesmf.readthedocs.io/en/latest/large_problems_on_HPC.html\n\n"
            + "Afterwards, we run the 'expt.tidy_bathymetry' method to skip the expensive interpolation step, and finishing metadata, encoding and cleanup.\n\n\n"
        )
        regridder = rgd.create_regridder(
            bathymetry_output, empty_bathy, locstream_out=False
        )
        bathymetry = regridder(bathymetry_output)
        if write_to_file:
            bathymetry.to_netcdf(
                self.mom_input_dir / "bathymetry_unfinished.nc",
                mode="w",
                engine="netcdf4",
            )
        print(
            "Regridding successful! Now calling `tidy_bathymetry` method for some finishing touches..."
        )

        print("setup bathymetry has finished successfully.")
        return self.tidy_bathymetry(
            fill_channels,
            positive_down,
            bathymetry=bathymetry,
            write_to_file=write_to_file,
        )

    def tidy_bathymetry(
        self,
        fill_channels=False,
        positive_down=False,
        vertical_coordinate_name="depth",
        bathymetry=None,
        write_to_file=True,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
    ):
        """
        An auxiliary method for bathymetry used to fix up the metadata and remove inland
        lakes after regridding the bathymetry. Having :func:`~tidy_bathymetry` as a separate
        method from :func:`~setup_bathymetry` allows for the regridding to be done separately,
        since regridding can be really expensive for large domains.

        If the bathymetry is already regridded and what is left to be done is fixing the metadata
        or fill in some channels, then :func:`~tidy_bathymetry` directly can read the existing
        ``bathymetry_unfinished.nc`` file that should be in the input directory.

        Arguments:
            fill_channels (Optional[bool]): Whether to fill in diagonal channels.
                This removes more narrow inlets, but can also connect extra islands to land.
                Default: ``False``.
            positive_down (Optional[bool]): If ``False`` (default), assume that
                bathymetry vertical coordinate is positive down, as is the case in GEBCO for example.
            bathymetry (Optional[xr.Dataset]): The bathymetry dataset to tidy up. If not provided,
                it will read the bathymetry from the file ``bathymetry_unfinished.nc`` in the input directory
                that was created by :func:`~setup_bathymetry`.
        """

        ## reopen bathymetry to modify
        print(
            "Tidy bathymetry: Reading in regridded bathymetry to fix up metadata...",
            end="",
        )
        if read_bathy_from_file := bathymetry is None:
            bathymetry = xr.open_dataset(
                self.mom_input_dir / "bathymetry_unfinished.nc", engine="netcdf4"
            )

        ## Ensure correct encoding
        bathymetry = xr.Dataset(
            {"depth": (["ny", "nx"], bathymetry[vertical_coordinate_name].values)},
            coords={
                "lon": (["ny", "nx"], bathymetry[longitude_coordinate_name].values),
                "lat": (["ny", "nx"], bathymetry[latitude_coordinate_name].values),
            },
        )
        bathymetry.attrs["depth"] = "meters"
        bathymetry.attrs["standard_name"] = "bathymetric depth at T-cell centers"
        bathymetry.attrs["coordinates"] = "zi"

        bathymetry.expand_dims("tiles", 0)

        if not positive_down:
            ## Ensure that coordinate is positive down!
            bathymetry["depth"] *= -1

        ## Make a land mask based on the bathymetry
        ocean_mask = xr.where(bathymetry.depth <= 0, 0, 1)
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
        bathymetry = bathymetry.fillna(
            0
        )  # After min_depth filtering, change the land values to zero

        if write_to_file:
            bathymetry.expand_dims({"ntiles": 1}).to_netcdf(
                self.mom_input_dir / "bathymetry.nc",
                mode="w",
                encoding={"depth": {"_FillValue": None}},
            )
        return bathymetry

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
                str(self.fre_tools_dir / "make_solo_mosaic")
                + " --num_tiles 1 --dir . --mosaic_name ocean_mosaic --tile_file hgrid.nc",
                shell=True,
                cwd=self.mom_input_dir,
            ),
            sep="\n\n",
        )

        print(
            "OUTPUT FROM QUICK MOSAIC:",
            subprocess.run(
                str(self.fre_tools_dir / "make_quick_mosaic")
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
                str(self.fre_tools_dir / "check_mask")
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
    ):
        """
        Set up the run directory for MOM6. Either copy a pre-made set of files, or modify
        existing files in the 'rundir' directory for the experiment.

        Arguments:
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
                "Perhaps the package was imported directly rather than installed with conda. Checking if this is the case... ",
                end="",
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
            tidal_files_exist = any(Path(self.mom_input_dir).rglob("tu*"))

            if not tidal_files_exist:
                raise ValueError(
                    "No files with 'tu' in their names found in the forcing or input directory. If you meant to use tides, please run the setup_boundary_tides method first. That does output some tidal files."
                )

        # Set local var
        ncpus = None

        # 3 different cases to handle:
        #   1. User is creating a new run directory from scratch. Here we copy across all files and modify.
        #   2. User has already created a run directory, and wants to modify it. Here we only modify the MOM_layout file.
        #   3. User has already created a run directory, and wants to overwrite it. Here we copy across all files and modify. This requires overwrite = True

        if not overwrite:
            for file in base_run_dir.glob(
                "*"
            ):  ## copy each file individually if it doesn't already exist
                if not (self.mom_run_dir / file.name).exists():
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
        MOM_layout_dict = mpt.read_MOM_file_as_dict("MOM_layout", self.mom_run_dir)
        if "MASKTABLE" in MOM_layout_dict:
            MOM_layout_dict["MASKTABLE"]["value"] = (
                mask_table or " # MASKTABLE = no mask table"
            )
        if (
            "LAYOUT" in MOM_layout_dict
            and "IO_Layout" not in MOM_layout_dict
            and layout != None
        ):
            MOM_layout_dict["LAYOUT"]["value"] = str(layout[1]) + "," + str(layout[0])
        if "NIGLOBAL" in MOM_layout_dict:
            MOM_layout_dict["NIGLOBAL"]["value"] = self.hgrid.nx.shape[0] // 2
        if "NJGLOBAL" in MOM_layout_dict:
            MOM_layout_dict["NJGLOBAL"]["value"] = self.hgrid.ny.shape[0] // 2

        MOM_input_dict = mpt.read_MOM_file_as_dict("MOM_input", self.mom_run_dir)
        MOM_override_dict = mpt.read_MOM_file_as_dict("MOM_override", self.mom_run_dir)
        # The number of boundaries is reflected in the number of segments setup in setup_ocean_state_boundary under expt.segments.
        # The setup_boundary_tides function currently only works with rectangular grids amd sets up 4 segments, but DOESN"T save them to expt.segments.
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
            self.boundaries
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
        for seg in self.boundaries:
            ind_seg = self.find_MOM6_rectangular_orientation(seg)
            key_start = f"OBC_SEGMENT_00{ind_seg}"
            ## Position and Config
            key_POSITION = key_start

            rect_MOM6_index_dir = {
                "south": '"J=0,I=0:N',
                "north": '"J=N,I=N:0',
                "east": '"I=N,J=0:N',
                "west": '"I=0,J=N:0',
            }
            index_str = rect_MOM6_index_dir[seg]

            MOM_override_dict[key_POSITION]["value"] = (
                index_str + ',FLATHER,ORLANSKI,NUDGED,ORLANSKI_TAN,NUDGED_TAN"'
            )

            # Nudging Key
            key_NUDGING = key_start + "_VELOCITY_NUDGING_TIMESCALES"
            MOM_override_dict[key_NUDGING]["value"] = "0.3, 360.0"

            # Data Key
            key_DATA = key_start + "_DATA"
            file_num_obc = str(
                self.find_MOM6_rectangular_orientation(seg)
            )  # 1,2,3,4 for rectangular boundaries, BUT if we have less than 4 segments we use the index to specific the number, but keep filenames as if we had four boundaries

            obc_string = (
                f'"U=file:forcing_obc_segment_00{file_num_obc}.nc(u),'
                f"V=file:forcing_obc_segment_00{file_num_obc}.nc(v),"
                f"SSH=file:forcing_obc_segment_00{file_num_obc}.nc(eta),"
                f"TEMP=file:forcing_obc_segment_00{file_num_obc}.nc(temp),"
                f"SALT=file:forcing_obc_segment_00{file_num_obc}.nc(salt)"
            )
            MOM_override_dict[key_DATA]["value"] = obc_string
            if with_tides:
                tides_addition = (
                    f",Uamp=file:tu_segment_00{file_num_obc}.nc(uamp),"
                    f"Uphase=file:tu_segment_00{file_num_obc}.nc(uphase),"
                    f"Vamp=file:tu_segment_00{file_num_obc}.nc(vamp),"
                    f"Vphase=file:tu_segment_00{file_num_obc}.nc(vphase),"
                    f"SSHamp=file:tz_segment_00{file_num_obc}.nc(zamp),"
                    f'SSHphase=file:tz_segment_00{file_num_obc}.nc(zphase)"'
                )
                MOM_override_dict[key_DATA]["value"] = (
                    MOM_override_dict[key_DATA]["value"] + tides_addition
                )
            else:
                MOM_override_dict[key_DATA]["value"] = (
                    MOM_override_dict[key_DATA]["value"] + '"'
                )
        if type(self.date_range[0]) == str:
            self.date_range[0] = dt.datetime.strptime(
                self.date_range[0], "%Y-%m-%d %H:%M:%S"
            )
            self.date_range[1] = dt.datetime.strptime(
                self.date_range[1], "%Y-%m-%d %H:%M:%S"
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
            MOM_override_dict["OBC_TIDE_REF_DATE"]["value"] = self.date_range[
                0
            ].strftime("%Y, %m, %d")

        for key, val in MOM_override_dict.items():
            if isinstance(val, dict) and key != "original":
                MOM_override_dict[key]["override"] = True
        mpt.write_MOM_file(MOM_input_dict, self.mom_run_dir)
        mpt.write_MOM_file(MOM_override_dict, self.mom_run_dir)
        mpt.write_MOM_file(MOM_layout_dict, self.mom_run_dir)

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

        # Edit Diag Table Date
        # Read the file
        with open(self.mom_run_dir / "diag_table", "r") as file:
            lines = file.readlines()

        # The date is the second line
        lines[1] = self.date_range[0].strftime("%Y %-m %-d %-H %-M %-S\n")

        # Write the file
        with open(self.mom_run_dir / "diag_table", "w") as file:
            file.writelines(lines)

        return

    def setup_era5(self, era5_path):
        """
        Setup the ERA5 forcing files for the experiment. This assumes that
        all of the ERA5 data in the prescribed date range are downloaded.
        We need the following fields: "2t", "10u", "10v", "sp", "2d", "msdwswrf",
        "msdwlwrf", "lsrr", and "crr".

        Arguments:
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
            #          paths_per_year = [Path(era5_path / fname / year) for year in years]
            paths_per_year = [Path(f"{era5_path}/{fname}/{year}/") for year in years]
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
                "calendar": "gregorian",
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
    and puts it on gregorian calendar.

    Note:
        Only supports z-star (z*) vertical coordinate.

    Arguments:
        hgrid (xarray.Dataset): The horizontal grid used for domain.

        outfolder (Union[str, Path]): Path to folder where the model inputs will
            be stored.
        segment_name (str): Name of the segment, e.g., ``'segment_001'``.
        orientation (str): Cardinal direction (lowercase) of the boundary segment,
            i.e., ``'east'``, ``'west'``, ``'north'``, or ``'south'``.
        startdate (str): The starting date to use in the segment calendar.
        time_units (str): The units used by the raw forcing files, e.g., ``hours``,
            ``days`` (default).
        repeat_year_forcing (Optional[bool]): When ``True`` the experiment runs with repeat-year
            forcing. When ``False`` (default) then inter-annual forcing is used.
    """

    def __init__(
        self,
        *,
        hgrid,
        outfolder,
        segment_name,
        orientation,
        startdate,
        bathymetry_path=None,
        repeat_year_forcing=False,
    ):
        self.startdate = startdate

        ## Store other data
        orientation = orientation.lower()
        if orientation not in ("north", "south", "east", "west"):
            raise ValueError(
                "orientation only supported for one of: 'north', 'south', 'east', or 'west'. If you are using a non-rectangular grid, please modify the code accordingly."
            )
        self.orientation = orientation
        self.outfolder = outfolder
        self.hgrid = hgrid
        try:
            self.bathymetry = xr.open_dataset(bathymetry_path)
        except:
            self.bathymetry = None
        self.segment_name = segment_name
        self.repeat_year_forcing = repeat_year_forcing

    def regrid_velocity_tracers(
        self,
        infile,
        varnames: dict,
        arakawa_grid="A",
        rotational_method=rot.RotationMethod.EXPAND_GRID,
        regridding_method="bilinear",
        time_units="days",
        calendar="gregorian",
        fill_method=rgd.fill_missing_data,
    ):
        """
        Cut out and interpolate the velocities and tracers.

        Arguments:
            rotational_method (rot.RotationMethod): The method to use for rotation of the velocities. Currently, the default method, ``EXPAND_GRID``, works even with non-rotated grids.
            infile (Union[str, Path]): Path to the raw, unprocessed boundary segment.
            varnames (Dict[str, str]): Mapping between the variable/dimension names and
            standard naming convention of this pipeline, e.g., ``{"xq": "longitude,
            "yh": "latitude", "salt": "salinity", ...}``. Key "tracers" points to nested
            dictionary of tracers to include in boundary.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the boundary forcing.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
            regridding_method (str): regridding method to use throughout the function. Default is ``'bilinear'``
            fill_method (Function): Fill method to use throughout the function. Default is ``rgd.fill_missing_data``

        """
        reprocessed_var_map = apply_arakawa_grid_mapping(
            var_mapping=varnames, arakawa_grid=arakawa_grid
        )

        # Create weights directory
        (self.outfolder / "weights").mkdir(exist_ok=True)

        rawseg = xr.open_mfdataset(infile, decode_times=False, engine="netcdf4")

        coords = rgd.coords(self.hgrid, self.orientation, self.segment_name)

        regridders = create_vt_regridders(
            reprocessed_var_map,
            rawseg,
            coords,
            Path(self.outfolder),
            regridding_method,
            self.orientation,
        )

        u_regridded = regridders["u"](
            rawseg[reprocessed_var_map["u_var_name"]].rename(
                {
                    reprocessed_var_map["u_x_coord"]: "lon",
                    reprocessed_var_map["u_y_coord"]: "lat",
                }
            )
        )
        v_regridded = regridders["v"](
            rawseg[reprocessed_var_map["v_var_name"]].rename(
                {
                    reprocessed_var_map["v_x_coord"]: "lon",
                    reprocessed_var_map["v_y_coord"]: "lat",
                }
            )
        )
        tracers_regridded = regridders["tracers"](
            rawseg[
                [reprocessed_var_map["eta_var_name"]]
                + list(reprocessed_var_map["tracer_var_names"].values())
            ].rename(
                {
                    reprocessed_var_map["tracer_x_coord"]: "lon",
                    reprocessed_var_map["tracer_y_coord"]: "lat",
                }
            )
        )

        rotated_u, rotated_v = rotate(
            u_regridded,
            v_regridded,
            radian_angle=np.radians(
                rot.get_rotation_angle(
                    rotational_method, self.hgrid, orientation=self.orientation
                ).values
            ),
        )

        rotated_u.name = reprocessed_var_map["u_var_name"]
        rotated_v.name = reprocessed_var_map["v_var_name"]
        segment_out = xr.merge([rotated_u, rotated_v, tracers_regridded])

        ## segment out now contains our interpolated boundary.
        ## Now, we need to fix up all the metadata and save
        segment_out = segment_out.rename(
            {"lon": f"lon_{self.segment_name}", "lat": f"lat_{self.segment_name}"}
        )

        ## Convert temperatures to celsius # use pint
        depth_coord = reprocessed_var_map["depth_coord"]
        if type(reprocessed_var_map["depth_coord"]) == list:
            for dc in reprocessed_var_map["depth_coord"]:
                if (
                    dc
                    in segment_out[reprocessed_var_map["tracer_var_names"]["temp"]].dims
                ):  # At least one must be true
                    depth_coord = dc
        if (
            np.nanmin(
                segment_out[reprocessed_var_map["tracer_var_names"]["temp"]].isel(
                    {
                        reprocessed_var_map["time_var_name"]: 0,
                        depth_coord: 0,
                    }
                )
            )
            > 100
        ):
            segment_out[reprocessed_var_map["tracer_var_names"]["temp"]] -= 273.15
            segment_out[reprocessed_var_map["tracer_var_names"]["temp"]].attrs[
                "units"
            ] = "degrees Celsius"

        # fill in NaNs
        segment_out = fill_method(
            segment_out,
            xdim=f"{coords.attrs['parallel']}_{self.segment_name}",
            zdim=reprocessed_var_map["depth_coord"],
        )
        if "since" not in time_units:
            times = xr.DataArray(
                np.arange(
                    0,  #! Indexing everything from start of experiment = simple but maybe counterintutive?
                    segment_out[reprocessed_var_map["time_var_name"]].shape[
                        0
                    ],  ## Time is indexed from start date of window
                    dtype=float,
                ),  # Import pandas for this shouldn't be a big deal b/c it's already kinda required somewhere deep in some tree.
                dims=["time"],
            )
            segment_out.time.attrs = {
                "calendar": calendar,
                "units": f"{time_units} since {self.startdate}",
            }
            # This to change the time coordinate.
            segment_out = rgd.add_or_update_time_dim(
                segment_out, times, reprocessed_var_map["depth_coord"]
            )
        else:
            segment_out.time.attrs = {
                "calendar": calendar,
                "units": time_units,
            }

        # This to change the time coordinate.
        segment_out = rgd.add_or_update_time_dim(segment_out, times)
        segment_out.time.attrs = {
            "calendar": calendar,
            "units": f"{self.time_units} since {self.startdate}",
        }
        # Here, keep in mind that 'var' keeps track of the mom6 variable names we want, and self.tracers[var]
        # will return the name of the variable from the original data

        allfields = {
            **reprocessed_var_map["tracer_var_names"],
            "u": reprocessed_var_map["u_var_name"],
            "v": reprocessed_var_map["v_var_name"],
            "eta": reprocessed_var_map["eta_var_name"],
        }  ## Combine all fields into one flattened dictionary to iterate over as we fix metadata

        for (
            var
        ) in (
            allfields
        ):  ## Replace with more generic list of tracer variables that might be included?
            v = f"{var}_{self.segment_name}"
            ## Rename each variable in dataset
            segment_out = segment_out.rename({allfields[var]: v})

            # Find out if the tracer has depth, and if so, what is it's z dimension (z dimension being a list is an edge case for MARBL BGC)
            variable_has_depth = False
            depth_coord = None
            if type(reprocessed_var_map["depth_coord"]) != list:
                dc_list = [reprocessed_var_map["depth_coord"]]
            else:
                dc_list = reprocessed_var_map["depth_coord"]

            for dc in dc_list:
                if dc in segment_out[v].dims:
                    depth_coord = dc
                    variable_has_depth = True
                    break

            if variable_has_depth:
                segment_out = rgd.vertical_coordinate_encoding(
                    segment_out,
                    v,
                    self.segment_name,
                    depth_coord,
                )

            segment_out = rgd.add_secondary_dimension(
                segment_out, v, coords, self.segment_name
            )
            if variable_has_depth:
                segment_out = rgd.generate_layer_thickness(
                    segment_out,
                    v,
                    self.segment_name,
                    depth_coord,
                )
        # Overwrite the actual lat/lon values in the dimensions, replace with incrementing integers

        segment_out[f"{coords.attrs['perpendicular']}_{self.segment_name}"] = [0]

        segment_out[f"{coords.attrs['parallel']}_{self.segment_name}"] = np.arange(
            segment_out[f"{coords.attrs['parallel']}_{self.segment_name}"].size
        )

        encoding_dict = {
            "time": {"dtype": "double"},
            f"nx_{self.segment_name}": {
                "dtype": "int32",
            },
            f"ny_{self.segment_name}": {
                "dtype": "int32",
            },
        }
        segment_out = rgd.mask_dataset(
            segment_out,
            self.bathymetry,
            self.orientation,
        )
        encoding_dict = rgd.generate_encoding(
            segment_out,
            encoding_dict,
            default_fill_value=1.0e20,
        )
        segment_out.load().to_netcdf(
            self.outfolder / f"forcing_obc_{self.segment_name}.nc",
            encoding=encoding_dict,
            unlimited_dims="time",
        )

        return segment_out, encoding_dict

    def regrid_tides(
        self,
        tpxo_v,
        tpxo_u,
        tpxo_h,
        times,
        rotational_method=rot.RotationMethod.EXPAND_GRID,
        regridding_method="bilinear",
        fill_method=rgd.fill_missing_data,
    ):
        """
        Regrids and interpolates the tidal data for MOM6. Steps include:

        - Read raw tidal data (all constituents)
        - Perform minor transformations/conversions
        - Regrid the tidal elevation, and tidal velocity
        - Encode the output

        General Description:
            The tidal data functions sourced from the GFDL's code above were changed in the following ways:

            - Converted code for regional-mom6 segment class
            - Implemented horizontal subsetting
            - Combined all functions of NWA25 into a four function process (in the style of regional-mom6), i.e.,
              :func:`~experiment.setup_boundary_tides`, :func:`~regional_mom6.regridding.coords`, :func:`segment.regrid_tides`, and
              :func:`segment.encode_tidal_files_and_output`.

        Arguments:
            infile_td (str): Raw tidal file/directory.
            tpxo_v, tpxo_u, tpxo_h (xarray.Dataset): Specific adjusted for MOM6 tpxo datasets (Adjusted with :func:`~experiment.setup_boundary_tides`)
            times (pd.DateRange): The start date of our model period.
            rotational_method (rot.RotationMethod): The method to use for rotation of the velocities.
                The default method, ``EXPAND_GRID``, works even with non-rotated grids.
            regridding_method (str): regridding method to use throughout the function. Default is ``'bilinear'``
            fill_method (Function): Fill method to use throughout the function. Default is ``rgd.fill_missing_data``

        Returns:
            netCDF files: Regridded tidal velocity and elevation files in 'inputdir/forcing'

        Code credit:

        .. code-block:: bash

            Author(s): GFDL, James Simkins, Rob Cermak, and contributors
            Year: 2022
            Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
            Version: N/A
            Type: Python Functions, Source Code
            Web Address: https://github.com/jsimkins2/nwa25
        """

        # Create weights directory
        (self.outfolder / "weights").mkdir(exist_ok=True)

        # Establish Coords
        coords = rgd.coords(self.hgrid, self.orientation, self.segment_name)

        ########## Tidal Elevation: Horizontally interpolate elevation components ############
        regrid = rgd.create_regridder(
            tpxo_h[["lon", "lat", "hRe"]],
            coords,
            Path(
                self.outfolder / "forcing" / f"regrid_{self.segment_name}_tidal_elev.nc"
            ),
            method=regridding_method,
        )

        redest = regrid(tpxo_h[["lon", "lat", "hRe"]])
        imdest = regrid(tpxo_h[["lon", "lat", "hIm"]])

        # Fill missing data.
        # Need to do this first because complex would get converted to real
        redest = fill_method(
            redest, xdim=f"{coords.attrs['parallel']}_{self.segment_name}", zdim=None
        )
        redest = redest["hRe"]
        imdest = fill_method(
            imdest, xdim=f"{coords.attrs['parallel']}_{self.segment_name}", zdim=None
        )
        imdest = imdest["hIm"]

        # Convert complex
        cplex = redest + 1j * imdest

        # Convert to real amplitude and phase.
        ds_ap = xr.Dataset({f"zamp_{self.segment_name}": np.abs(cplex)})

        # np.angle doesn't return dataarray
        ds_ap[f"zphase_{self.segment_name}"] = (
            ("constituent", f"{coords.attrs['parallel']}_{self.segment_name}"),
            -1 * np.angle(cplex),
        )  # radians

        # Add time coordinate and transpose so that time is first,
        # so that it can be the unlimited dimension
        times = xr.DataArray(
            pd.date_range(
                self.startdate, periods=1
            ),  # Import pandas for this shouldn't be a big deal b/c it's already kinda required somewhere deep in some tree.
            dims=["time"],
        )

        ds_ap = rgd.add_or_update_time_dim(ds_ap, times)
        ds_ap = ds_ap.transpose(
            "time", "constituent", f"{coords.attrs['parallel']}_{self.segment_name}"
        )

        self.encode_tidal_files_and_output(ds_ap, "tz")

        ########### Regrid Tidal Velocity ######################

        regrid_u = rgd.create_regridder(
            tpxo_u[["lon", "lat", "uRe"]], coords, method=regridding_method
        )
        regrid_v = rgd.create_regridder(
            tpxo_v[["lon", "lat", "vRe"]], coords, method=regridding_method
        )

        # Interpolate each real and imaginary parts to self.
        uredest = regrid_u(tpxo_u[["lon", "lat", "uRe"]])["uRe"]
        uimdest = regrid_u(tpxo_u[["lon", "lat", "uIm"]])["uIm"]
        vredest = regrid_v(tpxo_v[["lon", "lat", "vRe"]])["vRe"]
        vimdest = regrid_v(tpxo_v[["lon", "lat", "vIm"]])["vIm"]

        # Fill missing data.
        # Need to do this first because complex would get converted to real
        uredest = fill_method(
            uredest, xdim=f"{coords.attrs['parallel']}_{self.segment_name}", zdim=None
        )
        uimdest = fill_method(
            uimdest, xdim=f"{coords.attrs['parallel']}_{self.segment_name}", zdim=None
        )
        vredest = fill_method(
            vredest, xdim=f"{coords.attrs['parallel']}_{self.segment_name}", zdim=None
        )
        vimdest = fill_method(
            vimdest, xdim=f"{coords.attrs['parallel']}_{self.segment_name}", zdim=None
        )

        # Convert to complex, remaining separate for u and v.
        ucplex = uredest + 1j * uimdest
        vcplex = vredest + 1j * vimdest

        # Convert complex u and v to ellipse,
        # rotate ellipse from earth-relative to model-relative,
        # and convert ellipse back to amplitude and phase.
        SEMA, ECC, INC, PHA = ap2ep(ucplex, vcplex)

        # Rotate
        INC -= np.radians(
            rot.get_rotation_angle(
                rotational_method, self.hgrid, orientation=self.orientation
            ).data[np.newaxis, :]
        )

        ua, va, up, vp = ep2ap(SEMA, ECC, INC, PHA)
        # Convert to real amplitude and phase.

        ds_ap = xr.Dataset(
            {f"uamp_{self.segment_name}": ua, f"vamp_{self.segment_name}": va}
        )
        # up, vp aren't dataarraysf
        ds_ap[f"uphase_{self.segment_name}"] = (
            ("constituent", f"{coords.attrs['parallel']}_{self.segment_name}"),
            up,
        )  # radians
        ds_ap[f"vphase_{self.segment_name}"] = (
            ("constituent", f"{coords.attrs['parallel']}_{self.segment_name}"),
            vp,
        )  # radians

        times = xr.DataArray(
            pd.date_range(
                self.startdate, periods=1
            ),  # Import pandas for this shouldn't be a big deal b/c it's already kinda required somewhere deep in some tree.
            dims=["time"],
        )
        ds_ap = rgd.add_or_update_time_dim(ds_ap, times)
        ds_ap = ds_ap.transpose(
            "time", "constituent", f"{coords.attrs['parallel']}_{self.segment_name}"
        )

        # Some things may have become missing during the transformation
        ds_ap = fill_method(
            ds_ap, xdim=f"{coords.attrs['parallel']}_{self.segment_name}", zdim=None
        )

        self.encode_tidal_files_and_output(ds_ap, "tu")

        return

    def encode_tidal_files_and_output(self, ds, filename):
        """
        This method:

        - Expands the dimensions (with the segment name)
        - Renames some dimensions to be more specific to the segment
        - Provides an output file encoding
        - Exports the files.

        Arguments:
            self.outfolder (str/path): The output folder to save the tidal files into
            dataset (xarray.Dataset): The processed tidal dataset
            filename (str): The output file name

        Returns:
            netCDF files: Regridded [FILENAME] files in 'self.outfolder/[filename]_[segmentname].nc'

        General Description:
            This tidal data functions are sourced from the GFDL NWA25 and changed in the following ways:

            - Converted code for regional-mom6 segment class
            - Implemented horizontal Subsetting
            - Combined all functions of NWA25 into a four function process (in the style of regional-mom6), i.e.,
              :func:`~experiment.setup_boundary_tides`, :func:`~regional_mom6.regridding.coords`, :func:`segment.regrid_tides`, and
              :func:`segment.encode_tidal_files_and_output`.

        Code credit:

        .. code-block:: bash

            Author(s): GFDL, James Simkins, Rob Cermak, and contributors
            Year: 2022
            Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
            Version: N/A
            Type: Python Functions, Source Code
            Web Address: https://github.com/jsimkins2/nwa25
        """

        coords = rgd.coords(self.hgrid, self.orientation, self.segment_name)

        ## Expand Tidal Dimensions ##

        for var in ds:

            ds = rgd.add_secondary_dimension(ds, str(var), coords, self.segment_name)

        ## Rename Tidal Dimensions ##
        ds = ds.rename(
            {"lon": f"lon_{self.segment_name}", "lat": f"lat_{self.segment_name}"}
        )

        if self.bathymetry is not None:
            print(
                "Bathymetry has been provided to the regridding tides function. "
                "Masking tides dataset with bathymetry may result in errors like large surface values one timestep in. "
                " To avoid masking tides, do not pass in bathymetry path to the tides function."
            )
        ds = rgd.mask_dataset(ds, self.bathymetry, self.orientation)
        ## Perform Encoding ##

        fname = f"{filename}_{self.segment_name}.nc"
        # Set format and attributes for coordinates, including time if it does not already have calendar attribute
        # (may change this to detect whether time is a time type or a float).
        # Need to include the fillvalue or it will be back to nan
        encoding = {
            "time": dict(dtype="float64", calendar="gregorian", _FillValue=1.0e20),
            f"lon_{self.segment_name}": dict(dtype="float64", _FillValue=1.0e20),
            f"lat_{self.segment_name}": dict(dtype="float64", _FillValue=1.0e20),
        }
        encoding = rgd.generate_encoding(ds, encoding, default_fill_value=1.0e20)

        ## Export Files ##
        ds.to_netcdf(
            Path(self.outfolder / fname),
            engine="netcdf4",
            encoding=encoding,
            unlimited_dims="time",
        )
        return ds


def create_vt_regridders(
    reprocessed_var_map: dict,
    rawseg: xr.Dataset,
    coords: xr.Dataset,
    outfolder: str,
    regridding_method: str,
    id: str = "",
) -> dict[str, xe.Regridder]:
    """
    Create regridders for velocity and tracer variables based on the specified Arakawa grid.

    This function uses a validated variable mapping to create one or more
    `xesmf.Regridder` objects for velocity (`u`, `v`) and tracer fields,
    depending on the detected Arakawa grid type.

    Args:
        reprocessed_var_map: Mapping of variable and coordinate names, including nested
            tracer variable names (e.g., {"tracers": {"salt": "salt", "temp": "temp"}}).
        raw_seg: The source dataset containing the original variables.
        coords: The target grid coordinates dataset.
        outfolder: Path to the output folder where regridding weights are saved.
        regridding_method: The interpolation method (default: "bilinear").
        id: Optional string identifier appended to output weight filenames.

    Returns:
        dict[str, xe.Regridder]: A dictionary containing the created regridders with keys:
            - "tracers"
            - "u"
            - "v"
    """
    regridders = {}
    arakawa_grid = identify_arakawa_grid(reprocessed_var_map)
    outfolder = Path(outfolder)
    regridders["tracers"] = rgd.create_regridder(
        rawseg[reprocessed_var_map["tracer_var_names"]["salt"]].rename(
            {
                reprocessed_var_map["tracer_lon_coord"]: "lon",
                reprocessed_var_map["tracer_lat_coord"]: "lat",
            }
        ),
        coords,
        outfolder / f"weights/bilinear_tracer_weights_{id}.nc",
        method=regridding_method,
    )

    if arakawa_grid == "B" or arakawa_grid == "C":
        regridders["u"] = rgd.create_regridder(
            rawseg[reprocessed_var_map["u_var_name"]].rename(
                {
                    reprocessed_var_map["u_lon_coord"]: "lon",
                    reprocessed_var_map["u_lat_coord"]: "lat",
                }
            ),
            coords,
            outfolder / f"weights/bilinear_u_weights_{id}.nc",
            method=regridding_method,
        )
    else:  # Arakawa A
        regridders["u"] = regridders["tracers"]

    if arakawa_grid == "C":
        regridders["v"] = rgd.create_regridder(
            rawseg[reprocessed_var_map["v_var_name"]].rename(
                {
                    reprocessed_var_map["v_lon_coord"]: "lon",
                    reprocessed_var_map["v_lat_coord"]: "lat",
                }
            ),
            coords,
            outfolder / f"weights/bilinear_v_weights_{id}.nc",
            method=regridding_method,
        )
    else:  # Arakawa A and B
        regridders["v"] = regridders["u"]

    return regridders


def apply_arakawa_grid_mapping(var_mapping: dict, arakawa_grid: str = None) -> dict:
    """
    Map variable and coordinate names according to the specified Arakawa grid type.

    This function checks the provided Arakawa grid type and constructs a consistent
    mapping between standard variable keys (e.g., tracer, velocity components) and
    their corresponding actual names. It raises an error if any required variable
    names are missing for the specified grid type.

    Args:
        var_mappings (Dict[str, str]):
            A dictionary mapping standardized variable/dimension names to their actual
            names. Input names can use either the ``xh/xq`` convention with a specific arakawa grid or the exact output
            format produced by this function without the arakawa_grid specified (which it will only then do the sanity checks).
        arakawa_grid (str):
            The Arakawa grid staggering type of the boundary forcing. Must be one of:
            ``'A'``, ``'B'``, or ``'C'``.

    Returns:
        Dict[str, Any]:
            A dictionary containing variable names mapped according to the specified
            Arakawa grid type. The returned dictionary includes the following keys:
                - ``u_x_coord``
                - ``u_y_coord``
                - ``v_x_coord``
                - ``v_y_coord``
                - ``tracer_x_coord``
                - ``tracer_y_coord``
                - ``u_lon_coord``
                - ``u_lat_coord``
                - ``v_lon_coord``
                - ``v_lat_coord``
                - ``tracer_lon_coord``
                - ``tracer_lat_coord``
                - ``depth_coord``
                - ``u_var_name``
                - ``v_var_name``
                - ``tracer_var_names`` (a nested dict with keys ``"salt"`` and ``"temp"``)
    """

    if arakawa_grid is None:
        # If no arakawa_grid is provided, assume the mapping is already in the correct format
        print(
            "No arakawa_grid provided, assuming the variable mapping for your data product is already in correct format."
        )
        validate_var_mapping(var_mapping, is_xhyh=False)
        arakawa_grid = identify_arakawa_grid(var_mapping)
        print("Arakawa {} grid detected in variable mapping".format(arakawa_grid))
        return var_mapping
    else:
        if arakawa_grid not in ("A", "B", "C"):
            raise ValueError("arakawa_grid must be one of: 'A', 'B', or 'C'")

        # Validate basic var mapping structure
        validate_var_mapping(var_mapping, is_xhyh=True)

        reprocessed_var_map = {
            "tracer_x_coord": var_mapping["xh"],
            "tracer_y_coord": var_mapping["yh"],
            "u_var_name": var_mapping["u"],
            "v_var_name": var_mapping["v"],
            "eta_var_name": var_mapping["eta"],
            "time_var_name": var_mapping["time"],
            "depth_coord": var_mapping["zl"],
            "tracer_var_names": {
                "salt": var_mapping["tracers"]["salt"],
                "temp": var_mapping["tracers"]["temp"],
            },
        }

        if arakawa_grid == "A":
            print(
                "Applying Arakawa A grid variable mapping, which is velocities and tracers on the same grid"
            )
            reprocessed_var_map["u_x_coord"] = reprocessed_var_map["tracer_x_coord"]
            reprocessed_var_map["u_y_coord"] = reprocessed_var_map["tracer_y_coord"]
            reprocessed_var_map["v_x_coord"] = reprocessed_var_map["tracer_x_coord"]
            reprocessed_var_map["v_y_coord"] = reprocessed_var_map["tracer_y_coord"]

        elif arakawa_grid == "B":
            print(
                "Applying Arakawa B grid variable mapping, which is velocities on xq, yq and tracers on xh, yh."
            )
            if var_mapping["xq"] is None or var_mapping["yq"] is None:
                raise ValueError(
                    "For Arakawa B grid, variable mapping must include 'xq' and 'yq' coordinate names."
                )
            reprocessed_var_map["u_x_coord"] = var_mapping["xq"]
            reprocessed_var_map["u_y_coord"] = var_mapping["yq"]
            reprocessed_var_map["v_x_coord"] = var_mapping["xq"]
            reprocessed_var_map["v_y_coord"] = var_mapping["yq"]

        elif arakawa_grid == "C":
            print(
                "Applying Arakawa C grid variable mapping, which is u-velocity on xq, yh; v-velocity on xh, yq; and tracers on xh, yh."
            )
            if var_mapping["xq"] is None or var_mapping["yq"] is None:
                raise ValueError(
                    "For Arakawa C grid, variable mapping must include 'xq' and 'yq' coordinate names."
                )
            reprocessed_var_map["u_x_coord"] = var_mapping["xq"]
            reprocessed_var_map["u_y_coord"] = var_mapping["yh"]
            reprocessed_var_map["v_x_coord"] = var_mapping["xh"]
            reprocessed_var_map["v_y_coord"] = var_mapping["yq"]

        # Because curvilinear grids will have different x.y versus lat/lon but this version of the var_mapping assumes they are rectilinear, we set the
        # x/y coord to lon/lat
        # If you did want to use curvilinear in/out data, you would not use this xh/yh version of the var mapping and instead use the reprocessed variable mapping, which is the if part of this if/else statement
        reprocessed_var_map["u_lon_coord"] = reprocessed_var_map["u_x_coord"]
        reprocessed_var_map["u_lat_coord"] = reprocessed_var_map["u_y_coord"]
        reprocessed_var_map["v_lon_coord"] = reprocessed_var_map["v_x_coord"]
        reprocessed_var_map["v_lat_coord"] = reprocessed_var_map["v_y_coord"]
        reprocessed_var_map["tracer_lon_coord"] = reprocessed_var_map["tracer_x_coord"]
        reprocessed_var_map["tracer_lat_coord"] = reprocessed_var_map["tracer_y_coord"]

        # One last sanity check
        validate_var_mapping(reprocessed_var_map, is_xhyh=False)
        return reprocessed_var_map


def validate_var_mapping(var_map: dict, is_xhyh: bool = False) -> None:
    """
    Validate the structure and completeness of a variable mapping dictionary.

    This function checks that all expected keys and subkeys are present in the
    dictionary returned by the Arakawa grid variable mapping function.

    Args:
        var_map (Dict[str, Any]): The dictionary to validate.
        is_xhyh (bool): If True, expects the input dictionary to use the ``xh/xq`` regional_mom6 format

    Raises:
        ValueError: If any required keys or subkeys are missing, or if the dictionary
                    structure does not match the expected format.
    """
    if not is_xhyh:
        required_keys = {
            "time_var_name",
            "u_x_coord",
            "u_y_coord",
            "v_x_coord",
            "v_y_coord",
            "u_lon_coord",
            "u_lat_coord",
            "v_lon_coord",
            "v_lat_coord",
            "tracer_x_coord",
            "tracer_y_coord",
            "tracer_lon_coord",
            "tracer_lat_coord",
            "depth_coord",
            "u_var_name",
            "v_var_name",
            "eta_var_name",
            "tracer_var_names",
        }

    else:
        required_keys = {"time", "xh", "zl", "u", "v", "tracers", "eta"}

    missing = required_keys - var_map.keys()
    if missing:
        raise ValueError(
            f"Missing required keys in var_map: {', '.join(sorted(missing))}"
        )
    if not is_xhyh:
        tracer_map = var_map.get("tracer_var_names")
    else:
        tracer_map = var_map.get("tracers")
    # Validate nested tracer variable names

    if not isinstance(tracer_map, dict):
        raise ValueError("Expected tracers to be a dictionary.")

    required_tracers = {"salt", "temp"}
    missing_tracers = required_tracers - tracer_map.keys()
    if missing_tracers:
        raise ValueError(
            f"Missing required tracer variable names: {', '.join(sorted(missing_tracers))}"
        )


def identify_arakawa_grid(var_mapping):
    """identify the arakawa grid from the variable mapping"""
    if (
        var_mapping["v_x_coord"] == var_mapping["u_x_coord"]
        and var_mapping["u_x_coord"] == var_mapping["tracer_x_coord"]
    ):
        return "A"
    elif (
        var_mapping["v_x_coord"] == var_mapping["u_x_coord"]
        and var_mapping["u_x_coord"] != var_mapping["tracer_x_coord"]
    ):
        return "B"
    elif (
        var_mapping["v_x_coord"] != var_mapping["u_x_coord"]
        and var_mapping["u_x_coord"] != var_mapping["tracer_x_coord"]
        and var_mapping["v_x_coord"] != var_mapping["tracer_x_coord"]
    ):
        return "C"
    else:
        raise ValueError(
            "Could not determine Arakawa grid type from provided variable mapping. Something's wrong! Please specify variable mapping correctly"
        )
