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
import pandas as pd
from pathlib import Path
import json
from ruamel.yaml import YAML
from regional_mom6 import MOM_parameter_tools as mpt
from regional_mom6 import regridding as rgd
from regional_mom6.config import Config
from regional_mom6.grid import Grid
from regional_mom6.vgrid import VGrid
from regional_mom6.topo import Topo
from regional_mom6.chl import interpolate_and_fill_seawifs
from regional_mom6.segment import Segment
from regional_mom6.utils import (
    rotate,
    find_files_by_pattern,
    try_pint_convert,
)
from mom6_forge._supergrid import SupergridBase
from mom6_forge.utils import longitude_slicer
from mom6_forge.vgrid import *
from mom6_forge.grid import *
from mom6_forge.topo import *
from regional_mom6.validate import validate_obc_file, validate_general_file

warnings.filterwarnings("ignore")

__all__ = [
    "experiment",
    "Segment",
    "get_glorys_data",
    "Grid",
    "Topo",
    "VGrid",
]


# Maximum tolerated disagreement (in degrees) between an hgrid.nc file's stored
# `angle_dx` and the angle MOM6's expanded-supergrid method computes from the same
# grid's x/y coordinates, before `experiment.hgrid` refuses to use it. A large
# disagreement usually means the file's `angle_dx` came from a different tool or
# rotation convention than mom6_forge/MOM6 expect.
ANGLE_DX_DISCREPANCY_THRESHOLD_DEGREES = 5.0


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

    lines.append(f"""
copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m --variable so --variable thetao --variable uo --variable vo --variable zos --start-datetime {str(timerange[0]).replace(" ","T")} --end-datetime {str(timerange[1]).replace(" ","T")} --minimum-longitude {longitude_extent[0] - buffer} --maximum-longitude {longitude_extent[1] + buffer} --minimum-latitude {latitude_extent[0] - buffer} --maximum-latitude {latitude_extent[1] + buffer} --minimum-depth 0 --maximum-depth 6000 -o {str(path)} -f {segment_name}.nc\n
""")
    file.writelines(lines)
    file.close()
    return Path(path / "get_glorys_data.sh")


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
        mom_run_dir (str): Path of the MOM6 control directory.
        mom_input_dir (str): Path of the MOM6 input directory, to receive the forcing files.
        resolution (float, optional): Lateral resolution of the domain (in degrees). Required
            only when ``hgrid_type`` doesn't already provide a ``Grid`` object (i.e., you need
            one to be generated from ``longitude_extent``/``latitude_extent``).
        number_vertical_layers (int, optional): Number of vertical layers. Required only when
            ``vgrid_type`` doesn't already provide a ``VGrid`` object.
        layer_thickness_ratio (float, optional): Ratio of largest to smallest layer thickness;
            used as input in :func:`~hyperbolictan_thickness_profile`. Required only when
            ``vgrid_type`` doesn't already provide a ``VGrid`` object.
        depth (float, optional): Depth of the domain. Required only when ``vgrid_type`` doesn't
            already provide a ``VGrid`` object.
        fre_tools_dir (str): Path of GFDL's FRE tools (https://github.com/NOAA-GFDL/FRE-NCtools)
            binaries.
        longitude_extent (Tuple[float], optional): Extent of the region in longitude (in degrees). For
            example: ``(40.5, 50.0)``. Required only when ``hgrid_type`` doesn't already provide a
            ``Grid`` object.
        latitude_extent (Tuple[float], optional): Extent of the region in latitude (in degrees). For
            example: ``(-20.0, 30.0)``. Required only when ``hgrid_type`` doesn't already provide a
            ``Grid`` object.
        hgrid_type (str or Grid): Type of horizontal grid to generate. Currently, only ``'even_spacing'`` is supported.
            Setting this argument to ``'from_file'`` lazily reads ``hgrid.nc`` from ``mom_input_dir`` the first time
            the ``hgrid`` property is accessed. You can also pass a mom6_forge ``Grid`` object directly, in which case
            ``hgrid`` is derived from it instead of touching disk, and ``resolution``/``longitude_extent``/
            ``latitude_extent`` are not required.
        vgrid_type (str or VGrid): Type of vertical grid to generate. Currently, only ``'hyperbolic_tangent'`` is
            supported. Setting this argument to ``'from_file'`` lazily reads ``vgrid.nc`` from ``mom_input_dir`` the
            first time the ``vgrid`` property is accessed. You can also pass a mom6_forge ``VGrid`` object directly, in
            which case ``vgrid`` is derived from it instead of touching disk, and ``number_vertical_layers``/
            ``layer_thickness_ratio``/``depth`` are not required.
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
        expt.m6f_hgrid = None
        expt.m6f_vgrid = None
        expt.m6f_bathymetry = None
        expt._hgrid = None
        expt._vgrid = None
        expt._bathymetry = None
        return expt

    def __init__(
        self,
        *,
        date_range,
        mom_run_dir,
        mom_input_dir,
        resolution=None,
        number_vertical_layers=None,
        layer_thickness_ratio=None,
        depth=None,
        fre_tools_dir=None,
        longitude_extent=None,
        latitude_extent=None,
        hgrid_type="even_spacing",
        vgrid_type="hyperbolic_tangent",
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
        # `self.m6f_hgrid`/`self.m6f_vgrid`/`self.m6f_bathymetry` are the mom6_forge
        # class objects backing the `hgrid`/`vgrid`/`bathymetry` properties (see below).
        # When one isn't supplied directly, the properties lazily read it from
        # `mom_input_dir` on first access.
        self.m6f_hgrid = None
        self.m6f_vgrid = None
        self.m6f_bathymetry = None
        self._hgrid = None
        self._vgrid = None
        self._bathymetry = None

        if isinstance(hgrid_type, Grid):
            self.m6f_hgrid = hgrid_type
            self.longitude_extent = (
                float(self.hgrid.x.min()),
                float(self.hgrid.x.max()),
            )
            self.latitude_extent = (
                float(self.hgrid.y.min()),
                float(self.hgrid.y.max()),
            )
        elif hgrid_type == "from_file":
            # `self.hgrid` lazily reads `mom_input_dir/hgrid.nc` the first time it's
            # accessed. A rotation-angle discrepancy here is only warned about (not
            # raised), so construction can still succeed and the user has a live
            # `experiment` to call `recalculate_rotation_angle()` on afterward.
            hgrid = self.hgrid
            hgrid = self.m6f_hgrid.supergrid.to_ds()
            self.longitude_extent = (float(hgrid.x.min()), float(hgrid.x.max()))
            self.latitude_extent = (float(hgrid.y.min()), float(hgrid.y.max()))
        else:
            assert (
                resolution is not None
                and longitude_extent is not None
                and latitude_extent is not None
            ), (
                "`resolution`, `longitude_extent`, and `latitude_extent` are required "
                "to generate an hgrid; pass a mom6_forge `Grid` object via `hgrid_type` "
                "instead if you don't want to specify them."
            )
            self.longitude_extent = tuple(longitude_extent)
            self.latitude_extent = tuple(latitude_extent)
            self._make_hgrid()  # sets `self.m6f_hgrid`; `self.hgrid` derives from it

        if isinstance(vgrid_type, VGrid):
            self.m6f_vgrid = vgrid_type
        elif vgrid_type == "from_file":
            # `self.vgrid` lazily reads `mom_input_dir/vgrid.nc` the first time
            # it's accessed.
            pass
        else:
            assert (
                number_vertical_layers is not None
                and layer_thickness_ratio is not None
                and depth is not None
            ), (
                "`number_vertical_layers`, `layer_thickness_ratio`, and `depth` are "
                "required to generate a vgrid; pass a mom6_forge `VGrid` object via "
                "`vgrid_type` instead if you don't want to specify them."
            )
            self._make_vgrid()  # sets `self.m6f_vgrid`; `self.vgrid` derives from it

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
    def hgrid(self):
        """The horizontal supergrid, as an ``xarray.Dataset``, always regenerated live
        from ``self.m6f_hgrid`` (a mom6_forge ``Grid`` object) -- so it stays in sync
        with any in-place edits made to ``m6f_hgrid``.

        If ``m6f_hgrid`` hasn't been supplied yet -- passed in directly via
        ``hgrid_type``, or generated by ``_make_hgrid`` -- it's lazily built from
        ``hgrid.nc`` in ``mom_input_dir`` the first time this property is accessed. On
        that first load, the file's ``angle_dx`` is checked against the angle MOM6's
        expanded-supergrid method would compute from the grid's ``x``/``y``
        coordinates; see :meth:`recalculate_rotation_angle`.
        """
        if self.m6f_hgrid is None:
            hgrid_path = self.mom_input_dir / "hgrid.nc"
            if not hgrid_path.exists():
                raise FileNotFoundError(
                    f"Horizontal grid {hgrid_path} not found. Make sure `hgrid.nc` "
                    f"exists in {self.mom_input_dir} directory, or pass in a Grid "
                    "object via `hgrid_type`."
                )
            self.m6f_hgrid = Grid.from_supergrid(hgrid_path)
            self._validate_hgrid_rotation_angle(source=hgrid_path)
        return self.m6f_hgrid.supergrid.to_ds()

    def _validate_hgrid_rotation_angle(self, source):
        """Compare the loaded hgrid's stored ``angle_dx`` against the angle MOM6's
        expanded-supergrid method computes from its ``x``/``y`` coordinates, and raise
        if they disagree by more than :data:`ANGLE_DX_DISCREPANCY_THRESHOLD_DEGREES`.
        """
        supergrid = self.m6f_hgrid.supergrid
        expected_angle_dx = SupergridBase.calc_supergrid_rotation_angles_using_expanded_supergrid_method(
            supergrid.x, supergrid.y
        )
        max_discrepancy = float(
            np.nanmax(np.abs(supergrid.angle_dx - expected_angle_dx))
        )
        if max_discrepancy > ANGLE_DX_DISCREPANCY_THRESHOLD_DEGREES:
            warnings.warn(
                f"The `angle_dx` stored in {source} disagrees with the angle MOM6's "
                f"expanded-supergrid method computes from the grid's x/y coordinates "
                f"by up to {max_discrepancy:.2f} degrees (threshold: "
                f"{ANGLE_DX_DISCREPANCY_THRESHOLD_DEGREES} degrees). This usually means "
                "the hgrid.nc came from a different tool or rotation convention. If the "
                "MOM6-consistent angle is what you actually want, call "
                "`recalculate_rotation_angle()` to overwrite it."
            )

    def recalculate_rotation_angle(self):
        """Recompute ``angle_dx`` for the current hgrid using MOM6's
        expanded-supergrid method, overwriting whatever is currently stored.

        Call this after hand-editing the hgrid's coordinates (e.g. via
        ``TopoEditor``/manual rotation), or if :attr:`hgrid` raised a discrepancy
        error and the MOM6-consistent angle is what you want.
        """
        assert self.hgrid is not None, "No hgrid available"
        supergrid = self.m6f_hgrid.supergrid
        supergrid.angle_dx = SupergridBase.calc_supergrid_rotation_angles_using_expanded_supergrid_method(
            supergrid.x, supergrid.y
        )

    @property
    def vgrid(self):
        """The vertical coordinate dataset (interface/cell-center depths), as an
        ``xarray.Dataset``, always regenerated live from ``self.m6f_vgrid`` (a
        mom6_forge ``VGrid`` object) -- so it stays in sync with any in-place edits
        made to ``m6f_vgrid``.

        If ``m6f_vgrid`` hasn't been supplied yet -- passed in directly via
        ``vgrid_type``, or generated by ``_make_vgrid`` -- it's lazily built from
        ``vgrid.nc`` in ``mom_input_dir`` the first time this property is accessed.

        Note: each access rewrites ``vcoord.nc`` in ``mom_input_dir`` via
        ``VGrid.write_z_file``.
        """
        if self.m6f_vgrid is None:
            vgrid_path = self.mom_input_dir / "vgrid.nc"
            if not vgrid_path.exists():
                raise FileNotFoundError(
                    f"Vertical grid {vgrid_path} not found. Make sure `vgrid.nc` "
                    f"exists in {self.mom_input_dir} directory, or pass in a VGrid "
                    "object via `vgrid_type`."
                )
            self.m6f_vgrid = VGrid.from_file(vgrid_path)
            if len(self.m6f_vgrid.zi) > 2 and self.minimum_depth < self.m6f_vgrid.zi[2]:
                print(
                    f"Warning: Minimum depth of {self.minimum_depth}m is less than the depth of the third interface ({self.m6f_vgrid.zi[2]}m)!\n"
                    + "This means that some areas may only have one or two layers between the surface and sea floor. \n"
                    + "For increased stability, consider increasing the minimum depth, or adjusting the vertical coordinate to add more layers near the surface."
                )
        return self.m6f_vgrid.write_z_file(self.mom_input_dir / "vcoord.nc")

    @property
    def bathymetry(self):
        """The bathymetry, as an ``xarray.Dataset``, always regenerated live from
        ``self.m6f_bathymetry`` (a mom6_forge ``Topo`` object) -- so it stays in sync
        with any in-place edits made to ``m6f_bathymetry`` (e.g. via ``TopoEditor``).

        If ``m6f_bathymetry`` hasn't been supplied yet -- generated by
        ``setup_bathymetry``/``tidy_bathymetry`` -- it's lazily built from
        ``bathymetry.nc`` in ``mom_input_dir`` (via ``Topo.from_topo_file``, using
        ``self.hgrid``) the first time this property is accessed.
        """
        if self.m6f_bathymetry is None:
            bathymetry_path = self.mom_input_dir / "bathymetry.nc"
            if not bathymetry_path.exists():
                raise FileNotFoundError(
                    f"Bathymetry {bathymetry_path} not found. Make sure you've "
                    "successfully run the setup_bathymetry method, or copied a "
                    f"bathymetry.nc file into {self.mom_input_dir}."
                )
            self.hgrid  # ensures `m6f_hgrid` is populated (from disk if needed)
            self.m6f_bathymetry = Topo.from_topo_file(
                self.m6f_hgrid,
                bathymetry_path,
                min_depth=self.minimum_depth,
                git=False,
            )
        return self.m6f_bathymetry.gen_topo_ds()

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

    def _get_segment(self, orientation, bathymetry_path=None):
        """
        Build (or reuse a cached) :class:`~regional_mom6.segment.Segment` for the
        given cardinal ``orientation``.

        The first call for a given ``orientation`` builds the ``Segment`` (masking
        with the bathymetry at ``bathymetry_path`` if given) and caches it in
        ``self.segments``; subsequent calls for the same ``orientation`` reuse the
        cached ``Segment`` regardless of ``bathymetry_path`` -- this lets
        :func:`~setup_ocean_state_boundaries` and :func:`~setup_boundary_tides` share
        one ``Segment`` per orientation instead of each re-deriving it from the grid.
        """
        if orientation in self.segments:
            return self.segments[orientation]

        topo = None
        if bathymetry_path is not None:
            try:
                self.hgrid  # ensures `m6f_hgrid` is populated (from disk if needed)
                topo = Topo.from_topo_file(
                    self.m6f_hgrid,
                    bathymetry_path,
                    min_depth=self.minimum_depth,
                    git=False,
                )
            except Exception:
                topo = None

        segment_name = "segment_{:03d}".format(
            self.find_MOM6_rectangular_orientation(orientation)
        )
        segment = Segment.cardinal(self.hgrid, orientation, segment_name, topo=topo)
        self.segments[orientation] = segment
        return segment

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
            self.m6f_hgrid = Grid(
                resolution=self.resolution,  # in degrees
                xstart=self.longitude_extent[0],  # min longitude in [0, 360]
                lenx=self.longitude_extent[1]
                - self.longitude_extent[0],  # longitude extent in degrees
                ystart=self.latitude_extent[0],  # min latitude in [-90, 90]
                leny=self.latitude_extent[1]
                - self.latitude_extent[0],  # latitude extent in degrees
                name=self.expt_name,
                type="rectilinear_cartesian",  # m6b name for even_spacing
            )

            return self.m6f_hgrid.write_supergrid(self.mom_input_dir / "hgrid.nc")

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
            self.m6f_vgrid = VGrid.hyperbolic(
                self.number_vertical_layers, self.depth, self.layer_thickness_ratio
            )
            thicknesses = self.m6f_vgrid.dz
        else:
            self.m6f_vgrid = VGrid(thicknesses)

        ## Check whether the minimum depth is less than the first three layers

        if len(self.m6f_vgrid.zi) > 2 and self.minimum_depth < self.m6f_vgrid.zi[2]:
            print(
                f"Warning: Minimum depth of {self.minimum_depth}m is less than the depth of the third interface ({self.m6f_vgrid.zi[2]}m)!\n"
                + "This means that some areas may only have one or two layers between the surface and sea floor. \n"
                + "For increased stability, consider increasing the minimum depth, or adjusting the vertical coordinate to add more layers near the surface."
            )
        ds = self.m6f_vgrid.write_z_file(self.mom_input_dir / "vcoord.nc")

        return ds

    def setup_initial_condition(
        self,
        raw_ic_path,
        varnames,
        arakawa_grid="A",
        vcoord_type="height",
        regridding_method=None,
    ):
        """
        Reads the initial condition from files in ``raw_ic_path``, interpolates to the
        model grid, fixes up metadata, and saves back to the input directory.

        Arguments:
            raw_ic_path (Union[str, Path]): Path to raw initial condition file to read in.
            varnames (Dict[str, str]): Mapping from MOM6 variable/coordinate names to the names
                in the input dataset. For example, ``{'xq': 'lonq', 'yh': 'lath', 'salt': 'so', ...}``.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the initial condition.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
            vcoord_type (Optional[str]): The type of vertical coordinate used in the forcing files.
                Either ``'height'`` or ``'thickness'``.
            regridding_method (Optional[str]): The type of regridding method to use. Defaults to self.regridding_method
        """
        if regridding_method is None:
            regridding_method = self.regridding_method

        reprocessed_var_map = rgd.apply_arakawa_grid_mapping(
            var_mapping=varnames, arakawa_grid=arakawa_grid
        )

        if not Path(raw_ic_path).exists():
            raise FileNotFoundError(
                f"Initial condition file not found at {raw_ic_path}. Please ensure that the files are named in the format `ic_unprocessed.nc`."
            )
        ic_raw = xr.open_dataset(raw_ic_path)

        # There is a case where MARBL tracers have multiple zdims, this is not supported for initial conditions:
        if type(reprocessed_var_map["depth_coord"]) == list:
            reprocessed_var_map["depth_coord"] = reprocessed_var_map["depth_coord"][0]

        # Convert zdim if possible & needed
        ic_raw[reprocessed_var_map["depth_coord"]] = try_pint_convert(
            ic_raw[reprocessed_var_map["depth_coord"]],
            "m",
            reprocessed_var_map["depth_coord"],
        )

        # Convert values
        for var in rgd.main_field_target_units:
            if var == "temp" or var == "salt":
                value_name = reprocessed_var_map["tracer_var_names"][var]
            else:
                value_name = reprocessed_var_map[var + "_var_name"]
            ic_raw[value_name] = try_pint_convert(
                ic_raw[value_name], rgd.main_field_target_units[var], var
            )
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
        ## Although we now attempt a pint convert, we're leaving this manual conversion in for now
        ## just in case, as K->C is absolutely necessary, and for some inputs pint may fail where this won't.
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

        ic_raw_eta = ic_raw_eta.rename(
            {
                reprocessed_var_map["tracer_lat_coord"]: "lat",
                reprocessed_var_map["tracer_lon_coord"]: "lon",
            }
        )

        hgrid = self.hgrid
        hgrid["lon"] = hgrid["x"]
        hgrid["lat"] = hgrid["y"]
        tgrid = (
            rgd.get_hgrid_arakawa_c_points(hgrid, "t")
            .rename({"tlon": "lon", "tlat": "lat", "nxp": "nx", "nyp": "ny"})
            .set_coords(["lat", "lon"])
        )

        ## Make our three horizontal regridders

        regridder_u = rgd.create_regridder(
            ic_raw_u, hgrid, locstream_out=False, method=regridding_method
        )
        regridder_v = rgd.create_regridder(
            ic_raw_v, hgrid, locstream_out=False, method=regridding_method
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
            radian_angle=np.radians(hgrid.angle_dx.values),
        )

        # Slice the velocites to the u and v grid.
        u_points = rgd.get_hgrid_arakawa_c_points(hgrid, "u")
        v_points = rgd.get_hgrid_arakawa_c_points(hgrid, "v")
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

        encoding = {
            var: {"_FillValue": -1e20, "missing_value": -1e20}
            for var in reprocessed_var_map["tracer_var_names"].keys()
        }
        tracers_out.to_netcdf(
            self.mom_input_dir / "init_tracers.nc",
            mode="w",
            encoding=encoding,
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

        validate_general_file(
            xr.Dataset({"eta_t": eta_out}),
            ["eta_t"],
            {
                "eta_t": {"_FillValue": None},
            },
        )
        validate_general_file(
            tracers_out,
            list(reprocessed_var_map["tracer_var_names"].keys()),
            encoding,
        )
        validate_general_file(
            vel_out,
            ["u", "v"],
            {
                "u": {"_FillValue": netCDF4.default_fillvals["f4"]},
                "v": {"_FillValue": netCDF4.default_fillvals["f4"]},
            },
        )
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
                self.date_range[0] + dt.timedelta(days=1),
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
        bgc_tracer_names: dict = None,
        arakawa_grid="A",
        bathymetry_path=None,
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
            bgc_tracer_names (Dict[str, str]): Specify the BGC tracer names to the name in the
                input dataset, can also be specified in the varnames dict but this is here so we can reformat the output into seperate files. For example, ``{'oxygen': 'o2', 'phosphate': 'po4', ...}``.
            boundaries (List[str]): List of cardinal directions for which to create boundary forcing files.
                Default is ``["south", "north", "west", "east"]``.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the boundary forcing.
                Either ``'A'`` (default), ``'B'``, or ``'C'``.
            bathymetry_path (Optional[str]): Path to the bathymetry file. Default is ``None``, in which case the
                boundary condition is not masked.
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
                "This method only supports up to four boundaries. To set up more complex boundary shapes, construct a "
                "regional_mom6.segment.Segment directly (e.g. via Segment.from_hgrid) and call its "
                "regrid_velocity_tracers method for each boundary."
            )

        if bgc_tracer_names is None:
            bgc_tracer_names = {}

        # In the future, we should change varnames to physical_varnames
        physical_varnames = varnames
        all_varnames = physical_varnames.copy()

        # Merge the bgc tracer names into the varnames
        if bgc_tracer_names:
            key = "tracers" if "tracers" in physical_varnames else "tracer_var_names"
            all_varnames[key] = {**physical_varnames[key], **bgc_tracer_names}

        # Now iterate through our four boundaries
        for orientation in self.boundaries:
            self.setup_single_boundary(
                Path(raw_boundaries_path / (orientation + "_unprocessed.nc")),
                all_varnames,
                orientation,  # The cardinal direction of the boundary
                self.find_MOM6_rectangular_orientation(
                    orientation
                ),  # A number to identify the boundary; indexes from 1
                arakawa_grid=arakawa_grid,
                bathymetry_path=bathymetry_path,
                regridding_method=regridding_method,
                fill_method=fill_method,
            )

        # Scrape the bgc var names into their own files for the boundary conditions (required for generic tracers at the moment, Apr 2026)
        self.reformat_bgc_tracers_into_files(bgc_tracer_names)

    def reformat_bgc_tracers_into_files(self, bgc_tracer_names: dict = None):
        """
        Reformat the boundary condition files so that the BGC tracers are in separate files from the physical tracers. This is required for generic tracers at the moment (Apr 2026) but may not be in the future as we add more flexibility to the code.

        Arguments:
            bgc_tracer_names (Dict[str, str]): Specify the BGC tracer names to the name in the
                input dataset, can also be specified in the varnames dict but this is here so we can reformat the output into seperate files. For example, ``{'oxygen': 'o2', 'phosphate': 'po4', ...}``.
        """

        if bgc_tracer_names is None or bgc_tracer_names == {}:
            return

        # Read in the forcing datasets
        datasets = {}
        for boundary in self.boundaries:
            num = str(self.find_MOM6_rectangular_orientation(boundary)).zfill(3)
            datasets[num] = xr.open_dataset(
                self.mom_input_dir / f"forcing_obc_segment_{num}.nc"
            )

        # Get base variable names
        base_vars = list(bgc_tracer_names.keys())
        for var in base_vars:
            ds_var = xr.Dataset()
            for key, ds in datasets.items():
                var_name = f"{var}_segment_{key}"  # 001, 002, 003, 004
                ds_var[var_name] = ds[var_name]
                dz_var_name = f"dz_{var_name}"
                if dz_var_name in ds:
                    ds_var[dz_var_name] = ds[dz_var_name]
            output_file = self.mom_input_dir / f"{var}_obc_segment.nc"
            ds_var.to_netcdf(output_file, unlimited_dims="time")
            print("Saved BGC tracer {} to file {}".format(var, output_file))

    def setup_single_boundary(
        self,
        path_to_bc,
        varnames,
        orientation,
        segment_number,
        arakawa_grid="A",
        bathymetry_path=None,
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
        if not Path(path_to_bc).exists():
            raise FileNotFoundError(
                f"Boundary file not found at {path_to_bc}. Please ensure that the files are named in the format `east_unprocessed.nc`."
            )
        segment = self._get_segment(orientation, bathymetry_path=bathymetry_path)

        segment.regrid_velocity_tracers(
            infile=path_to_bc,  # location of raw boundary
            varnames=varnames,
            outfolder=self.mom_input_dir,
            startdate=self.date_range[0],
            arakawa_grid=arakawa_grid,
            regridding_method=regridding_method,
            fill_method=fill_method,
            repeat_year_forcing=self.repeat_year_forcing,
        )

        print("Done.")
        return

    def setup_boundary_tides(
        self,
        tpxo_elevation_filepath,
        tpxo_velocity_filepath,
        tidal_constituents=None,
        bathymetry_path=None,
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
            regridding_method (Optional[str]): The type of regridding method to use. Defaults to self.regridding_method
            fill_method (Function): Fill method to use throughout the function. Default is ``self.fill_method``

        Returns:
            netCDF files: Regridded tidal velocity and elevation files in 'inputdir/forcing'

        The tidal data functions are sourced from the GFDL NWA25 and modified so that:

        - Converted code for regional-mom6 :class:`~regional_mom6.segment.Segment` class
        - Implemented horizontal subsetting.
        - Combined all functions of NWA25 into a four function process (in the style of regional-mom6), i.e.,
          :func:`~experiment.setup_boundary_tides`, :meth:`Segment.from_hgrid`, :meth:`Segment.regrid_tides`, and
          :meth:`Segment.encode_tidal_files_and_output`.

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

            # If ocean-state setup already built this segment, reuse it instead of
            # re-deriving it from the grid again.
            segment = self._get_segment(b, bathymetry_path=bathymetry_path)

            # Output and regrid tides
            segment.regrid_tides(
                tpxo_v,
                tpxo_u,
                tpxo_h,
                times,
                outfolder=self.mom_input_dir,
                startdate=self.date_range[0],
                regridding_method=regridding_method,
                repeat_year_forcing=self.repeat_year_forcing,
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
        depth_method="xesmf",
        mask_method="dataset",
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
            depth_method (Optional[str]): Method used to set the depth: ``'stats'`` (statistic from
                sub-sampled source data), ``'xesmf'`` (direct xESMF regrid of the source depth), or
                ``'cressman'`` (Cressman interpolation). Default: ``'xesmf'``.
            mask_method (Optional[str]): Method used to distinguish ocean from land: ``'naturalearth'``,
                ``'ocean_frac'``, ``'dataset'``, or ``'manual'`` (uses ``self.m6f_bathymetry.user_mask``, which must
                already be set). Default: ``'dataset'``.
        """

        print(
            "Setting up bathymetry...if this fails, please follow the printed instructions with your experiment's m6f_bathymetry object, like this: [experiment_obj].m6f_bathymetry. For example, if the output tells you to run mpi_set_from_dataset instead of set_from_dataset. You would do: [experiment_obj].m6f_bathymetry.mpi_set_from_dataset(...)"
        )
        if regridding_method is None:
            regridding_method = self.regridding_method

        self.m6f_bathymetry = Topo(
            grid=self.m6f_hgrid, min_depth=self.minimum_depth, git=False
        )
        self._bathymetry = None  # invalidate any cached `bathymetry` view

        self.m6f_bathymetry.set_from_dataset(
            bathymetry_path=bathymetry_path,
            output_dir=self.mom_input_dir,
            longitude_coordinate_name=longitude_coordinate_name,
            latitude_coordinate_name=latitude_coordinate_name,
            vertical_coordinate_name=vertical_coordinate_name,
            regridding_method=regridding_method,
            fill_channels=fill_channels,
            is_input_positive_below_msl=positive_down,
            write_to_file=write_to_file,
            depth_method=depth_method,
            mask_method=mask_method,
        )
        self.m6f_bathymetry.write_topo(self.mom_input_dir / "bathymetry.nc")
        return self.m6f_bathymetry.gen_topo_ds()

    def tidy_bathymetry(
        self,
        fill_channels=False,
    ):
        if fill_channels:
            self.m6f_bathymetry.fill_inland_lakes_and_channels()
        self.m6f_bathymetry.write_topo(
            self.mom_input_dir / "bathymetry.nc",
        )
        return self.m6f_bathymetry.gen_topo_ds()

    def setup_chl(self, processed_seawifs_path, output_path=None):
        """
        Interpolate and fill the SeaWiFS chlorophyll climatology onto the experiment's grid.

        Output is saved in the input directory of the experiment, unless a different
        ``output_path`` is provided.

        Arguments:
            processed_seawifs_path (str): Path to the preprocessed SeaWiFS chlorophyll dataset.
            output_path (Optional[str]): Path to save the output NetCDF file. Defaults to
                ``mom_input_dir / f"seawifs-clim-1997-2010-{expt_name}.nc"``.
        """
        self.hgrid  # ensures `m6f_hgrid` is populated (from disk if needed)
        self.bathymetry  # ensures `m6f_bathymetry` is populated (from disk if needed)

        if output_path is None:
            output_path = (
                self.mom_input_dir / f"seawifs-clim-1997-2010-{self.expt_name}.nc"
            )

        return interpolate_and_fill_seawifs(
            self.m6f_hgrid,
            self.m6f_bathymetry,
            processed_seawifs_path,
            output_path=output_path,
        )

    def run_FRE_tools(self):
        """
        A wrapper for FRE Tools ``check_mask``, ``make_solo_mosaic``, and ``make_quick_mosaic``.

        This method is not needed if you're running under NUOPC (e.g., NCAR/CROCODILE or most ACCESS/COSIMA people). However, if you're not using the auto-mask table, then this is the only way within the regional-mom6 package to generate a cpu mask file.

        The FRE tools require some additional attributes and dimensions on the bathymetry and hgrid files, which are added here before calling the tools.

        """

        if not (self.mom_input_dir / "bathymetry.nc").exists():
            print("No bathymetry file! Need to run setup_bathymetry method first")
            return

        for p in self.mom_input_dir.glob("mask_table*"):
            p.unlink()

        # If ntiles not present in hgrid & topography, add them
        if "ntiles" not in self.bathymetry.dims:
            self.bathymetry.expand_dims({"ntiles": 1}).to_netcdf(
                self.mom_input_dir / "bathymetry.nc",
                mode="w",
            )

        if "tile" not in self.hgrid:
            # `self.hgrid` always regenerates from `m6f_hgrid`, so the "tile" coord is
            # added to a local copy and written straight to `hgrid.nc` here, rather than
            # persisted back onto `self.hgrid` (which would just be regenerated away).
            hgrid_with_tile = self.hgrid.assign(
                {
                    "tile": (
                        (),
                        np.array(b"tile1", dtype="|S255"),
                        {
                            "standard_name": "grid_tile_spec",
                            "geometry": "spherical",
                            "north_pole": "0.0 90.0",
                            "discretization": "logically_rectangular",
                            "conformal": "true",
                        },
                    )
                }
            )
            hgrid_with_tile.to_netcdf(
                self.mom_input_dir / "hgrid.nc", format="NETCDF3_64BIT", mode="w"
            )

        print(
            "Running GFDL's FRE Tools. The following information is all printed by the FRE tools themselves"
        )
        # First run the make solo mosaic. This reads hgrid and outputs the ocean_mosaic.nc file
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

        # Next the quick mosaic function takes the mosaic we just made and the bathymetry to make the grid_spec file
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

    def setup_generic(self, ncpus=100, mask_land_cpus=True):
        """
        Set up the run directory for the model run. This is a multi step process - given that NUOPC vs FMS based runs are quite different, this function handles all of the setup steps that they share in common, with more specific steps performed in setup_rOM3 and setup_FMS_version respectively.

        The main thing this function does is manage the MOM_override file, as this is common to both use cases.

        Arguments:
            ncpus (Optional[int]): The number of PEs to use
            mask_land_cpus (Optional[bool]): If your domain has enough land in it that some processors would only have land to deal with, set to True. If a mostly water domain, set to False otherwise the automatic mask table throws a fatal (see issue: https://github.com/issues/created?issue=mom-ocean%7CMOM6%7C1686)
        """
        # Check if we need tides
        with_tides = len(self.tidal_constituents) > 0
        if with_tides:
            tidal_files_exist = any(Path(self.mom_input_dir).rglob("tu*"))

            if not tidal_files_exist:
                raise ValueError(
                    "No files with 'tu' in their names found in the forcing or input directory. If you meant to use tides, please run the setup_boundary_tides method first to create tidal files. If you didn't, set ``tidal_constituants = []`` when defining experiment."
                )

        ### Make symlinks between run and input directories ###
        inputdir_in_rundir = self.mom_run_dir / "inputdir"
        rundir_in_inputdir = self.mom_input_dir / "rundir"

        inputdir_in_rundir.unlink(missing_ok=True)
        inputdir_in_rundir.symlink_to(self.mom_input_dir)

        rundir_in_inputdir.unlink(missing_ok=True)
        rundir_in_inputdir.symlink_to(self.mom_run_dir)

        ### Write to the MOM_override file ###

        MOM_override_dict = mpt.read_MOM_file_as_dict("MOM_override", self.mom_run_dir)

        MOM_override_dict["MINIMUM_DEPTH"]["value"] = float(self.minimum_depth)

        # Define spatial dimensions
        nx = self.hgrid.nx.shape[0] // 2
        ny = self.hgrid.ny.shape[0] // 2
        MOM_override_dict["NK"]["value"] = len(self.vgrid.zl.values)
        MOM_override_dict["NIGLOBAL"]["value"] = nx
        MOM_override_dict["NJGLOBAL"]["value"] = ny

        # If we're not using the Auto Mask Table feature, need a mask table:
        if mask_land_cpus == True:
            MOM_override_dict["AUTO_MASKTABLE"]["value"] = True
        else:
            # No mask table at all
            MOM_override_dict["AUTO_MASKTABLE"]["value"] = False
            MOM_override_dict["MASKTABLE"]["value"] = "None"

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

            index_str = '"' + self._get_segment(seg).mom6_obc_position_string()

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
            MOM_override_dict["TIDES"][
                "comment"
            ] = "This turns on body tidal forcing in the interior of domain."
            for constituent in self.tidal_constituents:
                MOM_override_dict[f"TIDE_{constituent.upper()}"]["value"] = "True"

            # OBC tides
            MOM_override_dict["OBC_TIDE_CONSTITUENTS"]["value"] = (
                '"' + ", ".join(self.tidal_constituents) + '"'
            )
            MOM_override_dict["OBC_TIDE_CONSTITUENTS"][
                "comment"
            ] = "OBC_TIDE constituent settings define the tidal forcing at boundaries"
            MOM_override_dict["OBC_TIDE_ADD_EQ_PHASE"]["value"] = "True"
            MOM_override_dict["OBC_TIDE_N_CONSTITUENTS"]["value"] = len(
                self.tidal_constituents
            )
            MOM_override_dict["OBC_TIDE_REF_DATE"]["value"] = self.date_range[
                0
            ].strftime("%Y, %m, %d")

        # Chlorophyll shortwave penetration, if setup_chl has been run
        chl_files = list(Path(self.mom_input_dir).glob("seawifs-clim-*.nc"))
        if chl_files:
            MOM_override_dict["CHL_FILE"]["value"] = f'"{chl_files[0].name}"'
            MOM_override_dict["CHL_FROM_FILE"]["value"] = "True"
            MOM_override_dict["VAR_PEN_SW"]["value"] = "True"
            MOM_override_dict["PEN_SW_NBANDS"]["value"] = 3

        for key, val in MOM_override_dict.items():
            if isinstance(val, dict) and key != "original":
                MOM_override_dict[key]["override"] = True
        mpt.write_MOM_file(MOM_override_dict, self.mom_run_dir)

        # Modify the config.yaml file. This is the same whether NUOPC or FMS
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        yaml.indent(
            mapping=4, sequence=4, offset=4
        )  # Preserve the 4 space indent formatting
        yaml.width = 4096  # Prevent line wrapping
        with open(self.mom_run_dir / "config.yaml", "r") as file:
            config = yaml.load(file)
        config["ncpus"] = ncpus
        config["jobname"] = self.mom_run_dir.name
        config["input"][0] = str(self.mom_input_dir)
        with open(self.mom_run_dir / "config.yaml", "w") as file:
            yaml.dump(config, file)

        return

    def setup_rOM3(self, ncpus=208, mask_land_cpus=True, overwrite=True):
        """
        Set up the run directory for an ACCESS-regional-ocean-model-3 experiment. This function copies existing configuration files (MOM_input,config.yaml etc.) from an ACCESS-NRI supported source to ensure that users have access to the latest executable and fixes.


        Arguments:
            ncpus (Optional[int]): The number of PEs to use
            mask_land_cpus (Optional[bool]): If your domain has enough land in it that some processors would only have land to deal with, set to True. If a mostly water domain, set to False otherwise the automatic mask table throws a fatal (see issue: https://github.com/issues/created?issue=mom-ocean%7CMOM6%7C1686)
            overwrite (Optional[bool]): If true, reset the run directory. Set to False to attempt to attempt to modify the files in an exsiting run directory.
        """
        if os.path.exists(self.mom_run_dir) and overwrite:
            shutil.rmtree(self.mom_run_dir)
        else:
            print(
                "Overwrite set to False. I'll attempt to modify existing files in the directory rather than re-populate it from scratch. \nIf there are issues, try re-run with overwrite=True, or make your intended changes manually."
            )

        # First, make the ESMF mesh file required for all NUOPC based runs, like rom3
        self.topo.write_esmf_mesh(self.mom_input_dir / "access-rom3-ESMFmesh.nc")
        # Now modify to make a mask free version
        maskmesh = xr.open_dataset(self.mom_input_dir / "access-rom3-ESMFmesh.nc")
        maskmesh.elementMask[:] = 1
        maskmesh.to_netcdf(self.mom_input_dir / "access-rom3-nomask-ESMFmesh.nc")

        #! PLACEHOLDER
        #! need to implement something like:
        #! payu clone stencil_name self.mom_run_dir.
        #!
        shutil.copytree(
            "/g/data/ol01/ab8992/access-om3-configs",
            self.mom_run_dir,
            dirs_exist_ok=True,
        )
        #!
        #! END PLACEHOLDER

        # Run the generic setup that's required for all rmom6 runs

        self.setup_generic(ncpus=ncpus, mask_land_cpus=mask_land_cpus)

        nx = self.hgrid.nx.shape[0] // 2
        ny = self.hgrid.ny.shape[0] // 2
        with open(f"{self.mom_run_dir}/nuopc.runconfig", "r") as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if "     start_ymd" in lines[i]:
                    lines[i] = (
                        f"     start_ymd = {self.date_range[0].strftime('%Y%m%d')}\n"
                    )
                if "ocn_nx" in lines[i]:
                    lines[i] = f"     ocn_nx = {nx}\n"
                if "ocn_ny" in lines[i]:
                    lines[i] = f"     ocn_ny = {ny}\n"
        with open(f"{self.mom_run_dir}/nuopc.runconfig", "w") as file:
            file.writelines(lines)

        # Modify the drof / datm files to all have the right number of x and y points
        datm = f90nml.read(self.mom_run_dir / "datm_in")
        datm["datm_nml"]["nx_global"]

        for i in ["drof", "datm"]:
            file = f90nml.read(self.mom_run_dir / f"{i}_in")
            file[f"{i}_nml"]["nx_global"] = nx
            file[f"{i}_nml"]["ny_global"] = ny
            file.write(self.mom_run_dir / f"{i}_in", force=True)

        return

    def setup_fms_version(self, ncpus=100, surface_forcing=None, mask_land_cpus=True):
        """
        Set up the run directory for MOM6. Either copy a pre-made set of files, or modify
        existing files in the 'rundir' directory for the experiment.

        Arguments:
            surface_forcing (Optional[str]): Specify the choice of surface forcing, one
                of: ``'jra'`` or ``'era5'``. If not prescribed then constant fluxes are used.
            mask_land_cpus (Optional[bool]): If your domain has enough land in it that some processors would only have land to deal with, set to True. If a mostly water domain, set to False otherwise the automatic mask table throws a fatal (see issue: https://github.com/issues/created?issue=mom-ocean%7CMOM6%7C1686)
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

        shutil.copytree(base_run_dir, self.mom_run_dir, dirs_exist_ok=True)
        if overwrite_run_dir != False:
            shutil.copytree(overwrite_run_dir, self.mom_run_dir, dirs_exist_ok=True)

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

        self.setup_generic(ncpus=ncpus, mask_land_cpus=mask_land_cpus)

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
