from regional_mom6 import utils
from regional_mom6.regridding import get_hgrid_arakawa_c_points, coords

rotation_logger = utils.setup_logger(__name__, set_handler=False)
# An Enum is like a dropdown selection for a menu, it essentially limits the type of input parameters. It comes with additional complexity, which of course is always a challenge.
from enum import Enum
import xarray as xr
import numpy as np


class RotationMethod(Enum):
    """Prescribes the rotational method to be used in boundary conditions.

    The main regional-mom6 class passes this ``Enum`` to ``regrid_tides`` and ``regrid_velocity_tracers`` to determine the method used.

    Attributes:
        EXPAND_GRID (int): This method is used with the basis that we can find the angles at the q-u-v points by pretending we have another row/column of the ``hgrid`` with the same distances as the t-point to u/v points in the actual grid then use the four points to calculate the angle. This method replicates exactly what MOM6 does.
        GIVEN_ANGLE (int): Expects a pre-given angle called ``angle_dx``.
        NO_ROTATION (int): Grid is along lines of constant latitude-longitude and therefore no rotation is required.
    """

    EXPAND_GRID = 1
    GIVEN_ANGLE = 2
    NO_ROTATION = 3

def initialize_grid_rotation_angles_using_expanded_hgrid(
    hgrid: xr.Dataset,
) -> xr.Dataset:
    """
    Calculate the ``angle_dx`` in degrees from the true x (east?) direction counter-clockwise) and return as a dataarray.

    Parameters
    ----------
    hgrid: xr.Dataset
        The hgrid dataset

    Returns
    -------
    xr.DataArray
        The t-point angles
    """
    # Get expanded (pseudo) grid
    expanded_hgrid = create_expanded_hgrid(hgrid)

    return mom6_angle_calculation_method(
        expanded_hgrid.x.max() - expanded_hgrid.x.min(),
        expanded_hgrid.isel(nyp=slice(2, None), nxp=slice(0, -2)),
        expanded_hgrid.isel(nyp=slice(2, None), nxp=slice(2, None)),
        expanded_hgrid.isel(nyp=slice(0, -2), nxp=slice(0, -2)),
        expanded_hgrid.isel(nyp=slice(0, -2), nxp=slice(2, None)),
        hgrid,
    )


def initialize_grid_rotation_angle(hgrid: xr.Dataset) -> xr.DataArray:
    """
    Calculate the ``angle_dx`` in degrees from the true x (east?) direction counter-clockwise) and return as a dataarray.

    Parameters
    ----------
    hgrid: xr.Dataset
        The hgrid dataset

    Returns
    -------
    xr.DataArray
        The t-point angles
    """
    ds_t = get_hgrid_arakawa_c_points(hgrid, "t")
    ds_q = get_hgrid_arakawa_c_points(hgrid, "q")

    # Reformat into x, y comps
    t_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), ds_t.tlon.data),
            "y": (("nyp", "nxp"), ds_t.tlat.data),
        }
    )
    q_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), ds_q.qlon.data),
            "y": (("nyp", "nxp"), ds_q.qlat.data),
        }
    )

    return mom6_angle_calculation_method(
        hgrid.x.max() - hgrid.x.min(),
        q_points.isel(nyp=slice(1, None), nxp=slice(0, -1)),
        q_points.isel(nyp=slice(1, None), nxp=slice(1, None)),
        q_points.isel(nyp=slice(0, -1), nxp=slice(0, -1)),
        q_points.isel(nyp=slice(0, -1), nxp=slice(1, None)),
        t_points,
    )


def modulo_around_point(x, xc, Lx):
    """
    Calculate the modulo around a point. Return the modulo value of x in an interval [xc-(Lx/2), xc+(Lx/2)]. If Lx<=0, then it returns x without applying modulo arithmetic.

    Parameters
    ----------
    x: float
        Value to which to apply modulo arithmetic
    xc: float
        Center of modulo range
    Lx: float
        Modulo range width

    Returns
    -------
    float
        x shifted by an integer multiple of Lx to be close to xc,
    """
    if Lx <= 0:
        return x
    else:
        return ((x - (xc - 0.5 * Lx)) % Lx) - Lx / 2 + xc


def mom6_angle_calculation_method(
    len_lon,
    top_left: xr.DataArray,
    top_right: xr.DataArray,
    bottom_left: xr.DataArray,
    bottom_right: xr.DataArray,
    point: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate the angle of the point using the MOM6 method in ``initialize_grid_rotation_angle``. Built for vectorized calculations.

    Parameters
    ----------
    len_lon: float
        The length of the longitude of the regional domain
    top_left, top_right, bottom_left, bottom_right: xr.DataArray
        The four points around the point to calculate the angle from the hgrid requires an x and y component
    point: xr.DataArray
        The point to calculate the angle from the hgrid

    Returns
    -------
    xr.DataArray
        The angle of the point
    """
    rotation_logger.info("Calculating grid rotation angle")
    # Direct Translation
    pi_720deg = (
        np.arctan(1) / 180
    )  # One quarter the conversion factor from degrees to radians

    # Compute lonB for all points
    lonB = np.zeros((2, 2, len(point.nyp), len(point.nxp)))

    # Vectorized computation of lonB
    # Vectorized computation of lonB
    lonB[0][0] = modulo_around_point(bottom_left.x, point.x, len_lon)  # Bottom Left
    lonB[1][0] = modulo_around_point(top_left.x, point.x, len_lon)  # Top Left
    lonB[1][1] = modulo_around_point(top_right.x, point.x, len_lon)  # Top Right
    lonB[0][1] = modulo_around_point(bottom_right.x, point.x, len_lon)  # Bottom Right

    # Compute lon_scale
    lon_scale = np.cos(
        pi_720deg * ((bottom_left.y + bottom_right.y) + (top_right.y + top_left.y))
    )

    # Compute angle
    angle = np.arctan2(
        lon_scale * ((lonB[0, 1] - lonB[1, 0]) + (lonB[1, 1] - lonB[0, 0])),
        (bottom_left.y - top_right.y) + (top_left.y - bottom_right.y),
    )
    # Assign angle to angles_arr
    angles_arr = np.rad2deg(angle) - 90

    # Assign angles_arr to hgrid
    t_angles = xr.DataArray(
        angles_arr,
        dims=["nyp", "nxp"],
        coords={
            "nyp": point.nyp.values,
            "nxp": point.nxp.values,
        },
    )
    return t_angles


def create_expanded_hgrid(hgrid: xr.Dataset, expansion_width=1) -> xr.Dataset:
    """
    Adds an additional boundary to the hgrid to allow for the calculation of the ``angle_dx`` for the boundary points using the method in MOM6.
    """
    if expansion_width != 1:
        raise NotImplementedError("Only expansion_width = 1 is supported")

    pseudo_hgrid_x = np.full((len(hgrid.nyp) + 2, len(hgrid.nxp) + 2), np.nan)
    pseudo_hgrid_y = np.full((len(hgrid.nyp) + 2, len(hgrid.nxp) + 2), np.nan)

    ## Fill Boundaries
    pseudo_hgrid_x[1:-1, 1:-1] = hgrid.x.values
    pseudo_hgrid_x[0, 1:-1] = hgrid.x.values[0, :] - (
        hgrid.x.values[1, :] - hgrid.x.values[0, :]
    )  # Bottom Fill
    pseudo_hgrid_x[-1, 1:-1] = hgrid.x.values[-1, :] + (
        hgrid.x.values[-1, :] - hgrid.x.values[-2, :]
    )  # Top Fill
    pseudo_hgrid_x[1:-1, 0] = hgrid.x.values[:, 0] - (
        hgrid.x.values[:, 1] - hgrid.x.values[:, 0]
    )  # Left Fill
    pseudo_hgrid_x[1:-1, -1] = hgrid.x.values[:, -1] + (
        hgrid.x.values[:, -1] - hgrid.x.values[:, -2]
    )  # Right Fill

    pseudo_hgrid_y[1:-1, 1:-1] = hgrid.y.values
    pseudo_hgrid_y[0, 1:-1] = hgrid.y.values[0, :] - (
        hgrid.y.values[1, :] - hgrid.y.values[0, :]
    )  # Bottom Fill
    pseudo_hgrid_y[-1, 1:-1] = hgrid.y.values[-1, :] + (
        hgrid.y.values[-1, :] - hgrid.y.values[-2, :]
    )  # Top Fill
    pseudo_hgrid_y[1:-1, 0] = hgrid.y.values[:, 0] - (
        hgrid.y.values[:, 1] - hgrid.y.values[:, 0]
    )  # Left Fill
    pseudo_hgrid_y[1:-1, -1] = hgrid.y.values[:, -1] + (
        hgrid.y.values[:, -1] - hgrid.y.values[:, -2]
    )  # Right Fill

    ## Fill Corners
    pseudo_hgrid_x[0, 0] = hgrid.x.values[0, 0] - (
        hgrid.x.values[1, 1] - hgrid.x.values[0, 0]
    )  # Bottom Left
    pseudo_hgrid_x[-1, 0] = hgrid.x.values[-1, 0] - (
        hgrid.x.values[-2, 1] - hgrid.x.values[-1, 0]
    )  # Top Left
    pseudo_hgrid_x[0, -1] = hgrid.x.values[0, -1] - (
        hgrid.x.values[1, -2] - hgrid.x.values[0, -1]
    )  # Bottom Right
    pseudo_hgrid_x[-1, -1] = hgrid.x.values[-1, -1] - (
        hgrid.x.values[-2, -2] - hgrid.x.values[-1, -1]
    )  # Top Right

    pseudo_hgrid_y[0, 0] = hgrid.y.values[0, 0] - (
        hgrid.y.values[1, 1] - hgrid.y.values[0, 0]
    )  # Bottom Left
    pseudo_hgrid_y[-1, 0] = hgrid.y.values[-1, 0] - (
        hgrid.y.values[-2, 1] - hgrid.y.values[-1, 0]
    )  # Top Left
    pseudo_hgrid_y[0, -1] = hgrid.y.values[0, -1] - (
        hgrid.y.values[1, -2] - hgrid.y.values[0, -1]
    )  # Bottom Right
    pseudo_hgrid_y[-1, -1] = hgrid.y.values[-1, -1] - (
        hgrid.y.values[-2, -2] - hgrid.y.values[-1, -1]
    )  # Top Right

    pseudo_hgrid = xr.Dataset(
        {
            "x": (["nyp", "nxp"], pseudo_hgrid_x),
            "y": (["nyp", "nxp"], pseudo_hgrid_y),
        }
    )
    return pseudo_hgrid


def get_rotation_angle(
    rotational_method: RotationMethod, hgrid: xr.Dataset, orientation=None
):
    """
    Returns the rotation angle - THIS IS ALWAYS BASED ON THE ASSUMPTION OF DEGREES - based on the rotational method and provided hgrid, if orientation & coords are provided, it will assume the boundary is requested.

    Parameters
    ----------
    rotational_method: RotationMethod
        The rotational method to use
    hgrid: xr.Dataset
        The hgrid dataset
    orientation: xr.Dataset
        The orientation, which also lets us now that we are on a boundary

    Returns
    -------
    xr.DataArray
        angle in degrees
    """
    rotation_logger.info("Getting rotation angle")
    boundary = False
    if orientation != None:
        rotation_logger.debug(
            "The rotational angle is requested for the boundary: {}".format(orientation)
        )
        boundary = True

    if rotational_method == RotationMethod.NO_ROTATION:
        rotation_logger.debug("Using NO_ROTATION method")
        if not utils.is_rectilinear_hgrid(hgrid):
            raise ValueError("NO_ROTATION method only works with rectilinear grids")
        angles = xr.zeros_like(hgrid.x)

        if boundary:
            # Subset to just boundary
            # Add zeroes to hgrid
            hgrid["zero_angle"] = angles

            # Cut to boundary
            zero_angle = coords(
                hgrid,
                orientation,
                "doesnt_matter",
                angle_variable_name="zero_angle",
            )["angle"]

            return zero_angle
        else:
            return angles
    elif rotational_method == RotationMethod.GIVEN_ANGLE:
        rotation_logger.debug("Using GIVEN_ANGLE method")
        if boundary:
            return coords(
                hgrid, orientation, "doesnt_matter", angle_variable_name="angle_dx"
            )["angle"]
        else:
            return hgrid["angle_dx"]
    elif rotational_method == RotationMethod.EXPAND_GRID:
        rotation_logger.debug("Using EXPAND_GRID method")
        hgrid["angle_dx_rm6"] = initialize_grid_rotation_angles_using_expanded_hgrid(
            hgrid
        )

        if boundary:
            degree_angle = coords(
                hgrid,
                orientation,
                "doesnt_matter",
                angle_variable_name="angle_dx_rm6",
            )["angle"]
            return degree_angle
        else:
            return hgrid["angle_dx_rm6"]
    else:
        raise ValueError("Invalid rotational method")
