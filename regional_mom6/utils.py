import numpy as np
import logging
import sys
import xarray as xr
from regional_mom6 import regridding as rgd
from pathlib import Path
import pint
import pint_xarray
import importlib.resources
# from pint_xarray.errors import PintExceptionGroup # This is only supported when pint_xarray is 0.6.0, which is not currently supported in the CI

# Handle Unit Registry (only done once)
ureg = pint.UnitRegistry(
    force_ndarray_like=True
)  # The force option is required for pint_xarray

# Open the unit definitions file from the package safely
with importlib.resources.open_text("regional_mom6", "rm6_unit_defs.txt") as f:
    ureg.load_definitions(f)


def try_pint_convert(da, target_units, var_name=None):
    """
    Try to quantify and convert a DataArray using pint.
    Falls back if it fails or if units are invalid.
    Returns a plain (non-pint) DataArray.
    """
    try:
        # Attempt to pintify the dataarray
        source_units = da.attrs.get("units", None)
        if not source_units:
            raise ValueError(f"DataArray 'var_name' has no units; cannot quantify.")
        if not isinstance(da.data, pint.Quantity):
            da_quantified = da.pint.quantify(unit_registry=ureg)

            # This is only supported when pint_xarray is 0.6.0, which is not currently supported in the CI
            # except PintExceptionGroup as ex_group:
            #     # Each exception corresponds to a variable, coord, or dimension that failed
            #     print(
            #         f"PintExceptionGroup: could not quantify some elements of {var_name}"
            #     )
            #     for idx, exc in enumerate(ex_group.exceptions):
            #         print(f"  Sub-exception {idx+1}: {exc}")
            #     raise ex_group
        if source_units != target_units:
            da_converted = da_quantified.pint.to(target_units).pint.dequantify()
            utils_logger.warning(
                f"Converted {var_name} from {source_units} to {target_units}"
            )
            return da_converted
    except Exception:
        # If quantification fails (bad units, etc.), just return original
        utils_logger.warning(
            f"regional_mom6 could not use pint for data array {var_name}"
        )

    return da


def vecdot(v1, v2):
    """Return the dot product of vectors ``v1`` and ``v2``.
    ``v1`` and ``v2`` can be either numpy vectors or numpy.ndarrays
    in which case the last dimension is considered the dimension
    over which the dot product is taken.
    """
    return np.sum(v1 * v2, axis=-1)


def angle_between(v1, v2, v3):
    """Return the angle ``v2``-``v1``-``v3`` (in radians), where
    ``v1``, ``v2``, ``v3`` are 3-vectors. That is, the angle that
    is formed between vectors ``v2 - v1`` and vector ``v3 - v1``.

    Example:

        >>> from regional_mom6.utils import angle_between
        >>> v1 = (0, 0, 1)
        >>> v2 = (1, 0, 0)
        >>> v3 = (0, 1, 0)
        >>> angle_between(v1, v2, v3)
        1.5707963267948966
        >>> from numpy import rad2deg
        >>> rad2deg(angle_between(v1, v2, v3))
        90.0
    """

    v1xv2 = np.cross(v1, v2)
    v1xv3 = np.cross(v1, v3)

    norm_v1xv2 = np.sqrt(vecdot(v1xv2, v1xv2))
    norm_v1xv3 = np.sqrt(vecdot(v1xv3, v1xv3))

    cosangle = vecdot(v1xv2, v1xv3) / (norm_v1xv2 * norm_v1xv3)

    return np.arccos(cosangle)


def quadrilateral_area(v1, v2, v3, v4):
    """Return the area of a spherical quadrilateral on the unit sphere that
    has vertices on the 3-vectors ``v1``, ``v2``, ``v3``, ``v4``
    (counter-clockwise orientation is implied). The area is computed via
    the excess of the sum of the spherical angles of the quadrilateral from 2π.

    Example:

        Calculate the area that corresponds to half the Northern hemisphere
        of a sphere of radius *R*. This should be 1/4 of the sphere's total area,
        that is π *R*:sup:`2`.

        >>> from regional_mom6.utils import quadrilateral_area, latlon_to_cartesian
        >>> R = 434.3
        >>> v1 = latlon_to_cartesian(0, 0, R)
        >>> v2 = latlon_to_cartesian(0, 90, R)
        >>> v3 = latlon_to_cartesian(90, 0, R)
        >>> v4 = latlon_to_cartesian(0, -90, R)
        >>> quadrilateral_area(v1, v2, v3, v4)
        592556.1793298927
        >>> from numpy import pi
        >>> quadrilateral_area(v1, v2, v3, v4) == pi * R**2
        True
    """

    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    v4 = np.array(v4)

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
    """Convert latitude and longitude (in degrees) to Cartesian coordinates on
    a sphere of radius ``R``. By default ``R = 1``.

    Arguments:
        lat (float): Latitude (in degrees).
        lon (float): Longitude (in degrees).
        R (float): The radius of the sphere; default: 1.

    Returns:
        tuple: Tuple with the Cartesian coordinates ``x, y, z``

    Examples:

        Find the Cartesian coordinates that correspond to point with
        ``(lat, lon) = (0, 0)`` on a sphere with unit radius.

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> latlon_to_cartesian(0, 0)
        (1.0, 0.0, 0.0)

        Now let's do the same on a sphere with Earth's radius

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> R = 6371e3
        >>> latlon_to_cartesian(0, 0, R)
        (6371000.0, 0.0, 0.0)
    """

    x = R * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = R * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = R * np.sin(np.deg2rad(lat))

    return x, y, z


def quadrilateral_areas(lat, lon, R=1):
    """Return the area of spherical quadrilaterals on a sphere of radius ``R``.
    By default, ``R = 1``. The quadrilaterals are formed by constant latitude and
    longitude lines on the ``lat``-``lon`` grid provided.

    Arguments:
        lat (numpy.array): Array of latitude points (in degrees).
        lon (numpy.array): Array of longitude points (in degrees).
        R (float): The radius of the sphere; default: 1.

    Returns:
        numpy.array: Array with the areas of the quadrilaterals defined by the ``lat``-``lon`` grid
        provided. If the provided ``lat``, ``lon`` arrays are of dimension *m* :math:`\\times` *n*
        then returned areas array is of dimension (*m-1*) :math:`\\times` (*n-1*).

    Example:

        Let's construct a lat-lon grid on the sphere with 60 degree spacing.
        Then we compute the areas of each grid cell and confirm that the
        sum of the areas gives us the total area of the sphere.

        >>> from regional_mom6.utils import quadrilateral_areas
        >>> import numpy as np
        >>> λ = np.linspace(0, 360, 7)
        >>> φ = np.linspace(-90, 90, 4)
        >>> lon, lat = np.meshgrid(λ, φ)
        >>> lon
        array([[  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.]])
        >>> lat
        array([[-90., -90., -90., -90., -90., -90., -90.],
               [-30., -30., -30., -30., -30., -30., -30.],
               [ 30.,  30.,  30.,  30.,  30.,  30.,  30.],
               [ 90.,  90.,  90.,  90.,  90.,  90.,  90.]])
        >>> R = 6371e3
        >>> areas = quadrilateral_areas(lat, lon, R)
        >>> areas
        array([[1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13],
               [4.56284230e+13, 4.56284230e+13, 4.56284230e+13, 4.56284230e+13,
                4.56284230e+13, 4.56284230e+13],
               [1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13]])
        >>> np.isclose(areas.sum(), 4 * np.pi * R**2, atol=np.finfo(areas.dtype).eps)
        True
    """

    coords = np.dstack(latlon_to_cartesian(lat, lon, R))

    return quadrilateral_area(
        coords[:-1, :-1, :], coords[:-1, 1:, :], coords[1:, 1:, :], coords[1:, :-1, :]
    )


def ap2ep(uc, vc):
    """Convert complex tidal u and v to tidal ellipse.

    Adapted from ap2ep.m for Matlab. Copyright notice::

        Authorship:

        The author retains the copyright of this program, while  you are welcome
        to use and distribute it as long as you credit the author properly and respect
        the program name itself. Particularly, you are expected to retain the original
        author's name in this original version or any of its modified version that
        you might make. You are also expected not to essentially change the name of
        the programs except for adding possible extension for your own version you
        might create, e.g. ap2ep_xx is acceptable.  Any suggestions are welcome and
        enjoy my program(s)!

        Author Info:

        Zhigang Xu, Ph.D.
        (pronounced as Tsi Gahng Hsu)
        Research Scientist
        Coastal Circulation
        Bedford Institute of Oceanography
        1 Challenge Dr.
        P.O. Box 1006                    Phone  (902) 426-2307 (o)
        Dartmouth, Nova Scotia           Fax    (902) 426-7827
        CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca

        Release Date: Nov. 2000, Revised on May. 2002 to adopt Foreman's northern semi
        major axis convention.

    Arguments:
        uc: complex tidal u velocity
        vc: complex tidal v velocity

    Returns:
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
    """Convert tidal ellipse to real u and v amplitude and phase.

    Adapted from ep2ap.m for Matlab. Copyright notice::

        Authorship:

        The author of this program retains the copyright of this program, while
        you are welcome to use and distribute this program as long as you credit
        the author properly and respect the program name itself. Particularly,
        you are expected to retain the original author's name in this original
        version of the program or any of its modified version that you might make.
        You are also expected not to essentially change the name of the programs
        except for adding possible extension for your own version you might create,
        e.g. app2ep_xx is acceptable.  Any suggestions are welcome and enjoy my
        program(s)!

        Author Info:

        Zhigang Xu, Ph.D.
        (pronounced as Tsi Gahng Hsu)
        Research Scientist
        Coastal Circulation
        Bedford Institute of Oceanography
        1 Challenge Dr.
        P.O. Box 1006                    Phone  (902) 426-2307 (o)
        Dartmouth, Nova Scotia           Fax    (902) 426-7827
        CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca

        Release Date: Nov. 2000

    Arguments:
        SEMA: semi-major axis
        ECC: eccentricity
        INC: inclination [radians]
        PHA: phase [radians]

    Returns:
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


def setup_logger(
    name: str, set_handler=False, log_level=logging.INFO
) -> logging.Logger:
    """
    Setup general configuration for a logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if set_handler and not logger.hasHandlers():
        # Create a handler to print to stdout (Jupyter captures stdout)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Create a formatter (optional)
        formatter = logging.Formatter("%(name)s.%(funcName)s:%(levelname)s:%(message)s")
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)
    return logger


def rotate_complex(u, v, radian_angle):
    """
    Rotate velocities counter-clockwise by angle ``radian_angle`` (in radians) using complex number math (Same as :func:`rotate`.)

    Arguments:
        u (xarray.DataArray): The :math:`u`-component of the velocity.
        v (xarray.DataArray): The :math:`v`-component of the velocity.
        radian_angle (xarray.DataArray): The angle of rotation (in radians).

    Returns:
        Tuple[xarray.DataArray, xarray.DataArray]: The rotated :math:`u` and :math:`v` components of the velocity.
    """

    # complexify the velocity
    vel = u + v * 1j

    # rotate velocity by radian_angle
    vel = vel * np.exp(1j * radian_angle)

    # extract the rotated u and v components from vel
    u = np.real(vel)
    v = np.imag(vel)

    return u, v


def rotate(u, v, radian_angle):
    """
    Rotate the velocities counter-clockwise by an angle ``radian_angle`` (in radians).  (Same as :func:`rotate_complex`.)

    Arguments:
        u (xarray.DataArray): The :math:`u`-component of the velocity.
        v (xarray.DataArray): The :math:`v`-component of the velocity.
        radian_angle (xarray.DataArray): The angle of rotation (in radians).

    Returns:
        Tuple[xarray.DataArray, xarray.DataArray]: The rotated :math:`u` and :math:`v` components of the velocity.
    """

    u_rot = u * np.cos(radian_angle) - v * np.sin(radian_angle)
    v_rot = u * np.sin(radian_angle) + v * np.cos(radian_angle)

    return u_rot, v_rot


def is_rectilinear_hgrid(hgrid: xr.Dataset, rtol: float = 1e-3) -> bool:
    """
    Check if the ``hgrid`` is a rectilinear grid by comparing the first and last rows and columns of the tlon and tlat arrays.

    From ``mom6_bathy.grid.is_rectangular`` by Alper (Altuntas).

    Arguments:
        hgrid (xarray.Dataset): The horizontal grid dataset.
        rtol (float): Relative tolerance. Default is 1e-3.
    """
    ds_t = rgd.get_hgrid_arakawa_c_points(hgrid)
    if (
        np.allclose(ds_t.tlon[:, 0], ds_t.tlon[0, 0], rtol=rtol)
        and np.allclose(ds_t.tlon[:, -1], ds_t.tlon[0, -1], rtol=rtol)
        and np.allclose(ds_t.tlat[0, :], ds_t.tlat[0, 0], rtol=rtol)
        and np.allclose(ds_t.tlat[-1, :], ds_t.tlat[-1, 0], rtol=rtol)
    ):
        return True
    return False


def find_files_by_pattern(paths: list, patterns: list, error_message=None) -> list:
    """
    Function searchs paths for patterns and returns the list of the file paths with that pattern
    """
    # Use glob to find all files
    all_files = []
    for pattern in patterns:
        for path in paths:
            all_files.extend(Path(path).glob(pattern))

    if len(all_files) == 0:
        if error_message is None:
            raise ValueError(
                "No files found at the following paths: {} for the following patterns: {}".format(
                    paths, patterns
                )
            )
        else:
            raise ValueError(error_message)
    return all_files


def get_edge(ds, edge, x_name=None, y_name=None):
    edge = edge.lower()
    if edge not in {"north", "south", "east", "west"}:
        raise ValueError("edge must be one of: 'north', 'south', 'east', 'west'")

    # Infer x and y coordinate names if not given
    if x_name is None or y_name is None:
        for dim in ds.dims:
            if x_name is None and dim.lower() in ("x", "lon", "longitude", "nxp"):
                x_name = dim
            if y_name is None and dim.lower() in ("y", "lat", "latitude", "nyp"):
                y_name = dim
    if x_name is None or y_name is None:
        raise ValueError("Could not infer x/y coordinate names. Pass x_name/y_name.")

    if edge == "north":
        return ds.isel({y_name: -1})
    if edge == "south":
        return ds.isel({y_name: 0})
    if edge == "east":
        return ds.isel({x_name: -1})
    if edge == "west":
        return ds.isel({x_name: 0})


utils_logger = setup_logger(__name__, set_handler=False)
