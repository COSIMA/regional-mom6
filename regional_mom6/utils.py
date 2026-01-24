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

unit_path = Path(importlib.resources.files("regional_mom6") / "rm6_unit_defs.txt")
ureg.load_definitions(unit_path)


def try_pint_convert(da, target_units, var_name=None, debug=False):
    """
    Attempt to quantify and convert an xarray DataArray using Pint.

    Steps:
    1. Check if the DataArray has a 'units' attribute.
       - If not, Pint cannot quantify it, so we raise a ValueError.
    2. Quantify the DataArray using Pint (attach units).
       - This converts the DataArray into a pint-aware object.
       - If already a Pint Quantity, skip quantification.
    3. Convert the DataArray to the target units if necessary.
       - Uses Pint's `.to()` to perform unit conversion.
       - Dequantify afterwards to return a plain xarray DataArray.
    4. If any step fails (missing units, invalid units, etc.),
       - Log a warning and return the original DataArray unchanged.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to quantify and/or convert.
    target_units : str
        Units to convert the DataArray to.
    var_name : str, optional
        Name of the variable (used for logging messages).
    debug : bool, optional
        If True, print debug information about the error with subexceptions.

    Returns
    -------
    xarray.DataArray
        A DataArray with units converted if successful; otherwise the original.
    """
    try:
        # Get the units string from the DataArray attributes
        source_units = da.attrs.get("units", None)
        if not source_units:
            raise ValueError(f"DataArray '{var_name}' has no units; cannot quantify.")

        # Only quantify if not already a Pint Quantity
        if not isinstance(da.data, pint.Quantity):
            da_quantified = da.pint.quantify(unit_registry=ureg)

            # code for PintExceptionGroup (not supported in current CI until we can use pint 0.6.0)
            # Allows catching multiple quantification errors at once
            # except PintExceptionGroup as ex_group:
            #     print(f"PintExceptionGroup: could not quantify some elements of {var_name}")
            #     for idx, exc in enumerate(ex_group.exceptions):
            #         print(f"  Sub-exception {idx+1}: {exc}")
            #     raise ex_group

        # Convert to the target units if they differ
        if source_units != target_units:
            da_converted = da_quantified.pint.to(target_units).pint.dequantify()
            utils_logger.warning(
                f"Converted {var_name} from {source_units} to {target_units}"
            )
            return da_converted
        else:
            utils_logger.info(f"Units for {var_name} did not need to be converted")

    except Exception as e:
        # If any error occurs (bad units, missing Pint, etc.), fall back gracefully
        utils_logger.warning(
            f"regional_mom6 could not use pint for data array {var_name}, assuming it's in the correct units"
        )
        if debug:
            if hasattr(e, "exceptions"):
                for i, exc in enumerate(e.exceptions):
                    print(f"\n--- Sub-exception {i} ---")
                    print(type(exc).__name__, exc)
            else:
                print(e)

    # Return the original DataArray if quantification or conversion failed
    return da


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
