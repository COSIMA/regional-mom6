"""
MOM6 requires NetCDF files to be in a very specific format to pass validation, including fill value and missing value attributes. This module is designed to accept input files and warn users of potential issues with their files
If you can, leave proof in the form of the exact lines of Fortran code where the validation step is required!

"""

from pathlib import Path
import xarray as xr
import numpy as np
import re
from .utils import setup_logger

logger = setup_logger(__name__)


def get_file(file):
    """accept a filepath or xarray dataset and return the xarray dataset"""
    if isinstance(file, xr.Dataset):
        return file
    else:
        return xr.open_dataset(file)


def check(condition, warning):
    if not condition:
        logger.warning(warning)
    return condition


# Individual validation rule functions
def _check_fill_value(da: xr.DataArray):
    """Check that fill values are set correctly"""
    condition = check(
        "_FillValue" in da.attrs, f"{da.name} does not have a FillValue attribute"
    )

    if condition:
        check(
            not np.isnan(da.attrs["_FillValue"]),
            f"Fill Value for variable {da.name} is NaN (normally not wanted)",
        )


def _check_coordinates(ds: xr.Dataset, var_name: str):
    """Check that coordinates attribute exists and all listed coordinates are present in the dataset"""

    assert var_name in ds
    condition = check(
        "coordinates" in ds[var_name].attrs,
        f"{var_name} does not have a coordinates attribute",
    )
    if condition:
        coordinates = ds[var_name].attrs["coordinates"].strip()
        for coord in coordinates.split():
            check(
                coord in ds,
                f"Coordinate {coord} for variable {var_name} does not exist",
            )


def _check_required_dimensions(da: xr.DataArray, surface=False):
    """Check that required dimensions exist"""
    if not surface:
        check(len(da.dims) == 4, f"Variable {da.name} does not have 4 dimensions")
    else:
        check(
            len(da.dims) == 3, f"Surface Variable {da.name} does not have 3 dimensions"
        )


def validate_obc_file(
    file, variable_names: list, encoding_dict=None, surface_var="eta"
):
    """Validate boundary condition file specifically (requires additional segment number validation)"""
    if encoding_dict is None:
        encoding_dict = {}
    ds = get_file(file)

    print(
        "This function identifies variables by if they have the word 'segment' in the name and don't start with nz,dz,lon,lat."
    )

    for var in variable_names:

        # check variable name format
        check(
            ends_with_3_digits(var),
            f"Variable {var} does not end with a 3 digit number. OBC file variables must end with a number",
        )
        check(
            "segment" in var,
            f"Variable {var} does not contain 'segment'. OBC file variables must include 'segment'",
        )

        # Add encodings
        if var in encoding_dict:
            for key, value in encoding_dict[var].items():
                ds[var].attrs[key] = value

        # Check if there is a non-NaN fill value
        _check_fill_value(ds[var])

        # check coordinates
        _check_coordinates(ds, var_name=var)

        # Check the correct number of dimensions
        _check_required_dimensions(ds[var], surface=(var == surface_var))

        # Check for thickness variable
        if var != surface_var:
            check(
                f"dz_{var}" in ds,
                f"Cannot find thickness variable for var {var}, it should be of the form dz_{var}",
            )


def ends_with_3_digits(s: str) -> bool:
    return bool(re.search(r"_\d{3}$", s))
