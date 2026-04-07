"""
Test suite for the validate module
"""

import pytest
import xarray as xr
import numpy as np
from pathlib import Path
from regional_mom6.validate import (
    _check_fill_value,
    _check_coordinates,
    _check_required_dimensions,
    _check_ends_with_3_digits,
    validate_obc_file,
)


# _check_fill_value tests
def test_check_fill_value_valid(caplog):
    """DataArray with valid fill value logs no warnings"""
    da = xr.DataArray(
        [1, 2, 3], dims="x", name="temperature", attrs={"_FillValue": -999.0}
    )
    _check_fill_value(da)
    assert "FillValue" not in caplog.text


def test_check_fill_value_missing(caplog):
    """DataArray without _FillValue attribute logs warning"""
    da = xr.DataArray([1, 2, 3], dims="x", name="temperature")
    _check_fill_value(da)
    assert "FillValue" in caplog.text


# _check_coordinates tests
def test_check_coordinates_valid(caplog):
    """DataArray with valid coordinates attribute logs no warnings"""
    ds = xr.Dataset(
        {
            "temperature": (["x", "y"], np.random.rand(3, 4)),
            "lon": (["x", "y"], np.random.rand(3, 4)),
            "lat": (["x", "y"], np.random.rand(3, 4)),
        }
    )
    ds["temperature"].attrs["coordinates"] = "lon lat"
    _check_coordinates(ds, "temperature")
    assert "coordinate" not in caplog.text.lower()


def test_check_coordinates_missing_attribute(caplog):
    """DataArray without coordinates attribute logs warning"""
    ds = xr.Dataset({"temperature": (["x", "y"], np.random.rand(3, 4))})
    _check_coordinates(ds, "temperature")
    assert "coordinates" in caplog.text.lower()


def test_check_coordinates_missing_in_dataset(caplog):
    """Missing coordinate variable logs warning"""
    ds = xr.Dataset({"temperature": (["x", "y"], np.random.rand(3, 4))})
    ds["temperature"].attrs["coordinates"] = "lon lat"
    _check_coordinates(ds, "temperature")
    assert "does not exist" in caplog.text


# _check_required_dimensions tests
def test_check_required_dimensions_valid_4d(caplog):
    """4D variable passes check when surface=False"""
    da = xr.DataArray(
        np.random.rand(2, 3, 4, 5), dims=["time", "z", "x", "y"], name="temperature"
    )
    _check_required_dimensions(da, surface=False)
    assert "dimension" not in caplog.text.lower()


def test_check_required_dimensions_invalid_3d_for_4d(caplog):
    """3D variable fails check when surface=False"""
    da = xr.DataArray(np.random.rand(3, 4, 5), dims=["x", "y", "z"], name="temperature")
    _check_required_dimensions(da, surface=False)
    assert "dimension" in caplog.text.lower()


def test_check_required_dimensions_valid_3d_surface(caplog):
    """3D variable passes check when surface=True"""
    da = xr.DataArray(np.random.rand(2, 3, 4), dims=["time", "x", "y"], name="eta")
    _check_required_dimensions(da, surface=True)
    assert "dimension" not in caplog.text.lower()


def test_check_required_dimensions_invalid_4d_for_surface(caplog):
    """4D variable fails check when surface=True"""
    da = xr.DataArray(
        np.random.rand(2, 3, 4, 5), dims=["time", "z", "x", "y"], name="eta"
    )
    _check_required_dimensions(da, surface=True)
    assert "dimension" in caplog.text.lower()


# _check_ends_with_3_digits tests
def test_check_ends_with_3_digits_valid_cases():
    """String ending with 3 digits returns True"""
    assert _check_ends_with_3_digits("temp_001") is True
    assert _check_ends_with_3_digits("var_999") is True
    assert _check_ends_with_3_digits("_000") is True
    assert _check_ends_with_3_digits("temp_01") is False
    assert _check_ends_with_3_digits("temp_0001") is False
    assert _check_ends_with_3_digits("temp_abc") is False
    assert _check_ends_with_3_digits("temp") is False


# validate_obc_file tests


def test_validate_obc_file_valid(caplog):
    """Valid OBC file with all required attributes passes"""
    ds = xr.Dataset(
        {
            "temp_segment_001": (["time", "z", "x", "y"], np.random.rand(2, 3, 4, 5)),
            "dz_temp_segment_001": (
                ["time", "z", "x", "y"],
                np.random.rand(2, 3, 4, 5),
            ),
            "eta_segment_001": (["time", "x", "y"], np.random.rand(2, 4, 5)),
            "lon": (["x", "y"], np.random.rand(4, 5)),
            "lat": (["x", "y"], np.random.rand(4, 5)),
        }
    )

    for var in ds.data_vars:
        ds[var].attrs["_FillValue"] = -999.0
        ds[var].attrs["coordinates"] = "lon lat"

    validate_obc_file(
        ds, ["temp_segment_001", "eta_segment_001"], surface_var="eta_segment_001"
    )


def test_validate_obc_file_issues(caplog):
    """OBC file with missing segment and thickness logs warnings"""
    ds = xr.Dataset(
        {
            "temp_001": (["time", "z", "x", "y"], np.random.rand(2, 3, 4, 5)),
            "lon": (["x", "y"], np.random.rand(4, 5)),
            "lat": (["x", "y"], np.random.rand(4, 5)),
        }
    )
    ds["temp_001"].attrs["_FillValue"] = -999.0
    ds["temp_001"].attrs["coordinates"] = "lon lat"

    validate_obc_file(ds, ["temp_001"])
    assert "segment" in caplog.text
    assert "thickness" in caplog.text or "dz_temp_001" in caplog.text


def test_validate_obc_file_encoding_dict():
    """Encoding dict is applied to variables"""
    ds = xr.Dataset(
        {
            "temp_segment_001": (["time", "z", "x", "y"], np.random.rand(2, 3, 4, 5)),
            "dz_temp_segment_001": (
                ["time", "z", "x", "y"],
                np.random.rand(2, 3, 4, 5),
            ),
            "lon": (["x", "y"], np.random.rand(4, 5)),
            "lat": (["x", "y"], np.random.rand(4, 5)),
        }
    )
    ds["temp_segment_001"].attrs["coordinates"] = "lon lat"
    ds["dz_temp_segment_001"].attrs["_FillValue"] = -999.0

    encoding_dict = {"temp_segment_001": {"_FillValue": " -999.0"}}
    validate_obc_file(ds, ["temp_segment_001"], encoding_dict=encoding_dict)


def test_validate_general_file_valid(caplog):
    """Valid general file with all required attributes passes"""
    ds = xr.Dataset(
        {
            "temp": (["time", "z", "x", "y"], np.random.rand(2, 3, 4, 5)),
            "dz_temp": (
                ["time", "z", "x", "y"],
                np.random.rand(2, 3, 4, 5),
            ),
            "eta": (["time", "x", "y"], np.random.rand(2, 4, 5)),
            "lon": (["x", "y"], np.random.rand(4, 5)),
            "lat": (["x", "y"], np.random.rand(4, 5)),
        }
    )

    for var in ds.data_vars:
        ds[var].attrs["_FillValue"] = -999.0
        ds[var].attrs["coordinates"] = "lon lat"

    validate_obc_file(ds, ["temp", "eta"], surface_var="eta")
