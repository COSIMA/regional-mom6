"""
Test suite for everything involed in pr #12
"""

import regional_mom6 as rmom6
import os
import pytest
import logging
from pathlib import Path
import xarray as xr
import numpy as np
import shutil
import importlib

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# @pytest.mark.skipif(
#     IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions."
# )


@pytest.fixture(scope="module")
def dummy_tidal_data():
    nx = 100
    ny = 100
    nc = 15
    nct = 4

    # Define tidal constituents
    con_list = [
        "m2  ",
        "s2  ",
        "n2  ",
        "k2  ",
        "k1  ",
        "o1  ",
        "p1  ",
        "q1  ",
        "mm  ",
        "mf  ",
        "m4  ",
        "mn4 ",
        "ms4 ",
        "2n2 ",
        "s1  ",
    ]
    con_data = np.array([list(con) for con in con_list], dtype="S1")

    # Generate random data for the variables
    lon_z_data = np.tile(np.linspace(-180, 180, nx), (ny, 1)).T
    lat_z_data = np.tile(np.linspace(-90, 90, ny), (nx, 1))
    ha_data = np.random.rand(nc, nx, ny)
    hp_data = np.random.rand(nc, nx, ny) * 360  # Random phases between 0 and 360
    hRe_data = np.random.rand(nc, nx, ny)
    hIm_data = np.random.rand(nc, nx, ny)

    # Create the xarray dataset
    ds_h = xr.Dataset(
        {
            "con": (["nc", "nct"], con_data),
            "lon_z": (["nx", "ny"], lon_z_data),
            "lat_z": (["nx", "ny"], lat_z_data),
            "ha": (["nc", "nx", "ny"], ha_data),
            "hp": (["nc", "nx", "ny"], hp_data),
            "hRe": (["nc", "nx", "ny"], hRe_data),
            "hIm": (["nc", "nx", "ny"], hIm_data),
        },
        coords={
            "nc": np.arange(nc),
            "nct": np.arange(nct),
            "nx": np.arange(nx),
            "ny": np.arange(ny),
        },
        attrs={
            "type": "Fake OTIS tidal elevation file",
            "title": "Fake TPXO9.v1 2018 tidal elevation file",
        },
    )

    # Generate random data for the variables for u_tpxo9.v1
    lon_u_data = (
        np.random.rand(nx, ny) * 360 - 180
    )  # Random longitudes between -180 and 180
    lat_u_data = (
        np.random.rand(nx, ny) * 180 - 90
    )  # Random latitudes between -90 and 90
    lon_v_data = (
        np.random.rand(nx, ny) * 360 - 180
    )  # Random longitudes between -180 and 180
    lat_v_data = (
        np.random.rand(nx, ny) * 180 - 90
    )  # Random latitudes between -90 and 90
    Ua_data = np.random.rand(nc, nx, ny)
    ua_data = np.random.rand(nc, nx, ny)
    up_data = np.random.rand(nc, nx, ny) * 360  # Random phases between 0 and 360
    Va_data = np.random.rand(nc, nx, ny)
    va_data = np.random.rand(nc, nx, ny)
    vp_data = np.random.rand(nc, nx, ny) * 360  # Random phases between 0 and 360
    URe_data = np.random.rand(nc, nx, ny)
    UIm_data = np.random.rand(nc, nx, ny)
    VRe_data = np.random.rand(nc, nx, ny)
    VIm_data = np.random.rand(nc, nx, ny)

    # Create the xarray dataset for u_tpxo9.v1
    ds_u = xr.Dataset(
        {
            "con": (["nc", "nct"], con_data),
            "lon_u": (["nx", "ny"], lon_u_data),
            "lat_u": (["nx", "ny"], lat_u_data),
            "lon_v": (["nx", "ny"], lon_v_data),
            "lat_v": (["nx", "ny"], lat_v_data),
            "Ua": (["nc", "nx", "ny"], Ua_data),
            "ua": (["nc", "nx", "ny"], ua_data),
            "up": (["nc", "nx", "ny"], up_data),
            "Va": (["nc", "nx", "ny"], Va_data),
            "va": (["nc", "nx", "ny"], va_data),
            "vp": (["nc", "nx", "ny"], vp_data),
            "URe": (["nc", "nx", "ny"], URe_data),
            "UIm": (["nc", "nx", "ny"], UIm_data),
            "VRe": (["nc", "nx", "ny"], VRe_data),
            "VIm": (["nc", "nx", "ny"], VIm_data),
        },
        coords={
            "nc": np.arange(nc),
            "nct": np.arange(nct),
            "nx": np.arange(nx),
            "ny": np.arange(ny),
        },
        attrs={
            "type": "Fake OTIS tidal transport file",
            "title": "Fake TPXO9.v1 2018 WE/SN transports/currents file",
        },
    )

    return ds_h, ds_u


def test_tides(dummy_tidal_data, tmp_path):
    """
    Test the main setup tides function!
    """
    expt_name = "testing"

    expt = rmom6.experiment.create_empty(
        expt_name=expt_name,
        mom_input_dir=tmp_path,
        mom_run_dir=tmp_path,
    )
    # Generate Fake Tidal Data
    ds_h, ds_u = dummy_tidal_data

    # Save to Fake Folder
    ds_h.to_netcdf(tmp_path / "h_fake_tidal_data.nc")
    ds_u.to_netcdf(tmp_path / "u_fake_tidal_data.nc")

    # Set other required variables needed in setup_tides

    # Lat Long
    expt.longitude_extent = (-5, 5)
    expt.latitude_extent = (0, 30)
    # Grid Type
    expt.hgrid_type = "even_spacing"
    # Dates
    expt.date_range = ("2000-01-01", "2000-01-02")
    expt.segments = {}
    # Generate Hgrid Data
    expt.resolution = 0.1
    expt.hgrid = expt._make_hgrid()
    # Create Forcing Folder
    os.makedirs(tmp_path / "forcing", exist_ok=True)

    expt.setup_boundary_tides(
        tmp_path / "h_fake_tidal_data.nc",
        tmp_path / "u_fake_tidal_data.nc",
    )


def test_change_MOM_parameter(tmp_path):
    """
    Test the change MOM parameter function, as well as read_MOM_file and write_MOM_file under the hood.
    """
    expt_name = "testing"

    expt = rmom6.experiment.create_empty(
        expt_name=expt_name,
        mom_input_dir=tmp_path,
        mom_run_dir=tmp_path,
    )
    # Copy over the MOM Files to the dump_files_dir
    base_run_dir = Path(
        os.path.join(
            importlib.resources.files("regional_mom6").parent,
            "demos",
            "premade_run_directories",
        )
    )
    shutil.copytree(base_run_dir / "common_files", expt.mom_run_dir, dirs_exist_ok=True)
    MOM_override_dict = expt.read_MOM_file_as_dict("MOM_override")
    og = expt.change_MOM_parameter("DT", "30", "COOL COMMENT")
    MOM_override_dict_new = expt.read_MOM_file_as_dict("MOM_override")
    assert MOM_override_dict_new["DT"]["value"] == "30"
    assert MOM_override_dict["DT"]["value"] == og
    assert MOM_override_dict_new["DT"]["comment"] == "COOL COMMENT\n"
