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


@pytest.fixture(scope="module")
def dummy_tidal_data():
    nx = 2160
    ny = 1081
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


@pytest.fixture(scope="module")
def dummy_bathymetry_data():
    latitude_extent = [16.0, 27]
    longitude_extent = [192, 209]

    bathymetry = np.random.random((100, 100)) * (-100)
    bathymetry = xr.DataArray(
        bathymetry,
        dims=["silly_lat", "silly_lon"],
        coords={
            "silly_lat": np.linspace(
                latitude_extent[0] - 5, latitude_extent[1] + 5, 100
            ),
            "silly_lon": np.linspace(
                longitude_extent[0] - 5, longitude_extent[1] + 5, 100
            ),
        },
    )
    bathymetry.name = "silly_depth"
    return bathymetry


class TestAll:
    @classmethod
    def setup_class(self):  # tmp_path is a pytest fixture
        expt_name = "testing"
        ## User-1st, test if we can even read the angled nc files.
        self.dump_files_dir = Path("testing_outputs")
        os.makedirs(self.dump_files_dir, exist_ok=True)
        self.expt = rmom6.experiment.create_empty(
            expt_name=expt_name,
            mom_input_dir=self.dump_files_dir,
            mom_run_dir=self.dump_files_dir,
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.dump_files_dir)

    @pytest.fixture(scope="module")
    def full_legit_expt_setup(self, dummy_bathymetry_data):

        expt_name = "testing"

        latitude_extent = [16.0, 27]
        longitude_extent = [192, 209]

        date_range = ["2005-01-01 00:00:00", "2005-02-01 00:00:00"]

        ## Place where all your input files go
        input_dir = Path(
            os.path.join(
                expt_name,
                "inputs",
            )
        )

        ## Directory where you'll run the experiment from
        run_dir = Path(
            os.path.join(
                expt_name,
                "run_files",
            )
        )
        data_path = Path("data")
        for path in (run_dir, input_dir, data_path):
            os.makedirs(str(path), exist_ok=True)
        bathy_path = data_path / "bathymetry.nc"
        bathymetry = dummy_bathymetry_data
        bathymetry.to_netcdf(bathy_path)
        self.glorys_path = bathy_path
        ## User-1st, test if we can even read the angled nc files.
        expt = rmom6.experiment(
            longitude_extent=longitude_extent,
            latitude_extent=latitude_extent,
            date_range=date_range,
            resolution=0.05,
            number_vertical_layers=75,
            layer_thickness_ratio=10,
            depth=4500,
            minimum_depth=5,
            mom_run_dir=run_dir,
            mom_input_dir=input_dir,
            toolpath_dir="",
        )
        return expt

    def test_full_legit_expt_setup(self, full_legit_expt_setup):
        assert str(full_legit_expt_setup)

    # @pytest.mark.skipif(
    #     IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions."
    # )
    def test_tides(self, dummy_tidal_data):
        """
        Test the main setup tides function!
        """

        # Generate Fake Tidal Data
        ds_h, ds_u = dummy_tidal_data

        # Save to Fake Folder
        ds_h.to_netcdf(self.dump_files_dir / "h_fake_tidal_data.nc")
        ds_u.to_netcdf(self.dump_files_dir / "u_fake_tidal_data.nc")

        # Set other required variables needed in setup_tides

        # Lat Long
        self.expt.longitude_extent = (-5, 5)
        self.expt.latitude_extent = (0, 30)
        # Grid Type
        self.expt.hgrid_type = "even_spacing"
        # Dates
        self.expt.date_range = ("2000-01-01", "2000-01-02")
        self.expt.segments = {}
        # Generate Hgrid Data
        self.expt.resolution = 0.1
        self.expt.hgrid = self.expt._make_hgrid()
        # Create Forcing Folder
        os.makedirs(self.dump_files_dir / "forcing", exist_ok=True)

        self.expt.setup_boundary_tides(self.dump_files_dir, "fake_tidal_data.nc")

    def test_change_MOM_parameter(self):
        """
        Test the change MOM parameter function, as well as read_MOM_file and write_MOM_file under the hood.
        """

        # Copy over the MOM Files to the dump_files_dir
        base_run_dir = Path(
            os.path.join(
                importlib.resources.files("regional_mom6").parent,
                "demos",
                "premade_run_directories",
            )
        )
        shutil.copytree(
            base_run_dir / "common_files", self.expt.mom_run_dir, dirs_exist_ok=True
        )
        MOM_override_dict = self.expt.read_MOM_file_as_dict("MOM_override")
        og = self.expt.change_MOM_parameter("DT", "30", "COOL COMMENT")
        MOM_override_dict_new = self.expt.read_MOM_file_as_dict("MOM_override")
        assert MOM_override_dict_new["DT"]["value"] == "30"
        assert MOM_override_dict["DT"]["value"] == og
        assert MOM_override_dict_new["DT"]["comment"] == "COOL COMMENT\n"

    def test_properties_empty(self):
        """
        Test the properties
        """
        dss = self.expt.era5
        dss_2 = self.expt.tides_boundaries
        dss_3 = self.expt.ocean_state_boundaries
        dss_4 = self.expt.initial_condition
        dss_5 = self.expt.bathymetry_property
        print(dss, dss_2, dss_3, dss_4, dss_5)
