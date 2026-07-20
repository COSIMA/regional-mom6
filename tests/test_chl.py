import numpy as np
import pytest
import xarray as xr
from mom6_forge.topo import Topo


def _make_small_seawifs_ds():
    """Build a tiny synthetic dataset mirroring the format of the real SeaWiFS
    climatology product (SeaWIFS.L3m.MC.CHL.chlor_a.0.25deg.nc), but at a very
    coarse (5deg) global resolution so it stays small and fast to interpolate.
    """
    nlat, nlon, ntime = 36, 72, 12
    dlat, dlon = 180.0 / nlat, 360.0 / nlon

    # Descending latitude / ascending longitude, same convention as the real product.
    lat = (90.0 - (np.arange(nlat) + 0.5) * dlat).astype(np.float32)
    lon = (-180.0 + (np.arange(nlon) + 0.5) * dlon).astype(np.float32)
    time = np.array(
        [
            16.0,
            45.5,
            75.0,
            105.5,
            136.0,
            166.5,
            197.0,
            228.0,
            258.5,
            289.0,
            319.5,
            350.0,
        ]
    )

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    month = np.arange(ntime)

    chlor_a = (
        0.2
        + 0.05 * np.cos(lat_rad)[None, :, None]
        + 0.01 * np.cos(lon_rad)[None, None, :]
        + 0.01 * np.sin(2 * np.pi * month / ntime)[:, None, None]
    ).astype(np.float64)

    chl_attrs = {
        "long_name": "Chlorophyll Concentration, OCI Algorithm",
        "units": "mg m^-3",
        "standard_name": "mass_concentration_chlorophyll_concentration_in_sea_water",
        "valid_min": np.float32(0.001),
        "valid_max": np.float32(100.0),
        "display_scale": "log",
        "display_min": np.float32(0.01),
        "display_max": np.float32(20.0),
    }

    ds = xr.Dataset(
        {
            "chlor_a": xr.DataArray(
                chlor_a, dims=["time", "lat", "lon"], attrs=chl_attrs
            ),
            "chlor_a_cf": xr.DataArray(
                chlor_a.copy(), dims=["time", "lat", "lon"], attrs=chl_attrs
            ),
            "chlor_a_cfm": xr.DataArray(
                chlor_a.copy(), dims=["time", "lat", "lon"], attrs=chl_attrs
            ),
            "ocean_mask": xr.DataArray(
                np.ones((nlat, nlon), dtype=np.int8),
                dims=["lat", "lon"],
                attrs={
                    "long_name": "Ocean Mask",
                    "flag_values": "0, 1",
                    "flag_meanings": "0=Land or Inland Water, 1=Ocean",
                },
            ),
            "land_mask": xr.DataArray(
                np.zeros((nlat, nlon), dtype=np.int8),
                dims=["lat", "lon"],
                attrs={
                    "long_name": "Land Mask",
                    "flag_values": "0, 1",
                    "flag_meanings": "1=Land or Inland Water, 0=Ocean",
                },
            ),
        },
        coords={
            "time": xr.DataArray(
                time,
                dims="time",
                attrs={
                    "long_name": "Mid-Month Day of Climatological Year",
                    "units": "days",
                },
            ),
            "lat": xr.DataArray(
                lat,
                dims="lat",
                attrs={
                    "long_name": "Latitude",
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "valid_min": np.float32(-90.0),
                    "valid_max": np.float32(90.0),
                },
            ),
            "lon": xr.DataArray(
                lon,
                dims="lon",
                attrs={
                    "long_name": "Longitude",
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "valid_min": np.float32(-180.0),
                    "valid_max": np.float32(180.0),
                },
            ),
        },
    )
    return ds


@pytest.fixture
def small_seawifs_path(tmp_path):
    path = tmp_path / "small_seawifs.nc"
    _make_small_seawifs_ds().to_netcdf(path)
    return path


def test_setup_chl(simple_experiment, small_seawifs_path):
    """Test experiment.setup_chl, which wraps interpolate_and_fill_seawifs using the experiment's own grid/bathymetry."""

    topo = Topo(simple_experiment.m6f_hgrid, min_depth=10.0, git=False)
    topo.set_flat(1000.0)
    simple_experiment.m6f_bathymetry = topo

    chl_ds = simple_experiment.setup_chl(processed_seawifs_path=small_seawifs_path)

    expected_output = (
        simple_experiment.mom_input_dir
        / f"seawifs-clim-1997-2010-{simple_experiment.expt_name}.nc"
    )
    assert expected_output.exists()
    assert "CHL_A" in chl_ds

    chl_a = chl_ds["CHL_A"]
    assert chl_a.shape == (
        12,
        simple_experiment.m6f_hgrid.ny,
        simple_experiment.m6f_hgrid.nx,
    )
    assert np.all(np.isfinite(chl_a.values))
    assert np.all(chl_a.values > 0.0)
    assert np.all(chl_a.values < 100.0)
