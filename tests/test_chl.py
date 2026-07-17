import socket

import numpy as np
import pytest
import xarray as xr

from regional_mom6.chl import interpolate_and_fill_seawifs

SEAWIFS_PATH = "/glade/campaign/cesm/cesmdata/cseg/inputdata/ocn/mom/croc/chl/data/SeaWIFS.L3m.MC.CHL.chlor_a.0.25deg.nc"


def on_cisl_machine():
    """Return True if the current machine is a CISL machine, False otherwise."""
    return "hpc.ucar.edu" in socket.getfqdn()


requires_cisl_machine = pytest.mark.skipif(
    not on_cisl_machine(),
    reason="Requires the SeaWiFS chlorophyll climatology, only staged on NCAR/CISL machines",
)


def add_synthetic_bathymetry(expt, tmp_path):
    """Set up random bathymetry for `expt`, same pattern as test_expt_class.py::test_setup_bathymetry."""
    bathymetry_file = tmp_path / "bathymetry.nc"
    bathymetry = np.random.random((100, 100)) * (-100)
    bathymetry = xr.DataArray(
        bathymetry,
        dims=["silly_lat", "silly_lon"],
        coords={
            "silly_lat": np.linspace(
                expt.latitude_extent[0] - 5, expt.latitude_extent[1] + 5, 100
            ),
            "silly_lon": np.linspace(
                expt.longitude_extent[0] - 5, expt.longitude_extent[1] + 5, 100
            ),
        },
    )
    bathymetry.name = "silly_depth"
    bathymetry.to_netcdf(bathymetry_file)
    bathymetry.close()

    expt.setup_bathymetry(
        bathymetry_path=str(bathymetry_file),
        longitude_coordinate_name="silly_lon",
        latitude_coordinate_name="silly_lat",
        vertical_coordinate_name="silly_depth",
    )


@requires_cisl_machine
def test_interpolate_and_fill_seawifs(simple_experiment, tmp_path):
    """Test the creation of chl files directly via interpolate_and_fill_seawifs."""
    add_synthetic_bathymetry(simple_experiment, tmp_path)

    output_path = tmp_path / "seawifs-clim-1997-2010-chl_test_grid.nc"
    chl_ds = interpolate_and_fill_seawifs(
        simple_experiment.m6f_hgrid,
        simple_experiment.m6f_bathymetry,
        processed_seawifs_path=SEAWIFS_PATH,
        output_path=output_path,
    )

    assert output_path.exists()
    assert "CHL_A" in chl_ds


@requires_cisl_machine
def test_setup_chl(simple_experiment, tmp_path):
    """Test experiment.setup_chl, which wraps interpolate_and_fill_seawifs using the experiment's own grid/bathymetry."""
    add_synthetic_bathymetry(simple_experiment, tmp_path)

    chl_ds = simple_experiment.setup_chl(processed_seawifs_path=SEAWIFS_PATH)

    expected_output = (
        simple_experiment.mom_input_dir
        / f"seawifs-clim-1997-2010-{simple_experiment.expt_name}.nc"
    )
    assert expected_output.exists()
    assert "CHL_A" in chl_ds
