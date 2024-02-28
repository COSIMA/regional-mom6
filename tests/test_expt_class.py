import numpy as np
import pytest
from regional_mom6 import experiment
import xarray as xr


@pytest.mark.parametrize(
    (
        "longitude_extent",
        "latitude_extent",
        "date_range",
        "resolution",
        "number_vertical_layers",
        "layer_thickness_ratio",
        "depth",
        "mom_run_dir",
        "mom_input_dir",
        "toolpath_dir",
        "grid_type",
    ),
    [
        (
            (-5, 5),
            [0, 10],
            ["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
            0.1,
            5,
            1,
            1000,
            "rundir/",
            "inputdir/",
            "toolpath",
            "even_spacing",
        ),
    ],
)
def test_bathymetry(
    longitude_extent,
    latitude_extent,
    date_range,
    resolution,
    number_vertical_layers,
    layer_thickness_ratio,
    depth,
    mom_run_dir,
    mom_input_dir,
    toolpath_dir,
    grid_type,
    tmp_path,
):
    expt = experiment(
        longitude_extent=longitude_extent,
        latitude_extent=latitude_extent,
        date_range=date_range,
        resolution=resolution,
        number_vertical_layers=number_vertical_layers,
        layer_thickness_ratio=layer_thickness_ratio,
        depth=depth,
        mom_run_dir=mom_run_dir,
        mom_input_dir=mom_input_dir,
        toolpath_dir=toolpath_dir,
        grid_type=grid_type,
    )

    ## Generate some bathymetry to test on

    bathy_file = tmp_path / "bathy.nc"

    bathy = np.random.random((100, 100)) * (-100)
    bathy = xr.DataArray(
        bathy,
        dims=["lata", "lona"],
        coords={
            "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
            "lona": np.linspace(longitude_extent[0] - 5, longitude_extent[1] + 5, 100),
        },
    )
    bathy.name = "elevation"
    bathy.to_netcdf(bathy_file)
    bathy.close()

    # Now use this bathymetry as input in `expt.bathymetry()`
    expt.bathymetry(
        str(bathy_file),
        {"xh": "lona", "yh": "lata", "elevation": "elevation"},
        minimum_layers=1,
        chunks={"lat": 10, "lon": 10},
    )

    bathy_file.unlink()


@pytest.mark.parametrize(
    (
        "longitude_extent",
        "latitude_extent",
        "date_range",
        "resolution",
        "number_vertical_layers",
        "layer_thickness_ratio",
        "depth",
        "mom_run_dir",
        "mom_input_dir",
        "toolpath_dir",
        "grid_type",
    ),
    [
        (
            [-5, 5],
            (0, 10),
            ["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
            0.1,
            5,
            1,
            1000,
            "rundir/",
            "inputdir/",
            "toolpath",
            "even_spacing",
        ),
    ],
)
def test_ocean_forcing(
    longitude_extent,
    latitude_extent,
    date_range,
    resolution,
    number_vertical_layers,
    layer_thickness_ratio,
    depth,
    mom_run_dir,
    mom_input_dir,
    toolpath_dir,
    grid_type,
    tmp_path,
):
    expt = experiment(
        longitude_extent=longitude_extent,
        latitude_extent=latitude_extent,
        date_range=date_range,
        resolution=resolution,
        number_vertical_layers=number_vertical_layers,
        layer_thickness_ratio=layer_thickness_ratio,
        depth=depth,
        mom_run_dir=mom_run_dir,
        mom_input_dir=mom_input_dir,
        toolpath_dir=toolpath_dir,
        grid_type=grid_type,
    )

    ## Generate some initial condition to test on

    # initial condition includes, temp, salt, eta, u, v
    initial_cond = xr.Dataset(
        {
            "temp": xr.DataArray(
                np.random.random((100, 100, 10)),
                dims=["lata", "lona", "deepness"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[0] - 5, longitude_extent[1] + 5, 100),
                    "deepness": np.linspace(0, 1000, 10),
                },
            ),
            "eta": xr.DataArray(
                np.random.random((100, 100)),
                dims=["lata", "lona"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[0] - 5, longitude_extent[1] + 5, 100),
                },
            ),
            "salt": xr.DataArray(
                np.random.random((100, 100, 10)),
                dims=["lata", "lona", "deepness"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[0] - 5, longitude_extent[1] + 5, 100),
                    "deepness": np.linspace(0, 1000, 10),
                },
            ),
            "u": xr.DataArray(
                np.random.random((100, 100, 10)),
                dims=["lata", "lona", "deepness"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[0] - 5, longitude_extent[1] + 5, 100),
                    "deepness": np.linspace(0, 1000, 10),
                },
            ),
            "v": xr.DataArray(
                np.random.random((100, 100, 10)),
                dims=["lata", "lona", "deepness"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[0] - 5, longitude_extent[1] + 5, 100),
                    "deepness": np.linspace(0, 1000, 10),
                },
            ),
        }
    )

    # Generate boundary forcing

    initial_cond.to_netcdf(tmp_path / "ic_unprocessed")
    initial_cond.close()
    varnames = {
        "x": "lona",
        "y": "lata",
        "time": "time",
        "eta": "eta",
        "zl": "deepness",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    expt.initial_condition(
        tmp_path / "ic_unprocessed",
        varnames,
        gridtype="A",
    )


@pytest.mark.parametrize(
    (
        "longitude_extent",
        "latitude_extent",
        "date_range",
        "resolution",
        "number_vertical_layers",
        "layer_thickness_ratio",
        "depth",
        "mom_run_dir",
        "mom_input_dir",
        "toolpath_dir",
        "grid_type",
    ),
    [
        (
            [-5, 5],
            [0, 10],
            ["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
            0.1,
            5,
            1,
            1000,
            "rundir/",
            "inputdir/",
            "toolpath",
            "even_spacing",
        ),
    ],
)
def test_rectangular_boundary(
    longitude_extent,
    latitude_extent,
    date_range,
    resolution,
    number_vertical_layers,
    layer_thickness_ratio,
    depth,
    mom_run_dir,
    mom_input_dir,
    toolpath_dir,
    grid_type,
    tmp_path,
):

    eastern_boundary = xr.Dataset(
        {
            "temp": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["lata", "lona", "deepness", "time"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5),
                    "deepness": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "eta": xr.DataArray(
                np.random.random((100, 5, 10)),
                dims=["lata", "lona", "time"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "salt": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["lata", "lona", "deepness", "time"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5),
                    "deepness": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "u": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["lata", "lona", "deepness", "time"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5),
                    "deepness": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "v": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["lata", "lona", "deepness", "time"],
                coords={
                    "lata": np.linspace(latitude_extent[0] - 5, latitude_extent[1] + 5, 100),
                    "lona": np.linspace(longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5),
                    "deepness": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
        }
    )
    eastern_boundary.to_netcdf(tmp_path / "east_unprocessed")
    eastern_boundary.close()

    expt = experiment(
        longitude_extent=longitude_extent,
        latitude_extent=latitude_extent,
        date_range=date_range,
        resolution=resolution,
        number_vertical_layers=number_vertical_layers,
        layer_thickness_ratio=layer_thickness_ratio,
        depth=depth,
        mom_run_dir=mom_run_dir,
        mom_input_dir=mom_input_dir,
        toolpath_dir=toolpath_dir,
        grid_type=grid_type,
    )

    varnames = {
        "x": "lona",
        "y": "lata",
        "time": "time",
        "eta": "eta",
        "zl": "deepness",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    expt.rectangular_boundary(tmp_path / "east_unprocessed", varnames, "east", 1)
