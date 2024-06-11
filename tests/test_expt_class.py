import numpy as np
import pytest
from regional_mom6 import experiment
import xarray as xr

## Note:
## When creating test dataarrays we use 'silly' names for coordinates to
## ensure that the proper mapping to MOM6 names occurs correctly


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
def test_setup_bathymetry(
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

    ## Generate a bathymetry to use in tests

    bathymetry_file = tmp_path / "bathymetry.nc"

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
    bathymetry.to_netcdf(bathymetry_file)
    bathymetry.close()

    # Now provide the above bathymetry file as input in `expt.setup_bathymetry()`
    expt.setup_bathymetry(
        bathymetry_path=str(bathymetry_file),
        longitude_coordinate_name="silly_lon",
        latitude_coordinate_name="silly_lat",
        vertical_coordinate_name="silly_depth",
        minimum_layers=1,
        chunks={"longitude": 10, "latitude": 10},
    )

    bathymetry_file.unlink()


def number_of_gridpoints(longitude_extent, latitude_extent, resolution):
    nx = int((longitude_extent[-1] - longitude_extent[0]) / resolution)
    ny = int((latitude_extent[-1] - latitude_extent[0]) / resolution)

    return nx, ny


def generate_temperature_arrays(nx, ny, number_vertical_layers):

    # temperatures close to 0 ᵒC
    temp_in_C = np.random.randn(ny, nx, number_vertical_layers)

    temp_in_C_masked = np.copy(temp_in_C)
    if int(ny / 4 + 4) < ny - 1 and int(nx / 3 + 4) < nx + 1:
        temp_in_C_masked[
            int(ny / 3) : int(ny / 3 + 5), int(nx) : int(nx / 4 + 4), :
        ] = float("nan")
    else:
        raise ValueError("use bigger domain")

    temp_in_K = np.copy(temp_in_C) + 273.15
    temp_in_K_masked = np.copy(temp_in_C_masked) + 273.15

    # ensure we didn't mask the minimum temperature
    if np.nanmin(temp_in_C_masked) == np.min(temp_in_C):
        return temp_in_C, temp_in_C_masked, temp_in_K, temp_in_K_masked
    else:
        return generate_temperature_arrays(nx, ny, number_vertical_layers)


def generate_silly_coords(
    longitude_extent, latitude_extent, resolution, depth, number_vertical_layers
):
    nx, ny = number_of_gridpoints(longitude_extent, latitude_extent, resolution)

    horizontal_buffer = 5

    silly_lat = np.linspace(
        latitude_extent[0] - horizontal_buffer,
        latitude_extent[1] + horizontal_buffer,
        ny,
    )
    silly_lon = np.linspace(
        longitude_extent[0] - horizontal_buffer,
        longitude_extent[1] + horizontal_buffer,
        nx,
    )
    silly_depth = np.linspace(0, depth, number_vertical_layers)

    return silly_lat, silly_lon, silly_depth


longitude_extent = [-5, 3]
latitude_extent = (0, 10)
date_range = ["2003-01-01 00:00:00", "2003-01-01 00:00:00"]
resolution = 0.1
number_vertical_layers = 5
layer_thickness_ratio = 1
depth = 1000

silly_lat, silly_lon, silly_depth = generate_silly_coords(
    longitude_extent, latitude_extent, resolution, depth, number_vertical_layers
)

dims = ["silly_lat", "silly_lon", "silly_depth"]

coords = {"silly_lat": silly_lat, "silly_lon": silly_lon, "silly_depth": silly_depth}

mom_run_dir = "rundir/"
mom_input_dir = "inputdir/"
toolpath_dir = "toolpath"
grid_type = "even_spacing"

nx, ny = number_of_gridpoints(longitude_extent, latitude_extent, resolution)

temp_in_C, temp_in_C_masked, temp_in_K, temp_in_K_masked = generate_temperature_arrays(
    nx, ny, number_vertical_layers
)

temp_C = xr.DataArray(temp_in_C, dims=dims, coords=coords)
temp_K = xr.DataArray(temp_in_K, dims=dims, coords=coords)
temp_C_masked = xr.DataArray(temp_in_C_masked, dims=dims, coords=coords)
temp_K_masked = xr.DataArray(temp_in_K_masked, dims=dims, coords=coords)

maximum_temperature_in_C = np.max(temp_in_C)


@pytest.mark.parametrize(
    "temp_dataarray_initial_condition",
    [temp_C, temp_C_masked, temp_K, temp_K_masked],
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
            longitude_extent,
            latitude_extent,
            date_range,
            resolution,
            number_vertical_layers,
            layer_thickness_ratio,
            depth,
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
    temp_dataarray_initial_condition,
    tmp_path,
):

    silly_lat, silly_lon, silly_depth = generate_silly_coords(
        longitude_extent, latitude_extent, resolution, depth, number_vertical_layers
    )

    dims = ["silly_lat", "silly_lon", "silly_depth"]

    coords = {
        "silly_lat": silly_lat,
        "silly_lon": silly_lon,
        "silly_depth": silly_depth,
    }

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

    nx, ny = number_of_gridpoints(longitude_extent, latitude_extent, resolution)

    # initial condition includes, temp, salt, eta, u, v
    initial_cond = xr.Dataset(
        {
            "eta": xr.DataArray(
                np.random.random((ny, nx)),
                dims=["silly_lat", "silly_lon"],
                coords={
                    "silly_lat": silly_lat,
                    "silly_lon": silly_lon,
                },
            ),
            "temp": temp_dataarray_initial_condition,
            "salt": xr.DataArray(
                np.random.random((ny, nx, number_vertical_layers)),
                dims=dims,
                coords=coords,
            ),
            "u": xr.DataArray(
                np.random.random((ny, nx, number_vertical_layers)),
                dims=dims,
                coords=coords,
            ),
            "v": xr.DataArray(
                np.random.random((ny, nx, number_vertical_layers)),
                dims=dims,
                coords=coords,
            ),
        }
    )

    # Generate boundary forcing

    initial_cond.to_netcdf(tmp_path / "ic_unprocessed")
    initial_cond.close()
    varnames = {
        "x": "silly_lon",
        "y": "silly_lat",
        "time": "time",
        "eta": "eta",
        "zl": "silly_depth",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    expt.initial_condition(
        tmp_path / "ic_unprocessed",
        varnames,
        arakawa_grid="A",
    )

    # ensure that temperature is in degrees C
    assert np.nanmin(expt.ic_tracers["temp"]) < 100.0

    # max(temp) can be less maximum_temperature_in_C due to re-gridding
    assert np.nanmax(expt.ic_tracers["temp"]) <= maximum_temperature_in_C


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
def test_rectangular_boundaries(
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
                dims=["silly_lat", "silly_lon", "silly_depth", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "silly_depth": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "eta": xr.DataArray(
                np.random.random((100, 5, 10)),
                dims=["silly_lat", "silly_lon", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "salt": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["silly_lat", "silly_lon", "silly_depth", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "silly_depth": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "u": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["silly_lat", "silly_lon", "silly_depth", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "silly_depth": np.linspace(0, 1000, 10),
                    "time": np.linspace(0, 1000, 10),
                },
            ),
            "v": xr.DataArray(
                np.random.random((100, 5, 10, 10)),
                dims=["silly_lat", "silly_lon", "silly_depth", "time"],
                coords={
                    "silly_lat": np.linspace(
                        latitude_extent[0] - 5, latitude_extent[1] + 5, 100
                    ),
                    "silly_lon": np.linspace(
                        longitude_extent[1] - 0.5, longitude_extent[1] + 0.5, 5
                    ),
                    "silly_depth": np.linspace(0, 1000, 10),
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
        "x": "silly_lon",
        "y": "silly_lat",
        "time": "time",
        "eta": "eta",
        "zl": "silly_depth",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    expt.rectangular_boundaries(tmp_path, varnames)
