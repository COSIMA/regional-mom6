import pytest
import os
import xarray as xr
import numpy as np
import regional_mom6 as rmom6

# Define the path where the curvilinear hgrid file is expected in the Docker container
DOCKER_FILE_PATH = "/data/small_curvilinear_hgrid.nc"


# Define the local directory where the user might have added the curvilinear hgrid file
LOCAL_FILE_PATH = str(os.getenv("local_curvilinear_hgrid"))


@pytest.fixture
def get_curvilinear_hgrid():
    # Check if the file exists in the Docker-specific location
    if os.path.exists(DOCKER_FILE_PATH):
        return xr.open_dataset(DOCKER_FILE_PATH)

    # Check if the user has provided the file in a specific local directory
    elif os.path.exists(LOCAL_FILE_PATH):
        return xr.open_dataset(LOCAL_FILE_PATH)

    # If neither location contains the file, skip test
    else:
        pytest.skip(
            f"Required file 'hgrid.nc' not found in {DOCKER_FILE_PATH} or {LOCAL_FILE_PATH}"
        )


@pytest.fixture
def get_rectilinear_hgrid():
    lat = np.linspace(0, 10, 7)
    lon = np.linspace(0, 10, 13)
    rect_hgrid = rmom6.generate_rectangular_hgrid(lat, lon)
    return rect_hgrid


@pytest.fixture()
def generate_silly_vt_dataset():
    latitude_extent = [30, 40]
    longitude_extent = [-80, -70]
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
    eastern_boundary.time.attrs = {"units": "days"}
    return eastern_boundary


@pytest.fixture()
def generate_silly_ic_dataset():
    def _generate_silly_ic_dataset(
        longitude_extent,
        latitude_extent,
        resolution,
        number_vertical_layers,
        depth,
        temp_dataarray_initial_condition,
    ):
        nx, ny = number_of_gridpoints(longitude_extent, latitude_extent, resolution)
        silly_lat, silly_lon, silly_depth = generate_silly_coords(
            longitude_extent, latitude_extent, resolution, depth, number_vertical_layers
        )

        dims = ["silly_lat", "silly_lon", "silly_depth"]

        coords = {
            "silly_lat": silly_lat,
            "silly_lon": silly_lon,
            "silly_depth": silly_depth,
        }
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
        return initial_cond

    return _generate_silly_ic_dataset


@pytest.fixture()
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


def get_temperature_dataarrays(
    longitude_extent, latitude_extent, resolution, number_vertical_layers, depth
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

    fre_tools_dir = "toolpath"
    hgrid_type = "even_spacing"

    nx, ny = number_of_gridpoints(longitude_extent, latitude_extent, resolution)

    (
        temp_in_C,
        temp_in_C_masked,
        temp_in_K,
        temp_in_K_masked,
    ) = generate_temperature_arrays(nx, ny, number_vertical_layers)

    temp_C = xr.DataArray(temp_in_C, dims=dims, coords=coords)
    temp_K = xr.DataArray(temp_in_K, dims=dims, coords=coords)
    temp_C_masked = xr.DataArray(temp_in_C_masked, dims=dims, coords=coords)
    temp_K_masked = xr.DataArray(temp_in_K_masked, dims=dims, coords=coords)

    maximum_temperature_in_C = np.max(temp_in_C)
    return [temp_C, temp_C_masked, temp_K, temp_K_masked]


def number_of_gridpoints(longitude_extent, latitude_extent, resolution):
    nx = int((longitude_extent[-1] - longitude_extent[0]) / resolution)
    ny = int((latitude_extent[-1] - latitude_extent[0]) / resolution)

    return nx, ny


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
