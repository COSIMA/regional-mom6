import numpy as np
import pytest
from regional_mom6 import experiment
import xarray as xr
import xesmf as xe
import dask
from .conftest import (
    generate_temperature_arrays,
    generate_silly_coords,
    number_of_gridpoints,
    get_temperature_dataarrays,
)

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
        "fre_tools_dir",
        "hgrid_type",
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
    fre_tools_dir,
    hgrid_type,
    tmp_path,
):
    mom_run_dir = tmp_path / "rundir"
    mom_input_dir = tmp_path / "inputdir"
    expt = experiment(
        longitude_extent=longitude_extent,
        latitude_extent=latitude_extent,
        date_range=date_range,
        resolution=resolution,
        number_vertical_layers=number_vertical_layers,
        layer_thickness_ratio=layer_thickness_ratio,
        depth=depth,
        mom_run_dir=tmp_path / mom_run_dir,
        mom_input_dir=tmp_path / mom_input_dir,
        fre_tools_dir=fre_tools_dir,
        hgrid_type=hgrid_type,
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
    )

    bathymetry_file.unlink()


longitude_extent = [-5, 3]
latitude_extent = (0, 10)
date_range = ["2003-01-01 00:00:00", "2003-01-01 00:00:00"]
resolution = 0.1
number_vertical_layers = 5
layer_thickness_ratio = 1
depth = 1000


@pytest.mark.parametrize(
    "temp_dataarray_initial_condition",
    get_temperature_dataarrays(
        longitude_extent, latitude_extent, resolution, number_vertical_layers, depth
    ),
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
        "fre_tools_dir",
        "hgrid_type",
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
    fre_tools_dir,
    hgrid_type,
    temp_dataarray_initial_condition,
    tmp_path,
    generate_silly_ic_dataset,
):
    dask.config.set(scheduler="single-threaded")
    mom_run_dir = tmp_path / "rundir"
    mom_input_dir = tmp_path / "inputdir"
    expt = experiment(
        longitude_extent=longitude_extent,
        latitude_extent=latitude_extent,
        date_range=date_range,
        resolution=resolution,
        number_vertical_layers=number_vertical_layers,
        layer_thickness_ratio=layer_thickness_ratio,
        depth=depth,
        mom_run_dir=tmp_path / mom_run_dir,
        mom_input_dir=tmp_path / mom_input_dir,
        fre_tools_dir=fre_tools_dir,
        hgrid_type=hgrid_type,
    )

    # initial condition includes, temp, salt, eta, u, v
    initial_cond = generate_silly_ic_dataset(
        longitude_extent,
        latitude_extent,
        resolution,
        number_vertical_layers,
        depth,
        temp_dataarray_initial_condition,
    )

    initial_cond.to_netcdf(tmp_path / "ic_unprocessed")
    initial_cond.close()
    varnames = {
        "xh": "silly_lon",
        "yh": "silly_lat",
        "time": "time",
        "eta": "eta",
        "zl": "silly_depth",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    expt.setup_initial_condition(
        tmp_path / "ic_unprocessed",
        varnames,
        arakawa_grid="A",
    )

    # ensure that temperature is in degrees C
    assert np.nanmin(expt.ic_tracers["temp"]) < 100.0
    maximum_temperature_in_C = np.max(temp_dataarray_initial_condition)
    # max(temp) can be less maximum_temperature_in_C due to re-gridding
    assert np.nanmax(expt.ic_tracers["temp"]) <= maximum_temperature_in_C
    dask.config.set(scheduler=None)


def test_bgc_tracers_carried_through_initial_condition(
    tmp_path, generate_silly_ic_dataset
):
    """BGC tracers beyond temp/salt must appear in ic_tracers after setup_initial_condition."""
    lon_ext = [-5, 3]
    lat_ext = [0, 10]
    nz = 5

    initial_cond = generate_silly_ic_dataset(
        lon_ext,
        lat_ext,
        resolution,
        nz,
        depth,
        get_temperature_dataarrays(lon_ext, lat_ext, resolution, nz, depth)[0],
    )
    # Add a BGC tracer
    nx, ny = number_of_gridpoints(lon_ext, lat_ext, resolution)
    silly_lat, silly_lon, silly_depth = (
        initial_cond.silly_lat,
        initial_cond.silly_lon,
        initial_cond.silly_depth,
    )
    initial_cond["no3"] = xr.DataArray(
        np.random.random((ny, nx, nz)),
        dims=["silly_lat", "silly_lon", "silly_depth"],
        coords={
            "silly_lat": silly_lat,
            "silly_lon": silly_lon,
            "silly_depth": silly_depth,
        },
    )
    initial_cond.to_netcdf(tmp_path / "ic_bgc")
    initial_cond.close()

    expt = experiment(
        longitude_extent=lon_ext,
        latitude_extent=lat_ext,
        date_range=date_range,
        resolution=resolution,
        number_vertical_layers=nz,
        layer_thickness_ratio=layer_thickness_ratio,
        depth=depth,
        mom_run_dir=tmp_path / "rundir",
        mom_input_dir=tmp_path / "inputdir",
        fre_tools_dir="toolpath",
        hgrid_type="even_spacing",
    )
    varnames = {
        "xh": "silly_lon",
        "yh": "silly_lat",
        "time": "time",
        "eta": "eta",
        "zl": "silly_depth",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt", "no3": "no3"},
    }
    expt.setup_initial_condition(tmp_path / "ic_bgc", varnames, arakawa_grid="A")

    assert "no3" in expt.ic_tracers
    dask.config.set(scheduler=None)


@pytest.mark.parametrize(
    (
        "longitude_extent",
        "latitude_extent",
        "date_range",
        "resolution",
        "number_vertical_layers",
        "layer_thickness_ratio",
        "depth",
        "fre_tools_dir",
        "hgrid_type",
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
    fre_tools_dir,
    hgrid_type,
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
            "o2": xr.DataArray(
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
    eastern_boundary.to_netcdf(tmp_path / "east_unprocessed.nc")
    eastern_boundary.close()
    mom_run_dir = tmp_path / "rundir"
    mom_input_dir = tmp_path / "inputdir"
    expt = experiment(
        longitude_extent=longitude_extent,
        latitude_extent=latitude_extent,
        date_range=date_range,
        resolution=resolution,
        number_vertical_layers=number_vertical_layers,
        layer_thickness_ratio=layer_thickness_ratio,
        depth=depth,
        mom_run_dir=tmp_path / mom_run_dir,
        mom_input_dir=tmp_path / mom_input_dir,
        fre_tools_dir=fre_tools_dir,
        hgrid_type=hgrid_type,
        boundaries=["east"],
    )

    varnames = {
        "xh": "silly_lon",
        "yh": "silly_lat",
        "time": "time",
        "eta": "eta",
        "zl": "silly_depth",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    # Add test for bgc_tracer_names
    bgc_tracer_names = {"o2": "o2"}
    expt.setup_ocean_state_boundaries(
        tmp_path, varnames, bgc_tracer_names=bgc_tracer_names
    )
    assert (
        expt.mom_input_dir / "o2_obc_segment.nc"
    ).exists(), "BGC tracer file not created"


def test_reformat_bgc_tracers_into_files(tmp_path):
    """reformat_bgc_tracers_into_files writes one file per BGC tracer containing the tracer and its dz variable."""
    expt = experiment(
        longitude_extent=[-5, 5],
        latitude_extent=[0, 10],
        date_range=["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
        resolution=0.1,
        number_vertical_layers=5,
        layer_thickness_ratio=1,
        depth=1000,
        mom_run_dir=tmp_path / "rundir",
        mom_input_dir=tmp_path / "inputdir",
        fre_tools_dir="toolpath",
        hgrid_type="even_spacing",
        boundaries=["east", "west"],
    )

    # Write a fake forcing_obc_segment_001.nc with BGC tracer vars
    segs = ["001", "002"]
    for seg in segs:
        ds = xr.Dataset(
            {
                f"o2_segment_{seg}": xr.DataArray(
                    np.random.random((3, 5, 4)),
                    dims=["time", "nz", f"nx_segment_{seg}"],
                ),
                f"dz_o2_segment_{seg}": xr.DataArray(
                    np.random.random((3, 5, 4)),
                    dims=["time", "nz", f"nx_segment_{seg}"],
                ),
                f"temp_segment_{seg}": xr.DataArray(
                    np.random.random((3, 5, 4)),
                    dims=["time", "nz", f"nx_segment_{seg}"],
                ),
            }
        )
        ds.to_netcdf(expt.mom_input_dir / f"forcing_obc_segment_{seg}.nc")

    expt.reformat_bgc_tracers_into_files({"o2": "o2"})

    out_file = expt.mom_input_dir / "o2_obc_segment.nc"
    assert out_file.exists(), "output file for BGC tracer not created"
    result = xr.open_dataset(out_file)
    for seg in segs:
        assert f"o2_segment_{seg}" in result, "tracer variable missing from output"
        assert (
            f"dz_o2_segment_{seg}" in result
        ), "dz thickness variable missing from output"
        assert (
            f"temp_segment_{seg}" not in result
        ), "physical tracer should not be in BGC file"
