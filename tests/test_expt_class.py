import importlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from regional_mom6 import experiment
from regional_mom6 import MOM_parameter_tools as mpt
from regional_mom6.segment import Segment
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
    dask.config.set(scheduler="single-threaded")
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


def test_experiment_from_grid_and_vgrid_objects_without_scalar_args(
    tmp_path, grid, vgrid
):
    """Passing Grid/VGrid objects directly via hgrid_type/vgrid_type should not require
    resolution, longitude_extent, latitude_extent, number_vertical_layers,
    layer_thickness_ratio, or depth."""
    expt = experiment(
        date_range=["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
        mom_run_dir=tmp_path / "rundir",
        mom_input_dir=tmp_path / "inputdir",
        hgrid_type=grid,
        vgrid_type=vgrid,
    )

    assert expt.longitude_extent == (-5.0, 5.0)
    assert expt.latitude_extent == (0.0, 10.0)
    assert expt.m6f_hgrid is grid
    assert expt.m6f_vgrid is vgrid


def test_experiment_requires_hgrid_scalars_when_no_grid_object(tmp_path):
    """Without a Grid object, resolution/longitude_extent/latitude_extent are required
    to generate an hgrid."""
    with pytest.raises(AssertionError, match="resolution"):
        experiment(
            date_range=["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
            mom_run_dir=tmp_path / "rundir",
            mom_input_dir=tmp_path / "inputdir",
            number_vertical_layers=5,
            layer_thickness_ratio=1,
            depth=1000,
        )


def test_experiment_requires_vgrid_scalars_when_no_vgrid_object(tmp_path):
    """Without a VGrid object, number_vertical_layers/layer_thickness_ratio/depth are
    required to generate a vgrid."""
    with pytest.raises(AssertionError, match="number_vertical_layers"):
        experiment(
            longitude_extent=[-5, 5],
            latitude_extent=[0, 10],
            date_range=["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
            resolution=0.1,
            mom_run_dir=tmp_path / "rundir",
            mom_input_dir=tmp_path / "inputdir",
        )


def _write_hgrid_with_bad_angle_calc(tmp_path, grid, angle_offset_degrees):
    """Build a small hgrid via mom6_forge, inject an `angle_dx` discrepancy, and write
    it to `tmp_path/inputdir/hgrid.nc`. Returns the input dir."""
    ds = grid.supergrid.to_ds()
    ds["angle_dx"] = ds["angle_dx"] + angle_offset_degrees
    input_dir = tmp_path / "inputdir"
    input_dir.mkdir()
    ds.to_netcdf(input_dir / "hgrid.nc")
    return input_dir


def test_hgrid_property_raises_on_stale_angle_dx(tmp_path, grid, vgrid):
    """A large angle_dx discrepancy discovered on a *lazy* hgrid.nc load (i.e. not
    during __init__ itself) should hard-error, pointing at recalculate_rotation_angle.
    """
    input_dir = _write_hgrid_with_bad_angle_calc(
        tmp_path, grid, angle_offset_degrees=45.0
    )

    with pytest.warns(UserWarning, match="recalculate_rotation_angle"):
        expt = experiment(
            date_range=["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
            mom_run_dir=tmp_path / "rundir",
            mom_input_dir=input_dir,
            hgrid_type="from_file",
            vgrid_type=vgrid,
        )


def test_recalculate_rotation_angle_is_noop_for_consistent_grid(tmp_path, grid, vgrid):
    """Calling recalculate_rotation_angle() on an already-consistent grid should
    leave angle_dx unchanged."""
    expt = experiment(
        date_range=["2003-01-01 00:00:00", "2003-01-01 00:00:00"],
        mom_run_dir=tmp_path / "rundir",
        mom_input_dir=tmp_path / "inputdir",
        hgrid_type=grid,
        vgrid_type=vgrid,
    )

    before = expt.hgrid["angle_dx"].values.copy()
    expt.recalculate_rotation_angle()
    after = expt.hgrid["angle_dx"].values

    np.testing.assert_allclose(before, after)


def test_get_segment_reuses_cached_segment(tmp_path, monkeypatch):
    """_get_segment should build a Segment once per orientation and reuse the
    same object on subsequent calls -- this is what lets setup_ocean_state_boundaries
    and setup_boundary_tides share one Segment per orientation instead of each
    re-deriving it from the grid."""
    expt = experiment.create_empty(
        expt_name="cache_test",
        mom_input_dir=tmp_path,
        mom_run_dir=tmp_path,
    )
    expt.longitude_extent = (-5, 5)
    expt.latitude_extent = (0, 10)
    expt.hgrid_type = "even_spacing"
    expt.resolution = 0.1
    expt._make_hgrid()

    call_count = {"n": 0}
    real_cardinal = Segment.cardinal.__func__

    def counting_cardinal(cls, *args, **kwargs):
        call_count["n"] += 1
        return real_cardinal(cls, *args, **kwargs)

    monkeypatch.setattr(Segment, "cardinal", classmethod(counting_cardinal))

    segment_first = expt._get_segment("east")
    segment_second = expt._get_segment("east")

    assert call_count["n"] == 1, "Segment.cardinal should only be called once"
    assert segment_first is segment_second
    assert expt.segments["east"] is segment_first

    # A different orientation still builds its own Segment
    segment_north = expt._get_segment("north")
    assert call_count["n"] == 2
    assert segment_north is not segment_first


def test_setup_generic_writes_correct_obc_position_strings(tmp_path):
    """setup_generic's OBC_SEGMENT_00N position strings should come from
    Segment.mom6_obc_position_string, matching the values the old hardcoded
    rect_MOM6_index_dir produced for the 4 cardinal boundaries."""
    expt = experiment.create_empty(
        expt_name="obc_test",
        mom_input_dir=tmp_path / "inputdir",
        mom_run_dir=tmp_path / "rundir",
    )
    expt.mom_input_dir.mkdir()
    expt.mom_run_dir.mkdir()
    expt.longitude_extent = (-5, 5)
    expt.latitude_extent = (0, 10)
    expt.date_range = ["2000-01-01 00:00:00", "2000-01-02 00:00:00"]
    expt.hgrid_type = "even_spacing"
    expt.resolution = 0.5
    expt.number_vertical_layers = 5
    expt.layer_thickness_ratio = 1
    expt.depth = 1000
    expt.minimum_depth = 4
    expt.tidal_constituents = []
    expt.boundaries = ["south", "north", "east", "west"]
    expt._make_hgrid()
    expt._make_vgrid()

    module_path = Path(importlib.resources.files("regional_mom6"))
    demos_dir = (
        module_path / "demos"
        if (module_path / "demos").exists()
        else module_path.parent / "demos"
    )
    shutil.copytree(
        demos_dir / "premade_run_directories" / "common_files",
        expt.mom_run_dir,
        dirs_exist_ok=True,
    )

    expt.setup_generic(mask_land_cpus=False)

    MOM_override_dict = mpt.read_MOM_file_as_dict("MOM_override", expt.mom_run_dir)
    expected_prefix = {
        "south": '"J=0,I=0:N',
        "north": '"J=N,I=N:0',
        "east": '"I=N,J=0:N',
        "west": '"I=0,J=N:0',
    }
    for seg in expt.boundaries:
        ind_seg = expt.find_MOM6_rectangular_orientation(seg)
        value = MOM_override_dict[f"OBC_SEGMENT_00{ind_seg}"]["value"]
        assert value.startswith(
            expected_prefix[seg]
        ), f"{seg} position string {value!r} does not match legacy convention"
