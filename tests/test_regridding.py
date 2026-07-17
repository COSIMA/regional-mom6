import regional_mom6 as rmom6
import regional_mom6.regridding as rgd
import pytest
import warnings
import xarray as xr
import numpy as np
import pandas as pd
from regional_mom6.boundary import Boundary
import shutil
from mom6_forge.grid import *
from mom6_forge.topo import Topo


# Not testing get_arakawa_c_points, & create_regridder
def test_smoke_untested_funcs(get_curvilinear_hgrid, generate_silly_vt_dataset):
    hgrid = get_curvilinear_hgrid
    ds = generate_silly_vt_dataset
    ds["lat"] = ds.silly_lat
    ds["lon"] = ds.silly_lat
    assert rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    assert Boundary.cardinal(hgrid, "north", "segment_002")
    assert rgd.create_regridder(ds, ds)


def test_fill_missing_data(generate_silly_vt_dataset):
    """
    Only testing forward fill for now
    """
    ds = generate_silly_vt_dataset
    ds["temp"][0, 0, 6:10, 0] = np.nan
    ds.temp.attrs = {"units": "C"}
    ds = rgd.fill_missing_data(ds, "silly_depth", fill="f")
    assert ds.temp.attrs == {"units": "C"}  # Assert that attributes are retained
    assert (
        ds["temp"][0, 0, 6:10, 0] == (ds["temp"][0, 0, 5, 0])
    ).all()  # Assert if we are forward filling in time

    ds_2 = generate_silly_vt_dataset
    ds_2["temp"][0, 0, 6:10, 0] = ds["temp"][0, 0, 5, 0]
    assert (ds["temp"] == (ds_2["temp"])).all()  # Assert everything else is the same


def test_add_or_update_time_dim(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset

    ds = rgd.add_or_update_time_dim(ds, xr.DataArray([0]))
    assert ds.time.attrs == {"units": "days"}  # Assert that attributes are retained
    assert ds["time"].values == [0]  # Assert time is added
    assert ds["temp"].dims[0] == "time"  # Check time is first dim


def test_generate_dz(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset

    dz = rgd.generate_dz(ds, "silly_depth")
    assert ds.time.attrs == {"units": "days"}  # Assert that attributes are retained
    z = np.linspace(0, 1000, 10)
    dz_check = np.full(z.shape, z[1] - z[0])
    assert (
        (dz.values - dz_check) < 0.00001
    ).all()  # Assert dz is generated correctly (some rounding leniency)


def test_add_secondary_dimension(get_curvilinear_hgrid, generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    hgrid = get_curvilinear_hgrid

    # N/S Boundary
    boundary = Boundary.cardinal(hgrid, "north", "segment_002")
    ds = rgd.add_secondary_dimension(ds, "temp", boundary, "segment_002")
    assert ds.time.attrs == {"units": "days"}  # Assert that attributes are retained
    assert ds["temp"].dims == (
        "silly_lat",
        "ny_segment_002",
        "silly_lon",
        "silly_depth",
        "time",
    )

    # E/W Boundary
    boundary = Boundary.cardinal(hgrid, "east", "segment_003")
    ds = generate_silly_vt_dataset
    ds = rgd.add_secondary_dimension(ds, "v", boundary, "segment_003")
    assert ds["v"].dims == (
        "silly_lat",
        "silly_lon",
        "nx_segment_003",
        "silly_depth",
        "time",
    )

    # Beginning
    ds = generate_silly_vt_dataset
    ds = rgd.add_secondary_dimension(
        ds, "temp", boundary, "segment_003", to_beginning=True
    )
    assert ds["temp"].dims[0] == "nx_segment_003"

    # NZ dim E/W Boundary
    ds = generate_silly_vt_dataset
    ds = ds.rename({"silly_depth": "nz"})
    ds = rgd.add_secondary_dimension(ds, "u", boundary, "segment_003")
    assert ds["u"].dims == (
        "silly_lat",
        "silly_lon",
        "nz",
        "nx_segment_003",
        "time",
    )


def test_vertical_coordinate_encoding(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset

    ds = rgd.vertical_coordinate_encoding(ds, "temp", "segment_002", "silly_depth")
    assert ds.time.attrs == {"units": "days"}  # Assert that attributes are retained

    assert "nz_segment_002_temp" in ds["temp"].dims
    assert "nz_segment_002_temp" in ds
    assert (
        ds["nz_segment_002_temp"] == np.arange(ds[f"nz_segment_002_temp"].size)
    ).all()


def test_generate_layer_thickness(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    ds["temp"] = ds["temp"].transpose("time", "silly_depth", "silly_lat", "silly_lon")
    ds = rgd.generate_layer_thickness(ds, "temp", "segment_002", "silly_depth")
    assert ds.time.attrs == {"units": "days"}  # Assert that attributes are retained

    assert "dz_temp" in ds
    assert ds["dz_temp"].dims == ("time", "nz_temp", "ny_segment_002", "nx_segment_002")
    assert (
        ds["temp"]["silly_depth"].shape == ds["dz_temp"]["nz_temp"].shape
    )  # Make sure the depth dimension was broadcasted correctly


def test_generate_encoding(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    encoding_dict = {}
    ds["temp_segment_002"] = ds["temp"]
    ds.coords["temp_segment_003_nz_"] = ds.silly_depth
    encoding_dict = rgd.generate_encoding(ds, encoding_dict, default_fill_value="-3")
    assert (
        encoding_dict["temp_segment_002"]["_FillValue"] == "-3"
        and "dtype" not in encoding_dict["temp_segment_002"]
    )
    assert encoding_dict["temp_segment_003_nz_"]["dtype"] == "int32"


def _synthetic_grid_and_topo(land_ny_indices=()):
    """A small rectilinear synthetic grid + matching Topo, all-ocean except for
    the given t-point row indices (along ny), which are forced to land."""
    grid = Grid(
        resolution=2,
        xstart=2,
        lenx=10,
        ystart=2,
        leny=10,
        name="masktest",
        type="rectilinear_cartesian",
    )
    topo = Topo(grid, min_depth=5.0, git=False)
    topo.set_flat(100.0)
    if land_ny_indices:
        depth = topo.depth.values.copy()
        for j in land_ny_indices:
            depth[j, :] = 0.0
        topo.depth = depth
    return grid, topo


def test_boundary_mask_matches_direct_supergridmask_slice():
    """Boundary.from_hgrid's mask should be exactly topo.supergridmask sliced the
    same way as lon/lat/angle -- no separate resolution-conversion step needed."""
    grid, topo = _synthetic_grid_and_topo(land_ny_indices=[0])
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")

    boundary = Boundary.from_hgrid(
        hgrid, axis="nyp", index=-1, segment_name="segment_001", topo=topo
    )
    expected = topo.supergridmask.isel(nyp=-1)
    assert (boundary.mask.values == expected.values).all()


def test_mask_dataset_no_mask_fills_zero():
    """With no mask (topo=None equivalent), mask_dataset just fills NaNs with 0."""
    boundary = Boundary(
        lon=xr.DataArray([1.0, 2.0, 3.0], dims=["nx_segment_099"]),
        lat=xr.DataArray([1.0, 1.0, 1.0], dims=["nx_segment_099"]),
        angle=xr.DataArray([0.0, 0.0, 0.0], dims=["nx_segment_099"]),
        segment_name="segment_099",
        parallel="nx",
        perpendicular="ny",
        axis_to_expand=2,
        mask=None,
    )
    ds = xr.Dataset(
        {"temp": (("ny_segment_099", "nx_segment_099"), np.array([[1.0, np.nan, 3.0]]))}
    )
    ds = rgd.mask_dataset(ds, boundary)
    assert (ds["temp"].values == np.array([[1.0, 0.0, 3.0]])).all()


def test_mask_dataset_no_dilation():
    """Ocean/land transitions should NOT be dilated by one point -- unlike the old
    get_boundary_mask, boundary.mask (from Topo.supergridmask) is treated as ground
    truth with no post-processing."""
    boundary = Boundary(
        lon=xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=["nx_segment_099"]),
        lat=xr.DataArray([1.0] * 5, dims=["nx_segment_099"]),
        angle=xr.DataArray([0.0] * 5, dims=["nx_segment_099"]),
        segment_name="segment_099",
        parallel="nx",
        perpendicular="ny",
        axis_to_expand=2,
        mask=xr.DataArray(
            [0, 1, 1, 0, 1], dims=["nx_segment_099"]
        ),  # land,ocean,ocean,land,ocean
    )
    ds = xr.Dataset(
        {
            "temp": (
                ("ny_segment_099", "nx_segment_099"),
                np.array([[10.0, 20.0, 30.0, 40.0, 50.0]]),
            )
        }
    )
    fill_value = -999.0
    ds = rgd.mask_dataset(ds, boundary, fill_value=fill_value)
    # Land points (index 0 and 3) are masked out; ocean points keep their values --
    # in particular, index 2 (ocean, adjacent to land at index 3) is NOT dilated
    # into being masked, and index 0/3 are NOT left unmasked just because a
    # neighbor is ocean.
    assert ds["temp"].values[0, 0] == fill_value
    assert ds["temp"].values[0, 3] == fill_value
    assert ds["temp"].values[0, 1] == 20.0
    assert ds["temp"].values[0, 2] == 30.0
    assert ds["temp"].values[0, 4] == 50.0


def test_boundary_from_hgrid_missing_angle_dx_warns(get_rectilinear_hgrid):
    hgrid = get_rectilinear_hgrid.drop_vars("angle_dx")
    with pytest.warns(UserWarning, match="angle_dx"):
        boundary = Boundary.from_hgrid(
            hgrid, axis="nyp", index=-1, segment_name="segment_001"
        )
    assert (boundary.angle.values == 0).all()


def test_boundary_cardinal_invalid_orientation_raises(get_rectilinear_hgrid):
    with pytest.raises(ValueError, match="orientation must be one of"):
        Boundary.cardinal(get_rectilinear_hgrid, "northeast", "segment_001")


def test_boundary_from_hgrid_invalid_axis_raises(get_rectilinear_hgrid):
    with pytest.raises(ValueError, match="axis must be one of"):
        Boundary.from_hgrid(
            get_rectilinear_hgrid, axis="nx", index=0, segment_name="segment_001"
        )


def test_boundary_from_hgrid_arbitrary_axis_index_range(get_rectilinear_hgrid):
    hgrid = get_rectilinear_hgrid
    index_range = slice(2, 6)
    boundary = Boundary.from_hgrid(
        hgrid,
        axis="nyp",
        index=3,
        segment_name="segment_001",
        index_range=index_range,
    )
    expected_lon = hgrid["x"].isel(nyp=3).isel(nxp=index_range)
    expected_lat = hgrid["y"].isel(nyp=3).isel(nxp=index_range)
    assert np.allclose(boundary.lon.values, expected_lon.values)
    assert np.allclose(boundary.lat.values, expected_lat.values)
    assert boundary.lon.sizes["nx_segment_001"] == index_range.stop - index_range.start


def test_regrid_velocity_tracers(toy_glorys_ds, tmp_path):
    """
    Correctness test for Boundary.regrid_velocity_tracers.

    Checks:
    - Output OBC file is written
    - Variables follow the {var}_{segment_name} naming convention
    - Temperature is in Celsius (< 100)
    - 3D fields have companion dz_* variables
    - Vertical coordinate is re-encoded as incremental integers
    - Perpendicular dimension has size 1
    """

    grid = Grid(
        resolution=2,
        xstart=2,
        lenx=2,
        ystart=2,
        leny=2,
        name="test",
        type="rectilinear_cartesian",
    )
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")
    seg_name = "segment_001"
    outfolder = tmp_path / "inputdir"
    outfolder.mkdir()

    # Minimal synthetic boundary dataset covering the east edge of the hgrid (lon ≈ 10, lat 0-10)

    infile = tmp_path / "east_raw.nc"
    ds = toy_glorys_ds
    ds.to_netcdf(infile)
    ds.close()

    varnames = {
        "xh": "lon",
        "yh": "lat",
        "time": "time",
        "eta": "eta",
        "zl": "depth",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    boundary = Boundary.cardinal(hgrid, "east", seg_name)
    segment_out, _ = boundary.regrid_velocity_tracers(
        infile, varnames, outfolder, "2003-01-01 00:00:00", arakawa_grid="A"
    )

    # Salt is spatially constant, so all ocean points must match exactly.
    salt_vals = segment_out[f"salt_{seg_name}"].values
    np.testing.assert_allclose(salt_vals, 35.0, rtol=1e-4)

    # Temp is spatially varying (20–26 °C), so just check values are exact (hgrid overlap with seg exactly)
    temp_vals = segment_out[f"temp_{seg_name}"].values
    assert temp_vals[0, 0, 0, 0] == 22
    assert temp_vals[0, 0, 2, 0] == 26

    boundary_north = Boundary.cardinal(hgrid, "north", seg_name)
    segment_out_north, _ = boundary_north.regrid_velocity_tracers(
        infile, varnames, outfolder, "2003-01-01 00:00:00", arakawa_grid="A"
    )
    temp_vals = segment_out_north[f"temp_{seg_name}"].values
    assert temp_vals[0, 0, 0, 0] == 24
    assert temp_vals[0, 0, 0, 2] == 26

    # Mess with hgrid, subtract 1

    folder = outfolder / "weights"
    shutil.rmtree(
        folder
    )  # removes weights so they aren't saved with the old hgrid, and forces them to be recomputed with the new hgrid
    folder.mkdir()  # recreate the empty folder
    hgrid["x"] = hgrid.x + 1
    hgrid["y"] = hgrid.y + 1
    boundary_regrid = Boundary.cardinal(hgrid, "west", seg_name)
    seg_regridded, _ = boundary_regrid.regrid_velocity_tracers(
        infile,
        varnames,
        outfolder,
        "2003-01-01 00:00:00",
        arakawa_grid="A",
        regridding_method="bilinear",
    )
    temp_vals = seg_regridded[f"temp_{seg_name}"].values
    assert (
        np.abs(temp_vals[0, 0, 0, 0] - 23) < 0.01
    )  # bilinear at this point is nearly the average of the toy_glory_ds values (22 and 24 and 26 and 20)
    assert (
        temp_vals[0, 0, 2, 0] == 0
    )  # The bilinear regridding would be zero here because there isn't 4 points


def test_boundary_standalone_no_hgrid(toy_glorys_ds, tmp_path):
    """Prove regrid_velocity_tracers needs nothing beyond a hand-built Boundary --
    no mom6_forge.Grid, no hgrid, no Experiment involved at all."""
    outfolder = tmp_path / "inputdir"
    outfolder.mkdir()

    infile = tmp_path / "raw.nc"
    ds = toy_glorys_ds
    ds.to_netcdf(infile)
    ds.close()

    varnames = {
        "xh": "lon",
        "yh": "lat",
        "time": "time",
        "eta": "eta",
        "zl": "depth",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    seg_name = "segment_099"
    # A straight boundary line of 2 points, fully described by literal numbers --
    # lon fixed at 3.0 (inside toy_glorys_ds's [2, 4] lon range), lat spanning it.
    boundary = Boundary(
        lon=xr.DataArray([3.0, 3.0], dims=[f"ny_{seg_name}"]),
        lat=xr.DataArray([2.5, 3.5], dims=[f"ny_{seg_name}"]),
        angle=xr.DataArray([0.0, 0.0], dims=[f"ny_{seg_name}"]),
        segment_name=seg_name,
        parallel="ny",
        perpendicular="nx",
        axis_to_expand=3,
        mask=None,
    )

    segment_out, _ = boundary.regrid_velocity_tracers(
        infile, varnames, outfolder, "2003-01-01 00:00:00", arakawa_grid="A"
    )

    assert (outfolder / f"forcing_obc_{seg_name}.nc").exists()
    assert np.isfinite(segment_out[f"temp_{seg_name}"].values).all()
    np.testing.assert_allclose(segment_out[f"salt_{seg_name}"].values, 35.0, rtol=1e-4)


def test_boundary_regridders_manual_reuse(toy_glorys_ds, tmp_path, monkeypatch):
    """regrid_velocity_tracers should only rebuild regridders when regridders=None
    -- passing back a previously-returned dict skips rebuilding, matching today's
    documented manual-reuse pattern."""
    grid = Grid(
        resolution=2,
        xstart=2,
        lenx=2,
        ystart=2,
        leny=2,
        name="test",
        type="rectilinear_cartesian",
    )
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")
    outfolder = tmp_path / "inputdir"
    outfolder.mkdir()

    infile = tmp_path / "east_raw.nc"
    ds = toy_glorys_ds
    ds.to_netcdf(infile)
    ds.close()

    varnames = {
        "xh": "lon",
        "yh": "lat",
        "time": "time",
        "eta": "eta",
        "zl": "depth",
        "u": "u",
        "v": "v",
        "tracers": {"temp": "temp", "salt": "salt"},
    }

    boundary = Boundary.cardinal(hgrid, "east", "segment_001")

    call_count = {"n": 0}
    real_create_vt_regridders = rgd.create_vt_regridders

    def counting_create_vt_regridders(*args, **kwargs):
        call_count["n"] += 1
        return real_create_vt_regridders(*args, **kwargs)

    monkeypatch.setattr(rgd, "create_vt_regridders", counting_create_vt_regridders)

    _, _ = boundary.regrid_velocity_tracers(
        infile, varnames, outfolder, "2003-01-01 00:00:00", arakawa_grid="A"
    )
    assert call_count["n"] == 1
    cached_regridders = boundary._regridders

    _, _ = boundary.regrid_velocity_tracers(
        infile,
        varnames,
        outfolder,
        "2003-01-01 00:00:00",
        arakawa_grid="A",
        regridders=cached_regridders,
    )
    assert call_count["n"] == 1  # not rebuilt


def _synthetic_tidal_datasets(nc=2):
    """Minimal synthetic TPXO-like elevation/velocity datasets, already in the
    lon/lat/*Re/*Im form Boundary.regrid_tides expects -- skips the raw
    h_*/u_*-file parsing that experiment.setup_boundary_tides normally does
    (that parsing is covered separately by test_tides.py)."""
    nx, ny = 6, 6
    lon2d, lat2d = np.meshgrid(
        np.linspace(277, 283, nx), np.linspace(6, 11, ny), indexing="ij"
    )
    rng = np.random.default_rng(0)

    def _mk(*var_names):
        data = {
            "lon": (["nx", "ny"], lon2d),
            "lat": (["nx", "ny"], lat2d),
        }
        for name in var_names:
            data[name] = (["constituent", "nx", "ny"], rng.random((nc, nx, ny)))
        return xr.Dataset(data, coords={"constituent": np.arange(nc)})

    tpxo_h = _mk("hRe", "hIm")
    tpxo_u = _mk("uRe", "uIm")
    tpxo_v = _mk("vRe", "vIm")
    return tpxo_v, tpxo_u, tpxo_h


def test_regrid_tides_standalone(get_rectilinear_hgrid, tmp_path):
    """Boundary.regrid_tides should run against a plain Boundary (no Experiment
    involved) and write tz_*/tu_* files, mirroring the existing standalone
    coverage of regrid_velocity_tracers."""
    hgrid = get_rectilinear_hgrid
    seg_name = "segment_001"
    boundary = Boundary.cardinal(hgrid, "east", seg_name)

    tpxo_v, tpxo_u, tpxo_h = _synthetic_tidal_datasets()
    times = xr.DataArray(pd.date_range("2000-01-01", periods=1), dims=["time"])

    outfolder = tmp_path / "inputdir"
    outfolder.mkdir()

    boundary.regrid_tides(
        tpxo_v, tpxo_u, tpxo_h, times, outfolder, "2000-01-01 00:00:00"
    )

    assert (outfolder / f"tz_{seg_name}.nc").exists()
    assert (outfolder / f"tu_{seg_name}.nc").exists()

    tz = xr.open_dataset(outfolder / f"tz_{seg_name}.nc")
    assert np.isfinite(tz[f"zamp_{seg_name}"].values).all()


def test_regrid_tides_regridders_manual_reuse(
    get_rectilinear_hgrid, tmp_path, monkeypatch
):
    """regrid_tides should only rebuild its 3 regridders (elev/u/v) when
    regridders=None -- passing back boundary._tidal_regridders from a prior
    call skips rebuilding, matching the documented manual-reuse pattern for
    regrid_velocity_tracers."""
    hgrid = get_rectilinear_hgrid
    boundary = Boundary.cardinal(hgrid, "east", "segment_001")

    tpxo_v, tpxo_u, tpxo_h = _synthetic_tidal_datasets()
    times = xr.DataArray(pd.date_range("2000-01-01", periods=1), dims=["time"])

    outfolder = tmp_path / "inputdir"
    outfolder.mkdir()

    call_count = {"n": 0}
    real_create_regridder = rgd.create_regridder

    def counting_create_regridder(*args, **kwargs):
        call_count["n"] += 1
        return real_create_regridder(*args, **kwargs)

    monkeypatch.setattr(rgd, "create_regridder", counting_create_regridder)

    boundary.regrid_tides(
        tpxo_v, tpxo_u, tpxo_h, times, outfolder, "2000-01-01 00:00:00"
    )
    assert call_count["n"] == 3  # elev, u, v
    cached_regridders = boundary._tidal_regridders

    boundary.regrid_tides(
        tpxo_v,
        tpxo_u,
        tpxo_h,
        times,
        outfolder,
        "2000-01-01 00:00:00",
        regridders=cached_regridders,
    )
    assert call_count["n"] == 3  # not rebuilt
