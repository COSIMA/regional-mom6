import regional_mom6 as rmom6
import regional_mom6.rotation as rot
import regional_mom6.regridding as rgd
import pytest
import xarray as xr
import numpy as np
import pandas as pd
from regional_mom6.regional_mom6 import segment

# Not testing get_arakawa_c_points, coords, & create_regridder
def test_smoke_untested_funcs(get_curvilinear_hgrid, generate_silly_vt_dataset):
    hgrid = get_curvilinear_hgrid
    ds = generate_silly_vt_dataset
    ds["lat"] = ds.silly_lat
    ds["lon"] = ds.silly_lat
    assert rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    assert rgd.coords(hgrid, "north", "segment_002")
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
    coords = rgd.coords(hgrid, "north", "segment_002")
    ds = rgd.add_secondary_dimension(ds, "temp", coords, "segment_002")
    assert ds.time.attrs == {"units": "days"}  # Assert that attributes are retained
    assert ds["temp"].dims == (
        "silly_lat",
        "ny_segment_002",
        "silly_lon",
        "silly_depth",
        "time",
    )

    # E/W Boundary
    coords = rgd.coords(hgrid, "east", "segment_003")
    ds = generate_silly_vt_dataset
    ds = rgd.add_secondary_dimension(ds, "v", coords, "segment_003")
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
        ds, "temp", coords, "segment_003", to_beginning=True
    )
    assert ds["temp"].dims[0] == "nx_segment_003"

    # NZ dim E/W Boundary
    ds = generate_silly_vt_dataset
    ds = ds.rename({"silly_depth": "nz"})
    ds = rgd.add_secondary_dimension(ds, "u", coords, "segment_003")
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


def test_get_boundary_mask(get_curvilinear_hgrid):
    hgrid = get_curvilinear_hgrid
    t_points = rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    bathy = hgrid.isel(nyp=t_points.t_points_y, nxp=t_points.t_points_x)
    bathy["depth"] = (("t_points_y", "t_points_x"), (np.full(bathy.x.shape, 0)))
    north_mask = rgd.get_boundary_mask(
        bathy,
        "north",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )
    south_mask = rgd.get_boundary_mask(
        bathy,
        "south",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )
    east_mask = rgd.get_boundary_mask(
        bathy,
        "east",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )
    west_mask = rgd.get_boundary_mask(
        bathy,
        "west",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )

    # Check corner property of mask, and ensure each direction is following what we expect
    for mask in [north_mask, south_mask, east_mask, west_mask]:
        assert (mask == 0).all()  # Ensure all other points are land
    assert north_mask.shape == (hgrid.x[-1].shape)  # Ensure mask is the right shape
    assert south_mask.shape == (hgrid.x[0].shape)  # Ensure mask is the right shape
    assert east_mask.shape == (hgrid.x[:, -1].shape)  # Ensure mask is the right shape
    assert west_mask.shape == (hgrid.x[:, 0].shape)  # Ensure mask is the right shape

    ## Now we check if the coast masking is correct (remember we make 3 cells into the coast be ocean)
    start_ind = 6
    end_ind = 9
    for i in range(start_ind, end_ind + 1):
        bathy["depth"][-1][i] = 15
    north_mask = rgd.get_boundary_mask(
        bathy,
        "north",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )

    # Ensure coasts are ocean with a 1 cell buffer, for the 1-point (remember mask is on the hgrid boundary - so (6 *2 +2) - 1 -> (9 *2 +2) + 1)
    assert (
        north_mask[(((start_ind * 2) + 1) - 1) : (((end_ind * 2) + 1) + 1 + 1)] == 1
    ).all()
    assert (north_mask[0 : (((start_ind * 2) + 1) - 1)] == 0).all()  # Left Side
    assert (north_mask[(((end_ind * 2) + 1) + 1 + 1) :] == 0).all()  # Right Side

    # On E/W
    start_ind = 6
    end_ind = 9
    for i in range(start_ind, end_ind + 1):
        bathy["depth"][:, 0][i] = 15
    west_mask = rgd.get_boundary_mask(
        bathy,
        "west",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )
    # Ensure coasts are ocean with a 1 cell buffer, for the 1-point (remember mask is on the hgrid boundary - so (6 *2 +2) - 1 -> (9 *2 +2) + 1)
    assert (
        west_mask[(((start_ind * 2) + 1) - 1) : (((end_ind * 2) + 1) + 1 + 1)] == 1
    ).all()
    assert (west_mask[0 : (((start_ind * 2) + 1) - 1)] == 0).all()  # Left Side
    assert (west_mask[(((end_ind * 2) + 1) + 1 + 1) :] == 0).all()  # Right Side


def test_mask_dataset(get_curvilinear_hgrid):
    hgrid = get_curvilinear_hgrid
    t_points = rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    bathy = hgrid.isel(nyp=t_points.t_points_y, nxp=t_points.t_points_x)
    bathy["depth"] = (("t_points_y", "t_points_x"), (np.full(bathy.x.shape, 0)))
    ds = hgrid.copy(deep=True)
    ds = ds.drop_vars(("tile", "area", "y", "x", "angle_dx", "dy", "dx"))
    ds["temp"] = (("t_points_y", "t_points_x"), (np.full(hgrid.x.shape, 100)))
    ds["temp"] = ds["temp"].isel(t_points_y=-1)
    start_ind = 6
    end_ind = 9
    for i in range(start_ind, end_ind + 1):
        bathy["depth"][-1][i] = 15

    ds["temp"][start_ind * 2 + 2] = np.nan
    ds["temp"] = ds["temp"].expand_dims("y", axis=0)
    ds["temp"] = ds["temp"].expand_dims("nz_temp", axis=0)
    ds["temp"] = ds["temp"].expand_dims("time", axis=0)
    ds.temp.attrs = {"units": "C"}
    fill_value = 36
    ds = rgd.mask_dataset(
        ds,
        bathy,
        "north",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
        fill_value=fill_value,
    )
    assert ds.temp.attrs == {"units": "C"}

    assert (
        np.isnan(ds["temp"][0, 0, 0][start_ind * 2 + 2]) == False
    )  # Ensure missing value was filled
    assert (
        np.isnan(
            ds["temp"][0, 0, 0][
                (((start_ind * 2) + 1) - 1) : (((end_ind * 2) + 1) + 1 + 1)
            ]
        )
    ).all() == False  # Ensure data is kept in ocean area
    assert (
        (ds["temp"][0, 0, 0][1 : (((start_ind * 2) + 1) - 1)] == fill_value)
    ).all() == True and (
        (ds["temp"][0, 0, 0][(((end_ind * 2) + 1) + 1 + 1) : -1] == fill_value)
    ).all() == True  # Ensure data is not in land area


def test_regrid_velocity_tracers(get_rectilinear_hgrid, tmp_path):
    """
    Correctness test for segment.regrid_velocity_tracers.

    Checks:
    - Output OBC file is written
    - Variables follow the {var}_{segment_name} naming convention
    - Temperature is in Celsius (< 100)
    - 3D fields have companion dz_* variables
    - Vertical coordinate is re-encoded as incremental integers
    - Perpendicular dimension has size 1
    """

    hgrid = get_rectilinear_hgrid
    seg_name = "segment_001"
    outfolder = tmp_path / "inputdir"
    outfolder.mkdir()

    seg = segment(
        hgrid=hgrid,
        outfolder=outfolder,
        segment_name=seg_name,
        orientation="east",
        startdate="2003-01-01 00:00:00",
    )

    # Minimal synthetic boundary dataset covering the east edge of the hgrid (lon ≈ 10, lat 0-10)
    lat = np.linspace(0, 3, 3)
    lon = np.linspace(0, 3, 3)
    depth = np.linspace(0, 3, 3)
    time = np.arange(3, dtype=float)
    s4 = (len(lat), len(lon), len(depth), len(time))
    s3 = (len(lat), len(lon), len(time))
    c4 = {"lat": lat, "lon": lon, "depth": depth, "time": time}
    c3 = {"lat": lat, "lon": lon, "time": time}
    d4 = ["lat", "lon", "depth", "time"]
    d3 = ["lat", "lon", "time"]
    ds = xr.Dataset(
        {
            "temp": xr.DataArray(np.full(s4, 20.0), dims=d4, coords=c4),
            "salt": xr.DataArray(np.full(s4, 35.0), dims=d4, coords=c4),
            "u": xr.DataArray(np.zeros(s4), dims=d4, coords=c4),
            "v": xr.DataArray(np.zeros(s4), dims=d4, coords=c4),
            "eta": xr.DataArray(np.zeros(s3), dims=d3, coords=c3),
        }
    )
    ds.time.attrs = {"units": "days"}
    infile = tmp_path / "east_raw.nc"
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

    segment_out, _ = seg.regrid_velocity_tracers(infile, varnames, arakawa_grid="A")

    # Output file must exist
    assert (outfolder / f"forcing_obc_{seg_name}.nc").exists()

    # All expected variables must be present with correct naming
    for v in [f"temp_{seg_name}", f"salt_{seg_name}", f"u_{seg_name}", f"v_{seg_name}", f"eta_{seg_name}"]:
        assert v in segment_out, f"Missing variable {v}"

    # Temperature must be in Celsius
    assert float(segment_out[f"temp_{seg_name}"].max()) < 100.0

    # Regridding correctness: source fields are spatially constant, so interpolated
    # values must match the source value regardless of where the boundary falls.
    # NaNs can appear at land points (masked), so we check only ocean points.
    large_fill = 1.0e20
    for var, expected in [(f"temp_{seg_name}", 20.0), (f"salt_{seg_name}", 35.0)]:
        ocean_vals = segment_out[var].values
        ocean_vals = ocean_vals[np.abs(ocean_vals) < large_fill / 2]
        np.testing.assert_allclose(ocean_vals, expected, rtol=1e-4)

    # 3D fields must have dz companion variables
    for v in [f"temp_{seg_name}", f"salt_{seg_name}", f"u_{seg_name}", f"v_{seg_name}"]:
        assert f"dz_{v}" in segment_out, f"Missing dz variable for {v}"

    # Vertical coordinates must be incremental integers starting at 0
    for base in ["temp", "salt", "u", "v"]:
        nz_coord = f"nz_{seg_name}_{base}"
        nz_vals = segment_out[nz_coord].values
        np.testing.assert_array_equal(nz_vals, np.arange(nz_vals.size))

    # Perpendicular dimension (ny for east boundary) must have size 1
    assert segment_out[f"ny_{seg_name}"].size == 1



# def test_regrid_tides(get_rectilinear_hgrid, tmp_path):
#     """
#     Correctness test for segment.regrid_tides.

#     Checks:
#     - Output tidal elevation (tz_*.nc) and velocity (tu_*.nc) files are written
#     - Amplitude values are non-negative
#     - Phase values are in [-π, π]
#     - Expected variable names are present
#     """


#     hgrid = get_rectilinear_hgrid
#     seg_name = "segment_001"
#     outfolder = tmp_path / "inputdir"
#     (outfolder / "forcing").mkdir(parents=True)

#     seg = segment(
#         hgrid=hgrid,
#         outfolder=outfolder,
#         segment_name=seg_name,
#         orientation="east",
#         startdate="2003-01-01 00:00:00",
#     )

#     # Synthetic tpxo-style datasets: 2 constituents over the domain
#     n_const = 2
#     lat_1d = np.linspace(-5, 15, 30)
#     lon_1d = np.linspace(-5, 15, 30)
#     lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
#     shape = (n_const, len(lat_1d), len(lon_1d))
#     dims = ["constituent", "lat", "lon"]
#     coords = {
#         "constituent": np.arange(n_const),
#         "lat": (["lat", "lon"], lat2d),
#         "lon": (["lat", "lon"], lon2d),
#     }

#     def _tpxo(re_name, im_name):
#         return xr.Dataset(
#             {
#                 re_name: xr.DataArray(np.random.random(shape).astype(np.float32), dims=dims, coords=coords),
#                 im_name: xr.DataArray(np.random.random(shape).astype(np.float32), dims=dims, coords=coords),
#             }
#         )

#     times = xr.DataArray(pd.date_range("2003-01-01", periods=1), dims=["time"])
#     seg.regrid_tides(_tpxo("vRe", "vIm"), _tpxo("uRe", "uIm"), _tpxo("hRe", "hIm"), times)

#     tz_file = outfolder / f"tz_{seg_name}.nc"
#     tu_file = outfolder / f"tu_{seg_name}.nc"
#     assert tz_file.exists(), "Tidal elevation file not written"
#     assert tu_file.exists(), "Tidal velocity file not written"

#     tz = xr.open_dataset(tz_file)
#     tu = xr.open_dataset(tu_file)

#     # Amplitude must be non-negative (it's |complex|)
#     assert float(tz[f"zamp_{seg_name}"].min()) >= 0.0
#     assert float(tu[f"uamp_{seg_name}"].min()) >= 0.0
#     assert float(tu[f"vamp_{seg_name}"].min()) >= 0.0

#     # Phase must be in [-π, π]
#     for ds, var in [
#         (tz, f"zphase_{seg_name}"),
#         (tu, f"uphase_{seg_name}"),
#         (tu, f"vphase_{seg_name}"),
#     ]:
#         assert float(ds[var].min()) >= -np.pi - 1e-6
#         assert float(ds[var].max()) <= np.pi + 1e-6

#     # All expected tidal variables must be present
#     for v in [f"zamp_{seg_name}", f"zphase_{seg_name}"]:
#         assert v in tz
#     for v in [f"uamp_{seg_name}", f"uphase_{seg_name}", f"vamp_{seg_name}", f"vphase_{seg_name}"]:
#         assert v in tu

#     tz.close()
#     tu.close()
