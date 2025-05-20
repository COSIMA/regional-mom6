import regional_mom6 as rmom6
import regional_mom6.rotation as rot
import regional_mom6.regridding as rgd
import pytest
import xarray as xr
import numpy as np


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

    ds = rgd.fill_missing_data(ds, "silly_depth", fill="f")

    assert (
        ds["temp"][0, 0, 6:10, 0] == (ds["temp"][0, 0, 5, 0])
    ).all()  # Assert if we are forward filling in time

    ds_2 = generate_silly_vt_dataset
    ds_2["temp"][0, 0, 6:10, 0] = ds["temp"][0, 0, 5, 0]
    assert (ds["temp"] == (ds_2["temp"])).all()  # Assert everything else is the same


def test_add_or_update_time_dim(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    ds = rgd.add_or_update_time_dim(ds, xr.DataArray([0]))

    assert ds["time"].values == [0]  # Assert time is added
    assert ds["temp"].dims[0] == "time"  # Check time is first dim


def test_generate_dz(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    dz = rgd.generate_dz(ds, "silly_depth")
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
    assert "nz_segment_002_temp" in ds["temp"].dims
    assert "nz_segment_002_temp" in ds
    assert (
        ds["nz_segment_002_temp"] == np.arange(ds[f"nz_segment_002_temp"].size)
    ).all()


def test_generate_layer_thickness(generate_silly_vt_dataset):
    ds = generate_silly_vt_dataset
    ds["temp"] = ds["temp"].transpose("time", "silly_depth", "silly_lat", "silly_lon")
    ds = rgd.generate_layer_thickness(ds, "temp", "segment_002", "silly_depth")
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
        hgrid,
        bathy,
        "north",
        "segment_002",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )
    south_mask = rgd.get_boundary_mask(
        hgrid,
        bathy,
        "south",
        "segment_001",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )
    east_mask = rgd.get_boundary_mask(
        hgrid,
        bathy,
        "east",
        "segment_003",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )
    west_mask = rgd.get_boundary_mask(
        hgrid,
        bathy,
        "west",
        "segment_004",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )

    # Check corner property of mask, and ensure each direction is following what we expect
    for mask in [north_mask, south_mask, east_mask, west_mask]:
        assert (mask==0).all()  # Ensure all other points are land
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
        hgrid,
        bathy,
        "north",
        "segment_002",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )

    # Ensure coasts are ocean with a 1 cell buffer, for the 1-point (remember mask is on the hgrid boundary - so (6 *2 +2) - 1 -> (9 *2 +2) + 1)
    assert (
        north_mask[(((start_ind * 2) + 1)-1) : (((end_ind * 2) + 1) + 1+ 1)] == 1
    ).all()  
    assert (
        north_mask[0: (((start_ind * 2) + 1) - 1) ] == 0
    ).all()  # Left Side
    assert (
        north_mask[ (((end_ind * 2) + 1) + 1 + 1):] == 0
    ).all()  # Right Side

    # On E/W
    start_ind = 6
    end_ind = 9
    for i in range(start_ind, end_ind + 1):
        bathy["depth"][:, 0][i] = 15
    west_mask = rgd.get_boundary_mask(
        hgrid,
        bathy,
        "west",
        "segment_004",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
    )
    # Ensure coasts are ocean with a 1 cell buffer, for the 1-point (remember mask is on the hgrid boundary - so (6 *2 +2) - 1 -> (9 *2 +2) + 1)
    assert (
        west_mask[(((start_ind * 2) + 1)-1) : (((end_ind * 2) + 1) + 1+ 1)] == 1
    ).all()  
    assert (
        west_mask[0: (((start_ind * 2) + 1) - 1) ] == 0
    ).all()  # Left Side
    assert (
        west_mask[ (((end_ind * 2) + 1) + 1 + 1):] == 0
    ).all()  # Right Side


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

    ds["temp"][
        start_ind * 2 + 2
    ] = np.nan
    
    ds["temp"] = ds["temp"].expand_dims("nz_temp", axis=0)
    fill_value = 36
    ds = rgd.mask_dataset(
        ds,
        hgrid,
        bathy,
        "north",
        "segment_002",
        y_dim_name="t_points_y",
        x_dim_name="t_points_x",
        fill_value = fill_value
    )

    assert (
        np.isnan(ds["temp"][0][start_ind * 2 + 2]) == False
    )  # Ensure missing value was filled
    assert (
        np.isnan(
            ds["temp"][0][(((start_ind * 2) + 1) - 1) : (((end_ind * 2) + 1) + 1 + 1)]
        )
    ).all() == False  # Ensure data is kept in ocean area
    assert (
        (ds["temp"][0][1 : (((start_ind * 2) + 1) - 1)] == fill_value)
    ).all() == True and (
        (ds["temp"][0][(((end_ind * 2) + 1) + 1 + 1) : -1] == fill_value)
    ).all() == True  # Ensure data is not in land area
