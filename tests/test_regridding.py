import regional_mom6 as rmom6
import regional_mom6.regridding as rgd
import pytest
import warnings
import xarray as xr
import numpy as np
from regional_mom6.segment import Segment
from mom6_forge.grid import *


# Not testing get_arakawa_c_points, & create_regridder
def test_smoke_untested_funcs(get_curvilinear_hgrid, generate_silly_vt_dataset):
    hgrid = get_curvilinear_hgrid
    ds = generate_silly_vt_dataset
    ds["lat"] = ds.silly_lat
    ds["lon"] = ds.silly_lat
    assert rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    assert Segment.cardinal(hgrid, "north", "segment_002")
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

    # N/S Segment
    segment = Segment.cardinal(hgrid, "north", "segment_002")
    ds = rgd.add_secondary_dimension(ds, "temp", segment, "segment_002")
    assert ds.time.attrs == {"units": "days"}  # Assert that attributes are retained
    assert ds["temp"].dims == (
        "silly_lat",
        "ny_segment_002",
        "silly_lon",
        "silly_depth",
        "time",
    )

    # E/W Segment
    segment = Segment.cardinal(hgrid, "east", "segment_003")
    ds = generate_silly_vt_dataset
    ds = rgd.add_secondary_dimension(ds, "v", segment, "segment_003")
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
        ds, "temp", segment, "segment_003", to_beginning=True
    )
    assert ds["temp"].dims[0] == "nx_segment_003"

    # NZ dim E/W Segment
    ds = generate_silly_vt_dataset
    ds = ds.rename({"silly_depth": "nz"})
    ds = rgd.add_secondary_dimension(ds, "u", segment, "segment_003")
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


def test_mask_dataset_no_mask_fills_zero():
    """With no mask (topo=None equivalent), mask_dataset just fills NaNs with 0."""
    segment = Segment(
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
    ds = rgd.mask_dataset(ds, segment)
    assert (ds["temp"].values == np.array([[1.0, 0.0, 3.0]])).all()


def test_mask_dataset_no_dilation():
    """Ocean/land transitions should NOT be dilated by one point -- unlike the old
    get_boundary_mask, segment.mask (from Topo.supergridmask) is treated as ground
    truth with no post-processing."""
    segment = Segment(
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
    ds = rgd.mask_dataset(ds, segment, fill_value=fill_value)
    # Land points (index 0 and 3) are masked out; ocean points keep their values --
    # in particular, index 2 (ocean, adjacent to land at index 3) is NOT dilated
    # into being masked, and index 0/3 are NOT left unmasked just because a
    # neighbor is ocean.
    assert ds["temp"].values[0, 0] == fill_value
    assert ds["temp"].values[0, 3] == fill_value
    assert ds["temp"].values[0, 1] == 20.0
    assert ds["temp"].values[0, 2] == 30.0
    assert ds["temp"].values[0, 4] == 50.0
