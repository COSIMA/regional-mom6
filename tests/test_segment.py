"""Coverage for regional_mom6.segment.Segment: construction (from_hgrid,
cardinal, from_lonlat), validation, masking, MOM6 position-string generation,
spec serialization, and the regridding methods (regrid_velocity_tracers,
regrid_tides) -- all exercised standalone, without ever touching
regional_mom6.experiment.

Includes a multi-segment "sketch domain" scenario covering full edges,
partial edges, and fully interior lines next to irregular "discarded domain"
land patches (headlands), with these properties: extents of neighbouring
segments may overlap, segment construction order carries no meaning, and land
masking comes straight from a mom6_forge.Topo.
"""

import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from regional_mom6 import regridding as rgd
from regional_mom6.segment import Segment

# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


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


def test_segment_mask_matches_direct_supergridmask_slice():
    """Segment.from_hgrid's mask should be exactly topo.supergridmask sliced the
    same way as lon/lat/angle -- no separate resolution-conversion step needed."""
    grid, topo = _synthetic_grid_and_topo(land_ny_indices=[0])
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")

    segment = Segment.cardinal(hgrid, "south", "segment_001", topo=topo)
    expected = topo.supergridmask.isel(nyp=0)
    assert (segment.mask.values == expected.values).all()


def test_segment_from_hgrid_missing_angle_dx_warns(get_rectilinear_hgrid):
    hgrid = get_rectilinear_hgrid.drop_vars("angle_dx")
    with pytest.warns(UserWarning, match="angle_dx"):
        segment = Segment.cardinal(hgrid, "south", "segment_001")
    assert (segment.angle.values == 0).all()


def test_segment_cardinal_invalid_orientation_raises(get_rectilinear_hgrid):
    with pytest.raises(ValueError, match="orientation must be one of"):
        Segment.cardinal(get_rectilinear_hgrid, "northeast", "segment_001")


def test_segment_from_hgrid_invalid_axis_raises(get_rectilinear_hgrid):
    with pytest.raises(ValueError, match="axis must be one of"):
        Segment.from_hgrid(
            get_rectilinear_hgrid,
            axis="nx",
            index=0,
            segment_name="segment_001",
            ocean_side="north",
        )


def test_segment_from_hgrid_arbitrary_axis_index_range(get_rectilinear_hgrid):
    hgrid = get_rectilinear_hgrid
    # Odd-length (T-center-aligned) slice -- BRUSHCUTTER_MODE requires this;
    # from_hgrid rejects an even-length (corner-aligned) one.
    index_range = slice(2, 7)
    segment = Segment.from_hgrid(
        hgrid,
        axis="nyp",
        index=3,
        segment_name="segment_001",
        index_range=index_range,
        ocean_side="north",
    )
    expected_lon = hgrid["x"].isel(nyp=3).isel(nxp=index_range)
    expected_lat = hgrid["y"].isel(nyp=3).isel(nxp=index_range)
    assert np.allclose(segment.lon.values, expected_lon.values)
    assert np.allclose(segment.lat.values, expected_lat.values)
    assert segment.lon.sizes["nx_segment_001"] == index_range.stop - index_range.start


def test_ocean_side_rejects_direction_that_does_not_match_axis(get_rectilinear_hgrid):
    hgrid = get_rectilinear_hgrid

    with pytest.raises(ValueError, match="invalid for axis"):
        Segment.from_hgrid(
            hgrid,
            axis="nyp",
            index=11,
            segment_name="bad_side",
            ocean_side="east",
        )
    with pytest.raises(ValueError, match="invalid for axis"):
        Segment.from_hgrid(
            hgrid,
            axis="nxp",
            index=11,
            segment_name="bad_side",
            ocean_side="north",
        )


# ---------------------------------------------------------------------------
# from_lonlat -- building a segment from physical coordinates
# ---------------------------------------------------------------------------


def test_from_lonlat_nyp_resolves_nearest_row_and_lon_range(get_rectilinear_hgrid):
    """axis="nyp": the segment sits at the nearest T-row to fixed_lat, spanning
    the T-columns nearest lon_range -- exact on this uniform rectilinear grid
    (resolution=0.1, lon in [278, 282], lat in [7, 10])."""
    hgrid = get_rectilinear_hgrid
    segment = Segment.from_lonlat(
        hgrid,
        axis="nyp",
        fixed_lat=8.5,
        lon_range=(279.0, 280.0),
        segment_name="segment_001",
        ocean_side="north",
    )
    assert np.allclose(segment.lat.values, 8.5, atol=0.05)
    assert segment.lon.values.min() >= 278.95
    assert segment.lon.values.max() <= 280.05

    # Matches an equivalent from_hgrid call at the resolved supergrid index.
    equivalent = Segment.from_hgrid(
        hgrid,
        axis="nyp",
        index=segment._index,
        index_range=segment._index_range,
        segment_name="segment_001b",
        ocean_side="north",
    )
    assert np.array_equal(segment.lon.values, equivalent.lon.values)
    assert np.array_equal(segment.lat.values, equivalent.lat.values)


def test_from_lonlat_nxp_resolves_nearest_col_and_lat_range(get_rectilinear_hgrid):
    """axis="nxp": the mirror-image case -- fixed_lon resolves a T-column,
    lat_range resolves the spanned T-rows."""
    hgrid = get_rectilinear_hgrid
    segment = Segment.from_lonlat(
        hgrid,
        axis="nxp",
        fixed_lon=279.5,
        lat_range=(8.0, 9.0),
        segment_name="segment_002",
        ocean_side="west",
    )
    assert np.allclose(segment.lon.values, 279.5, atol=0.05)
    assert segment.lat.values.min() >= 7.95
    assert segment.lat.values.max() <= 9.05


def test_from_lonlat_missing_required_args_raises(get_rectilinear_hgrid):
    hgrid = get_rectilinear_hgrid
    with pytest.raises(ValueError, match="requires fixed_lat"):
        Segment.from_lonlat(hgrid, axis="nyp", segment_name="x", ocean_side="north")
    with pytest.raises(ValueError, match="requires fixed_lon"):
        Segment.from_lonlat(hgrid, axis="nxp", segment_name="x", ocean_side="west")
    with pytest.raises(ValueError, match="axis must be one of"):
        Segment.from_lonlat(hgrid, axis="bogus", segment_name="x", ocean_side="north")


# ---------------------------------------------------------------------------
# detect_open_cardinal_boundaries
# ---------------------------------------------------------------------------


def _flat_grid_and_topo(name):
    grid = Grid(
        resolution=1,
        xstart=0,
        lenx=10,
        ystart=0,
        leny=10,
        name=name,
        type="rectilinear_cartesian",
    )
    topo = Topo(grid, min_depth=5.0, git=False)
    topo.set_flat(100.0)
    return grid, topo


def test_all_edges_open_on_a_fully_wet_domain():
    _, topo = _flat_grid_and_topo("all_open")
    assert set(Segment.detect_open_cardinal_boundaries(topo)) == {
        "north",
        "south",
        "east",
        "west",
    }


def test_fully_land_edge_is_dropped():
    _, topo = _flat_grid_and_topo("south_closed")
    depth = topo.depth.values.copy()
    depth[0, :] = 0.0  # entire southern T-row is land
    topo.depth = depth

    open_boundaries = Segment.detect_open_cardinal_boundaries(topo)
    assert "south" not in open_boundaries
    assert set(open_boundaries) == {"north", "east", "west"}


def test_two_fully_land_edges_are_both_dropped():
    _, topo = _flat_grid_and_topo("south_and_west_closed")
    depth = topo.depth.values.copy()
    depth[0, :] = 0.0  # south
    depth[:, 0] = 0.0  # west
    topo.depth = depth

    assert set(Segment.detect_open_cardinal_boundaries(topo)) == {"north", "east"}


def test_partially_wet_edge_is_kept():
    _, topo = _flat_grid_and_topo("south_mostly_land")
    depth = topo.depth.values.copy()
    depth[0, :-1] = 0.0  # all but one T-cell of the southern row is land
    topo.depth = depth

    assert "south" in Segment.detect_open_cardinal_boundaries(topo)


# ---------------------------------------------------------------------------
# mom6_obc_position_string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orientation,expected",
    [
        ("south", "J=0,I=0:N"),
        ("north", "J=N,I=N:0"),
        ("east", "I=N,J=0:N"),
        ("west", "I=0,J=N:0"),
    ],
)
def test_mom6_obc_position_string_matches_legacy_cardinal_convention(
    get_rectilinear_hgrid, orientation, expected
):
    """Segment.cardinal's default MOM6 index string must match the values the
    old hardcoded rect_MOM6_index_dir produced for the 4 cardinal boundaries,
    including the ascending/descending convention and the 'N' sentinel."""
    hgrid = get_rectilinear_hgrid
    segment = Segment.cardinal(hgrid, orientation, "segment_001")
    assert segment.mom6_obc_position_string() == expected


def test_mom6_obc_position_string_arbitrary_segment(get_rectilinear_hgrid):
    """An interior/partial segment (not a full outer edge) should get numeric
    MOM6 model-grid indices -- half the supergrid index, per the supergrid's
    2x resolution relative to MOM6's own model grid -- instead of the 'N'
    sentinel, and reverse=True should flip the parallel index direction."""
    hgrid = get_rectilinear_hgrid
    # Odd-length (T-center-aligned) slice -- BRUSHCUTTER_MODE requires this.
    segment = Segment.from_hgrid(
        hgrid,
        axis="nyp",
        index=5,
        segment_name="segment_002",
        index_range=slice(2, 9),
        ocean_side="north",
    )
    assert segment.mom6_obc_position_string() == "J=2,I=1:4"
    assert segment.mom6_obc_position_string(reverse=True) == "J=2,I=4:1"


def test_mom6_obc_position_string_requires_grid_index():
    """A hand-built Segment (no from_hgrid/cardinal) has no grid to derive
    MOM6 indices from, so this should fail loudly rather than guess."""
    segment = Segment(
        lon=xr.DataArray([1.0, 2.0], dims=["nx_segment_099"]),
        lat=xr.DataArray([1.0, 1.0], dims=["nx_segment_099"]),
        angle=xr.DataArray([0.0, 0.0], dims=["nx_segment_099"]),
        segment_name="segment_099",
        parallel="nx",
        perpendicular="ny",
        axis_to_expand=2,
    )
    with pytest.raises(ValueError, match="grid-index bookkeeping"):
        segment.mom6_obc_position_string()


# ---------------------------------------------------------------------------
# to_spec / from_spec
# ---------------------------------------------------------------------------


def test_to_spec_from_spec_round_trip_cardinal(get_rectilinear_hgrid):
    hgrid = get_rectilinear_hgrid
    original = Segment.cardinal(hgrid, "east", "segment_001")
    spec = original.to_spec()
    rebuilt = Segment.from_spec(hgrid, spec, "segment_001")

    assert np.array_equal(original.lon.values, rebuilt.lon.values)
    assert np.array_equal(original.lat.values, rebuilt.lat.values)
    assert original.mom6_obc_position_string() == rebuilt.mom6_obc_position_string()


def test_to_spec_from_spec_round_trip_interior():
    """The ocean_side stored in the spec must be enough on its own to
    reproduce an interior segment's masking/position-string bookkeeping --
    exercised here specifically because ocean_side="south" is the
    own-position-masking case (see _compute_grid_index)."""
    grid, topo = _sketch_grid_and_topo()
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")
    original = Segment.from_hgrid(
        hgrid,
        segment_name="segment_002",
        topo=topo,
        **_SKETCH_SEGMENT_SPECS["segment_002"],
    )

    spec = original.to_spec()
    assert spec["ocean_side"] == "south"

    rebuilt = Segment.from_spec(hgrid, spec, "segment_002", topo=topo)
    assert np.array_equal(original.lon.values, rebuilt.lon.values)
    assert np.array_equal(original.mask.values, rebuilt.mask.values)
    assert original.mom6_obc_position_string() == rebuilt.mom6_obc_position_string()


def test_to_spec_requires_grid_index():
    """Mirrors mom6_obc_position_string's own bookkeeping requirement -- a
    hand-built Segment has nothing to serialize."""
    segment = Segment(
        lon=xr.DataArray([1.0, 2.0], dims=["nx_segment_099"]),
        lat=xr.DataArray([1.0, 1.0], dims=["nx_segment_099"]),
        angle=xr.DataArray([0.0, 0.0], dims=["nx_segment_099"]),
        segment_name="segment_099",
        parallel="nx",
        perpendicular="ny",
        axis_to_expand=2,
    )
    with pytest.raises(ValueError, match="axis/index bookkeeping"):
        segment.to_spec()


# ---------------------------------------------------------------------------
# Multi-segment "sketch domain" scenario: full edges, partial edges, and
# fully interior lines next to irregular headlands, built in arbitrary order.
# ---------------------------------------------------------------------------


def _sketch_grid_and_topo():
    """A 20x20 T-cell rectilinear domain with two irregular headlands carved
    out of it (each with a one-cell notch, so neither is a plain rectangle):
      - Headland A: rows (t-index) 15-19, cols 2-13, notch at (15, 10).
      - Headland B: rows 0-4, cols 15-19, notch at (2, 17).

    Headland A's west edge (col 2) reaches further west than the "interior
    line just south of headland A" (segment_002) actually needs for its own
    data -- that extra width exists purely so segment_002's west endpoint
    (T-column 3) is land-capped: _check_land_capped_endpoints requires land
    one column past that endpoint (col 2) on the row MOM6 force-masks (16,
    one row north of segment_002's own row 15, which sits at the T-center
    supergrid index 31 = 2 * 15 + 1).
    """
    grid = Grid(
        resolution=1,
        xstart=0,
        lenx=20,
        ystart=0,
        leny=20,
        name="sketch_domain",
        type="rectilinear_cartesian",
    )
    topo = Topo(grid, min_depth=5.0, git=False)
    topo.set_flat(100.0)

    depth = topo.depth.values.copy()
    depth[15:20, 2:14] = 0.0
    depth[15, 10] = 100.0  # notch back to ocean
    depth[0:5, 15:20] = 0.0
    depth[2, 17] = 100.0  # notch back to ocean
    topo.depth = depth
    return grid, topo


# (segment_name, kwargs for Segment.from_hgrid, or None to use Segment.cardinal)
# Every index_range below is odd-length (T-center-aligned), per
# BRUSHCUTTER_MODE -- from_hgrid rejects an even-length one outright.
_SKETCH_SEGMENT_SPECS = {
    "segment_001": dict(
        axis="nyp", index=-1, index_range=slice(0, 15), ocean_side="north"
    ),  # north edge, west of headland A
    "segment_002": dict(
        axis="nyp", index=31, index_range=slice(7, 40), ocean_side="south"
    ),  # interior line just south of headland A; overlaps segment_001
    "segment_003": dict(
        axis="nxp", index=-1, index_range=slice(10, 41), ocean_side="west"
    ),  # east edge, above headland B
    "segment_004": dict(
        axis="nxp", index=31, index_range=slice(0, 7), ocean_side="west"
    ),  # interior line, west edge of headland B -- T-rows 0-3, land-capped
    # by headland B itself at (row 4, col 16), one row past its own span
    "segment_005": dict(
        axis="nxp", index=-1, index_range=slice(0, 11), ocean_side="west"
    ),  # east edge, alongside headland B
}
_SKETCH_CARDINAL_SEGMENTS = {
    "segment_006": "south",
    "segment_007": "west",
}


def _build_all_segments(hgrid, topo, order):
    segments = {}
    for name in order:
        if name in _SKETCH_CARDINAL_SEGMENTS:
            segments[name] = Segment.cardinal(
                hgrid, _SKETCH_CARDINAL_SEGMENTS[name], name, topo=topo
            )
        else:
            segments[name] = Segment.from_hgrid(
                hgrid, segment_name=name, topo=topo, **_SKETCH_SEGMENT_SPECS[name]
            )
    return segments


def test_sketch_domain_seven_segments_built_standalone():
    """All 7 segments from the sketch build cleanly via Segment alone -- no
    regional_mom6.experiment involved -- covering full edges (006, 007),
    partial edges (001, 003, 005), and fully interior lines (002, 004)."""
    grid, topo = _sketch_grid_and_topo()
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")

    all_names = list(_SKETCH_SEGMENT_SPECS) + list(_SKETCH_CARDINAL_SEGMENTS)
    assert len(all_names) == 7

    segments = _build_all_segments(hgrid, topo, all_names)
    assert set(segments) == set(all_names)


def test_sketch_domain_segment_masks_match_supergridmask_slices():
    """Every segment's mask -- full-edge, partial-edge, or interior -- is
    exactly topo.supergridmask sliced the same way, headland notches included."""
    grid, topo = _sketch_grid_and_topo()
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")
    segments = _build_all_segments(
        hgrid, topo, list(_SKETCH_SEGMENT_SPECS) + list(_SKETCH_CARDINAL_SEGMENTS)
    )

    for name, seg in segments.items():
        if name in _SKETCH_CARDINAL_SEGMENTS:
            spec = None
            axis = (
                "nyp"
                if _SKETCH_CARDINAL_SEGMENTS[name] in ("south", "north")
                else "nxp"
            )
            index = {"south": 0, "north": -1, "west": 0, "east": -1}[
                _SKETCH_CARDINAL_SEGMENTS[name]
            ]
            index_range = None
        else:
            spec = _SKETCH_SEGMENT_SPECS[name]
            axis, index, index_range = spec["axis"], spec["index"], spec["index_range"]

        expected = topo.supergridmask.isel({axis: index})
        if index_range is not None:
            parallel_axis = "nxp" if axis == "nyp" else "nyp"
            expected = expected.isel({parallel_axis: index_range})
        assert (seg.mask.values == expected.values).all(), f"{name} mask mismatch"
        # Headland masking actually did something -- not every point is ocean.
        if name in ("segment_002", "segment_004"):
            assert (seg.mask.values == 0).any(), f"{name} should touch masked land"


def test_sketch_domain_segment_extents_may_overlap():
    """segment_001 (north edge, west portion) and segment_002 (interior line
    just south of headland A) deliberately share part of their column range --
    building both must not raise or otherwise reject the overlap."""
    grid, topo = _sketch_grid_and_topo()
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")
    segments = _build_all_segments(hgrid, topo, ["segment_001", "segment_002"])

    lon_1 = set(np.round(segments["segment_001"].lon.values, 6))
    lon_2 = set(np.round(segments["segment_002"].lon.values, 6))
    assert lon_1 & lon_2, "segment_001 and segment_002 should overlap in longitude"


def test_sketch_domain_segment_construction_order_is_irrelevant():
    """The sketch notes there's no required order to segment orientation/
    construction -- building the same 7 segments in two unrelated orders must
    give bit-identical results per segment name."""
    grid, topo = _sketch_grid_and_topo()
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")
    all_names = list(_SKETCH_SEGMENT_SPECS) + list(_SKETCH_CARDINAL_SEGMENTS)

    order_a = all_names  # 001..007 in sketch order
    order_b = [
        "segment_004",
        "segment_007",
        "segment_002",
        "segment_006",
        "segment_001",
        "segment_005",
        "segment_003",
    ]
    assert set(order_a) == set(order_b)

    segments_a = _build_all_segments(hgrid, topo, order_a)
    segments_b = _build_all_segments(hgrid, topo, order_b)

    for name in all_names:
        a, b = segments_a[name], segments_b[name]
        assert np.array_equal(a.lon.values, b.lon.values)
        assert np.array_equal(a.lat.values, b.lat.values)
        assert np.array_equal(a.mask.values, b.mask.values)
        assert a.mom6_obc_position_string() == b.mom6_obc_position_string()


def test_sketch_domain_position_strings_full_edge_vs_interior():
    """Full-edge segments emit MOM6's 'N' sentinel (matching the legacy
    cardinal convention); interior/partial segments emit numeric indices."""
    grid, topo = _sketch_grid_and_topo()
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")
    segments = _build_all_segments(
        hgrid, topo, list(_SKETCH_SEGMENT_SPECS) + list(_SKETCH_CARDINAL_SEGMENTS)
    )

    # Full outer edges: same convention as the 4 legacy cardinal boundaries.
    assert segments["segment_006"].mom6_obc_position_string() == "J=0,I=0:N"
    assert segments["segment_007"].mom6_obc_position_string() == "I=0,J=N:0"

    # Interior line (axis="nyp", index=31 -- T-center for T-row 15,
    # index_range=slice(7, 40), ocean_side="south"): fixed J = 31 // 2 = 15,
    # but ocean_side="south" is exactly the case MOM6's
    # open_boundary_impose_land_mask force-masks to land at J *itself*
    # rather than a neighbor (confirmed against a real MOM6 run) -- Segment
    # compensates by reporting J=16 so MOM6 masks J=16 (the true north
    # neighbor) instead of the segment's own row, J=15.
    # Parallel I = 7//2=3 .. (40-1)//2=19, reversed (ocean_side="south") -> 19:3.
    assert segments["segment_002"].mom6_obc_position_string() == "J=16,I=19:3"
    # Interior line adjacent to headland B (axis="nxp", index=31 -- T-center
    # for T-column 15, index_range=slice(0, 7)). fixed I = 31 // 2 = 15, but
    # ocean_side="west" is the mirror-image broken case (masks I itself) --
    # compensated the same way, reporting I=16 instead of the own value, 15.
    # Parallel J = 0//2=0 .. (7-1)//2=3, not reversed (ocean_side="west") -> 0:3.
    assert segments["segment_004"].mom6_obc_position_string() == "I=16,J=0:3"

    # Partial edges: the *fixed* coordinate is still "N" (it's a real edge row/
    # column), but the *parallel* (index_range-restricted) coordinate is only
    # "N" on the side that happens to reach the domain boundary -- never on
    # both sides at once the way a full cardinal edge would.
    assert segments["segment_001"].mom6_obc_position_string() == "J=N,I=0:7"
    assert segments["segment_003"].mom6_obc_position_string() == "I=N,J=5:N"
    assert segments["segment_005"].mom6_obc_position_string() == "I=N,J=0:5"
    for name in ("segment_001", "segment_003", "segment_005", "segment_004"):
        parallel = segments[name].mom6_obc_position_string().split(",")[1]
        assert (
            parallel != "I=0:N" and parallel != "J=0:N"
        ), f"{name} shouldn't span the *whole* parallel edge like a cardinal segment: {parallel}"


def test_interior_partial_segment_regrid_velocity_tracers(toy_glorys_ds, tmp_path):
    """Beyond geometry: an interior, partially-masked segment (not a cardinal
    edge) round-trips through the actual regridding path, same as a cardinal
    segment would."""
    grid = Grid(
        resolution=1,
        xstart=2,
        lenx=4,
        ystart=2,
        leny=4,
        name="interior_regrid_test",
        type="rectilinear_cartesian",
    )
    topo = Topo(grid, min_depth=5.0, git=False)
    topo.set_flat(100.0)
    depth = topo.depth.values.copy()
    depth[0, 2] = 0.0  # mask one interior T-cell along the segment's line to land
    depth[1, 0] = 0.0  # land-cap the segment's west endpoint (row north of it)
    topo.depth = depth
    hgrid = grid._supergrid.to_ds(name=grid.name, author="pytest")

    seg_name = "segment_010"
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

    # axis="nyp", index=1 (T-center for T-row 0, the row closest to
    # toy_glorys_ds's lat coverage so the regrid below doesn't extrapolate)
    # is an interior line (neither 0 nor the last index), index_range
    # restricts it to part of the domain's width -- exactly the "not a
    # cardinal edge" case the sketch is about.
    segment = Segment.from_hgrid(
        hgrid,
        axis="nyp",
        index=1,
        segment_name=seg_name,
        index_range=slice(2, 7),
        topo=topo,
        ocean_side="south",
    )
    assert (segment.mask.values == 0).any(), "segment should cross the masked cell"

    segment_out, _ = segment.regrid_velocity_tracers(
        infile, varnames, outfolder, "2003-01-01 00:00:00", arakawa_grid="A"
    )

    salt_vals = segment_out[f"salt_{seg_name}"].values
    ocean_points = segment.mask.values.astype(bool)
    # Salt is spatially constant at 35 in toy_glorys_ds -- ocean points should
    # match; masked land points are filled, not physically meaningful.
    np.testing.assert_allclose(salt_vals[..., ocean_points], 35.0, rtol=1e-4)
    assert np.isfinite(salt_vals).all()


# ---------------------------------------------------------------------------
# regrid_velocity_tracers / regrid_tides
# ---------------------------------------------------------------------------


def test_regrid_velocity_tracers(toy_glorys_ds, tmp_path):
    """
    Correctness test for Segment.regrid_velocity_tracers.

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

    # Minimal synthetic segment dataset covering the east edge of the hgrid (lon ≈ 10, lat 0-10)

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

    segment = Segment.cardinal(hgrid, "east", seg_name)
    segment_out, _ = segment.regrid_velocity_tracers(
        infile, varnames, outfolder, "2003-01-01 00:00:00", arakawa_grid="A"
    )

    # Salt is spatially constant, so all ocean points must match exactly.
    salt_vals = segment_out[f"salt_{seg_name}"].values
    np.testing.assert_allclose(salt_vals, 35.0, rtol=1e-4)

    # Temp is spatially varying (20–26 °C), so just check values are exact (hgrid overlap with seg exactly)
    temp_vals = segment_out[f"temp_{seg_name}"].values
    assert temp_vals[0, 0, 0, 0] == 22
    assert temp_vals[0, 0, 2, 0] == 26

    segment_north = Segment.cardinal(hgrid, "north", seg_name)
    segment_out_north, _ = segment_north.regrid_velocity_tracers(
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
    segment_regrid = Segment.cardinal(hgrid, "west", seg_name)
    seg_regridded, _ = segment_regrid.regrid_velocity_tracers(
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


def test_segment_standalone_no_hgrid(toy_glorys_ds, tmp_path):
    """Prove regrid_velocity_tracers needs nothing beyond a hand-built Segment --
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
    # A straight segment line of 2 points, fully described by literal numbers --
    # lon fixed at 3.0 (inside toy_glorys_ds's [2, 4] lon range), lat spanning it.
    segment = Segment(
        lon=xr.DataArray([3.0, 3.0], dims=[f"ny_{seg_name}"]),
        lat=xr.DataArray([2.5, 3.5], dims=[f"ny_{seg_name}"]),
        angle=xr.DataArray([0.0, 0.0], dims=[f"ny_{seg_name}"]),
        segment_name=seg_name,
        parallel="ny",
        perpendicular="nx",
        axis_to_expand=3,
        mask=None,
    )

    segment_out, _ = segment.regrid_velocity_tracers(
        infile, varnames, outfolder, "2003-01-01 00:00:00", arakawa_grid="A"
    )

    assert (outfolder / f"forcing_obc_{seg_name}.nc").exists()
    assert np.isfinite(segment_out[f"temp_{seg_name}"].values).all()
    np.testing.assert_allclose(segment_out[f"salt_{seg_name}"].values, 35.0, rtol=1e-4)


def test_segment_regridders_manual_reuse(toy_glorys_ds, tmp_path, monkeypatch):
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

    segment = Segment.cardinal(hgrid, "east", "segment_001")

    call_count = {"n": 0}
    real_create_vt_regridders = rgd.create_vt_regridders

    def counting_create_vt_regridders(*args, **kwargs):
        call_count["n"] += 1
        return real_create_vt_regridders(*args, **kwargs)

    monkeypatch.setattr(rgd, "create_vt_regridders", counting_create_vt_regridders)

    _, _ = segment.regrid_velocity_tracers(
        infile, varnames, outfolder, "2003-01-01 00:00:00", arakawa_grid="A"
    )
    assert call_count["n"] == 1
    cached_regridders = segment._regridders

    _, _ = segment.regrid_velocity_tracers(
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
    lon/lat/*Re/*Im form Segment.regrid_tides expects -- skips the raw
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
    """Segment.regrid_tides should run against a plain Segment (no Experiment
    involved) and write tz_*/tu_* files, mirroring the existing standalone
    coverage of regrid_velocity_tracers."""
    hgrid = get_rectilinear_hgrid
    seg_name = "segment_001"
    segment = Segment.cardinal(hgrid, "east", seg_name)

    tpxo_v, tpxo_u, tpxo_h = _synthetic_tidal_datasets()
    times = xr.DataArray(pd.date_range("2000-01-01", periods=1), dims=["time"])

    outfolder = tmp_path / "inputdir"
    outfolder.mkdir()

    segment.regrid_tides(
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
    regridders=None -- passing back segment._tidal_regridders from a prior
    call skips rebuilding, matching the documented manual-reuse pattern for
    regrid_velocity_tracers."""
    hgrid = get_rectilinear_hgrid
    segment = Segment.cardinal(hgrid, "east", "segment_001")

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

    segment.regrid_tides(
        tpxo_v, tpxo_u, tpxo_h, times, outfolder, "2000-01-01 00:00:00"
    )
    assert call_count["n"] == 3  # elev, u, v
    cached_regridders = segment._tidal_regridders

    segment.regrid_tides(
        tpxo_v,
        tpxo_u,
        tpxo_h,
        times,
        outfolder,
        "2000-01-01 00:00:00",
        regridders=cached_regridders,
    )
    assert call_count["n"] == 3  # not rebuilt
