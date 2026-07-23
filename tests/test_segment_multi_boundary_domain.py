"""Coverage for Segment.from_hgrid/cardinal on a domain with many segments --
full edges, partial edges, and fully interior lines -- next to irregular
"discarded domain" land patches (headlands), matching a hand-sketched design
with these properties: extents of neighbouring segments may overlap, segment
construction order carries no meaning, and land masking comes straight from a
mom6_forge.Topo -- all exercised without ever touching regional_mom6.experiment.
"""

import numpy as np
import pandas as pd
import xarray as xr

from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from regional_mom6.segment import Segment


def _sketch_grid_and_topo():
    """A 20x20 T-cell rectilinear domain with two irregular headlands carved
    out of it (each with a one-cell notch, so neither is a plain rectangle):
      - Headland A: rows (t-index) 15-19, cols 7-13, notch at (15, 10).
      - Headland B: rows 0-4, cols 15-19, notch at (2, 17).
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
    depth[15:20, 7:14] = 0.0
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
        axis="nyp", index=-1, index_range=slice(0, 15)
    ),  # north edge, west of headland A
    "segment_002": dict(
        axis="nyp", index=30, index_range=slice(7, 40), mom6_index_reverse=True
    ),  # interior line just south of headland A; overlaps segment_001
    "segment_003": dict(
        axis="nxp", index=-1, index_range=slice(10, 41)
    ),  # east edge, above headland B
    "segment_004": dict(
        axis="nxp", index=30, index_range=slice(0, 11)
    ),  # interior line, west edge of headland B
    "segment_005": dict(
        axis="nxp", index=-1, index_range=slice(0, 11)
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

    # Interior line (axis="nyp", index=30, index_range=slice(7, 40), reverse=True):
    # fixed J = 30 // 2 = 15, but reverse=True on axis="nyp" is the NORTH
    # direction, which MOM6's open_boundary_impose_land_mask force-masks to
    # land at J *itself* rather than a neighbor (confirmed against a real
    # MOM6 run) -- Segment compensates by reporting J=16 so MOM6 masks J=16
    # (the true north neighbor) instead of the segment's own row, J=15.
    # Parallel I = 7//2=3 .. (40-1)//2=19, reversed -> 19:3.
    assert segments["segment_002"].mom6_obc_position_string() == "J=16,I=19:3"
    # Interior line adjacent to headland B (axis="nxp", index=30, index_range=slice(0,11)).
    # fixed I = 30 // 2 = 15, but reverse=False on axis="nxp" is the EAST
    # direction, which is the mirror-image broken case (masks I itself) --
    # compensated the same way, reporting I=16 instead of the own value, 15.
    assert segments["segment_004"].mom6_obc_position_string() == "I=16,J=0:5"

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
    depth[2, 2] = 0.0  # mask one interior T-cell along the segment's line to land
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

    # axis="nyp", index=4 is an interior line (neither 0 nor the last index),
    # index_range restricts it to part of the domain's width -- exactly the
    # "not a cardinal edge" case the sketch is about.
    segment = Segment.from_hgrid(
        hgrid,
        axis="nyp",
        index=4,
        segment_name=seg_name,
        index_range=slice(2, 7),
        topo=topo,
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
