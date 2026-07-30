"""Coverage for Segment.detect_open_cardinal_boundaries: which of the 4
cardinal edges of a domain actually touch open ocean, per a mom6_forge.Topo's
supergridmask -- an edge that's entirely land needs no OBC segment."""

from mom6_forge.grid import Grid
from mom6_forge.topo import Topo

from regional_mom6.segment import Segment


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
