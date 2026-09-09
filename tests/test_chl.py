from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge.chl import gen_chl_empty_dataset, interpolate_and_fill_seawifs
import pytest
import os
from utils import on_cisl_machine


def test_chl(tmp_path, get_rect_grid):
    """Test the creation of chl files."""
    if not on_cisl_machine():
        pytest.skip("This test is only for the derecho and casper machines")
    # attempt to create a regional grid object from scratch
    grid = get_rect_grid
    grid.name = "pan2"
    # create a corresponding bathymetry object
    topo = Topo(
        grid=grid,
        min_depth=9.5,  # in meters
    )
    topo.set_spoon(1000, 10)

    interpolate_and_fill_seawifs(
        grid,
        topo,
        processed_seawifs_path="/glade/campaign/cesm/cesmdata/cseg/inputdata/ocn/mom/croc/chl/data/SeaWIFS.L3m.MC.CHL.chlor_a.0.25deg.nc",
        output_path=tmp_path / "seawifs-clim-1997-2010-pan-xesmf.nc",
    )

    assert os.path.exists(tmp_path / "seawifs-clim-1997-2010-pan-xesmf.nc")


@pytest.mark.parametrize(
    "calendar, expected_attr",
    [
        ("noleap", "noleap"),
        ("NOLEAP", "noleap"),
        ("365_day", "365_day"),
        ("gregorian", "gregorian"),
        ("standard", "gregorian"),  # CF's name for gregorian; FMS rejects "standard"
    ],
)
def test_chl_empty_dataset_calendar(calendar, expected_attr):
    """The calendar attribute is normalized to a name FMS accepts, and the TIME
    axis is the month midpoints of that calendar's climatological year."""
    ds = gen_chl_empty_dataset(None, [0.0, 1.0], [0.0, 1.0], calendar=calendar)

    assert ds.TIME.attrs["calendar"] == expected_attr
    assert ds.TIME.values == pytest.approx(
        [15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5]
    )


@pytest.mark.parametrize("calendar", ["all_leap", "366_day", "360_day", "julian"])
def test_chl_empty_dataset_unsupported_calendar(calendar):
    """Calendars we cannot yet build a matching TIME axis for are rejected rather
    than silently given a 365-day axis."""
    with pytest.raises(NotImplementedError, match="not supported"):
        gen_chl_empty_dataset(None, [0.0, 1.0], [0.0, 1.0], calendar=calendar)