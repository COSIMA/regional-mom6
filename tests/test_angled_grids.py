import regional_mom6 as rmom6
import os
from pathlib import Path


def test_angled_grids():
    """
    Test that the angled grid is correctly read in.
    """
    expt_name = "nwa12_read_grids"

    latitude_extent = [16.0, 27]
    longitude_extent = [192, 209]

    date_range = ["2005-01-01 00:00:00", "2005-02-01 00:00:00"]

    ## Place where all your input files go
    input_dir = Path(
        os.path.join(
            "/",
            "glade",
            "u",
            "home",
            "manishrv",
            "documents",
            "nwa12_0.1",
            "regional_mom_workflows",
            "rm6",
            expt_name,
            "inputs",
        )
    )

    ## Directory where you'll run the experiment from
    run_dir = Path(
        os.path.join(
            "/",
            "glade",
            "u",
            "home",
            "manishrv",
            "documents",
            "nwa12_0.1",
            "regional_mom_workflows",
            "rm6",
            expt_name,
            "run_files",
        )
    )
    for path in (run_dir, input_dir):
        os.makedirs(str(path), exist_ok=True)

    ## User-1st, test if we can even read the angled nc files.
    expt = rmom6.experiment(
        longitude_extent=longitude_extent,
        latitude_extent=latitude_extent,
        date_range=date_range,
        resolution=0.05,
        number_vertical_layers=75,
        layer_thickness_ratio=10,
        depth=4500,
        minimum_depth=5,
        mom_run_dir=run_dir,
        mom_input_dir=input_dir,
        toolpath_dir="",
        read_existing_grids=True,
    )

    ## Dev-2nd, test if the segment.coords function can properly give us the angles from this grid, which is at least called by rectangular_boundaries.

    ## User-2nd, test our ocean state boundary conditions

    ## User-3rd, test our tides boundary conditions
