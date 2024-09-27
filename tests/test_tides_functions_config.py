import regional_mom6 as rmom6
import os
import pytest
import logging
from pathlib import Path

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestAll:
    @classmethod
    def setup_class(self):
        expt_name = "testing"

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
        self.glorys_path = os.path.join(
            "/",
            "glade",
            "derecho",
            "scratch",
            "manishrv",
            "inputs_rm6_hawaii",
            "glorys",
        )
        ## User-1st, test if we can even read the angled nc files.
        self.expt = rmom6.experiment(
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
        )

    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions."
    )
    def test_initial_condition(self):
        ocean_varnames = {
            "time": "time",
            "yh": "latitude",
            "xh": "longitude",
            "zl": "depth",
            "eta": "zos",
            "u": "uo",
            "v": "vo",
            "tracers": {"salt": "so", "temp": "thetao"},
        }

        # Set up the initial condition
        self.expt.setup_initial_condition(
            Path(
                os.path.join(self.glorys_path, "ic_unprocessed.nc")
            ),  # directory where the unprocessed initial condition is stored, as defined earlier
            ocean_varnames,
            arakawa_grid="A",
        )
        d1, d2, d3 = self.expt.initial_condition
        print(d1, d2, d3)
