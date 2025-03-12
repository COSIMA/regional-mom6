from regional_mom6 import MOM_parameter_tools as mpt
from pathlib import Path
import regional_mom6 as rmom6
import importlib
import os
import shutil


def test_change_MOM_parameter(tmp_path):
    """
    Test the change MOM parameter function, as well as read_MOM_file and write_MOM_file under the hood.
    """
    expt_name = "testing"

    expt = rmom6.experiment.create_empty(
        expt_name=expt_name,
        mom_input_dir=tmp_path,
        mom_run_dir=tmp_path,
    )
    # Copy over the MOM Files to the dump_files_dir
    base_run_dir = Path(
        os.path.join(
            importlib.resources.files("regional_mom6").parent,
            "demos",
            "premade_run_directories",
        )
    )
    shutil.copytree(base_run_dir / "common_files", expt.mom_run_dir, dirs_exist_ok=True)
    MOM_override_dict = mpt.read_MOM_file_as_dict("MOM_override", expt.mom_run_dir)
    og = mpt.change_MOM_parameter(expt.mom_run_dir, "DT", "30", "COOL COMMENT")
    MOM_override_dict_new = mpt.read_MOM_file_as_dict("MOM_override", expt.mom_run_dir)
    assert MOM_override_dict_new["DT"]["value"] == "30"
    assert MOM_override_dict["DT"]["value"] == og
    assert MOM_override_dict_new["DT"]["comment"] == "COOL COMMENT\n"
