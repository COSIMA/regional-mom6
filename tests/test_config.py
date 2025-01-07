import pytest
import regional_mom6 as rmom6
from pathlib import Path
import os
import json
import shutil


def test_write_config():
    expt_name = "testing"

    latitude_extent = [16.0, 27]
    longitude_extent = [192, 209]

    date_range = ["2005-01-01 00:00:00", "2005-02-01 00:00:00"]

    ## Place where all your input files go
    input_dir = Path(
        os.path.join(
            expt_name,
            "inputs",
        )
    )

    ## Directory where you'll run the experiment from
    run_dir = Path(
        os.path.join(
            expt_name,
            "run_files",
        )
    )
    data_path = Path("data")
    for path in (run_dir, input_dir, data_path):
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
        minimum_depth=25,
        mom_run_dir=run_dir,
        mom_input_dir=input_dir,
        toolpath_dir="",
        expt_name="test",
        boundaries=["south", "north"],
    )
    config_dict = expt.write_config_file()
    assert config_dict["longitude_extent"] == tuple(longitude_extent)
    assert config_dict["latitude_extent"] == tuple(latitude_extent)
    assert config_dict["date_range"] == date_range
    assert config_dict["resolution"] == 0.05
    assert config_dict["number_vertical_layers"] == 75
    assert config_dict["layer_thickness_ratio"] == 10
    assert config_dict["depth"] == 4500
    assert config_dict["minimum_depth"] == 25
    assert config_dict["expt_name"] == "test"
    assert config_dict["hgrid_type"] == "even_spacing"
    assert config_dict["repeat_year_forcing"] == False
    assert config_dict["tidal_constituents"] == ["M2"]
    assert config_dict["expt_name"] == "test"
    assert config_dict["boundaries"] == ["south", "north"]
    shutil.rmtree(run_dir)
    shutil.rmtree(input_dir)
    shutil.rmtree(data_path)


def test_load_config():

    expt_name = "testing"

    latitude_extent = [16.0, 27]
    longitude_extent = [192, 209]

    date_range = ["2005-01-01 00:00:00", "2005-02-01 00:00:00"]

    ## Place where all your input files go
    input_dir = Path(
        os.path.join(
            expt_name,
            "inputs",
        )
    )

    ## Directory where you'll run the experiment from
    run_dir = Path(
        os.path.join(
            expt_name,
            "run_files",
        )
    )
    data_path = Path("data")
    for path in (run_dir, input_dir, data_path):
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
        minimum_depth=25,
        mom_run_dir=run_dir,
        mom_input_dir=input_dir,
        toolpath_dir="",
    )
    path = "testing_config.json"
    config_expt = expt.write_config_file(path)
    new_expt = rmom6.create_experiment_from_config(os.path.join(path))
    assert str(new_expt) == str(expt)
    print(new_expt.vgrid)
    print(expt.vgrid)
    assert new_expt.hgrid == expt.hgrid
    assert (new_expt.vgrid.zi == expt.vgrid.zi).all() & (
        new_expt.vgrid.zl == expt.vgrid.zl
    ).all()
    assert os.path.exists(new_expt.mom_run_dir) & os.path.exists(new_expt.mom_input_dir)
    assert os.path.exists(new_expt.mom_input_dir / "hgrid.nc") & os.path.exists(
        new_expt.mom_input_dir / "vcoord.nc"
    )
    shutil.rmtree(run_dir)
    shutil.rmtree(input_dir)
    shutil.rmtree(data_path)
    shutil.rmtree(new_expt.mom_run_dir)
    shutil.rmtree(new_expt.mom_input_dir)
