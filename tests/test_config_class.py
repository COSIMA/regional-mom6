import pytest
import regional_mom6 as rm6
from regional_mom6.config import Config
from pathlib import Path
import os
import json
import shutil
from regional_mom6 import regridding as rgd


@pytest.fixture
def create_expt(tmp_path):
    expt_name = "testing"

    latitude_extent = [16.0, 27]
    longitude_extent = [192, 209]

    date_range = ["2005-01-01 00:00:00", "2005-02-01 00:00:00"]

    ## Place where all your input files go
    input_dir = Path(
        os.path.join(
            tmp_path,
            expt_name,
            "inputs",
        )
    )

    ## Directory where you'll run the experiment from
    run_dir = Path(
        os.path.join(
            tmp_path,
            expt_name,
            "run_files",
        )
    )
    data_path = Path(tmp_path / "data")
    for path in (run_dir, input_dir, data_path):
        os.makedirs(str(path), exist_ok=True)

    ## User-1st, test if we can even read the angled nc files.
    expt = rm6.experiment(
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
        fre_tools_dir="",
        expt_name="test",
        boundaries=["south", "north"],
        regridding_method="bilinear",
        fill_method=rgd.fill_missing_data,
    )
    return expt


def test_write_config(create_expt, tmp_path):
    expt = create_expt
    Config.save_to_json(expt, tmp_path / "test_config.json")
    with open(tmp_path / "test_config.json", "r") as f:
        config_dict = json.load(f)
    assert config_dict["args"]["longitude_extent"]["value"]["values"] == list(
        expt.longitude_extent
    )
    assert config_dict["args"]["latitude_extent"]["value"]["values"] == list(
        expt.latitude_extent
    )
    assert config_dict["args"]["date_range"]["value"]["values"] == [
        expt.date_range[0].isoformat(),
        expt.date_range[1].isoformat(),
    ]
    assert config_dict["args"]["resolution"]["value"] == 0.05
    assert config_dict["args"]["number_vertical_layers"]["value"] == 75
    assert config_dict["args"]["layer_thickness_ratio"]["value"] == 10
    assert config_dict["args"]["depth"]["value"] == 4500
    assert config_dict["args"]["minimum_depth"]["value"] == 25
    assert config_dict["args"]["expt_name"]["value"] == "test"
    assert config_dict["args"]["hgrid_type"]["value"] == "even_spacing"
    assert config_dict["args"]["repeat_year_forcing"]["value"] == False
    assert config_dict["args"]["tidal_constituents"]["value"]["values"] == [
        "M2",
        "S2",
        "N2",
        "K2",
        "K1",
        "O1",
        "P1",
        "Q1",
        "MM",
        "MF",
    ]
    assert config_dict["args"]["expt_name"]["value"] == "test"
    assert config_dict["args"]["boundaries"]["value"]["values"] == ["south", "north"]
    assert config_dict["args"]["regridding_method"]["value"] == "bilinear"
    assert config_dict["args"]["fill_method"]["type"] == "function"


def test_read_config(create_expt, tmp_path):

    expt = create_expt
    path = tmp_path / "testing_config.json"
    Config.save_to_json(expt, path)
    new_expt = Config.load_from_json(
        os.path.join(path), mom_input_dir=tmp_path, mom_run_dir=tmp_path
    )

    assert new_expt.hgrid == expt.hgrid
    assert (new_expt.vgrid.zi == expt.vgrid.zi).all() & (
        new_expt.vgrid.zl == expt.vgrid.zl
    ).all()
    assert os.path.exists(new_expt.mom_run_dir) & os.path.exists(new_expt.mom_input_dir)
    assert os.path.exists(new_expt.mom_input_dir / "hgrid.nc") & os.path.exists(
        new_expt.mom_input_dir / "vcoord.nc"
    )


def test_str_dump(create_expt):
    expt = create_expt
    result = str(expt)
    assert isinstance(result, str)


def test_write_config_fail(create_expt):
    expt = create_expt
    with pytest.raises(ValueError):
        Config.save_to_json(expt, None, export=True)
