from regional_mom6 import regional_mom6 as rm6
import json
import importlib
import inspect
from pathlib import Path
import datetime as dt
import xarray as xr
from regional_mom6.utils import setup_logger

config_logger = setup_logger(__name__, set_handler=False)


class Config:
    """Handles saving and loading experiment class configurations, including a function."""

    @staticmethod
    def save_to_json(obj, path=None, export=True):
        """Write a ``json`` configuration file for the experiment. The ``json`` file contains
        the experiment variable information to allow for easy pass off to other users, with a strict computer
        independence restriction. It also makes information about the expirement readable, and
        can be also useful for simply printing out information about the experiment.

        Arguments:
            obj (rm6.Experiment): The experiment object to save.
            path (str): Path to write the config file to. If not provided, the file is written to the ``mom_run_dir`` directory.
            export (bool): If ``True`` (default), the configuration file is written to disk on the given ``path``
        Returns:
            Dict: A dictionary containing the configuration information."""

        if export and path is None:
            raise ValueError(
                "The 'path' argument must be provided when 'export' is True."
            )

        config = {"args": {}}

        for key, value in obj.__dict__.items():
            if inspect.isfunction(value):
                config["args"][key] = {
                    "type": "function",
                    "name": value.__name__,
                    "module": value.__module__,
                }
            else:
                config["args"][key] = {
                    "type": type(value).__name__,
                    "value": convert_value_to_string(value),
                }

        if export:
            with open(path, "w") as f:
                json.dump(config, f, indent=4)
        return config

    @staticmethod
    def load_from_json(
        filename,
        mom_input_dir="mom_input/from_config",
        mom_run_dir="mom_run/from_config",
        create_hgrid_and_vgrid=True,
    ):
        """
        Load experiment variables from a configuration file and generate the horizontal and vertical grids (``hgrid``/``vgrid``).

        (This is basically another way to initialize an experiment.)

        Arguments:
        config_file_path (str): Path to the config file.
        mom_input_dir (str): Path to the MOM6 input directory. Default: ``"mom_input/from_config"``.
        mom_run_dir (str): Path to the MOM6 run directory. Default: ``"mom_run/from_config"``.
        create_hgrid_and_vgrid (bool): Whether to create the hgrid and the vgrid. Default is True.

        Returns:
        An experiment object with the fields from the configuration at ``config_file_path``.
        """
        with open(filename, "r") as f:
            config = json.load(f)

            # Dynamically import class

        expt = rm6.experiment.create_empty()
        # Iterate over the config and set attributes on the experiment object
        for key, val in config["args"].items():
            value_type = val.get("type")
            value = val.get("value")

            # Handle special cases based on value type
            if value_type == "function":
                # Handle functions: You may need to load the function based on its module and name
                config_logger.info(
                    f"Loading function {val['name']} from module {val['module']}"
                )
                try:
                    module = importlib.import_module(val["module"])
                    func = getattr(module, val["name"], None)
                    setattr(expt, key, func)
                except Exception as e:
                    config_logger.error(
                        f"Could not load function {val['name']} from module {val['module']}. Replace function not loaded correctly."
                    )

            else:
                # Handle unsupported types or custom logic as needed
                setattr(expt, key, convert_string_to_value(value, value_type))
        if expt.mom_run_dir is None:
            expt.mom_run_dir = Path(mom_run_dir)
        if expt.mom_input_dir is None:
            expt.mom_input_dir = Path(mom_input_dir)
        expt.mom_run_dir.mkdir(parents=True, exist_ok=True)
        expt.mom_input_dir.mkdir(parents=True, exist_ok=True)

        if create_hgrid_and_vgrid:
            expt.hgrid = expt._make_hgrid()
            expt.vgrid = expt._make_vgrid()

        return expt


def convert_value_to_string(value):
    """Helper function to handle serialization of different types."""
    if isinstance(value, dt.datetime):
        return value.isoformat()
    elif isinstance(value, (tuple, list)):
        return {
            "type": type(value[0]).__name__,
            "values": [convert_value_to_string(v) for v in value],
        }  # Convert each element
    elif isinstance(value, (str, int, float, dict, bool, type(None))):
        return value
    else:
        return str(value)  # Fallback: Convert unknown types to strings


def convert_string_to_value(value, value_type: str):
    """Helper function to handle deserialization of different types."""
    if value_type == "datetime":
        return dt.datetime.fromisoformat(value)
    if value_type == "PosixPath":
        return Path(value)
    elif value_type == "tuple":
        return tuple(
            convert_string_to_value(v, value["type"]) for v in value["values"]
        )  # Handle tuple elements recursively
    elif value_type == "list":
        return [
            convert_string_to_value(v, value["type"]) for v in value["values"]
        ]  # Handle list elements recursively

    return value  # If the type is not handled, just return the value
