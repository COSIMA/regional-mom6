from pathlib import Path
import copy
from collections import defaultdict
import os


def change_MOM_parameter(
    directory, param_name, param_value=None, comment=None, override=True, delete=False
):
    """
    **Requires MOM parameter files present in the run directory**

    Changes a parameter in the ``MOM_input`` or ``MOM_override`` file. Returns original value, if there was one.
    If `delete` keyword argument is `True` the parameter is removed. But note, that **only** parameters from
    the ``MOM_override`` are be deleted since deleting parameters from ``MOM_input`` is not safe and can lead to errors.
    If the parameter does not exist, it will be added to the file.

    Arguments:
        param_name (str):
            Parameter name to modify
        directory (str):
            Directory where the MOM_input and MOM_override files are located
        param_value (Optional[str]):
            New assigned Value
        comment (Optional[str]):
            Any comment to add
        delete (Optional[bool]):
            Whether to delete the specified ``param_name``
    """
    if not delete and param_value is None:
        raise ValueError(
            "If not deleting a parameter, you must specify a new value for it."
        )

    MOM_override_dict = read_MOM_file_as_dict("MOM_override", directory)
    original_val = "No original val"
    if not delete:

        if param_name in MOM_override_dict:
            original_val = MOM_override_dict[param_name]["value"]
            print(
                "This parameter {} is being replaced from {} to {} in MOM_override".format(
                    param_name, original_val, param_value
                )
            )

        MOM_override_dict[param_name]["value"] = param_value
        MOM_override_dict[param_name]["comment"] = comment
        MOM_override_dict[param_name]["override"] = override
    else:
        if param_name in MOM_override_dict:
            original_val = MOM_override_dict[param_name]["value"]
            print("Deleting parameter {} from MOM_override".format(param_name))
            del MOM_override_dict[param_name]
        else:
            print(
                "Key to be deleted {} was not in MOM_override to begin with.".format(
                    param_name
                )
            )
    write_MOM_file(MOM_override_dict, directory)
    return original_val


def read_MOM_file_as_dict(
    filename,
    directory,
):
    """
    Read the MOM_input file from directory and return a dictionary of the variables and their values.
    """

    # Default information for each parameter
    default_layout = {"value": None, "override": False, "comment": None}

    if not os.path.exists(Path(directory / filename)):
        raise ValueError(
            f"File {filename} does not exist in the run directory {directory}"
        )
    with open(Path(directory / filename), "r") as file:
        lines = file.readlines()

        # Set the default initialization for a new key
        MOM_file_dict = defaultdict(lambda: default_layout.copy())
        MOM_file_dict["filename"] = filename
        dlc = copy.deepcopy(default_layout)
        for j in range(len(lines)):
            if "=" in lines[j] and not "===" in lines[j]:
                split = lines[j].split("=", 1)
                var = split[0]
                value = split[1]
                if "#override" in var:
                    var = var.split("#override")[1].strip()
                    dlc["override"] = True
                else:
                    dlc["override"] = False
                if "!" in value:
                    dlc["comment"] = value.split("!")[1]
                    dlc["value"] = value.split("!")[0].strip()
                else:
                    dlc["value"] = str(value.strip())
                    dlc["comment"] = None

                MOM_file_dict[var.strip()] = copy.deepcopy(dlc)

        # Save a copy of the original dictionary
        MOM_file_dict["original"] = copy.deepcopy(MOM_file_dict)
    return MOM_file_dict


def write_MOM_file(MOM_file_dict, directory):
    """
    Write the MOM_input file from a dictionary of variables and their values to directory. Does not support removing fields.
    """
    # Replace specific variable values
    original_MOM_file_dict = MOM_file_dict.pop("original")
    with open(Path(directory / MOM_file_dict["filename"]), "r") as file:
        lines = file.readlines()
        for jj in range(len(lines)):
            if "=" in lines[jj] and not "===" in lines[jj]:
                var = lines[jj].split("=", 1)[0].strip()
                if "#override" in var:
                    var = var.replace("#override", "")
                    var = var.strip()
                else:
                    # As in there wasn't an override before but we want one
                    if MOM_file_dict[var]["override"]:
                        lines[jj] = "#override " + lines[jj]
                        print("Added override to variable " + var + "!")
                if var in MOM_file_dict.keys() and (
                    str(MOM_file_dict[var]["value"])
                ) != str(original_MOM_file_dict[var]["value"]):
                    lines[jj] = lines[jj].replace(
                        str(original_MOM_file_dict[var]["value"]),
                        str(MOM_file_dict[var]["value"]),
                    )
                    if original_MOM_file_dict[var]["comment"] != None:
                        lines[jj] = lines[jj].replace(
                            original_MOM_file_dict[var]["comment"],
                            str(MOM_file_dict[var]["comment"]),
                        )
                    else:
                        lines[jj] = (
                            lines[jj].replace("\n", "")
                            + " !"
                            + str(MOM_file_dict[var]["comment"])
                            + "\n"
                        )

                    print(
                        "Changed "
                        + str(var)
                        + " from "
                        + str(original_MOM_file_dict[var]["value"])
                        + " to "
                        + str(MOM_file_dict[var]["value"])
                        + "in {}!".format(str(MOM_file_dict["filename"]))
                    )

    # Add new fields
    lines.append("! === Added with regional-mom6 ===\n")
    for key in MOM_file_dict.keys():
        if key not in original_MOM_file_dict.keys():
            if MOM_file_dict[key]["override"]:
                lines.append(
                    f"#override {key} = {MOM_file_dict[key]['value']} !{MOM_file_dict[key]['comment']}\n"
                )
            else:
                lines.append(
                    f"{key} = {MOM_file_dict[key]['value']} !{MOM_file_dict[key]['comment']}\n"
                )
            print(
                "Added",
                key,
                "to",
                MOM_file_dict["filename"],
                "with value",
                MOM_file_dict[key],
            )

    # Check any fields removed
    for key, entry in original_MOM_file_dict.items():
        if key not in MOM_file_dict:
            search_words = [
                key,
                entry["value"],
                entry["comment"],
            ]

            lines = [
                line for line in lines if not all(word in line for word in search_words)
            ]
            print(
                "Removed",
                key,
                "in",
                MOM_file_dict["filename"],
                "with value",
                original_MOM_file_dict[key],
            )

    with open(Path(directory / MOM_file_dict["filename"]), "w") as f:
        f.writelines(lines)
