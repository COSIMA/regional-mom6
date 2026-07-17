"""
Custom-built helper methods to regrid the boundary conditions and ensure proper encoding for MOM6.

Steps:

1. Initial Regridding -> Find the boundary of the ``hgrid``, and regrid the forcing variables to that boundary. Call (``initial_regridding``) and then use the ``xesmf.Regridder`` with any datasets you need.
2. Ensure temperatures are in Celsius.
3. Fill in NaNs. This step is important for MOM6 (``fill_missing_data``) -> This diverges between
4. For tides, we split the tides into an amplitude and a phase
5. In some cases, here is a great place to rotate the velocities to match a curved grid (tidal_velocity), velocity is also a good place to do this.
6. We then add the time coordinate
7. Add several depth-related coordinates to fields that are not related to the ocean's surface (like, e.g., surface wind stress).
    * Add a ``dz`` variable in layer thickness
    * Some metadata issues later on
8. Now we do up the metadata
9. Rename variables to var_segment_num
10. (For fields with vertical dimension) Rename the vertical coordinate of the variable to ``nz_segment_num_var``.
11. (For fields with vertical dimension) Declare this new vertical coordinate as a increasing series of integers.
12. Re-add the "perpendicular" dimension.
13. ....Add  layer thickness ``dz`` to the forcing fields with vertical dimension.
14. Add to encoding_dict a fill value(_FillValue) and zlib, dtype, for time, lat lon, ... and each variable (no type needed though).
"""

import xesmf as xe
import xarray as xr
from pathlib import Path
import dask.array as da
import numpy as np
import netCDF4
import logging
from os.path import isfile

regridding_logger = logging.getLogger(__name__)

# If the array is pint possible, ensure we have the right units for main fields (eta, u, v, temp),
# salinity and bgc tracers are a bit more abstract and should be already in the correct units, a TODO: would be to add functionality to convert these tracers
main_field_target_units = {
    "eta": "m",
    "u": "m/s",
    "v": "m/s",
    "temp": "degC",
}


def get_hgrid_arakawa_c_points(hgrid: xr.Dataset, point_type="t") -> xr.Dataset:
    """
    Get the Arakawa C points from the hgrid.

    Credit: Method originally by Fred Castruccio.

    Parameters
    ----------
    hgrid: xr.Dataset
        The hgrid dataset

    Returns
    -------
    xr.Dataset
        The specific points x, y, & point indexes
    """
    if point_type not in "uvqth":
        raise ValueError("point_type must be one of 'uvqht'")

    regridding_logger.debug("Getting {} points..".format(point_type))

    # Figure out the maths for the offset
    k = 2
    kp2 = k // 2
    offset_one_by_two_y = np.arange(kp2, len(hgrid.x.nyp), k)
    offset_one_by_two_x = np.arange(kp2, len(hgrid.x.nxp), k)
    by_two_x = np.arange(0, len(hgrid.x.nxp), k)
    by_two_y = np.arange(0, len(hgrid.x.nyp), k)

    # T point locations
    if point_type == "t" or point_type == "h":
        points = (offset_one_by_two_y, offset_one_by_two_x)
    # U point locations
    elif point_type == "u":
        points = (offset_one_by_two_y, by_two_x)
    # V point locations
    elif point_type == "v":
        points = (by_two_y, offset_one_by_two_x)
    # Corner point locations
    elif point_type == "q":
        points = (by_two_y, by_two_x)
    else:
        raise ValueError("Invalid Point Type (u, v, q, or t/h only)")

    point_dataset = xr.Dataset(
        {
            "{}lon".format(point_type): hgrid.x[points],
            "{}lat".format(point_type): hgrid.y[points],
            "{}_points_y".format(point_type): points[0],
            "{}_points_x".format(point_type): points[1],
        }
    )
    point_dataset.attrs["description"] = (
        "Arakawa C {}-points of supplied h-grid".format(point_type)
    )
    return point_dataset


def create_regridder(
    forcing_variables: xr.Dataset,
    output_grid: xr.Dataset,
    outfile: Path = None,
    method: str = "bilinear",
    locstream_out: bool = True,
    periodic: bool = False,
    reuse_weights: bool = False,
) -> xe.Regridder:
    """
    Basic regridder for any forcing variables. This is essentially a wrapper for
    the xesmf regridder with some default parameter choices.

    Parameters
    ----------
    forcing_variables : xr.Dataset
        The dataset of the forcing variables.
    output_grid : xr.Dataset
        The dataset of the output grid. This is the boundary of the ``hgrid``
    outfile : Path, optional
        The path to save regridding weights; default: ``None``. Weights are always
        written when a path is provided, even if ``reuse_weights=False``.
    method : str, optional
        The regridding method; default: ``"bilinear"``
    locstream_out : bool, optional
        Whether to output the locstream; default: ``True``
    periodic : bool, optional
        Whether the grid is periodic; default: ``False``
    reuse_weights : bool, optional
        If ``True`` and ``outfile`` exists on disk, load weights instead of
        recomputing them. Default ``False`` — always recompute so that stale
        weights from a previous grid do not silently produce wrong results.
        Set to ``True`` only when you are certain the grid has not changed,
        e.g. when reusing a :class:`segment` regridder across multiple time steps.

    Returns
    -------
    xe.Regridder
        The regridding object
    """
    regridding_logger.debug("Creating Regridder")

    if reuse_weights and bool(outfile) and isfile(outfile):
        regridding_logger.warning(
            f"Reusing existing weights file at {outfile}. Delete it if the grid has changed."
        )
    regridder = xe.Regridder(
        forcing_variables,
        output_grid,
        method=method,
        locstream_out=locstream_out,
        periodic=periodic,
        filename=outfile,
        reuse_weights=reuse_weights,
        unmapped_to_nan=True,
    )

    return regridder


def fill_missing_data(
    ds: xr.Dataset, xdim: str = "locations", zdim: str = "z", fill: str = "b"
) -> xr.Dataset:
    """
    Fill in missing values.

    Arguments:
        ds (xr.Dataset): The dataset to be filled in
        z_dim_name (str): The name of the ``z`` dimension

    Returns:
        xr.Dataset: The filled dataset

    Code credit:

    .. code-block:: bash

        Author(s): GFDL, James Simkins, Rob Cermak, and contributors
        Year: 2022
        Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
        Version: N/A
        Type: Python Functions, Source Code
        Web Address: https://github.com/jsimkins2/nwa25
    """
    regridding_logger.debug("Filling in missing data horizontally, then vertically")
    if fill == "f":
        filled = ds.ffill(dim=xdim, limit=None)
    elif fill == "b":
        filled = ds.bfill(dim=xdim, limit=None)
    if zdim is not None:
        if type(zdim) != list:
            zdim = [zdim]
        for z in zdim:
            filled = filled.ffill(dim=z, limit=None).fillna(0)
    return filled


def add_or_update_time_dim(ds: xr.Dataset, times, z_dims=None) -> xr.Dataset:
    """
    Add the time dimension to the dataset, in tides case can be one time step.

    Parameters:
        ds (xr.Dataset): The dataset to add the time dimension to
        times (list, np.Array, xr.DataArray): The list of times
        z_dims (list): z dimensions must go first, if they are provided that is enforced

    Returns:
        (xr.Dataset): The dataset with the time dimension added
    """
    regridding_logger.debug("Adding time dimension")

    regridding_logger.debug(f"Times: {times}")
    regridding_logger.debug(f"Make sure times is a DataArray")
    # Make sure times is an xr.DataArray
    times = xr.DataArray(times)

    if "time" in ds.dims:
        regridding_logger.debug("Time already in dataset, overwriting with new values")
        times.attrs = ds.time.attrs
        ds["time"] = times
    else:
        regridding_logger.debug("Time not in dataset, xr.Broadcasting time dimension")
        ds, _ = xr.broadcast(ds, times)

    # Make sure time is first....
    regridding_logger.debug("Transposing time to first dimension")
    if z_dims is not None:
        if type(z_dims) != list:
            z_dims = [z_dims]
        other_dims = [d for d in ds.dims if d not in ["time"] + z_dims]
        new_dims = ["time"] + z_dims + other_dims
    else:
        new_dims = ["time"] + [dim for dim in ds.dims if dim != "time"]
    ds = ds.transpose(*new_dims)

    return ds


def generate_dz(ds: xr.Dataset, z_dim_name: str) -> xr.Dataset:
    """
    Generate the vertical coordinate spacing.

    Parameters:
        ds (xr.Dataset): The dataset from which we extract the vertical coordinate.
        z_dim_name (str): The name of the vertical coordinate.

    Returns
        (xr.Dataset): The vertical spacing variable.
    """
    dz = ds[z_dim_name].diff(z_dim_name)
    dz.name = "dz"
    dz = xr.concat([dz, dz[-1]], dim=z_dim_name)
    return dz


def add_secondary_dimension(
    ds: xr.Dataset, var: str, boundary, segment_name: str, to_beginning=False
) -> xr.Dataset:
    """Add the perpendiciular dimension to the dataset, even if it is
    only one value since it is required.

    Parameters:
        ds (xr.Dataset): The dataset to add the perpendicular dimension to
        var (str): The variable to add the perpendicular dimension to
        boundary: A ``regional_mom6.boundary.Boundary`` (or any object exposing
            ``.perpendicular``/``.axis_to_expand``) describing the boundary's
            dimension layout, needed to add the perpendicular dimension.
        segment_name (str): The segment name
        to_beginning (bool, optional): Whether to add the perpendicular dimension to the
            beginning or to the selected position, by default False

    Returns

        (xr.Dataset): The dataset with the vertical dimension added
    """

    # Check if we need to insert the dim earlier or later
    regridding_logger.debug("Adding perpendicular dimension to {}".format(var))

    regridding_logger.debug(
        "Checking if nz or constituent is in dimensions, then we have to bump the perpendicular dimension up by one"
    )
    insert_behind_by = 0
    if not to_beginning:
        if any(
            coord.startswith("nz") or coord == "constituent" for coord in ds[var].dims
        ):
            regridding_logger.debug("Bump it by one")
            insert_behind_by = 0
        else:
            # Missing vertical dim or tidal coord means we don't need to offset the perpendicular
            insert_behind_by = 1
    else:
        insert_behind_by = (
            boundary.axis_to_expand
        )  # Just magic to add dim to the beginning

    regridding_logger.debug(f"Expand dimensions")
    ds[var] = ds[var].expand_dims(
        f"{boundary.perpendicular}_{segment_name}",
        axis=boundary.axis_to_expand - insert_behind_by,
    )
    return ds


def vertical_coordinate_encoding(
    ds: xr.Dataset, var: str, segment_name: str, old_vert_coord_name: str
) -> xr.Dataset:
    """
    Rename vertical coordinate to nz[additional-text] then change it to regular increments

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to rename the vertical coordinate in
    var : str
        The variable to rename the vertical coordinate in
    segment_name : str
        The segment name
    old_vert_coord_name : str
        The old vertical coordinate name
    """

    regridding_logger.debug("Renaming vertical coordinate to nz_... in {}".format(var))
    section = "_seg"
    base_var = var[: var.find(section)] if section in var else var
    ds[var] = ds[var].rename({old_vert_coord_name: f"nz_{segment_name}_{base_var}"})

    ## Replace the old depth coordinates with incremental integers
    regridding_logger.debug("Replacing old depth coordinates with incremental integers")
    ds[f"nz_{segment_name}_{base_var}"] = np.arange(
        ds[f"nz_{segment_name}_{base_var}"].size
    )

    return ds


def generate_layer_thickness(
    ds: xr.Dataset, var: str, segment_name: str, old_vert_coord_name: str
) -> xr.Dataset:
    """
    Generate Layer Thickness Variable, needed for vars with vertical dimensions
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to generate the layer thickness for
    var : str
        The variable to generate the layer thickness for
    segment_name : str
        The segment name
    old_vert_coord_name : str
        The old vertical coordinate name
    Returns
    -------
    xr.Dataset
        The dataset with the layer thickness variable added
    """
    regridding_logger.debug("Generating layer thickness variable for {}".format(var))
    dz = generate_dz(ds, old_vert_coord_name)
    ds[f"dz_{var}"] = (
        [
            "time",
            f"nz_{var}",
            f"ny_{segment_name}",
            f"nx_{segment_name}",
        ],
        da.broadcast_to(
            dz.data[None, :, None, None],
            ds[var].shape,
            chunks=(
                1,
                None,
                None,
                None,
            ),  ## Chunk in each time, and every 5 vertical layers
        ),
    )

    return ds


def mask_dataset(
    ds: xr.Dataset,
    boundary,
    fill_value=-1e20,
) -> xr.Dataset:
    """
    This function masks the dataset using the boundary's ocean(1)/land(0) mask
    (``boundary.mask``, already aligned point-for-point with the boundary's
    lon/lat). If ``boundary.mask`` is ``None``, it fills all NaNs with 0 instead.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to mask
    boundary : regional_mom6.boundary.Boundary
        The boundary supplying the ocean/land mask (``boundary.mask``) and the
        across-boundary dimension prefix (``boundary.perpendicular``).
    fill_value : float
        The value land points should be filled with
    """
    ## Add Boundary Mask ##
    if boundary.mask is not None:
        regridding_logger.debug(
            "Masking to the boundary's ocean/land mask. If you don't want this, don't pass a topo to Boundary construction."
        )
        mask = boundary.mask.values.astype(float).copy()
        mask[np.where(mask == 0)] = np.nan  # Convert Land Points to NaNs

        if boundary.perpendicular == "nx":
            mask = mask[:, np.newaxis]
        else:
            mask = mask[np.newaxis, :]

        for var in ds.data_vars.keys():
            # Drop to just the Boundary Dim
            da = ds[var].isel({dim: 0 for dim in list(ds[var].dims)[:-2]}).squeeze()

            nans_in_data = np.where(np.isnan(da))
            nans_in_mask = np.where(np.isnan(mask.squeeze()))

            # Check if all nans in the data are in the ocean and fill if so
            if not np.isin(nans_in_data, nans_in_mask).all():
                regridding_logger.warning(
                    f"NaNs in {var} not in mask. Which means there are NaNs over ocean. There shoudn't be NaNs after the regridding & filling functions. Please report to the regional_mom6 github repository as an issue."
                    + " These NaNs are filled with zeroes b/c they could cause issues with boundary conditions. Please check the final OBC files to make sure you're happy with this substitute!"
                )
                ds[var] = ds[var].fillna(0)

            # Apply the mask where land is NaN (using values because of conflicting indexes)
            ds[var].values = ds[var] * mask

            # Replace the land NaNs with a large FillValue
            ds[var].values = ds[var].fillna(fill_value)
    else:
        regridding_logger.warning(
            "All NaNs filled b/c no ocean/land mask was available. "
            + "Pass a topo to Boundary construction to avoid this."
        )
        ds = ds.fillna(
            0
        )  # Without a mask, we can't assume the nans will be allowed in Boundary Conditions
    return ds


def generate_encoding(
    ds: xr.Dataset, encoding_dict, default_fill_value=netCDF4.default_fillvals["f8"]
) -> dict:
    """
    Generate the encoding dictionary for the dataset.

    Parameters:

        ds (xr.Dataset): The dataset to generate the encoding for
        encoding_dict (dict): The starting encoding dict with some specifications needed
            for time and other vars, this will be updated with encodings in this function
        default_fill_value (float, optional): The default fill value; default: 1.0e20

    Returns:

        (dict): The encoding dictionary
    """
    regridding_logger.debug("Generating encoding dictionary")
    for var in ds:
        if "_segment_" in var and not "nz" in var:
            encoding_dict[var] = {
                "_FillValue": default_fill_value,
            }
    for var in ds.coords:
        if "nz_" in var:
            encoding_dict[var] = {
                "dtype": "int32",
            }

    return encoding_dict


def create_vt_regridders(
    reprocessed_var_map: dict,
    rawseg: xr.Dataset,
    coords: xr.Dataset,
    outfolder: str,
    regridding_method: str,
    id: str = "",
) -> dict[str, xe.Regridder]:
    """
    Create regridders for velocity and tracer variables based on the specified Arakawa grid.

    This function uses a validated variable mapping to create one or more
    `xesmf.Regridder` objects for velocity (`u`, `v`) and tracer fields,
    depending on the detected Arakawa grid type.

    Args:
        reprocessed_var_map: Mapping of variable and coordinate names, including nested
            tracer variable names (e.g., {"tracers": {"salt": "salt", "temp": "temp"}}).
        raw_seg: The source dataset containing the original variables.
        coords: The target grid coordinates dataset.
        outfolder: Path to the output folder where regridding weights are saved.
        regridding_method: The interpolation method (default: "bilinear").
        id: Optional string identifier appended to output weight filenames.

    Returns:
        dict[str, xe.Regridder]: A dictionary containing the created regridders with keys:
            - "tracers"
            - "u"
            - "v"
    """
    regridders = {}
    arakawa_grid = identify_arakawa_grid(reprocessed_var_map)
    outfolder = Path(outfolder)
    regridders["tracers"] = create_regridder(
        rawseg[reprocessed_var_map["tracer_var_names"]["salt"]].rename(
            {
                reprocessed_var_map["tracer_lon_coord"]: "lon",
                reprocessed_var_map["tracer_lat_coord"]: "lat",
            }
        ),
        coords,
        outfolder / f"weights/bilinear_tracer_weights_{id}.nc",
        method=regridding_method,
    )

    if arakawa_grid == "B" or arakawa_grid == "C":
        regridders["u"] = create_regridder(
            rawseg[reprocessed_var_map["u_var_name"]].rename(
                {
                    reprocessed_var_map["u_lon_coord"]: "lon",
                    reprocessed_var_map["u_lat_coord"]: "lat",
                }
            ),
            coords,
            outfolder / f"weights/bilinear_u_weights_{id}.nc",
            method=regridding_method,
        )
    else:  # Arakawa A
        regridders["u"] = regridders["tracers"]

    if arakawa_grid == "C":
        regridders["v"] = create_regridder(
            rawseg[reprocessed_var_map["v_var_name"]].rename(
                {
                    reprocessed_var_map["v_lon_coord"]: "lon",
                    reprocessed_var_map["v_lat_coord"]: "lat",
                }
            ),
            coords,
            outfolder / f"weights/bilinear_v_weights_{id}.nc",
            method=regridding_method,
        )
    else:  # Arakawa A and B
        regridders["v"] = regridders["u"]

    return regridders


def apply_arakawa_grid_mapping(var_mapping: dict, arakawa_grid: str = None) -> dict:
    """
    Map variable and coordinate names according to the specified Arakawa grid type.

    This function checks the provided Arakawa grid type and constructs a consistent
    mapping between standard variable keys (e.g., tracer, velocity components) and
    their corresponding actual names. It raises an error if any required variable
    names are missing for the specified grid type.

    Args:
        var_mappings (Dict[str, str]):
            A dictionary mapping standardized variable/dimension names to their actual
            names. Input names can use either the ``xh/xq`` convention with a specific arakawa grid or the exact output
            format produced by this function without the arakawa_grid specified (which it will only then do the sanity checks).
        arakawa_grid (str):
            The Arakawa grid staggering type of the boundary forcing. Must be one of:
            ``'A'``, ``'B'``, or ``'C'``.

    Returns:
        Dict[str, Any]:
            A dictionary containing variable names mapped according to the specified
            Arakawa grid type. The returned dictionary includes the following keys:
                - ``u_x_coord``
                - ``u_y_coord``
                - ``v_x_coord``
                - ``v_y_coord``
                - ``tracer_x_coord``
                - ``tracer_y_coord``
                - ``u_lon_coord``
                - ``u_lat_coord``
                - ``v_lon_coord``
                - ``v_lat_coord``
                - ``tracer_lon_coord``
                - ``tracer_lat_coord``
                - ``depth_coord``
                - ``u_var_name``
                - ``v_var_name``
                - ``tracer_var_names`` (a nested dict with keys ``"salt"`` and ``"temp"``)
    """

    if arakawa_grid is None:
        # If no arakawa_grid is provided, assume the mapping is already in the correct format
        print(
            "No arakawa_grid provided, assuming the variable mapping for your data product is already in correct format."
        )
        validate_var_mapping(var_mapping, is_xhyh=False)
        arakawa_grid = identify_arakawa_grid(var_mapping)
        print("Arakawa {} grid detected in variable mapping".format(arakawa_grid))
        return var_mapping
    else:
        if arakawa_grid not in ("A", "B", "C"):
            raise ValueError("arakawa_grid must be one of: 'A', 'B', or 'C'")

        # Validate basic var mapping structure
        validate_var_mapping(var_mapping, is_xhyh=True)

        reprocessed_var_map = {
            "tracer_x_coord": var_mapping["xh"],
            "tracer_y_coord": var_mapping["yh"],
            "u_var_name": var_mapping["u"],
            "v_var_name": var_mapping["v"],
            "eta_var_name": var_mapping["eta"],
            "time_var_name": var_mapping["time"],
            "depth_coord": var_mapping["zl"],
            "tracer_var_names": var_mapping[
                "tracers"
            ],  # validate_var_mapping will ensure this is a nested dict with "salt" and "temp" keys
        }

        if arakawa_grid == "A":
            print(
                "Applying Arakawa A grid variable mapping, which is velocities and tracers on the same grid"
            )
            reprocessed_var_map["u_x_coord"] = reprocessed_var_map["tracer_x_coord"]
            reprocessed_var_map["u_y_coord"] = reprocessed_var_map["tracer_y_coord"]
            reprocessed_var_map["v_x_coord"] = reprocessed_var_map["tracer_x_coord"]
            reprocessed_var_map["v_y_coord"] = reprocessed_var_map["tracer_y_coord"]

        elif arakawa_grid == "B":
            print(
                "Applying Arakawa B grid variable mapping, which is velocities on xq, yq and tracers on xh, yh."
            )
            if var_mapping["xq"] is None or var_mapping["yq"] is None:
                raise ValueError(
                    "For Arakawa B grid, variable mapping must include 'xq' and 'yq' coordinate names."
                )
            reprocessed_var_map["u_x_coord"] = var_mapping["xq"]
            reprocessed_var_map["u_y_coord"] = var_mapping["yq"]
            reprocessed_var_map["v_x_coord"] = var_mapping["xq"]
            reprocessed_var_map["v_y_coord"] = var_mapping["yq"]

        elif arakawa_grid == "C":
            print(
                "Applying Arakawa C grid variable mapping, which is u-velocity on xq, yh; v-velocity on xh, yq; and tracers on xh, yh."
            )
            if var_mapping["xq"] is None or var_mapping["yq"] is None:
                raise ValueError(
                    "For Arakawa C grid, variable mapping must include 'xq' and 'yq' coordinate names."
                )
            reprocessed_var_map["u_x_coord"] = var_mapping["xq"]
            reprocessed_var_map["u_y_coord"] = var_mapping["yh"]
            reprocessed_var_map["v_x_coord"] = var_mapping["xh"]
            reprocessed_var_map["v_y_coord"] = var_mapping["yq"]

        # Because curvilinear grids will have different x.y versus lat/lon but this version of the var_mapping assumes they are rectilinear, we set the
        # x/y coord to lon/lat
        # If you did want to use curvilinear in/out data, you would not use this xh/yh version of the var mapping and instead use the reprocessed variable mapping, which is the if part of this if/else statement
        reprocessed_var_map["u_lon_coord"] = reprocessed_var_map["u_x_coord"]
        reprocessed_var_map["u_lat_coord"] = reprocessed_var_map["u_y_coord"]
        reprocessed_var_map["v_lon_coord"] = reprocessed_var_map["v_x_coord"]
        reprocessed_var_map["v_lat_coord"] = reprocessed_var_map["v_y_coord"]
        reprocessed_var_map["tracer_lon_coord"] = reprocessed_var_map["tracer_x_coord"]
        reprocessed_var_map["tracer_lat_coord"] = reprocessed_var_map["tracer_y_coord"]

        # One last sanity check
        validate_var_mapping(reprocessed_var_map, is_xhyh=False)
        return reprocessed_var_map


def validate_var_mapping(var_map: dict, is_xhyh: bool = False) -> None:
    """
    Validate the structure and completeness of a variable mapping dictionary.

    This function checks that all expected keys and subkeys are present in the
    dictionary returned by the Arakawa grid variable mapping function.

    Args:
        var_map (Dict[str, Any]): The dictionary to validate.
        is_xhyh (bool): If True, expects the input dictionary to use the ``xh/xq`` regional_mom6 format

    Raises:
        ValueError: If any required keys or subkeys are missing, or if the dictionary
                    structure does not match the expected format.
    """
    if not is_xhyh:
        required_keys = {
            "time_var_name",
            "u_x_coord",
            "u_y_coord",
            "v_x_coord",
            "v_y_coord",
            "u_lon_coord",
            "u_lat_coord",
            "v_lon_coord",
            "v_lat_coord",
            "tracer_x_coord",
            "tracer_y_coord",
            "tracer_lon_coord",
            "tracer_lat_coord",
            "depth_coord",
            "u_var_name",
            "v_var_name",
            "eta_var_name",
            "tracer_var_names",
        }

    else:
        required_keys = {"time", "xh", "zl", "u", "v", "tracers", "eta"}

    missing = required_keys - var_map.keys()
    if missing:
        raise ValueError(
            f"Missing required keys in var_map: {', '.join(sorted(missing))}"
        )
    if not is_xhyh:
        tracer_map = var_map.get("tracer_var_names")
    else:
        tracer_map = var_map.get("tracers")
    # Validate nested tracer variable names

    if not isinstance(tracer_map, dict):
        raise ValueError("Expected tracers to be a dictionary.")

    required_tracers = {"salt", "temp"}
    missing_tracers = required_tracers - tracer_map.keys()
    if missing_tracers:
        raise ValueError(
            f"Missing required tracer variable names: {', '.join(sorted(missing_tracers))}"
        )


def identify_arakawa_grid(var_mapping):
    """identify the arakawa grid from the variable mapping"""
    if (
        var_mapping["v_x_coord"] == var_mapping["u_x_coord"]
        and var_mapping["u_x_coord"] == var_mapping["tracer_x_coord"]
    ):
        return "A"
    elif (
        var_mapping["v_x_coord"] == var_mapping["u_x_coord"]
        and var_mapping["u_x_coord"] != var_mapping["tracer_x_coord"]
    ):
        return "B"
    elif (
        var_mapping["v_x_coord"] != var_mapping["u_x_coord"]
        and var_mapping["u_x_coord"] != var_mapping["tracer_x_coord"]
        and var_mapping["v_y_coord"] != var_mapping["tracer_y_coord"]
    ):
        return "C"
    else:
        raise ValueError(
            "Could not determine Arakawa grid type from provided variable mapping. Something's wrong! Please specify variable mapping correctly"
        )
