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
from regional_mom6.utils import setup_logger, get_edge
from os.path import isfile

regridding_logger = setup_logger(__name__, set_handler=False)


def coords(
    hgrid: xr.Dataset,
    orientation: str,
    segment_name: str,
    coords_at_t_points=False,
    angle_variable_name="angle_dx",
) -> xr.Dataset:
    """
    Allows us to call the coords for use in the ``xesmf.Regridder`` in the :func:`~regrid_tides` function.
    ``self.coords`` gives us the subset of the ``hgrid`` based on the orientation.

    Arguments:
        hgrid (xr.Dataset): The horizontal grid dataset.
        orientation (str): The orientation of the boundary.
        segment_name (str): The name of the segment.
        coords_at_t_points (bool, optional): Whether to return the boundary t-points instead of
            the q/u/v of a general boundary for rotation. Default: ``False``.

    Returns:
        xr.Dataset: The correct coordinate space for the orientation

    Code credit:

    .. code-block:: bash

        Author(s): GFDL, James Simkins, Rob Cermak, and contributors
        Year: 2022
        Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
        Version: N/A
        Type: Python Functions, Source Code
        Web Address: https://github.com/jsimkins2/nwa25
    """

    dataset_to_get_coords = None

    if coords_at_t_points:
        regridding_logger.info("Creating coordinates of the boundary t-points")

        # Calculate t-point information
        ds = get_hgrid_arakawa_c_points(hgrid, "t")

        tangle_dx = hgrid[angle_variable_name][(ds.t_points_y, ds.t_points_x)]
        # Assign to dataset
        dataset_to_get_coords = xr.Dataset(
            {
                "x": ds.tlon,
                "y": ds.tlat,
                angle_variable_name: (("nyp", "nxp"), tangle_dx.values),
            },
            coords={"nyp": ds.nyp, "nxp": ds.nxp},
        )
    else:
        regridding_logger.info("Creating coordinates of the boundary q/u/v points")
        # Don't have to do anything because this is the actual boundary.
        # t-points are one-index deep and require managing.
        dataset_to_get_coords = hgrid

    # Rename nxp and nyp to locations
    if orientation == "south":
        rcoord = xr.Dataset(
            {
                "lon": dataset_to_get_coords["x"].isel(nyp=0),
                "lat": dataset_to_get_coords["y"].isel(nyp=0),
                "angle": dataset_to_get_coords[angle_variable_name].isel(nyp=0),
            }
        )
        rcoord = rcoord.rename_dims({"nxp": f"nx_{segment_name}"})
        rcoord.attrs["perpendicular"] = "ny"
        rcoord.attrs["parallel"] = "nx"
        rcoord.attrs[
            "axis_to_expand"
        ] = 2  ## Need to keep track of which axis the 'main' coordinate corresponds to when re-adding the 'secondary' axis
    elif orientation == "north":
        rcoord = xr.Dataset(
            {
                "lon": dataset_to_get_coords["x"].isel(nyp=-1),
                "lat": dataset_to_get_coords["y"].isel(nyp=-1),
                "angle": dataset_to_get_coords[angle_variable_name].isel(nyp=-1),
            }
        )
        rcoord = rcoord.rename_dims({"nxp": f"nx_{segment_name}"})
        rcoord.attrs["perpendicular"] = "ny"
        rcoord.attrs["parallel"] = "nx"
        rcoord.attrs["axis_to_expand"] = 2
    elif orientation == "west":
        rcoord = xr.Dataset(
            {
                "lon": dataset_to_get_coords["x"].isel(nxp=0),
                "lat": dataset_to_get_coords["y"].isel(nxp=0),
                "angle": dataset_to_get_coords[angle_variable_name].isel(nxp=0),
            }
        )
        rcoord = rcoord.rename_dims({"nyp": f"ny_{segment_name}"})
        rcoord.attrs["perpendicular"] = "nx"
        rcoord.attrs["parallel"] = "ny"
        rcoord.attrs["axis_to_expand"] = 3
    elif orientation == "east":
        rcoord = xr.Dataset(
            {
                "lon": dataset_to_get_coords["x"].isel(nxp=-1),
                "lat": dataset_to_get_coords["y"].isel(nxp=-1),
                "angle": dataset_to_get_coords[angle_variable_name].isel(nxp=-1),
            }
        )
        rcoord = rcoord.rename_dims({"nyp": f"ny_{segment_name}"})
        rcoord.attrs["perpendicular"] = "nx"
        rcoord.attrs["parallel"] = "ny"
        rcoord.attrs["axis_to_expand"] = 3

    # Make lat and lon coordinates
    rcoord = rcoord.assign_coords(lat=rcoord["lat"], lon=rcoord["lon"])

    return rcoord


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

    regridding_logger.info("Getting {} points..".format(point_type))

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
    point_dataset.attrs[
        "description"
    ] = "Arakawa C {}-points of supplied h-grid".format(point_type)
    return point_dataset


def create_regridder(
    forcing_variables: xr.Dataset,
    output_grid: xr.Dataset,
    outfile: Path = None,
    method: str = "bilinear",
    locstream_out: bool = True,
    periodic: bool = False,
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
        The path to the output file for weights; default: `None`
    method : str, optional
        The regridding method; default: ``"bilinear"``
    locstream_out : bool, optional
        Whether to output the locstream; default: ``True``
    periodic : bool, optional
        Whether the grid is periodic; default: ``False``

    Returns
    -------
    xe.Regridder
        The regridding object
    """
    regridding_logger.info("Creating Regridder")

    weights_exist = bool(outfile) and isfile(outfile)
    if weights_exist:
        regridding_logger.warning(
            f"Using existing weights file at {outfile} to save computing time. Delete weights file to regenerate weights."
        )
    regridder = xe.Regridder(
        forcing_variables,
        output_grid,
        method=method,
        locstream_out=locstream_out,
        periodic=periodic,
        filename=outfile,
        reuse_weights=weights_exist,
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
    regridding_logger.info("Filling in missing data horizontally, then vertically")
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
    regridding_logger.info("Adding time dimension")

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
    ds: xr.Dataset, var: str, coords, segment_name: str, to_beginning=False
) -> xr.Dataset:
    """Add the perpendiciular dimension to the dataset, even if it is
    only one value since it is required.

    Parameters:
        ds (xr.Dataset): The dataset to add the perpendicular dimension to
        var (str): The variable to add the perpendicular dimension to
        coords (xr.Dataset): The output xarray Dataset from the coords function. Contains information required to add the perpendicular dimension.
        segment_name (str): The segment name
        to_beginning (bool, optional): Whether to add the perpendicular dimension to the
            beginning or to the selected position, by default False

    Returns

        (xr.Dataset): The dataset with the vertical dimension added
    """

    # Check if we need to insert the dim earlier or later
    regridding_logger.info("Adding perpendicular dimension to {}".format(var))

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
        insert_behind_by = coords.attrs[
            "axis_to_expand"
        ]  # Just magic to add dim to the beginning

    regridding_logger.debug(f"Expand dimensions")
    ds[var] = ds[var].expand_dims(
        f"{coords.attrs['perpendicular']}_{segment_name}",
        axis=coords.attrs["axis_to_expand"] - insert_behind_by,
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

    regridding_logger.info("Renaming vertical coordinate to nz_... in {}".format(var))
    section = "_seg"
    base_var = var[: var.find(section)] if section in var else var
    ds[var] = ds[var].rename({old_vert_coord_name: f"nz_{segment_name}_{base_var}"})

    ## Replace the old depth coordinates with incremental integers
    regridding_logger.info("Replacing old depth coordinates with incremental integers")
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


def get_boundary_mask(
    bathy: xr.Dataset,
    side: str,
    minimum_depth=0,
    x_dim_name="nx",
    y_dim_name="ny",
) -> np.ndarray:
    """
    Mask out the boundary conditions based on the bathymetry. We don't want to have boundary conditions on land.
    Parameters
    ----------
    bathy : xr.Dataset
        The bathymetry dataset
    side : str
        The side of the boundary, "north", "south", "east", or "west"
    minimum_depth : float, optional
        The minimum depth to consider land, by default 0
    Returns
    -------
    np.ndarray
        The boundary mask
    """

    # Get the boundary depth
    depth = get_edge(bathy, side, x_name=x_dim_name, y_name=y_dim_name).depth
    # Force loading and copying to avoid shared references or lazy arrays
    depth = depth.load().copy()

    # If ntiles in bathymetry, remove it.
    if "ntiles" in depth.dims:
        depth = depth.isel({"ntiles": 0})

    # Mask fill values
    land = 0.0
    ocean = 1.0

    # Create mask of all ocean
    boundary_mask = np.full(depth.shape[0] * 2 + 1, ocean)

    # Fill with MOM6 version of mask (wet, wet_u,wet_c, wet_v)
    for i in range(len(depth)):
        if depth[i] <= minimum_depth:
            # The points to the left and right of this t-point are land points
            boundary_mask[(i * 2) + 2] = land
            boundary_mask[(i * 2) + 1] = land
            boundary_mask[(i * 2)] = land

    # Add Exceptions. The MOM6 mask (wet, not wet) does not include the neighboring q point as ocean. However, that point is used at the boundary.
    boundary_mask_og = boundary_mask.copy()
    for index in range(1, len(boundary_mask) - 1):
        if boundary_mask_og[index - 1] == land and boundary_mask_og[index] == ocean:
            boundary_mask[index - 1] = ocean
        elif boundary_mask_og[index + 1] == land and boundary_mask_og[index] == ocean:
            boundary_mask[index + 1] = ocean

    return boundary_mask


def mask_dataset(
    ds: xr.Dataset,
    bathymetry: xr.Dataset,
    orientation,
    y_dim_name="ny",
    x_dim_name="nx",
    fill_value=-1e20,
) -> xr.Dataset:
    """
    This function masks the dataset to the provided bathymetry. If bathymetry is not provided, it fills all NaNs with 0.
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to mask
    bathymetry : xr.Dataset
        The bathymetry dataset
    orientation : str
        The orientation of the boundary
    fill_value : float
        The value land points should be filled with
    """
    ## Add Boundary Mask ##
    if bathymetry is not None:
        regridding_logger.info(
            "Masking to bathymetry. If you don't want this, set bathymetry_path to None in the segment class."
        )
        mask = get_boundary_mask(
            bathymetry,
            orientation,
            minimum_depth=0,
            x_dim_name=x_dim_name,
            y_dim_name=y_dim_name,
        )

        mask[np.where(mask == 0)] = np.nan  # Convert Land Points to NaNs

        if orientation in ["east", "west"]:
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
            "All NaNs filled b/c bathymetry wasn't provided to the function. "
            + "Add bathymetry_path to the segment class to avoid this"
        )
        ds = ds.fillna(
            0
        )  # Without bathymetry, we can't assume the nans will be allowed in Boundary Conditions
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
    regridding_logger.info("Generating encoding dictionary")
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
