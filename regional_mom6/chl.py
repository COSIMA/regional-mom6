from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
import xarray as xr
import numpy as np
import xesmf as xe
from datetime import datetime
from pathlib import Path
from os.path import isfile
from mom6_forge.utils import fill_missing_data, compute_subsampling_factor
from mom6_forge.mapping import regrid_with_subsampling


def interpolate_and_fill_seawifs(
    grid: Grid,
    topo: Topo,
    processed_seawifs_path: Path | str,
    output_path: Path | str = None,
):
    """
    Interpolate and fill SeaWiFS chlorophyll data to a model grid and save to NetCDF.

    This function assumes a NO_LEAP calendar.

    This function takes gridded SeaWiFS chlorophyll data, interpolates it onto a
    super-sampled model grid, applies ocean masking, fills missing values, and writes
    the processed data to a NetCDF file. Global attributes are added to the output
    dataset to document provenance and authorship.

    Parameters
    ----------
    grid : Grid
        Model grid object containing tracer longitude and latitude arrays (`grid.tlon`, `grid.tlat`)
        and other grid arrays (`grid.qlon`, `grid.qlat`, `grid.nx`, `grid.ny`).
    topo : Topo
        Topography object containing the ocean mask (`topo.tmask`).
    processed_seawifs_path : Path or str
        Path to the preprocessed SeaWiFS chlorophyll dataset.
    output_path : Path or str, optional
        Path to save the output NetCDF file. If not provided, it is created in the same
        directory as `processed_seawifs_path` using the grid name.

    Returns
    -------
    xarray.Dataset
        A dataset containing the interpolated and filled chlorophyll concentration with
        appropriate global attributes and coordinate variables.
    """
    if grid.name is None:
        grid.name = "UnknownGridName"

    assert (
        grid.is_rectangular()
    ), "This function currently only supports rectangular grids."
    ocn_mask = topo.tmask
    ocn_nj, ocn_ni = ocn_mask.shape
    src_nc = xr.open_dataset(processed_seawifs_path)
    src_data = src_nc["chlor_a"]
    src_nj, src_ni = src_data.shape[-2], src_data.shape[-1]

    src_lon = src_nc[src_data.dims[-1]]
    src_lat = ((np.arange(src_nj) + 0.5) / src_nj - 0.5) * 180.0  # Recompute as doubles

    src_x0 = int((src_lon[0] + src_lon[-1]) / 2 + 0.5) - 180.0
    src_lon = (
        (np.arange(src_ni) + 0.5) / src_ni
    ) * 360.0 + src_x0  # Recompute as doubles

    ny_sub, nx_sub = compute_subsampling_factor(src_nj, src_ni, ocn_nj, ocn_ni)

    # Set output path
    if output_path is None:
        output_path = (
            Path(processed_seawifs_path).parent
            / f"seawifs-clim-1997-2010-{grid.name}.nc"
        )
    else:
        output_path = Path(output_path)

    # Generate empty dataset to fill for chlorophyll
    fill_value = np.float32(-1.0e34)
    chla = gen_chl_empty_dataset(
        output_path,
        grid.tlon[int(grid.ny / 2), :].values,
        grid.tlat[:, int(grid.nx / 2)].values,
        fill_value,
    )
    chlor_a = chla["CHL_A"]

    regridder = None
    for t in range(src_data.shape[0]):

        # Build source dataset for this timestep
        src_ds = xr.Dataset(
            {
                "chlor_a": xr.DataArray(
                    src_data[t, ::-1, :].values,
                    dims=["lat", "lon"],
                    coords={"lat": src_lat, "lon": src_lon},
                )
            }
        )

        # Regrid to super-sampled sub-point grid and average back to model grid
        q_sub, regridder = regrid_with_subsampling(
            input_dataset=src_ds,
            qlon=grid.qlon.values,
            qlat=grid.qlat.values,
            nx_sub=nx_sub,
            ny_sub=ny_sub,
            regridding_method="bilinear",
            regridder=regridder,
        )
        q_int = q_sub["chlor_a"].mean(dim=["ny_sub", "nx_sub"]).values

        # Fill any missing data
        q = q_int * ocn_mask
        q_nan = np.where((q == 0) | np.isnan(q), np.nan, q)
        chlor_a[t, :] = fill_missing_data(q_nan, ocn_mask)

    # Global attributes
    chla.attrs["title"] = (
        "Chlorophyll Concentration, OCI Algorithm, interpolated and objectively filled to "
        + grid.name
    )
    chla.attrs["repository"] = (
        "https://github.com/NCAR/SeaWIFS_MOM6 and https://github.com/NCAR/mom6_forge"
    )
    chla.attrs["authors"] = (
        "Gustavo Marques (gmarques@ucar.edu) and Frank Bryan (bryan@ucar.edu)"
    )
    chla.attrs["date"] = datetime.now().isoformat()

    # Assign variable data
    chla["CHL_A"].data[:] = chlor_a
    chla["LON"] = grid.tlon[0, :]
    chla["LAT"] = grid.tlat[:, 0]

    # Write to NetCDF
    chla.to_netcdf(
        output_path,
        unlimited_dims=["TIME"],
        encoding={
            "CHL_A": {
                "_FillValue": fill_value,
            }
        },
        format="NETCDF3_64BIT",
    )
    print(f"Wrote interpolated and filled SeaWiFS data to:\n{output_path}")

    # Clean up weights file if it exists
    Path("bilin_weights.nc").unlink(missing_ok=True)

    return chla


def gen_chl_empty_dataset(output_path, lon, lat, fill_value=-1.0e34, no_leap=True):
    """
    Generate an empty NetCDF dataset for SeaWiFS chlorophyll climatology and save it to disk.

    This function assumes a NO_LEAP calendar.

    This function creates a synthetic xarray dataset with a CHL_A variable (monthly mean chlorophyll)
    initialized entirely with fill values. The dataset uses the provided longitude and latitude
    arrays to define the spatial grid and includes a fixed climatological time axis.

    Parameters
    ----------
    output_path : str or Path
        Path to save the output NetCDF file.

    lon : array-like
        1D array of longitude values (in degrees east) defining the spatial X-axis.

    lat : array-like
        1D array of latitude values (in degrees north) defining the spatial Y-axis.

    Returns
    -------
    ds : xarray.Dataset
        The generated empty dataset containing the CHL_A variable and coordinate metadata.

    Notes
    -----
    - The CHL_A variable is filled with the placeholder value -1e34.
    - The TIME dimension is fixed and marked as unlimited in the NetCDF file.
    - TIME values represent the approximate midpoint of each month in a climatological year.
    - This dataset structure mimics that of SeaWiFS chlorophyll climatology products and is intended
      as a template or placeholder.
    """

    # === Coordinates ===

    if no_leap:
        time = np.array(
            [15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5]
        )
    else:
        time = np.array(
            [15.5, 45.5, 75.5, 106, 136.5, 167, 197.5, 228.5, 259, 289.5, 320, 350.5]
        )

    # === Placeholder data for CHL_A (all fill values) ===
    fill_value = np.float32(-1.0e34)
    chl_a_data = np.empty((len(time), len(lat), len(lon)), dtype=np.float32)

    # === xarray Dataset ===
    ds = xr.Dataset(
        {
            "CHL_A": xr.DataArray(
                chl_a_data,
                dims=["TIME", "LAT", "LON"],
                coords={"TIME": time, "LAT": lat, "LON": lon},
                attrs={
                    "long_name": "CHL_A = monthly mean",
                    "units": "mg/m^3",
                    "missing_value": fill_value,
                },
            )
        },
        coords={
            "LON": xr.DataArray(
                lon, dims="LON", attrs={"units": "degrees_east", "axis": "X"}
            ),
            "LAT": xr.DataArray(
                lat, dims="LAT", attrs={"units": "degrees_north", "axis": "Y"}
            ),
            "TIME": xr.DataArray(
                time,
                dims="TIME",
                attrs={
                    "units": "days since 0001-01-01 00:00:00",
                    "calendar": "NOLEAP",
                    "modulo": " ",
                    "axis": "T",
                    "cartesian_axis": "T",
                },
            ),
        },
        attrs={  # Global attributes
            "title": "SeaWiFS Chlorophyll Climatology (1997–2010)",
            "institution": "Generated by xarray",
            "source": "SeaWifs and NASA",
            "history": "",
        },
    )

    return ds


if __name__ == "__main__":
    interpolate_and_fill_seawifs()
