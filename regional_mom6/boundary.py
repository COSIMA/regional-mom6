"""
Boundary: a self-contained description of a single straight, index-aligned
MOM6 open boundary condition (OBC) line, plus the regridding/rotation/masking/
writing logic needed to turn raw forcing data into MOM6 OBC segment files.

A ``Boundary`` only ever needs a small slice of a horizontal grid (and,
optionally, a matching ``mom6_forge.Topo`` for masking) -- it never needs the
whole ``experiment``/rest of the domain, so it can be built and exercised on
its own, independent of ``experiment``.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from regional_mom6 import regridding as rgd
from regional_mom6.utils import ap2ep, ep2ap, rotate, try_pint_convert
from regional_mom6.validate import validate_obc_file

# Legacy full-edge cases: (axis held fixed, index along that axis)
_CARDINAL_AXES = {
    "south": ("nyp", 0),
    "north": ("nyp", -1),
    "west": ("nxp", 0),
    "east": ("nxp", -1),
}


class Boundary:
    """
    The geometry of a single straight boundary line, plus the ability to
    regrid raw forcing data onto it and write MOM6-ready segment files.

    Fields (set once at construction, never mutated):
        lon (xarray.DataArray): 1-D longitude along the boundary.
        lat (xarray.DataArray): 1-D latitude along the boundary, same dims as ``lon``.
        angle (xarray.DataArray): Rotation angle (degrees) at each point along the
            boundary, same dims as ``lon``.
        segment_name (str): Name of the segment, e.g., ``'segment_001'``.
        parallel (str): ``"nx"`` or ``"ny"`` -- the along-boundary logical dim prefix.
        perpendicular (str): ``"ny"`` or ``"nx"`` -- the across-boundary logical dim prefix.
        axis_to_expand (int): Which axis the "main" coordinate corresponds to when
            re-adding the perpendicular axis during regridding.
        mask (Optional[xarray.DataArray]): 1-D ocean(1)/land(0) mask along the boundary,
            same dims as ``lon``. ``None`` means no masking is applied.
    """

    def __init__(
        self,
        *,
        lon,
        lat,
        angle,
        segment_name,
        parallel,
        perpendicular,
        axis_to_expand,
        mask=None,
    ):
        self.lon = lon
        self.lat = lat
        self.angle = angle
        self.segment_name = segment_name
        self.parallel = parallel
        self.perpendicular = perpendicular
        self.axis_to_expand = axis_to_expand
        self.mask = mask
        self._regridders = None
        self._tidal_regridders = None

    @classmethod
    def from_hgrid(
        cls,
        hgrid: xr.Dataset,
        *,
        axis: str,
        index: int,
        segment_name: str,
        index_range=None,
        topo=None,
    ) -> "Boundary":
        """
        Slice a straight boundary line out of ``hgrid``.

        Arguments:
            hgrid (xarray.Dataset): The horizontal supergrid dataset.
            axis (str): Which supergrid axis is held fixed -- ``"nyp"`` for a boundary
                that runs east-west (varies in x), ``"nxp"`` for one that runs
                north-south (varies in y).
            index (int): The fixed index along ``axis``. ``0`` or ``-1`` gives a full
                outer edge (the legacy north/south/east/west cases); any other value
                gives an interior/arbitrary line.
            segment_name (str): Name of the segment, e.g., ``'segment_001'``.
            index_range (slice, optional): Restrict the boundary to part of the line
                along the other (parallel) axis instead of its full length.
                Default ``None`` (whole line).
            topo (mom6_forge.topo.Topo, optional): A ``Topo`` instance for the same
                grid. If given, its ``supergridmask`` is sliced the same way to build
                the boundary's ocean/land mask. Default ``None`` (no masking).
        """
        if axis not in ("nyp", "nxp"):
            raise ValueError("axis must be one of: 'nyp', 'nxp'")

        lon = hgrid["x"].isel({axis: index})
        lat = hgrid["y"].isel({axis: index})
        if "angle_dx" in hgrid:
            angle = hgrid["angle_dx"].isel({axis: index})
        else:
            warnings.warn(
                "hgrid has no 'angle_dx' variable -- assuming zero rotation for this "
                "boundary. If your grid is rotated, compute angle_dx yourself (e.g. "
                "via mom6_forge's grid generation) before calling from_hgrid."
            )
            angle = xr.zeros_like(lon)
        mask = topo.supergridmask.isel({axis: index}) if topo is not None else None

        parallel_axis = "nxp" if axis == "nyp" else "nyp"
        if index_range is not None:
            lon = lon.isel({parallel_axis: index_range})
            lat = lat.isel({parallel_axis: index_range})
            angle = angle.isel({parallel_axis: index_range})
            if mask is not None:
                mask = mask.isel({parallel_axis: index_range})

        parallel, perpendicular, axis_to_expand = (
            ("nx", "ny", 2) if axis == "nyp" else ("ny", "nx", 3)
        )
        new_dim_name = f"{parallel}_{segment_name}"
        lon = lon.rename({parallel_axis: new_dim_name})
        lat = lat.rename({parallel_axis: new_dim_name})
        angle = angle.rename({parallel_axis: new_dim_name})
        if mask is not None:
            mask = mask.rename({parallel_axis: new_dim_name})

        return cls(
            lon=lon,
            lat=lat,
            angle=angle,
            segment_name=segment_name,
            parallel=parallel,
            perpendicular=perpendicular,
            axis_to_expand=axis_to_expand,
            mask=mask,
        )

    @classmethod
    def cardinal(
        cls,
        hgrid: xr.Dataset,
        orientation: str,
        segment_name: str,
        index_range=None,
        topo=None,
    ) -> "Boundary":
        """
        Convenience wrapper for the 4 legacy full-edge cases.

        Arguments:
            hgrid (xarray.Dataset): The horizontal supergrid dataset.
            orientation (str): One of ``'north'``, ``'south'``, ``'east'``, ``'west'``
                (case-insensitive).
            segment_name (str): Name of the segment, e.g., ``'segment_001'``.
            index_range (slice, optional): Restrict the boundary to part of the edge.
                Default ``None`` (the whole edge).
            topo (mom6_forge.topo.Topo, optional): See :meth:`from_hgrid`.
        """
        orientation = orientation.lower()
        if orientation not in _CARDINAL_AXES:
            raise ValueError(
                "orientation must be one of: 'north', 'south', 'east', or 'west'. "
                "For a boundary that isn't a full outer edge, use Boundary.from_hgrid directly."
            )
        axis, index = _CARDINAL_AXES[orientation]
        return cls.from_hgrid(
            hgrid,
            axis=axis,
            index=index,
            segment_name=segment_name,
            index_range=index_range,
            topo=topo,
        )

    @property
    def _coords_ds(self) -> xr.Dataset:
        """The boundary's lon/lat as a small xarray.Dataset, for use as the
        ``xesmf.Regridder`` output grid. ``lon``/``lat`` must be actual
        coordinates (not just data variables) for ``xe.Regridder``'s
        ``locstream_out`` mode to carry them through to the regridded output."""
        ds = xr.Dataset({"lon": self.lon, "lat": self.lat})
        return ds.assign_coords(lat=ds["lat"], lon=ds["lon"])

    def regrid_velocity_tracers(
        self,
        infile,
        varnames: dict,
        outfolder,
        startdate,
        arakawa_grid="A",
        regridding_method="bilinear",
        time_units="days",
        calendar="gregorian",
        fill_method=rgd.fill_missing_data,
        regridders=None,
        repeat_year_forcing=False,
    ):
        """
        Cut out and interpolate the velocities and tracers onto this boundary.

        Arguments:
            infile (Union[str, Path]): Path to the raw, unprocessed boundary segment.
            varnames (Dict[str, str]): Mapping between the variable/dimension names and
                standard naming convention of this pipeline, e.g., ``{"xh": "longitude",
                "yh": "latitude", "salt": "salinity", ...}``. Key "tracers" points to a
                nested dictionary of tracers to include in boundary.
            outfolder (Union[str, Path]): Path to folder where the model inputs will
                be stored.
            startdate (str): The starting date to use in the segment calendar.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the boundary
                forcing. Either ``'A'`` (default), ``'B'``, or ``'C'``.
            regridding_method (str): Regridding method to use throughout the function.
                Default is ``'bilinear'``.
            time_units (str): The units used by the raw forcing files. Default ``'days'``.
            calendar (str): Calendar to use for the time coordinate. Default ``'gregorian'``.
            fill_method (Function): Fill method to use throughout the function. Default
                is ``rgd.fill_missing_data``.
            regridders (dict, optional): Pre-built regridders with keys ``"tracers"``,
                ``"u"``, ``"v"``. If provided, regridder creation is skipped entirely --
                useful when calling this method multiple times for different time
                windows on the same boundary (pass ``boundary._regridders`` from a
                prior call). Default ``None`` -- regridders are built and cached on
                ``self._regridders`` for reuse on the next call.
            repeat_year_forcing (Optional[bool]): When ``True`` the experiment runs
                with repeat-year forcing. When ``False`` (default) inter-annual
                forcing is used.
        """
        reprocessed_var_map = rgd.apply_arakawa_grid_mapping(
            var_mapping=varnames, arakawa_grid=arakawa_grid
        )

        outfolder = Path(outfolder)
        (outfolder / "weights").mkdir(exist_ok=True)

        rawseg = xr.open_mfdataset(infile, decode_times=False, engine="netcdf4")

        # Convert z coordinates to meters if pint-enabled
        if type(reprocessed_var_map["depth_coord"]) != list:
            dc_list = [reprocessed_var_map["depth_coord"]]
        else:
            dc_list = reprocessed_var_map["depth_coord"]
        for dc in dc_list:
            rawseg[dc] = try_pint_convert(rawseg[dc], "m", dc)

        if regridders is None:
            regridders = rgd.create_vt_regridders(
                reprocessed_var_map,
                rawseg,
                self._coords_ds,
                outfolder,
                regridding_method,
                self.segment_name,
            )
        self._regridders = regridders

        u_regridded = regridders["u"](
            rawseg[reprocessed_var_map["u_var_name"]].rename(
                {
                    reprocessed_var_map["u_x_coord"]: "lon",
                    reprocessed_var_map["u_y_coord"]: "lat",
                }
            )
        )
        v_regridded = regridders["v"](
            rawseg[reprocessed_var_map["v_var_name"]].rename(
                {
                    reprocessed_var_map["v_x_coord"]: "lon",
                    reprocessed_var_map["v_y_coord"]: "lat",
                }
            )
        )
        tracers_regridded = regridders["tracers"](
            rawseg[
                [reprocessed_var_map["eta_var_name"]]
                + list(reprocessed_var_map["tracer_var_names"].values())
            ].rename(
                {
                    reprocessed_var_map["tracer_x_coord"]: "lon",
                    reprocessed_var_map["tracer_y_coord"]: "lat",
                }
            )
        )

        rotated_u, rotated_v = rotate(
            u_regridded,
            v_regridded,
            radian_angle=np.radians(self.angle.values),
        )

        rotated_u.name = reprocessed_var_map["u_var_name"]
        rotated_v.name = reprocessed_var_map["v_var_name"]
        segment_out = xr.merge([rotated_u, rotated_v, tracers_regridded])

        ## segment out now contains our interpolated boundary.
        ## Now, we need to fix up all the metadata and save
        segment_out = segment_out.rename(
            {"lon": f"lon_{self.segment_name}", "lat": f"lat_{self.segment_name}"}
        )

        ## Convert temperatures to celsius # use pint
        depth_coord = reprocessed_var_map["depth_coord"]
        if type(reprocessed_var_map["depth_coord"]) == list:
            for dc in reprocessed_var_map["depth_coord"]:
                if (
                    dc
                    in segment_out[reprocessed_var_map["tracer_var_names"]["temp"]].dims
                ):  # At least one must be true
                    depth_coord = dc

        if "since" not in time_units:
            times = xr.DataArray(
                np.arange(
                    0,  #! Indexing everything from start of experiment = simple but maybe counterintutive?
                    segment_out[reprocessed_var_map["time_var_name"]].shape[
                        0
                    ],  ## Time is indexed from start date of window
                    dtype=float,
                ),
                dims=["time"],
            )

            # This to change the time coordinate.
            segment_out = rgd.add_or_update_time_dim(
                segment_out, times, reprocessed_var_map["depth_coord"]
            )

            segment_out.time.attrs = {
                "calendar": calendar,
                "units": f"{time_units} since {startdate}",
            }
        else:
            segment_out.time.attrs = {
                "calendar": calendar,
                "units": time_units,
            }

        # Here, keep in mind that 'var' keeps track of the mom6 variable names we want, and self.tracers[var]
        # will return the name of the variable from the original data
        output_var_list = []
        allfields = {
            **reprocessed_var_map["tracer_var_names"],
            "u": reprocessed_var_map["u_var_name"],
            "v": reprocessed_var_map["v_var_name"],
            "eta": reprocessed_var_map["eta_var_name"],
        }  ## Combine all fields into one flattened dictionary to iterate over as we fix metadata

        for (
            var
        ) in (
            allfields
        ):  ## Replace with more generic list of tracer variables that might be included?
            v = f"{var}_{self.segment_name}"
            ## Rename each variable in dataset
            segment_out = segment_out.rename({allfields[var]: v})
            output_var_list.append(v)

            # Try Pint Conversion
            if var in rgd.main_field_target_units:
                # Apply raw data units if they exist
                units = rawseg[allfields[var]].attrs.get("units")
                if units is not None:
                    segment_out[v].attrs["units"] = units

                segment_out[v] = try_pint_convert(
                    segment_out[v], rgd.main_field_target_units[var], var
                )

            # Find out if the tracer has depth, and if so, what is it's z dimension (z dimension being a list is an edge case for MARBL BGC)
            variable_has_depth = False
            depth_coord = None
            if type(reprocessed_var_map["depth_coord"]) != list:
                dc_list = [reprocessed_var_map["depth_coord"]]
            else:
                dc_list = reprocessed_var_map["depth_coord"]

            for dc in dc_list:
                if dc in segment_out[v].dims:
                    depth_coord = dc
                    variable_has_depth = True
                    break

            if variable_has_depth:
                segment_out = rgd.vertical_coordinate_encoding(
                    segment_out,
                    v,
                    self.segment_name,
                    depth_coord,
                )

            segment_out = rgd.add_secondary_dimension(
                segment_out, v, self, self.segment_name
            )
            if variable_has_depth:
                segment_out = rgd.generate_layer_thickness(
                    segment_out,
                    v,
                    self.segment_name,
                    depth_coord,
                )

        # Here, do a foolproof (hopefully) manual conversion from K -> C just in case
        # pint doesn't manage to do so. Pint is finicky, but required for BGC fields. However,
        # we're making sure that temp will always be in C not K as this is a big problem!
        if (
            np.nanmin(
                segment_out[f"temp_{self.segment_name}"].isel(
                    {
                        reprocessed_var_map["time_var_name"]: 0,
                        f"nz_{self.segment_name}_temp": 0,
                    }
                )
            )
            > 100
        ):
            segment_out[f"temp_{self.segment_name}"] -= 273.15
            segment_out[f"temp_{self.segment_name}"].attrs["units"] = "degrees Celsius"

        # fill in NaNs
        segment_out = fill_method(
            segment_out,
            xdim=f"{self.parallel}_{self.segment_name}",
            zdim=reprocessed_var_map["depth_coord"],
        )

        # Overwrite the actual lat/lon values in the dimensions, replace with incrementing integers
        segment_out[f"{self.perpendicular}_{self.segment_name}"] = [0]

        segment_out[f"{self.parallel}_{self.segment_name}"] = np.arange(
            segment_out[f"{self.parallel}_{self.segment_name}"].size
        )
        segment_out[f"ny_{self.segment_name}"].attrs["axis"] = "Y"
        segment_out[f"nx_{self.segment_name}"].attrs["axis"] = "X"
        encoding_dict = {
            "time": {"dtype": "double"},
            f"nx_{self.segment_name}": {
                "dtype": "int32",
            },
            f"ny_{self.segment_name}": {
                "dtype": "int32",
            },
        }
        segment_out = rgd.mask_dataset(segment_out, self)
        encoding_dict = rgd.generate_encoding(
            segment_out,
            encoding_dict,
            default_fill_value=1.0e20,
        )
        # If repeat-year forcing, add modulo coordinate
        if repeat_year_forcing:
            segment_out["time"] = segment_out["time"].assign_attrs({"modulo": " "})
        segment_out.load().to_netcdf(
            outfolder / f"forcing_obc_{self.segment_name}.nc",
            encoding=encoding_dict,
            unlimited_dims="time",
        )

        validate_obc_file(
            segment_out,
            output_var_list,
            encoding_dict,
            surface_var=f"eta_{self.segment_name}",
        )

        return segment_out, encoding_dict

    def regrid_tides(
        self,
        tpxo_v,
        tpxo_u,
        tpxo_h,
        times,
        outfolder,
        startdate,
        regridding_method="bilinear",
        fill_method=rgd.fill_missing_data,
        regridders=None,
        repeat_year_forcing=False,
    ):
        """
        Regrids and interpolates the tidal data for MOM6 onto this boundary. Steps
        include:

        - Read raw tidal data (all constituents)
        - Perform minor transformations/conversions
        - Regrid the tidal elevation, and tidal velocity
        - Encode the output

        Arguments:
            tpxo_v, tpxo_u, tpxo_h (xarray.Dataset): Specific adjusted for MOM6 tpxo
                datasets (adjusted with :func:`~experiment.setup_boundary_tides`).
            times (pd.DateRange): The start date of our model period.
            outfolder (Union[str, Path]): Path to folder where the model inputs will
                be stored.
            startdate (str): The starting date to use in the segment calendar.
            regridding_method (str): Regridding method to use throughout the function.
                Default is ``'bilinear'``.
            fill_method (Function): Fill method to use throughout the function.
                Default is ``rgd.fill_missing_data``.
            regridders (dict, optional): Pre-built regridders with keys ``"elev"``,
                ``"u"``, ``"v"``. If provided, regridder creation is skipped entirely
                -- useful when calling this method multiple times on the same
                boundary (pass ``boundary._tidal_regridders`` from a prior call).
                Default ``None`` -- regridders are built and cached on
                ``self._tidal_regridders`` for reuse on the next call.
            repeat_year_forcing (Optional[bool]): Unused for tides (kept for
                signature symmetry with :meth:`regrid_velocity_tracers`).

        Returns:
            netCDF files: Regridded tidal velocity and elevation files in ``outfolder``.

        Code credit:

        .. code-block:: bash

            Author(s): GFDL, James Simkins, Rob Cermak, and contributors
            Year: 2022
            Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
            Version: N/A
            Type: Python Functions, Source Code
            Web Address: https://github.com/jsimkins2/nwa25
        """
        outfolder = Path(outfolder)
        (outfolder / "weights").mkdir(exist_ok=True)

        if regridders is None:
            regridders = {
                "elev": rgd.create_regridder(
                    tpxo_h[["lon", "lat", "hRe"]],
                    self._coords_ds,
                    outfolder
                    / "weights"
                    / f"bilinear_tidal_elev_weights_{self.segment_name}.nc",
                    method=regridding_method,
                ),
                "u": rgd.create_regridder(
                    tpxo_u[["lon", "lat", "uRe"]],
                    self._coords_ds,
                    outfolder
                    / "weights"
                    / f"bilinear_tidal_u_weights_{self.segment_name}.nc",
                    method=regridding_method,
                ),
                "v": rgd.create_regridder(
                    tpxo_v[["lon", "lat", "vRe"]],
                    self._coords_ds,
                    outfolder
                    / "weights"
                    / f"bilinear_tidal_v_weights_{self.segment_name}.nc",
                    method=regridding_method,
                ),
            }
        self._tidal_regridders = regridders
        regrid = regridders["elev"]

        ########## Tidal Elevation: Horizontally interpolate elevation components ############

        redest = regrid(tpxo_h[["lon", "lat", "hRe"]])
        imdest = regrid(tpxo_h[["lon", "lat", "hIm"]])

        # Fill missing data.
        # Need to do this first because complex would get converted to real
        redest = fill_method(
            redest, xdim=f"{self.parallel}_{self.segment_name}", zdim=None
        )
        redest = redest["hRe"]
        imdest = fill_method(
            imdest, xdim=f"{self.parallel}_{self.segment_name}", zdim=None
        )
        imdest = imdest["hIm"]

        # Convert complex
        cplex = redest + 1j * imdest

        # Convert to real amplitude and phase.
        ds_ap = xr.Dataset({f"zamp_{self.segment_name}": np.abs(cplex)})

        # np.angle doesn't return dataarray
        ds_ap[f"zphase_{self.segment_name}"] = (
            ("constituent", f"{self.parallel}_{self.segment_name}"),
            -1 * np.angle(cplex),
        )  # radians

        # Add time coordinate and transpose so that time is first,
        # so that it can be the unlimited dimension
        times = xr.DataArray(
            pd.date_range(startdate, periods=1),
            dims=["time"],
        )

        ds_ap = rgd.add_or_update_time_dim(ds_ap, times)
        ds_ap = ds_ap.transpose(
            "time", "constituent", f"{self.parallel}_{self.segment_name}"
        )

        self.encode_tidal_files_and_output(ds_ap, "tz", outfolder)

        ########### Regrid Tidal Velocity ######################

        regrid_u = regridders["u"]
        regrid_v = regridders["v"]

        # Interpolate each real and imaginary parts to self.
        uredest = regrid_u(tpxo_u[["lon", "lat", "uRe"]])["uRe"]
        uimdest = regrid_u(tpxo_u[["lon", "lat", "uIm"]])["uIm"]
        vredest = regrid_v(tpxo_v[["lon", "lat", "vRe"]])["vRe"]
        vimdest = regrid_v(tpxo_v[["lon", "lat", "vIm"]])["vIm"]

        # Fill missing data.
        # Need to do this first because complex would get converted to real
        uredest = fill_method(
            uredest, xdim=f"{self.parallel}_{self.segment_name}", zdim=None
        )
        uimdest = fill_method(
            uimdest, xdim=f"{self.parallel}_{self.segment_name}", zdim=None
        )
        vredest = fill_method(
            vredest, xdim=f"{self.parallel}_{self.segment_name}", zdim=None
        )
        vimdest = fill_method(
            vimdest, xdim=f"{self.parallel}_{self.segment_name}", zdim=None
        )

        # Convert to complex, remaining separate for u and v.
        ucplex = uredest + 1j * uimdest
        vcplex = vredest + 1j * vimdest

        # Convert complex u and v to ellipse,
        # rotate ellipse from earth-relative to model-relative,
        # and convert ellipse back to amplitude and phase.
        SEMA, ECC, INC, PHA = ap2ep(ucplex, vcplex)

        # Rotate
        INC -= np.radians(self.angle.data[np.newaxis, :])

        ua, va, up, vp = ep2ap(SEMA, ECC, INC, PHA)
        # Convert to real amplitude and phase.

        ds_ap = xr.Dataset(
            {f"uamp_{self.segment_name}": ua, f"vamp_{self.segment_name}": va}
        )
        # up, vp aren't dataarraysf
        ds_ap[f"uphase_{self.segment_name}"] = (
            ("constituent", f"{self.parallel}_{self.segment_name}"),
            up,
        )  # radians
        ds_ap[f"vphase_{self.segment_name}"] = (
            ("constituent", f"{self.parallel}_{self.segment_name}"),
            vp,
        )  # radians

        times = xr.DataArray(
            pd.date_range(startdate, periods=1),
            dims=["time"],
        )
        ds_ap = rgd.add_or_update_time_dim(ds_ap, times)
        ds_ap = ds_ap.transpose(
            "time", "constituent", f"{self.parallel}_{self.segment_name}"
        )

        # Some things may have become missing during the transformation
        ds_ap = fill_method(
            ds_ap, xdim=f"{self.parallel}_{self.segment_name}", zdim=None
        )

        self.encode_tidal_files_and_output(ds_ap, "tu", outfolder)

        return

    def encode_tidal_files_and_output(self, ds, filename, outfolder):
        """
        This method:

        - Expands the dimensions (with the segment name)
        - Renames some dimensions to be more specific to the segment
        - Provides an output file encoding
        - Exports the files.

        Arguments:
            ds (xarray.Dataset): The processed tidal dataset
            filename (str): The output file name
            outfolder (Union[str, Path]): The output folder to save the tidal files into

        Returns:
            netCDF files: Regridded [FILENAME] files in ``outfolder/[filename]_[segmentname].nc``

        Code credit:

        .. code-block:: bash

            Author(s): GFDL, James Simkins, Rob Cermak, and contributors
            Year: 2022
            Title: "NWA25: Northwest Atlantic 1/25th Degree MOM6 Simulation"
            Version: N/A
            Type: Python Functions, Source Code
            Web Address: https://github.com/jsimkins2/nwa25
        """
        outfolder = Path(outfolder)

        ## Expand Tidal Dimensions ##
        output_var_list = []
        for var in ds:
            ds = rgd.add_secondary_dimension(ds, str(var), self, self.segment_name)
            output_var_list.append(var)

        ## Rename Tidal Dimensions ##
        ds = ds.rename(
            {"lon": f"lon_{self.segment_name}", "lat": f"lat_{self.segment_name}"}
        )

        if self.mask is not None:
            print(
                "A land/ocean mask has been provided to the regridding tides function. "
                "Masking tides dataset with it may result in errors like large surface values one timestep in. "
                "To avoid masking tides, don't pass a topo to Boundary construction for tidal-only use."
            )
        ds = rgd.mask_dataset(ds, self)
        ## Perform Encoding ##

        fname = f"{filename}_{self.segment_name}.nc"
        # Set format and attributes for coordinates, including time if it does not already have calendar attribute
        # (may change this to detect whether time is a time type or a float).
        # Need to include the fillvalue or it will be back to nan
        encoding = {
            "time": dict(dtype="float64", calendar="gregorian", _FillValue=1.0e20),
            f"lon_{self.segment_name}": dict(dtype="float64", _FillValue=1.0e20),
            f"lat_{self.segment_name}": dict(dtype="float64", _FillValue=1.0e20),
        }
        encoding = rgd.generate_encoding(ds, encoding, default_fill_value=1.0e20)

        validate_obc_file(
            ds,
            output_var_list,
            encoding,
            surface_var="",
        )

        ## Export Files ##
        ds.to_netcdf(
            outfolder / fname,
            engine="netcdf4",
            encoding=encoding,
            unlimited_dims="time",
        )
        return ds
