"""
Segment: a self-contained description of a single straight, index-aligned
MOM6 open boundary condition (OBC) segment, plus the regridding/rotation/
masking/writing logic needed to turn raw forcing data into MOM6 OBC segment
files.

A ``Segment`` only ever needs a small slice of a horizontal grid (and,
optionally, a matching ``mom6_forge.Topo`` for masking) -- it never needs the
whole ``experiment``/rest of the domain, so it can be built and exercised on
its own, independent of ``experiment``.

Indexing conventions
---------------------
Two different grids, and two different sub-grids within MOM6's own grid, are
involved here. Getting them confused is the single easiest way to build a
segment that looks right in Python but is silently wrong (or outright
crashes) in MOM6 -- so this is worth reading carefully before touching any of
the index arithmetic below.

1.  **Supergrid vs. native grid.** ``hgrid`` (dims ``nyp``/``nxp``) is at 2x
    MOM6's native resolution: even supergrid indices are cell corners, odd
    supergrid indices are T-cell (tracer) centers. ``from_hgrid``'s ``index``
    and ``index_range`` arguments are always supergrid indices, because
    that's what's needed to slice ``hgrid``/``topo.supergridmask`` directly --
    for native T-cell row/column ``k``, the corresponding supergrid index is
    ``2 * k + 1``. Everywhere below, "T-row/T-column ``k``" means this native,
    non-supergrid index (``supergrid_index // 2``).

2.  **T-points vs. U-/V-points.** MOM6's own ``OBC_SEGMENT_00N =
    "J=...,I=..."`` runtime string is in its native grid too, but -- and this
    is the part that's easy to get backwards -- **the fixed coordinate of
    that string is not a T-point index**. A segment with a fixed ``J``
    (``axis="nyp"``, a horizontal line running east-west) sits on the
    **V-point** grid, because the velocity component that flows *through* a
    horizontal boundary is the meridional one, V. A segment with a fixed
    ``I`` (``axis="nxp"``, a vertical line running north-south) sits on the
    **U-point** grid, for the same reason with the zonal component, U. This
    isn't a convention invented here -- it's how MOM6 itself is written: the
    Fortran routines that parse these strings in ``MOM_open_boundary.F90``
    are literally named ``setup_v_point_obc`` (for ``"J="`` strings) and
    ``setup_u_point_obc`` (for ``"I="`` strings). The *parallel* range
    (``I=lo:hi`` on a ``J=``-fixed segment, or vice versa) stays in plain
    T-cell units, which is why it needs no special adjustment -- only the
    fixed coordinate does (see point 3).

3.  **Every U- or V-point line sits on the face between two T-cells, T-row/
    T-column ``k`` and ``k + 1``, and MOM6 forcibly masks one of those two
    T-cells to land at runtime for every segment that isn't a full outer
    edge -- regardless of what the bathymetry file says.** Which of the two
    gets masked depends on the segment's direction, and -- this is the
    non-obvious, previously-undiscovered part, confirmed against a real MOM6
    run -- it is genuinely asymmetric between the two directions of a given
    axis; it is not simply "always the neighbor on this side." See
    :meth:`Segment._compute_grid_index` for the exact rule MOM6 follows and
    the compensation ``Segment`` applies automatically so callers never have
    to reason about it themselves -- callers only need to get
    ``mom6_index_reverse`` right (see :meth:`Segment.from_hgrid`), which
    T-cell specifically ends up masked is handled internally.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from mom6_forge.grid import Grid

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

# Whether the parallel (along-segment) MOM6 index counts up or down for each
# legacy full-edge case, matching MOM6's requirement that the domain interior
# stay on the same side all the way around the perimeter: walking south (I
# ascending) -> east (J ascending) -> north (I descending) -> west (J
# descending) traces the perimeter counterclockwise, keeping the interior on
# the left throughout.
_CARDINAL_REVERSE = {"south": False, "east": False, "north": True, "west": True}


class Segment:
    """
    The geometry of a single straight MOM6 OBC segment, plus the ability to
    regrid raw forcing data onto it and write MOM6-ready segment files.

    Fields (set once at construction, never mutated):
        lon (xarray.DataArray): 1-D longitude along the segment.
        lat (xarray.DataArray): 1-D latitude along the segment, same dims as ``lon``.
        angle (xarray.DataArray): Rotation angle (degrees) at each point along the
            segment, same dims as ``lon``.
        segment_name (str): Name of the segment, e.g., ``'segment_001'``.
        parallel (str): ``"nx"`` or ``"ny"`` -- the along-segment logical dim prefix.
        perpendicular (str): ``"ny"`` or ``"nx"`` -- the across-segment logical dim prefix.
        axis_to_expand (int): Which axis the "main" coordinate corresponds to when
            re-adding the perpendicular axis during regridding.
        mask (Optional[xarray.DataArray]): 1-D ocean(1)/land(0) mask along the segment,
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
        grid_index=None,
        axis=None,
        index=None,
        index_range=None,
    ):
        self.lon = lon
        self.lat = lat
        self.angle = angle
        self.segment_name = segment_name
        self.parallel = parallel
        self.perpendicular = perpendicular
        self.axis_to_expand = axis_to_expand
        self.mask = mask
        self._grid_index = grid_index
        self._axis = axis
        self._index = index
        self._index_range = index_range
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
        mom6_index_reverse: bool = False,
    ) -> "Segment":
        """
        Slice a straight segment out of ``hgrid``.

        Arguments:
            hgrid (xarray.Dataset): The horizontal supergrid dataset.
            axis (str): Which supergrid axis is held fixed -- ``"nyp"`` for a
                *horizontal* segment that runs east-west (varies in x, sits on
                MOM6's V-point grid), ``"nxp"`` for a *vertical* one that runs
                north-south (varies in y, sits on MOM6's U-point grid). See
                the module docstring's "Indexing conventions" section for why.
            index (int): The fixed index along ``axis``, as a **supergrid**
                index -- i.e. ``2 * k + 1`` for native T-row/T-column ``k``,
                never the native index ``k`` directly. ``0`` or ``-1`` gives a
                full outer edge (the legacy north/south/east/west cases,
                where ``index`` lands on a supergrid *corner*, not a T-cell
                center); any other value gives an interior/arbitrary line and
                should be a T-center-aligned (odd) supergrid index so the
                segment's data actually comes from the T-row/T-column you
                intend. Always pick this to point at the row/column you want
                the segment's own wet data on -- ``Segment`` handles the
                separate MOM6-side masking arithmetic (see point 3 in the
                module docstring, and :meth:`_compute_grid_index`)
                internally; you never need to shift ``index`` yourself to
                compensate for it.
            segment_name (str): Name of the segment, e.g., ``'segment_001'``.
            index_range (slice, optional): Restrict the segment to part of the line
                along the other (parallel) axis instead of its full length.
                Default ``None`` (whole line). Must resolve to an *odd* number
                of supergrid points (MOM6's BRUSHCUTTER_MODE requirement,
                checked at construction) -- for T-cell indices ``lo``/``hi``,
                use ``slice(2 * lo + 1, 2 * hi + 2)``, not
                ``slice(2 * lo, 2 * hi + 2)``.
            topo (mom6_forge.topo.Topo, optional): A ``Topo`` instance for the same
                grid. If given, its ``supergridmask`` is sliced the same way to build
                the segment's ocean/land mask. Default ``None`` (no masking).
            mom6_index_reverse (bool): Which of the two possible directions
                this segment faces, i.e. which side of the line is the
                interior (ocean) side -- ``False`` emits the parallel range
                ascending in :meth:`mom6_obc_position_string` (MOM6's SOUTH
                direction for ``axis="nyp"``, EAST for ``axis="nxp"``);
                ``True`` emits it descending (NORTH / WEST respectively).
                Concretely, in terms of which side stays open:
                    ``axis="nyp"`` (horizontal line): ``False`` -> interior is
                        NORTH of the line; ``True`` -> interior is SOUTH.
                    ``axis="nxp"`` (vertical line): ``False`` -> interior is
                        WEST of the line; ``True`` -> interior is EAST.
                Get this backwards and MOM6 will force the *wrong* side of
                the line to land at runtime, since it always trusts this
                direction over the bathymetry file. Default ``False``.
        """
        if axis not in ("nyp", "nxp"):
            raise ValueError("axis must be one of: 'nyp', 'nxp'")

        parallel_axis = "nxp" if axis == "nyp" else "nyp"
        if index_range is not None:
            parallel_size = hgrid.sizes[parallel_axis]
            p_start = index_range.start if index_range.start is not None else 0
            p_stop = index_range.stop if index_range.stop is not None else parallel_size
            if (p_stop - p_start) % 2 == 0:
                raise ValueError(
                    f"index_range={index_range!r} for segment {segment_name!r} spans "
                    f"{p_stop - p_start} supergrid points (even) -- MOM6's "
                    "BRUSHCUTTER_MODE requires an odd number, aligned to T-cell "
                    "centers rather than raw corners. For T-cell indices lo/hi, use "
                    "slice(2 * lo + 1, 2 * hi + 2), not slice(2 * lo, 2 * hi + 2)."
                )

        lon = hgrid["x"].isel({axis: index})
        lat = hgrid["y"].isel({axis: index})
        if "angle_dx" in hgrid:
            angle = hgrid["angle_dx"].isel({axis: index})
        else:
            warnings.warn(
                "hgrid has no 'angle_dx' variable -- assuming zero rotation for this "
                "segment. If your grid is rotated, compute angle_dx yourself (e.g. "
                "via mom6_forge's grid generation) before calling from_hgrid."
            )
            angle = xr.zeros_like(lon)
        mask = topo.supergridmask.isel({axis: index}) if topo is not None else None

        grid_index = cls._compute_grid_index(
            hgrid, axis, index, parallel_axis, index_range, mom6_index_reverse
        )

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
            grid_index=grid_index,
            axis=axis,
            index=index,
            index_range=index_range,
        )

    @staticmethod
    def _compute_grid_index(hgrid, axis, index, parallel_axis, index_range, reverse):
        """Bookkeeping for :meth:`mom6_obc_position_string`: convert a supergrid
        axis/index/index_range into MOM6's own (half-resolution) model-grid I/J
        numbering, resolved once at construction time.

        MOM6's own ``open_boundary_impose_land_mask`` (MOM_open_boundary.F90) is
        asymmetric: for a J-fixed (nyp) segment it force-masks T-row ``J-1`` to
        land for a SOUTH-direction segment, but T-row ``J`` itself (the
        segment's *own* row) for a NORTH-direction one; symmetrically, for an
        I-fixed (nxp) segment it masks T-column ``I-1`` for WEST but ``I``
        itself for EAST. For a full outer edge this lands on a harmless
        sentinel/out-of-bounds value either way, which is why this was never
        caught before -- but for a genuine interior line, the "masks its own
        row/column" direction would force the segment's own wet cell to land,
        breaking it outright (confirmed against a real MOM6 run). We
        compensate by shifting ``fixed_value`` by one for exactly that
        direction, on interior lines only, so MOM6 always ends up masking the
        intended neighbor instead of the segment itself.
        """
        nyp_size = hgrid.sizes["nyp"]
        nxp_size = hgrid.sizes["nxp"]
        NJ = (nyp_size - 1) // 2
        NI = (nxp_size - 1) // 2

        fixed_size = nyp_size if axis == "nyp" else nxp_size
        resolved_index = index if index >= 0 else fixed_size + index
        fixed_value = resolved_index // 2
        is_full_edge = resolved_index in (0, fixed_size - 1)

        # axis=="nyp": reverse=True -> NORTH direction masks its own row.
        # axis=="nxp": reverse=False -> EAST direction masks its own column.
        masks_own_position = (axis == "nyp") == reverse
        if not is_full_edge and masks_own_position:
            fixed_value += 1

        if axis == "nyp":
            fixed_letter, fixed_max = "J", NJ
            parallel_letter, parallel_size = "I", NI
        else:
            fixed_letter, fixed_max = "I", NI
            parallel_letter, parallel_size = "J", NJ

        parallel_full_size = nxp_size if parallel_axis == "nxp" else nyp_size
        if index_range is None:
            p_start_super, p_stop_super = 0, parallel_full_size
        else:
            p_start_super = index_range.start if index_range.start is not None else 0
            p_stop_super = (
                index_range.stop if index_range.stop is not None else parallel_full_size
            )

        return {
            "fixed_letter": fixed_letter,
            "fixed_value": fixed_value,
            "fixed_max": fixed_max,
            "fixed_is_edge": is_full_edge,
            "parallel_letter": parallel_letter,
            "parallel_start": p_start_super // 2,
            "parallel_stop": (p_stop_super - 1) // 2,
            "parallel_size": parallel_size,
            "reverse": reverse,
        }

    def mom6_obc_position_string(self, reverse: bool = None) -> str:
        """
        This segment's MOM6 ``OBC_SEGMENT_00N`` grid-index position string, e.g.
        ``"J=45,I=5:15"``, or ``"J=0,I=0:N"`` for a full south edge (``"N"`` is
        MOM6's own last-index sentinel, resolved by MOM6 at runtime).

        MOM6 numbers I/J on its own (half-resolution) model grid, not the
        supergrid this ``Segment`` was built from -- this converts automatically.

        Arguments:
            reverse (bool, optional): Direction of the along-segment (parallel)
                index -- ``False`` counts up, ``True`` counts down. Default
                ``None`` uses the segment's own default (set automatically by
                :meth:`cardinal`; ``False`` for a segment built directly via
                :meth:`from_hgrid` unless ``mom6_index_reverse`` was passed there).
                MOM6 requires the domain interior to stay on the same side all the
                way around a segment; get this wrong and the segment runs backwards.
        """
        if self._grid_index is None:
            raise ValueError(
                "mom6_obc_position_string() needs grid-index bookkeeping that's only "
                "recorded by Segment.from_hgrid/Segment.cardinal -- this Segment "
                "was constructed directly and has no underlying grid to derive it from."
            )
        info = self._grid_index
        if reverse is None:
            reverse = info["reverse"]

        def fmt(value, vmax):
            return "N" if value == vmax else str(value)

        # A full outer edge still needs fmt()'s value==vmax check (to tell a
        # J=0/I=0 edge from a J=N/I=N one) -- but an interior line must never
        # print "N" for the fixed coordinate, even if the own-position
        # compensation above happens to push fixed_value up to fixed_max,
        # since it isn't actually a full edge.
        if info["fixed_is_edge"]:
            fixed_value_str = fmt(info["fixed_value"], info["fixed_max"])
        else:
            fixed_value_str = str(info["fixed_value"])
        fixed_str = f"{info['fixed_letter']}={fixed_value_str}"
        start, stop = info["parallel_start"], info["parallel_stop"]
        if reverse:
            start, stop = stop, start
        parallel_str = (
            f"{info['parallel_letter']}={fmt(start, info['parallel_size'])}:"
            f"{fmt(stop, info['parallel_size'])}"
        )
        return f"{fixed_str},{parallel_str}"

    @classmethod
    def cardinal(
        cls,
        hgrid: xr.Dataset,
        orientation: str,
        segment_name: str,
        index_range=None,
        topo=None,
    ) -> "Segment":
        """
        Convenience wrapper for the 4 legacy full-edge cases.

        Arguments:
            hgrid (xarray.Dataset): The horizontal supergrid dataset.
            orientation (str): One of ``'north'``, ``'south'``, ``'east'``, ``'west'``
                (case-insensitive).
            segment_name (str): Name of the segment, e.g., ``'segment_001'``.
            index_range (slice, optional): Restrict the segment to part of the edge.
                Default ``None`` (the whole edge).
            topo (mom6_forge.topo.Topo, optional): See :meth:`from_hgrid`.
        """
        orientation = orientation.lower()
        if orientation not in _CARDINAL_AXES:
            raise ValueError(
                "orientation must be one of: 'north', 'south', 'east', or 'west'. "
                "For a segment that isn't a full outer edge, use Segment.from_hgrid directly."
            )
        axis, index = _CARDINAL_AXES[orientation]
        return cls.from_hgrid(
            hgrid,
            axis=axis,
            index=index,
            segment_name=segment_name,
            index_range=index_range,
            topo=topo,
            mom6_index_reverse=_CARDINAL_REVERSE[orientation],
        )

    @classmethod
    def from_lonlat(
        cls,
        hgrid: xr.Dataset,
        *,
        axis: str,
        segment_name: str,
        fixed_lat: float = None,
        fixed_lon: float = None,
        lon_range=None,
        lat_range=None,
        topo=None,
        mom6_index_reverse: bool = False,
    ) -> "Segment":
        """
        Build a segment from physical coordinates instead of raw supergrid
        indices, resolving the nearest T-cell on ``hgrid`` itself (via a
        ``mom6_forge.Grid`` built from it) and delegating to
        :meth:`from_hgrid`.

        For ``axis="nyp"`` (a line running east-west): pass ``fixed_lat`` and
        ``lon_range=(lon0, lon1)``. For ``axis="nxp"`` (a line running
        north-south): pass ``fixed_lon`` and ``lat_range=(lat0, lat1)``.

        This is a nearest-T-cell approximation, exact on a uniform
        rectilinear grid (where ``j`` depends only on latitude and ``i``
        only on longitude); on a curvilinear/rotated grid it still resolves
        to a single straight index-aligned line, but that line may drift
        from the requested fixed coordinate away from the two endpoints.

        Arguments:
            hgrid (xarray.Dataset): The horizontal supergrid dataset.
            axis (str): ``"nyp"`` or ``"nxp"``, as in :meth:`from_hgrid`.
            segment_name (str): Name of the segment, e.g., ``'segment_001'``.
            fixed_lat (float): Required for ``axis="nyp"``.
            fixed_lon (float): Required for ``axis="nxp"``.
            lon_range (tuple[float, float]): Required for ``axis="nyp"``.
            lat_range (tuple[float, float]): Required for ``axis="nxp"``.
            topo (mom6_forge.topo.Topo, optional): See :meth:`from_hgrid`.
            mom6_index_reverse (bool): See :meth:`from_hgrid`.
        """
        grid = Grid.from_supergrid_ds(hgrid)

        if axis == "nyp":
            if fixed_lat is None or lon_range is None:
                raise ValueError("axis='nyp' requires fixed_lat and lon_range")
            j0, i0 = grid.get_indices(fixed_lat, lon_range[0])
            _, i1 = grid.get_indices(fixed_lat, lon_range[1])
            index = 2 * j0 + 1
            # T-center-aligned (odd-length) slice, not a raw corner-to-corner
            # one -- see from_hgrid's index_range docs.
            index_range = slice(2 * min(i0, i1) + 1, 2 * max(i0, i1) + 2)
        elif axis == "nxp":
            if fixed_lon is None or lat_range is None:
                raise ValueError("axis='nxp' requires fixed_lon and lat_range")
            j0, i0 = grid.get_indices(lat_range[0], fixed_lon)
            j1, _ = grid.get_indices(lat_range[1], fixed_lon)
            index = 2 * i0 + 1
            # T-center-aligned (odd-length) slice, not a raw corner-to-corner
            # one -- see from_hgrid's index_range docs.
            index_range = slice(2 * min(j0, j1) + 1, 2 * max(j0, j1) + 2)
        else:
            raise ValueError("axis must be one of: 'nyp', 'nxp'")

        return cls.from_hgrid(
            hgrid,
            axis=axis,
            index=index,
            segment_name=segment_name,
            index_range=index_range,
            topo=topo,
            mom6_index_reverse=mom6_index_reverse,
        )

    def to_spec(self) -> dict:
        """
        A small, JSON-serializable dict describing how this segment was cut
        from its ``hgrid`` -- everything needed to reconstruct an equivalent
        ``Segment`` later via :meth:`from_spec`, without holding onto the
        ``hgrid``/``topo`` objects themselves.
        """
        if self._axis is None:
            raise ValueError(
                "to_spec() needs the axis/index bookkeeping recorded by "
                "Segment.from_hgrid/Segment.cardinal/Segment.from_lonlat -- "
                "this Segment was constructed directly and has nothing to "
                "serialize."
            )
        return {
            "axis": self._axis,
            "index": self._index,
            "index_range": (
                [self._index_range.start, self._index_range.stop]
                if self._index_range is not None
                else None
            ),
            "mom6_index_reverse": self._grid_index["reverse"],
        }

    @classmethod
    def from_spec(
        cls, hgrid: xr.Dataset, spec: dict, segment_name: str, topo=None
    ) -> "Segment":
        """Rebuild a ``Segment`` from a dict produced by :meth:`to_spec`."""
        index_range = slice(*spec["index_range"]) if spec.get("index_range") else None
        return cls.from_hgrid(
            hgrid,
            axis=spec["axis"],
            index=spec["index"],
            segment_name=segment_name,
            index_range=index_range,
            topo=topo,
            mom6_index_reverse=spec.get("mom6_index_reverse", False),
        )

    @property
    def _coords_ds(self) -> xr.Dataset:
        """The segment's lon/lat as a small xarray.Dataset, for use as the
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
        Cut out and interpolate the velocities and tracers onto this segment.

        Arguments:
            infile (Union[str, Path]): Path to the raw, unprocessed segment forcing file.
            varnames (Dict[str, str]): Mapping between the variable/dimension names and
                standard naming convention of this pipeline, e.g., ``{"xh": "longitude",
                "yh": "latitude", "salt": "salinity", ...}``. Key "tracers" points to a
                nested dictionary of tracers to include in the segment.
            outfolder (Union[str, Path]): Path to folder where the model inputs will
                be stored.
            startdate (str): The starting date to use in the segment calendar.
            arakawa_grid (Optional[str]): Arakawa grid staggering type of the segment
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
                windows on the same segment (pass ``segment._regridders`` from a
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

        ## segment_out now contains our interpolated segment.
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
        Regrids and interpolates the tidal data for MOM6 onto this segment.
        Steps include:

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
                segment (pass ``segment._tidal_regridders`` from a prior call).
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
                "To avoid masking tides, don't pass a topo to Segment construction for tidal-only use."
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
