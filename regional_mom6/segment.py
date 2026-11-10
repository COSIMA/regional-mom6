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

**The mental model:** you specify a T-point, and the segment ends up enforced
on the U/V-point face right next to it. ``index`` (a T-center-aligned, odd
supergrid value -- point 1 below) is used *only* to pull the segment's own
data: lon/lat/mask/forcing values come straight from that T-row/T-column's
own coordinates and wetness. But every T-cell has two adjacent U/V-point
faces along the fixed axis -- e.g. for ``axis="nyp"``, T-row ``k`` has a
south face (the V-point between row ``k - 1`` and ``k``) and a north face
(between ``k`` and ``k + 1``) -- and ``ocean_side`` (point 3 below) tells
``Segment`` which of those two faces is the real boundary MOM6 enforces, by
saying which side of T-row/T-column ``k`` the open ocean is on. You never
touch the U/V arithmetic yourself; T-point in, correct U/V face out.

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
    (``I=lo:hi`` on a ``J=``-fixed segment, or vice versa) counts cell
    corners, so it opens T-cells ``lo`` to ``hi - 1``. MOM6 reads a segment's
    file from its first corner: corner values sit at the even positions and
    face values (tracers, normal velocity, SSH) at the odd ones. So an
    ``index_range`` must start and end on a corner (an even supergrid index);
    one that starts on a T-cell center puts every value half a cell off.

3.  **Every U- or V-point line sits on the face between two T-cells, T-row/
    T-column ``k`` and ``k + 1``, and MOM6 forcibly masks one of those two
    T-cells to land at runtime for every segment that isn't a full outer
    edge -- regardless of what the bathymetry file says.** Which of the two
    gets masked depends on which side of the line the ocean is on, and --
    this is the non-obvious, previously-undiscovered part, confirmed against
    a real MOM6 run -- it is genuinely asymmetric between the two sides of a
    given axis (``ocean_side="south"``/``"west"`` masks the segment's own
    T-row/T-column; ``"north"``/``"east"`` masks the true neighbor); it is
    not simply "always the neighbor on this side." See
    :meth:`Segment._compute_grid_index` for the exact rule MOM6 follows and
    the compensation ``Segment`` applies automatically so callers never have
    to reason about it themselves -- callers only need to say which side is
    ocean via ``ocean_side`` (see :meth:`Segment.from_hgrid`) -- which T-cell
    specifically ends up masked is handled internally.
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

# The ocean_side for each legacy full-edge case -- always the compass
# direction opposite the edge's own name, since the domain interior sits on
# the opposite side of a full outer edge from the edge itself (a "south"
# edge has its ocean to the north, etc).
_CARDINAL_OCEAN_SIDE = {
    "south": "north",
    "north": "south",
    "west": "east",
    "east": "west",
}

# Which compass side is the ocean/interior on, for each axis, mapped to
# whether the parallel (along-segment) MOM6 index counts up or down. A "nyp"
# (horizontal, J-fixed) segment's interior is either "north" or "south" of
# the line; a "nxp" (vertical, I-fixed) segment's is either "west" or "east".
# For the 4 cardinal cases this also matches MOM6's requirement that the
# domain interior stay on the same side all the way around the perimeter:
# walking south (I ascending) -> east (J ascending) -> north (I descending)
# -> west (J descending) traces the perimeter counterclockwise, keeping the
# interior on the left throughout.
_OCEAN_SIDE_REVERSE = {
    "nyp": {"north": False, "south": True},
    "nxp": {"west": False, "east": True},
}

# The two ocean_side values (one per axis) for which MOM6's
# open_boundary_impose_land_mask masks the segment's own T-row/T-column
# rather than its neighbor's -- see _compute_grid_index.
_OCEAN_SIDE_MASKS_OWN_POSITION = frozenset({"south", "west"})


# The helpers below find interior boundaries on a 2-D ocean mask ``w`` whose
# rows run along the fixed axis: T-rows for a "nyp" segment, T-columns (the
# transposed mask) for a "nxp" one. Face ``f`` is the face between rows ``f``
# and ``f + 1``, and a segment across it keeps the "low" side (row ``f``,
# south/west) or the "high" side (row ``f + 1``, north/east).

# The (low, high) ocean_side names for each axis.
_LOW_HIGH_SIDES = {"nyp": ("south", "north"), "nxp": ("west", "east")}


def _open_face_runs(w):
    """For each face, the column ranges ``(lo, hi)`` that are ocean on both sides."""
    faces = np.zeros((w.shape[0] - 1, w.shape[1] + 2), dtype=np.int8)
    faces[:, 1:-1] = w[:-1] & w[1:]
    steps = np.diff(faces, axis=1)
    return [
        list(
            zip(
                np.flatnonzero(row == 1).tolist(),
                (np.flatnonzero(row == -1) - 1).tolist(),
            )
        )
        for row in steps
    ]


def _channel_chains(runs):
    """Follow each channel across successive faces.

    Runs on neighboring faces join a chain only when they overlap one to one,
    so a chain stops where its channel splits or merges and every branch gets
    its own chain. Each chain is a list of ``(face, lo, hi)``.
    """
    successor, has_predecessor = {}, set()
    for f in range(len(runs) - 1):
        a_runs, b_runs = runs[f], runs[f + 1]
        pairs, i, j = [], 0, 0
        while i < len(a_runs) and j < len(b_runs):
            (a_lo, a_hi), (b_lo, b_hi) = a_runs[i], b_runs[j]
            if a_lo <= b_hi and b_lo <= a_hi:
                pairs.append((i, j))
            if a_hi < b_hi:
                i += 1
            else:
                j += 1
        a_counts = np.bincount([i for i, _ in pairs], minlength=len(a_runs))
        b_counts = np.bincount([j for _, j in pairs], minlength=len(b_runs))
        for i, j in pairs:
            if a_counts[i] == 1 and b_counts[j] == 1:
                successor[(f, i)] = (f + 1, j)
                has_predecessor.add((f + 1, j))

    chains = []
    for f, face_runs in enumerate(runs):
        for i in range(len(face_runs)):
            if (f, i) in has_predecessor:
                continue
            node, chain = (f, i), []
            while node is not None:
                chain.append((node[0], *runs[node[0]][node[1]]))
                node = successor.get(node)
            chains.append(chain)
    return chains


def _necks(chain):
    """The necks of a channel: places where it narrows, then widens again.

    A neck counts when the channel is wider on both sides by at least 2 cells
    and an eighth of the neck's width, so a ragged coast doesn't make a string
    of near-duplicates. Each neck is yielded as the crossings that
    are narrower than both sides, narrowest first, since the narrowest one
    may not pass the land-cap check where the coast runs diagonally.
    """
    from scipy.signal import find_peaks

    widths = np.array([hi - lo + 1 for _, lo, hi in chain])
    if widths.size < 3:
        return
    peaks, props = find_peaks(-widths, prominence=1)
    for peak, prominence, left, right in zip(
        peaks, props["prominences"], props["left_bases"], props["right_bases"]
    ):
        if prominence < max(2, widths[peak] / 8):
            continue
        shoulder = min(widths[left], widths[right])
        crossings = [k for k in range(left + 1, right) if widths[k] < shoulder]
        crossings.sort(key=lambda k: (widths[k], abs(k - peak)))
        yield [chain[k] for k in crossings]


def _land_row(face, keep_low):
    """The row MOM6 turns to land for a segment across ``face``."""
    return face + 1 if keep_low else face


def _close_face(w, face, lo, hi, keep_low):
    """Close ``face`` over columns ``lo``..``hi`` the way MOM6 closes an interior
    segment: it turns the T-cells on the far side of the line to land.

    Returns ``(labels, kept)`` -- the connected ocean regions afterwards and the
    label of the side that is kept -- if that cuts the ocean in two, otherwise
    ``None``.
    """
    from scipy import ndimage

    n_rows, n_cols = w.shape
    land_row = _land_row(face, keep_low)
    data_row = face if keep_low else face + 1
    beyond = 2 * land_row - data_row
    if lo == 0 or hi == n_cols - 1 or not 0 <= beyond < n_rows:
        return None  # runs into the domain edge
    closed = w.copy()
    closed[land_row, lo : hi + 1] = False
    labels, _ = ndimage.label(closed)
    kept = labels[data_row, lo]
    cut_off = set(np.unique(labels[beyond, lo : hi + 1]).tolist()) - {0}
    if not cut_off or kept in cut_off:
        return None  # nothing beyond, or the ocean joins up around it
    return labels, kept


def _boundary_label(n):
    """0 -> "A", 25 -> "Z", 26 -> "AA", ..."""
    label = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        label = chr(ord("A") + r) + label
    return label


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
        ocean_side: str,
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
                **must** be a T-center-aligned (odd) supergrid index --
                ``from_hgrid`` raises ``ValueError`` immediately for an even,
                corner-aligned ``index`` outside the two edge cases, since the
                segment's data would otherwise be silently pulled from the
                corner instead of the T-row/T-column you intend. Always pick
                this to point at the row/column you want
                the segment's own wet data on -- ``Segment`` handles the
                separate MOM6-side masking arithmetic (see point 3 in the
                module docstring, and :meth:`_compute_grid_index`)
                internally; you never need to shift ``index`` yourself to
                compensate for it.
            segment_name (str): Name of the segment, e.g., ``'segment_001'``.
            index_range (slice, optional): Restrict the segment to part of the line
                along the other (parallel) axis instead of its full length.
                Default ``None`` (whole line). Must start and end on a cell
                corner (an even supergrid index), since MOM6 reads the
                segment's data from its first corner. To open T-cells ``lo``
                to ``hi``, use ``slice(2 * lo, 2 * hi + 3)``.
            topo (mom6_forge.topo.Topo, optional): A ``Topo`` instance for the same
                grid. If given, its ``supergridmask`` is sliced the same way to build
                the segment's ocean/land mask. Default ``None`` (no masking).
            ocean_side (str): Which side of the line is the interior (ocean)
                side, in plain compass terms. For ``axis="nyp"`` (horizontal
                line): ``"north"`` or ``"south"``. For ``axis="nxp"``
                (vertical line): ``"east"`` or ``"west"``. Passing a
                direction that doesn't belong to ``axis`` (e.g. ``"east"``
                with ``axis="nyp"``) raises ``ValueError`` immediately,
                rather than silently building a segment MOM6 will mask on
                the wrong side. Get this backwards and MOM6 will force the
                *wrong* side of the line to land at runtime, since it always
                trusts this direction over the bathymetry file.
        """
        if axis not in ("nyp", "nxp"):
            raise ValueError("axis must be one of: 'nyp', 'nxp'")

        fixed_size = hgrid.sizes[axis]
        resolved_index = index if index >= 0 else fixed_size + index
        is_full_edge = resolved_index in (0, fixed_size - 1)
        if not is_full_edge and resolved_index % 2 == 0:
            raise ValueError(
                f"index={index!r} for segment {segment_name!r} is an even "
                "supergrid index (a grid corner), not a T-cell center. "
                "Interior/arbitrary lines must use an odd, T-center-aligned "
                "index (2 * k + 1 for native T-row/T-column k) -- otherwise "
                "the segment's own data is silently pulled from the corner "
                "instead of the T-cell you intend, even though the reported "
                "MOM6 J/I value is unchanged (both floor-divide to the same "
                "native row/column). Full outer edges (index=0 or index=-1) "
                "are exempt."
            )

        valid_sides = _OCEAN_SIDE_REVERSE[axis]
        if ocean_side not in valid_sides:
            raise ValueError(
                f"ocean_side={ocean_side!r} is invalid for axis={axis!r} "
                f"on segment {segment_name!r} -- axis={axis!r} segments "
                f"take ocean_side in {sorted(valid_sides)}."
            )
        mom6_index_reverse = valid_sides[ocean_side]

        parallel_axis = "nxp" if axis == "nyp" else "nyp"
        if index_range is not None:
            parallel_size = hgrid.sizes[parallel_axis]
            p_start = index_range.start if index_range.start is not None else 0
            p_stop = index_range.stop if index_range.stop is not None else parallel_size
            if p_start % 2 or (p_stop - 1) % 2:
                raise ValueError(
                    f"index_range={index_range!r} for segment {segment_name!r} must "
                    "start and end on a cell corner (an even supergrid index). MOM6 "
                    "reads a segment's data from its first corner, so a range that "
                    "starts on a T-cell center puts every value half a cell off. To "
                    "open T-cells lo to hi, use slice(2 * lo, 2 * hi + 3)."
                )

        if topo is not None:
            cls._check_land_capped_endpoints(
                hgrid, axis, index, index_range, ocean_side, topo, segment_name
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
            hgrid,
            axis,
            index,
            parallel_axis,
            index_range,
            mom6_index_reverse,
            ocean_side,
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
    def _check_land_capped_endpoints(
        hgrid, axis, index, index_range, ocean_side, topo, segment_name
    ):
        """Require an interior segment's two endpoints to be flanked by land.

        This is independent of the own-position masking rule documented on
        :meth:`_compute_grid_index` -- that rule is about which T-cell
        *straddling the line* MOM6 masks; this is about the T-cell one step
        *past each end* of the line, on the land side. A segment that ends
        flush against open water there -- with no land buffer past its own
        span -- puts that open-water cell in a concave notch against the
        land wall. That alone is enough to blow MOM6 up: it produces wildly
        different velocities on opposite faces of that one cell (a
        masking/velocity-point inconsistency), giving extreme SSH and
        velocity right there within the model's first timestep, even with
        fully correct segment indexing/orientation and a bathymetry file
        that already passes MOM6's usual "T-cell on the land side must be
        land" check (confirmed against a real MOM6 run).

        Only applies to genuine interior segments -- a full cardinal edge's
        endpoints are the domain's own corners, a separate, already-robust
        case that predates this check.
        """
        if index_range is None:
            return  # full-length line along this axis -- not interior

        fixed_size = hgrid.sizes[axis]
        resolved_index = index if index >= 0 else fixed_size + index
        if resolved_index in (0, fixed_size - 1):
            return  # full cardinal edge -- endpoints are the domain's corners
        t_index = resolved_index // 2  # native T-row/T-column the segment sits on

        depth = topo.depth.values
        ny, nx = depth.shape

        parallel_axis = "nxp" if axis == "nyp" else "nyp"
        p_start = index_range.start if index_range.start is not None else 0
        p_stop = (
            index_range.stop
            if index_range.stop is not None
            else hgrid.sizes[parallel_axis]
        )
        # The corners at the segment's two ends. It opens T-cells lo to hi - 1,
        # so the T-cells just past its ends are lo - 1 and hi.
        lo, hi = p_start // 2, (p_stop - 1) // 2

        if axis == "nyp":
            land_row = t_index + 1 if ocean_side == "south" else t_index - 1
            if not (0 <= land_row < ny):
                return  # land side is outside the domain entirely
            if lo <= 0 and hi >= nx:
                return  # full-width line, not an interior segment
            checks = [(land_row, lo - 1, "west"), (land_row, hi, "east")]
        else:
            land_col = t_index + 1 if ocean_side == "west" else t_index - 1
            if not (0 <= land_col < nx):
                return
            if lo <= 0 and hi >= ny:
                return
            checks = [(lo - 1, land_col, "south"), (hi, land_col, "north")]

        for row, col, side in checks:
            if 0 <= row < ny and 0 <= col < nx and depth[row, col] > 0:
                raise ValueError(
                    f"Segment {segment_name!r} isn't land-capped at its {side} end: "
                    f"(row={row}, col={col}) is open ocean, one cell past the "
                    f"segment's own span. MOM6 can produce spurious velocity/SSH "
                    f"right at that corner even though the segment's own index/"
                    f"orientation is correct -- widen the land mask so this cell "
                    f"is land too, or shorten the segment so its endpoint lands "
                    f"under land."
                )

    @staticmethod
    def _compute_grid_index(
        hgrid, axis, index, parallel_axis, index_range, reverse, ocean_side
    ):
        """Bookkeeping for :meth:`mom6_obc_position_string`: convert a supergrid
        axis/index/index_range into MOM6's own (half-resolution) model-grid I/J
        numbering, resolved once at construction time.

        MOM6's own ``open_boundary_impose_land_mask`` (MOM_open_boundary.F90) is
        asymmetric: for a J-fixed (nyp) segment it force-masks T-row ``J-1`` to
        land when the ocean is to the north, but T-row ``J`` itself (the
        segment's *own* row) when the ocean is to the south; symmetrically, for
        an I-fixed (nxp) segment it masks T-column ``I-1`` when the ocean is to
        the east but ``I`` itself when the ocean is to the west. For a full
        outer edge this lands on a harmless sentinel/out-of-bounds value either
        way, which is why this was never caught before -- but for a genuine
        interior line, the "masks its own row/column" case would force the
        segment's own wet cell to land, breaking it outright (confirmed
        against a real MOM6 run). We compensate by shifting ``fixed_value`` by
        one for exactly that case, on interior lines only, so MOM6 always ends
        up masking the intended neighbor instead of the segment itself.
        """
        nyp_size = hgrid.sizes["nyp"]
        nxp_size = hgrid.sizes["nxp"]
        NJ = (nyp_size - 1) // 2
        NI = (nxp_size - 1) // 2

        fixed_size = nyp_size if axis == "nyp" else nxp_size
        resolved_index = index if index >= 0 else fixed_size + index
        fixed_value = resolved_index // 2
        is_full_edge = resolved_index in (0, fixed_size - 1)

        # ocean_side="south"/"west" is exactly the case MOM6 masks its own
        # T-row/T-column rather than the neighbor's -- see docstring above.
        masks_own_position = ocean_side in _OCEAN_SIDE_MASKS_OWN_POSITION
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
            "ocean_side": ocean_side,
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
                ``None`` uses the segment's own default -- the direction implied
                by the ``ocean_side`` passed to :meth:`from_hgrid`/:meth:`cardinal`
                at construction time. MOM6 requires the domain interior to stay
                on the same side all the way around a segment; get this wrong
                and the segment runs backwards.
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
            ocean_side=_CARDINAL_OCEAN_SIDE[orientation],
        )

    @classmethod
    def detect_open_cardinal_boundaries(cls, topo) -> list:
        """
        The subset of the 4 cardinal edges (``'north'``, ``'south'``,
        ``'east'``, ``'west'``) that touch at least one ocean point in
        ``topo.supergridmask`` -- an edge that's entirely land needs no OBC
        segment at all, since MOM6 has nothing to force open there.

        Whether a cardinal edge is open is a property of the grid and
        bathymetry, not something a caller should have to spell out by hand;
        this is meant as a sensible default for callers building a segment
        list, e.g. CrocoDash's ``configure_forcings(boundaries=None, ...)``.

        Doesn't apply to custom/interior boundaries -- those can't be
        inferred from topo alone and must always be specified explicitly.

        Arguments:
            topo (mom6_forge.topo.Topo): A ``Topo`` instance for the grid in
                question; only its ``supergridmask`` is used.
        """
        mask = topo.supergridmask
        return [
            orientation
            for orientation, (axis, index) in _CARDINAL_AXES.items()
            if bool(mask.isel({axis: index}).any())
        ]

    @classmethod
    def find_interior_boundaries(cls, topo, keep=None) -> dict:
        """
        Find the straits and channel necks where an interior segment could cut
        the ocean in two, labelled ``"A"``, ``"B"``, ... for picking from
        :meth:`plot_interior_boundaries`.

        Each one is a straight run of open faces at the narrowest point of a
        channel, land-capped at both ends, that separates the ocean on one side
        from the ocean on the other. Lines that only cut a bay's head, or that
        the ocean flows around (e.g. coast to island), are left out.

        MOM6 only closes the row of T-cells just past the line, so the ocean
        beyond it stays in the bathymetry as an unforced, cut-off basin. Fill it
        with land (e.g. with mom6_forge's topo editor) if you don't want it.

        Arguments:
            topo (mom6_forge.topo.Topo): The domain's bathymetry; its ``tmask``
                says what is ocean.
            keep (tuple[float, float], optional): A ``(lon, lat)`` point in the
                ocean you want to model. Each boundary keeps the side this point
                is on, and boundaries that don't separate it from anything are
                left out. Default ``None`` keeps the larger side of each one.

        Returns:
            dict: ``{label: spec}``, labelled north to south, where each spec is a
            :meth:`to_spec` dict. Build a segment with
            ``Segment.from_spec(hgrid, spec, segment_name, topo=topo)``.
        """
        from scipy import ndimage

        wet = np.asarray(topo.tmask).astype(bool)
        has_depth = np.asarray(topo.depth) > 0
        # Channels are followed coast to coast with islands filled in, so a
        # strait with an island in it is still one crossing; the segment spans
        # the island.
        no_islands = ndimage.binary_fill_holes(wet)
        keep_cell = None
        if keep is not None:
            keep_cell = tuple(topo._grid.get_indices(keep[1], keep[0]))

        found = []  # (sort key, spec)
        for axis, (low, high) in _LOW_HIGH_SIDES.items():
            w = wet if axis == "nyp" else wet.T
            capped = has_depth if axis == "nyp" else has_depth.T
            filled = no_islands if axis == "nyp" else no_islands.T
            cell = keep_cell if axis == "nyp" or keep_cell is None else keep_cell[::-1]
            for chain in _channel_chains(_open_face_runs(filled)):
                for stretch in _necks(chain):
                    for face, lo, hi in stretch:
                        # Trim island cells off the ends.
                        open_faces = np.flatnonzero(
                            w[face, lo : hi + 1] & w[face + 1, lo : hi + 1]
                        )
                        if open_faces.size == 0:
                            continue
                        lo, hi = lo + int(open_faces[0]), lo + int(open_faces[-1])
                        keep_low = cls._side_to_keep(w, capped, face, lo, hi, cell)
                        if keep_low is None:
                            continue
                        data_row = face if keep_low else face + 1
                        middle = (lo + hi) / 2
                        row, col = (
                            (data_row, middle) if axis == "nyp" else (middle, data_row)
                        )
                        spec = {
                            "axis": axis,
                            "index": 2 * data_row + 1,
                            "index_range": [2 * lo, 2 * hi + 3],
                            "ocean_side": low if keep_low else high,
                        }
                        found.append(((-row, col), spec))
                        break

        found.sort(key=lambda item: item[0])
        return {_boundary_label(n): spec for n, (_, spec) in enumerate(found)}

    @staticmethod
    def _side_to_keep(w, has_depth, face, lo, hi, keep_cell):
        """Whether a cut keeps the low side (``True``) or the high side
        (``False``), or ``None`` if it can't keep the side wanted: the one with
        ``keep_cell`` in it, or else the larger one.

        The wanted side is picked before the land-cap check, so a crossing
        that can only be land-capped for the other side is passed over rather
        than flipping which side is kept. ``has_depth`` is ``depth > 0``, as
        in :meth:`_check_land_capped_endpoints`.
        """
        cuts = {}
        for keep_low in (True, False):
            closed = _close_face(w, face, lo, hi, keep_low)
            if closed is not None:
                cuts[keep_low] = closed
        if keep_cell is not None:
            wanted = [
                k for k, (labels, kept) in cuts.items() if labels[keep_cell] == kept
            ]
        else:
            wanted = sorted(
                cuts, key=lambda k: np.count_nonzero(cuts[k][0] == cuts[k][1])
            )[-1:]
        for keep_low in wanted:
            land_row = _land_row(face, keep_low)
            if not (has_depth[land_row, lo - 1] or has_depth[land_row, hi + 1]):
                return keep_low
        return None

    @staticmethod
    def plot_interior_boundaries(topo, boundaries: dict, ax=None):
        """
        Draw the boundaries from :meth:`find_interior_boundaries` on the
        domain's land mask, each with its letter on the side it keeps.

        Arguments:
            topo (mom6_forge.topo.Topo): The bathymetry the boundaries came from.
            boundaries (dict): ``{label: spec}`` from :meth:`find_interior_boundaries`.
            ax (matplotlib.axes.Axes, optional): Axes to draw on. Default ``None``
                makes a new figure.

        Returns:
            matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        grid = topo._grid
        qlon, qlat = np.asarray(grid.qlon), np.asarray(grid.qlat)
        tlon, tlat = np.asarray(grid.tlon), np.asarray(grid.tlat)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6), layout="constrained")

        land, ocean, line_color, ink = "#d3d1c7", "#eef3f9", "#eb6834", "#0b0b0b"
        ax.pcolormesh(
            qlon,
            qlat,
            np.asarray(topo.tmask),
            cmap=ListedColormap([land, ocean]),
            vmin=0,
            vmax=1,
            shading="flat",
        )
        ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(tlat))))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Possible interior boundaries (letter on the side kept)")

        to_pixels = ax.transData.transform
        for label, spec in boundaries.items():
            k = (spec["index"] - 1) // 2  # the kept T-row/column next to the line
            lo, hi = spec["index_range"][0] // 2, (spec["index_range"][1] - 3) // 2
            line = k + 1 if spec["ocean_side"] in ("south", "west") else k
            if spec["axis"] == "nyp":
                x, y = qlon[line, lo : hi + 2], qlat[line, lo : hi + 2]
                kx, ky = tlon[k, lo : hi + 1], tlat[k, lo : hi + 1]
            else:
                x, y = qlon[lo : hi + 2, line], qlat[lo : hi + 2, line]
                kx, ky = tlon[lo : hi + 1, k], tlat[lo : hi + 1, k]
            ax.plot(x, y, color=line_color, lw=2.5, solid_capstyle="round")

            middle = (x.mean(), y.mean())
            toward = to_pixels((kx.mean(), ky.mean())) - to_pixels(middle)
            toward = toward / (np.hypot(*toward) or 1.0)
            ax.annotate(
                label,
                middle,
                xytext=tuple(11 * toward),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=ink,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )
        return ax

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
        ocean_side: str,
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
            ocean_side (str): See :meth:`from_hgrid`.
        """
        grid = Grid.from_supergrid_ds(hgrid)

        if axis == "nyp":
            if fixed_lat is None or lon_range is None:
                raise ValueError("axis='nyp' requires fixed_lat and lon_range")
            j0, i0 = grid.get_indices(fixed_lat, lon_range[0])
            _, i1 = grid.get_indices(fixed_lat, lon_range[1])
            index = 2 * j0 + 1
            # Corner to corner, opening every T-column from i0 to i1 -- see
            # from_hgrid's index_range docs.
            index_range = slice(2 * min(i0, i1), 2 * max(i0, i1) + 3)
        elif axis == "nxp":
            if fixed_lon is None or lat_range is None:
                raise ValueError("axis='nxp' requires fixed_lon and lat_range")
            j0, i0 = grid.get_indices(lat_range[0], fixed_lon)
            j1, _ = grid.get_indices(lat_range[1], fixed_lon)
            index = 2 * i0 + 1
            # Corner to corner, opening every T-row from j0 to j1 -- see
            # from_hgrid's index_range docs.
            index_range = slice(2 * min(j0, j1), 2 * max(j0, j1) + 3)
        else:
            raise ValueError("axis must be one of: 'nyp', 'nxp'")

        return cls.from_hgrid(
            hgrid,
            axis=axis,
            index=index,
            segment_name=segment_name,
            index_range=index_range,
            topo=topo,
            ocean_side=ocean_side,
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
            "ocean_side": self._grid_index["ocean_side"],
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
            ocean_side=spec["ocean_side"],
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
        ignore_degenerate=False,
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
            ignore_degenerate (bool): Passed through to regridder construction --
                skip degenerate source cells instead of raising. Default ``False``.
                Needed for source datasets with duplicate/collapsed cells near the
                poles; see ``rgd.create_regridder``. Ignored when ``regridders`` is
                supplied, since no regridder is built.
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
                ignore_degenerate=ignore_degenerate,
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
            dim="all",
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
        redest = fill_method(redest, dim=f"{self.parallel}_{self.segment_name}")
        redest = redest["hRe"]
        imdest = fill_method(imdest, dim=f"{self.parallel}_{self.segment_name}")
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
        uredest = fill_method(uredest, dim=f"{self.parallel}_{self.segment_name}")
        uimdest = fill_method(uimdest, dim=f"{self.parallel}_{self.segment_name}")
        vredest = fill_method(vredest, dim=f"{self.parallel}_{self.segment_name}")
        vimdest = fill_method(vimdest, dim=f"{self.parallel}_{self.segment_name}")

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
        ds_ap = fill_method(ds_ap, dim=f"{self.parallel}_{self.segment_name}")

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
