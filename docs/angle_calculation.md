# Rotated grids and the angle calculation

For rotated horizontal grids, that is grids whose coordinates do not align with lines of constant latitude and longitude, we have to rotate the boundary conditions appropriately by the angle that the grid is rotated from the latitude-longitude coordinates, i.e., the angle formed by the curved grid's constant ``x`` coordinate compared to the true north.

**Issue:** Although horizontal grids supplied by users usually _do_ contain an `angle_dx` field, MOM6 ignores `angle_dx`  entirely and re-calculates the angles that correspond to each grid point.

**Solution:** To be consistent with MOM6's treatment of grid angles, when we rotate our boundary conditions, we implemented MOM6 angle calculation in a file called "rotation.py", and included this in the the boundary regridding functions by default.

## Default Behavior
regional-mom6 by default computes the the angle of curved horizontal grids (``hgrids``) using the same algorithm as MOM6 does.
This algorithm is detailed below.

## Detailed Explanation

Here we explain the implementation of MOM6 angle calculation in regional-mom6, which is the process by which regional-mom6 calculates the angle of curved horizontal grids (``hgrids``).

## Boundary rotation algorithm
Steps 1-4 replicate the angle calculation as done by MOM6. Step 5 is an additional step required to apply this algorithm to the boundary points.

1. Figure out the longitudinal extent of our domain, or periodic range of longitudes. For global cases it is len_lon = 360, for our regional cases it is given by the hgrid.
2. At each ``t``-point on the `hgrid`, we find the four adjacent ``q``-points. We adjust each of these longitudes to be in the range of len_lon around the point itself. ({meth}`rotation.modulo_around_point <regional_mom6.rotation.modulo_around_point>`)
3. We then find the lon_scale, which is the "trigonometric scaling factor converting changes in longitude to equivalent distances in latitudes". What that means is we add the latitude of all four of these points from part 2, average it, and convert to radians. We then take the cosine of it. It's a factor we can use to convert a difference in longitude to equivalent latitude difference.
4. Then we calculate the angle. This is a simple arctan2 so y/x.
    1. The "y" component is the addition of the difference between the diagonals in longitude (adjusted by modulo_around_point in step 2) multiplied by the lon_scale, which is our conversion to latitude.
    2. The "x" component is the same addition of differences in latitude.
    3. Thus, given the same units, we can call arctan to get the angle in degrees

5. **Additional step to apply to boundaries**
Since the boundaries for a regional MOM6 domain are on the `q` points and not on the `t` points, to calculate the angle at the boundary points we need to expand the grid. This is implemented in the {meth}`rotation.create_expanded_hgrid <regional_mom6.rotation.create_expanded_hgrid>` method.

## Convert this method to boundary angles - 2 Options

1. **EXPAND_GRID**: Compute grid angle replicating MOM6 calculations. Calculate another boundary row/column points around the hgrid using simple difference techniques. Use the new points to calculate the angle at the boundaries. This works because we can now access the four points needed to calculate the angle, where previously at boundaries we would be missing at least two.
2. **GIVEN_ANGLE**: Do not calculate the grid angle but rather use the user-provided field in the horizontal grid called `angle_dx`.


### Force using the provided `angle_dx`

To enforce using the provided angles instead of the default algorithm, when calling the regridding methods {meth}`segment.regrid_velocity_tracers <regional_mom6.regional_mom6.segment.regrid_velocity_tracers>` and {meth}`segment.regrid_tides <regional_mom6.regional_mom6.segment.regrid_tides>`, set the optional keyword argument `rotational method = GIVEN_ANGLE`
