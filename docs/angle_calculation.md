# Rotation and angle calculation in regional-mom6 using MOM6 Angle Calculation

Here we explain the implementation of MOM6 angle calculation in regional-mom6, which is the process by which regional-mom6 calculates the angle of curved horizontal grids (``hgrids``).

**Issue:** On a curved hgrid, we have to rotate the boundary conditions according to the angle the grid is rotated from lat-lon coordinates (true north vs model north). Although horizontal grids supplied by users will contain an `angle_dx` field, MOM6 ignores this field entirely and calculates its own grid angles internally. 

**Solution:** To be consistent with MOM6's treatement of grid angles, when we rotate our boundary conditions, we implemented MOM6 angle calculation in a file called "rotation.py", and included this in the the boundary regridding functions by default.


## Boundary rotation algorithm
Steps 1-5 replicate the angle calculation as done by MOM6. Step 6 is an additional step required to apply this algorithm to the boundary points.

1. Calculate pi/4rads / 180 degrees  = Gives a 1/4 conversion of degrees to radians. I.E. multiplying an angle in degrees by this gives the conversion to radians at 1/4 the value. 
2. Figure out the longitudunal extent of our domain, or periodic range of longitudes. For global cases it is len_lon = 360, for our regional cases it is given by the hgrid.
3. At each point on our hgrid, we find the q-point to the top left, bottom left, bottom right, top right. We adjust each of these longitudes to be in the range of len_lon around the point itself. (module_around_point)
4. We then find the lon_scale, which is the "trigonometric scaling factor converting changes in longitude to equivalent distances in latitudes". Whatever that actually means is we add the latitude of all four of these points from part 3 and basically average it and convert to radians. We then take the cosine of it. As I understand it, it's a conversion of longitude to equivalent latitude distance. 
5. Then we calculate the angle. This is a simple arctan2 so y/x. 
    1. The "y" component is the addition of the difference between the diagonals in longitude (adjusted by modulo_around_point in step 3) multiplied by the lon_scale, which is our conversion to latitude.
    2. The "x" component is the same addition of differences in latitude.
    3. Thus, given the same units, we can call arctan to get the angle in degrees

6. **Additional step to apply to boundaries**
Since the boundaries for a regional MOM6 domain are on the `q` points and not on the `t` points, to calculate the angle at the boundary points we need to expand the grid. This is implemented in the `create_expanded_hgrid` method.

## Convert this method to boundary angles - 2 Options
1. **EXPAND_GRID**: Compute grid angle replicating MOM6 calculations. Calculate another boundary row/column points around the hgrid using simple difference techniques. Use the new points to calculate the angle at the boundaries. This works because we can now access the four points needed to calculate the angle, where previously at boundaries we would be missing at least two.
2. **GIVEN_ANGLE**: Don't calculate the angle and use the user-provided field in the hgrid called `angle_dx`.


## Force the usage of the provided angle_dx values

To use the provided angles instead of the default algorithm, when calling the regridding functions `regrid_velocity_tracers` and `regrid_tides`, set the optional argument `rotational method = given_angle` 

## Code structure 

Most calculation code is implemented in the `rotation.py`, which is called by the regridding functions if rotation is required. 
