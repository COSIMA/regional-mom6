# Rotation and angle calculation in regional-mom6 using MOM6 Angle Calculation

Here we explain the implementation of MOM6 angle calculation in regional-mom6, which is the process by which regional-mom6 calculates the angle of curved horizontal grids (``hgrids``).

**Issue:** On a curved hgrid, we have to rotate the boundary conditions according to the angle the grid is rotated from lat-lon coordinates (true north vs model north). MOM6 calculates the angle internally. ``rotation.py`` copies that MOM6 calculation into python so we can rotate it based on that calculation. The issue is that usually users provide hgrids in MOM6 with an angle field (``angle_dx``) and we were using that field. We don't necessarily trust that the user calculation doesn't have slight differences, so we implemented a way to use MOM6s angle calculation instead. In short, MOM6 doesn't actually use the user-provided ``angle_dx`` field in input hgrids, but internally calculates the angle. 

**Solution:** To accomodate this fact, when we rotate our boundary conditions, we implemented MOM6 angle calculation in a file called "rotation.py", and adjusted functions where we regrid the boundary conditions.


## MOM6 process of angle calculation (T-point only)
1. Calculate pi/4rads / 180 degrees  = Gives a 1/4 conversion of degrees to radians. I.E. multiplying an angle in degrees by this gives the conversion to radians at 1/4 the value. 
2. Figure out the longitudunal extent of our domain, or periodic range of longitudes. For global cases it is len_lon = 360, for our regional cases it is given by the hgrid.
3. At each point on our hgrid, we find the q-point to the top left, bottom left, bottom right, top right. We adjust each of these longitudes to be in the range of len_lon around the point itself. (module_around_point)
4. We then find the lon_scale, which is the "trigonometric scaling factor converting changes in longitude to equivalent distances in latitudes". Whatever that actually means is we add the latitude of all four of these points from part 3 and basically average it and convert to radians. We then take the cosine of it. As I understand it, it's a conversion of longitude to equivalent latitude distance. 
5. Then we calculate the angle. This is a simple arctan2 so y/x. 
    1. The "y" component is the addition of the difference between the diagonals in longitude (adjusted by modulo_around_point in step 3) multiplied by the lon_scale, which is our conversion to latitude.
    2. The "x" component is the same addition of differences in latitude.
    3. Thus, given the same units, we can call arctan to get the angle in degrees


## Problem
MOM6 only calculates the angle at t-points. For boundary rotation, we need the angle at the boundary, which is q/u/v points. Because we need the points to the left, right, top, and bottom of the point, this method won't work for the boundary.


## Convert this method to boundary angles - 3 Options
1. **GIVEN_ANGLE**: Don't calculate the angle and use the user-provided field in the hgrid called "angle_dx"
2. **EXPAND_GRID**: Calculate another boundary row/column points around the hgrid using simple difference techniques. Use the new points to calculate the angle at the boundaries. This works because we can now access the four points needed to calculate the angle, where previously at boundaries we would be missing at least two. 


## Code Description

Most calculation code is implemented in the rotation.py script, and the functional uses are in regrid_velocity_tracers and regrid_tides functions in the segment class of regional-mom6.


### Calculation Code (rotation.py)
1. **Rotational Method Definition**:  Rotational Methods are defined in the enum class "Rotational Method" in rotation.py.
2. **MOM6 Angle Calculation**: The method is implemented in "mom6_angle_calculation_method" in rotation.py and the direct t-point angle calculation is "initialize_grid_rotation_angle". 
3. **Fred's Pseudo Grid Expansion**: The method to add the additional boundary row/columns is referenced in "pseudo_hgrid" functions in rotation.py

### Implementation Code (regional_mom6.py)
Both regridding functions (regrid_velocity_tracers, regrid_tides) accept a parameter called "rotational_method" which takes the Enum class defining the rotational method.

We then define each method with a bunch of if statements. Here are the processes:

1. Given angle is the default method of accepting the hgrid's angle_dx
2. The EXPAND_GRID method is the least code, and we simply swap out the hgrid angle with the generated one we calculate right where we do the rotation.
