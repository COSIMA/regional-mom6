import numpy as np


def vecdot(v1, v2):
    """Return the dot product of vectors ``v1`` and ``v2``.
    ``v1`` and ``v2`` can be either numpy vectors or numpy.ndarrays
    in which case the last dimension is considered the dimension
    over which the dot product is taken of.
    """
    return np.sum(v1 * v2, axis=-1)


def angle_between(v1, v2, v3):
    """Returns the angle v2-v1-v3 (in radians). That is the angle between vectors v1-v2 and v1-v3.

    Example:

        >>> from regional_mom6.utils import angle_between
        >>> v1 = (0, 0, 1)
        >>> v2 = (1, 0, 0)
        >>> v3 = (0, 1, 0)
        >>> angle_between(v1, v2, v3)
        1.5707963267948966
        >>> from numpy import rad2deg
        >>> rad2deg(angle_between(v1, v2, v3))
        90.0
    """

    v1xv2 = np.cross(v1, v2)
    v1xv3 = np.cross(v1, v3)

    norm_v1xv2 = np.sqrt(vecdot(v1xv2, v1xv2))
    norm_v1xv3 = np.sqrt(vecdot(v1xv3, v1xv3))

    cosangle = vecdot(v1xv2, v1xv3) / (norm_v1xv2 * norm_v1xv3)

    return np.arccos(cosangle)


def quadrilateral_area(v1, v2, v3, v4):
    """Returns area of a spherical quadrilateral on the unit sphere that
    has vertices on 3-vectors `v1`, `v2`, `v3`, `v4` (counter-clockwise
    orientation is implied). The area is computed via the excess of the
    sum of the spherical angles of the quadrilateral from 2π.

    Example:

        Calculate the area that corresponds to half the Northern hemisphere
        of a sphere of radius *R*. This should be 1/4 of the sphere's total area,
        that is π *R*:sup:`2`.

        >>> from regional_mom6.utils import quadrilateral_area, latlon_to_cartesian
        >>> R = 434.3
        >>> v1 = latlon_to_cartesian(0, 0, R)
        >>> v2 = latlon_to_cartesian(0, 90, R)
        >>> v3 = latlon_to_cartesian(90, 0, R)
        >>> v4 = latlon_to_cartesian(0, -90, R)
        >>> quadrilateral_area(v1, v2, v3, v4)
        592556.1793298927
        >>> from numpy import pi
        >>> quadrilateral_area(v1, v2, v3, v4) == pi * R**2
        True
    """

    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    v4 = np.array(v4)

    if not (
        np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v3, v3)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v4, v4)))
    ):
        raise ValueError("vectors provided must have the same length")

    R = np.sqrt(vecdot(v1, v1))

    a1 = angle_between(v1, v2, v4)
    a2 = angle_between(v2, v3, v1)
    a3 = angle_between(v3, v4, v2)
    a4 = angle_between(v4, v1, v3)

    return (a1 + a2 + a3 + a4 - 2 * np.pi) * R**2


def latlon_to_cartesian(lat, lon, R=1):
    """Convert latitude-longitude (in degrees) to Cartesian coordinates on a
    sphere of radius `R`. By default `R = 1`.

    Examples:

        Find the Cartesian coordinates that correspond to point with
        ``(lat, lon) = (0, 0)`` on a sphere with unit radius.

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> latlon_to_cartesian(0, 0)
        (1.0, 0.0, 0.0)

        Now let's do the same on a sphere with Earth's radius

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> R = 6371e3
        >>> latlon_to_cartesian(0, 0, R)
        (6371000.0, 0.0, 0.0)
    """

    x = R * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = R * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = R * np.sin(np.deg2rad(lat))

    return x, y, z


def quadrilateral_areas(lat, lon, R=1):
    """Returns area of spherical quadrilaterals on a sphere of radius `R`. By default, `R = 1`.
    The quadrilaterals are formed by constant latitude and longitude lines on the `lat`-`lon` grid provided.

    Args:
        lat (array): Array of latitude points (in degrees)
        lon (array): Array of longitude points (in degrees)

    Returns:
        areas (array): Array with the areas of the quadrilaterals defined by the
                       `lat`-`lon` grid provided. If the `lat`-`lon` are `m x n`
                       then `areas` is `(m-1) x (n-1)`.

    Example:

        Let's construct a lat-lon grid on the sphere with 60 degree spacing.
        Then we compute the areas of each grid cell and confirm that the
        sum of the areas gives us the total area of the sphere.

        >>> from regional_mom6.utils import quadrilateral_areas
        >>> import numpy as np
        >>> λ = np.linspace(0, 360, 7)
        >>> φ = np.linspace(-90, 90, 4)
        >>> lon, lat = np.meshgrid(λ, φ)
        >>> lon
        array([[  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.]])
        >>> lat
        array([[-90., -90., -90., -90., -90., -90., -90.],
               [-30., -30., -30., -30., -30., -30., -30.],
               [ 30.,  30.,  30.,  30.,  30.,  30.,  30.],
               [ 90.,  90.,  90.,  90.,  90.,  90.,  90.]])
        >>> R = 6371e3
        >>> areas = quadrilateral_areas(lat, lon, R)
        >>> areas
        array([[1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13],
               [4.56284230e+13, 4.56284230e+13, 4.56284230e+13, 4.56284230e+13,
                4.56284230e+13, 4.56284230e+13],
               [1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13]])
        >>> np.isclose(areas.sum(), 4 * np.pi * R**2, atol=np.finfo(areas.dtype).eps)
        True
    """

    coords = np.dstack(latlon_to_cartesian(lat, lon, R))

    return quadrilateral_area(
        coords[:-1, :-1, :], coords[:-1, 1:, :], coords[1:, 1:, :], coords[1:, :-1, :]
    )
