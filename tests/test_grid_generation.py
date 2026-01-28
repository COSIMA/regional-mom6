import numpy as np
import pytest

from regional_mom6 import hyperbolictan_thickness_profile
from regional_mom6 import generate_rectangular_hgrid
from regional_mom6 import longitude_slicer

from regional_mom6.utils import angle_between
from regional_mom6.utils import latlon_to_cartesian
from regional_mom6.utils import quadrilateral_area
from regional_mom6.utils import quadrilateral_areas

import xarray as xr

## Note:
## When creating test dataarrays we use 'silly' names for coordinates to
## ensure that the proper mapping to MOM6 names occurs correctly


@pytest.mark.parametrize(
    ("nlayers", "ratio", "total_depth"),
    [
        (20, 1 / 3, 1000),
        (20, 2, 1000),
        (20, 10, 1000),
        (20, 2, 3000),
        (50, 1 / 3, 1000),
        (50, 2, 1000),
        (50, 10, 1000),
        (50, 2, 3000),
    ],
)
def test_hyperbolictan_thickness_profile_symmetric(nlayers, ratio, total_depth):
    assert np.isclose(
        hyperbolictan_thickness_profile(nlayers, ratio, total_depth),
        np.float64(
            np.flip(
                hyperbolictan_thickness_profile(
                    nlayers, 1 / ratio, np.float64(total_depth)
                )
            )
        ),
    ).all()


@pytest.mark.parametrize(
    ("nlayers", "total_depth"),
    [
        (23, 2000),
        (50, 1000),
        (50, 3000),
    ],
)
def test_hyperbolictan_thickness_profile_equispaced(nlayers, total_depth):
    assert np.isclose(
        hyperbolictan_thickness_profile(nlayers, 1, total_depth),
        np.float64(np.ones(nlayers) * total_depth / nlayers),
    ).all()


@pytest.mark.parametrize(
    ("lat", "lon", "true_xyz"),
    [
        (0, 0, (1, 0, 0)),
        (90, 0, (0, 0, 1)),
        (0, 90, (0, 1, 0)),
        (-90, 0, (0, 0, -1)),
    ],
)
def test_latlon_to_cartesian(lat, lon, true_xyz):
    assert np.isclose(latlon_to_cartesian(lat, lon), np.float64(true_xyz)).all()


@pytest.mark.parametrize(
    ("v1", "v2", "v3", "v4", "true_area"),
    [
        (
            np.dstack(latlon_to_cartesian(0, 0)),
            np.dstack(latlon_to_cartesian(0, 90)),
            np.dstack(latlon_to_cartesian(90, 0)),
            np.dstack(latlon_to_cartesian(0, -90)),
            np.pi,
        ),
        (
            np.dstack(latlon_to_cartesian(0, 0)),
            np.dstack(latlon_to_cartesian(90, 0)),
            np.dstack(latlon_to_cartesian(0, 90)),
            np.dstack(latlon_to_cartesian(-90, 0)),
            np.pi,
        ),
    ],
)
def test_quadrilateral_area(v1, v2, v3, v4, true_area):
    rhs = np.float64(quadrilateral_area(v1, v2, v3, v4))
    lhs = np.float64(true_area)
    assert np.isclose(rhs, lhs)


#    assert np.isclose(quadrilateral_area(v1, v2, v3, v4), np.float64(true_area))


v1 = np.dstack(latlon_to_cartesian(0, 0, R=2))
v2 = np.dstack(latlon_to_cartesian(90, 0, R=2))
v3 = np.dstack(latlon_to_cartesian(0, 90, R=2))
v4 = np.dstack(latlon_to_cartesian(-90, 0, R=2.1))


def test_quadrilateral_area_exception():
    with pytest.raises(ValueError) as excinfo:
        quadrilateral_area(v1, v2, v3, v4)

    assert str(excinfo.value) == "vectors provided must have the same length"


# create a lat-lon mesh that covers 1/4 of the North Hemisphere
lon1, lat1 = np.meshgrid(np.linspace(0, 90, 5), np.linspace(0, 90, 5))
area1 = 1 / 8 * (4 * np.pi)

# create a lat-lon mesh that covers 1/4 of the whole globe
lon2, lat2 = np.meshgrid(np.linspace(-45, 45, 5), np.linspace(-90, 90, 5))
area2 = 1 / 4 * (4 * np.pi)


@pytest.mark.parametrize(
    ("lat", "lon", "true_area"),
    [
        (lat1, lon1, area1),
        (lat2, lon2, area2),
    ],
)
def test_quadrilateral_areas(lat, lon, true_area):
    assert np.isclose(
        np.sum(quadrilateral_areas(lat, lon)).item(), np.float64(true_area)
    )


# a simple test that rectangular_hgrid runs without erroring
@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (np.linspace(0, 10, 7), np.linspace(0, 10, 13)),
    ],
)
def test_rectangular_hgrid(lat, lon):
    assert isinstance(generate_rectangular_hgrid(lon, lat), xr.Dataset)


def test_longitude_slicer():
    with pytest.raises(AssertionError):
        nx, ny, nt = 4, 14, 5

        latitude_extent = (10, 20)
        longitude_extent = (12, 18)

        dims = ["silly_lat", "silly_lon", "time"]

        dλ = (longitude_extent[1] - longitude_extent[0]) / 2

        data = xr.DataArray(
            np.random.random((ny, nx, nt)),
            dims=dims,
            coords={
                "silly_lat": np.linspace(latitude_extent[0], latitude_extent[1], ny),
                "silly_lon": np.array(
                    [
                        longitude_extent[0],
                        longitude_extent[0] + 1.5 * dλ,
                        longitude_extent[0] + 2.6 * dλ,
                        longitude_extent[1],
                    ]
                ),
                "time": np.linspace(0, 1000, nt),
            },
        )

        longitude_slicer(data, longitude_extent, "silly_lon")
