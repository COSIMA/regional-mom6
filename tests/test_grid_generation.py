import regional_mom6; nlayers, target_depth = 10, 0.4; ratio = 4; dz = regional_mom6.hyperbolictan_thickness_profile(nlayers, ratio, target_depth); dz

import numpy as np
import pytest
from regional_mom6 import hyperbolictan_thickness_profile
from regional_mom6 import angle_between
from regional_mom6 import latlon_to_cartesian
from regional_mom6 import quadrilateral_area
from regional_mom6 import quadrilateral_areas
from regional_mom6 import rectangular_hgrid
import xarray as xr


@pytest.mark.parametrize(
    ("nlayers", "ratio", "target_depth"),
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
def test_hyperbolictan_thickness_profile_symmetric(nlayers, ratio, target_depth):
    assert np.isclose(
        hyperbolictan_thickness_profile(nlayers, ratio, target_depth),
        np.flip(hyperbolictan_thickness_profile(nlayers, 1 / ratio, target_depth)),
    ).all()

@pytest.mark.parametrize(
    ("nlayers", "target_depth"),
    [
        (23, 2000),
        (50, 1000),
        (50, 3000),
    ],
)
def test_hyperbolictan_thickness_profile_equispaced(nlayers, target_depth):
    assert np.isclose(
        hyperbolictan_thickness_profile(nlayers, 1, target_depth),
        np.ones(nlayers) * target_depth / nlayers,
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
    assert np.isclose(latlon_to_cartesian(lat, lon), true_xyz).all()


@pytest.mark.parametrize(
    ("v1", "v2", "v3", "true_angle"),
    [
        ([1, 0, 0], [0, 1, 0], [0, 0, 1], np.pi / 2),
        ([1, 0, 0], [1, 1, 0], [0, 1, 1], np.pi / 4),
        ([1, 0, 0], [1, 1, 1], [0, 0, 1], np.pi / 4),
        ([1, 1, 1], [1, 1, 0], [0, 1, 1], 2 * np.pi / 3),
    ],
)
def test_angle_between(v1, v2, v3, true_angle):
    assert np.isclose(angle_between(v1, v2, v3), true_angle)


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
    assert np.isclose(quadrilateral_area(v1, v2, v3, v4), true_area)


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
    assert np.isclose(np.sum(quadrilateral_areas(lat, lon)), true_area)


# a simple test that rectangular_hgrid runs without erroring
@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (np.linspace(0, 10, 7), np.linspace(0, 10, 13)),
    ],
)
def test_rectangular_hgrid(lat, lon):
    assert isinstance(rectangular_hgrid(lat, lon), xr.Dataset)
