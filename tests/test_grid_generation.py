import numpy as np
import pytest
from regional_mom6 import angle_between
from regional_mom6 import quad_area


@pytest.mark.parametrize(
    ("v1", "v2", "v3", "true_angle"),
    [
        ([1, 0, 0], [0, 1, 0], [0, 0, 1], np.pi / 2),
        ([1, 0, 0], [1, 1, 0], [0, 1, 1], np.pi / 4),
    ],
)
def test_angle_between(v1, v2, v3, true_angle):
    assert np.isclose(angle_between(v1, v2, v3), true_angle)


X, Y = np.meshgrid(np.linspace(0, 90, 100), np.linspace(0, 90, 100))
# create a lat-lon mesh that covers 1/4 of the North Hemisphere
lon, lat = np.meshgrid(np.linspace(0, 90, 5), np.linspace(0, 90, 5))


@pytest.mark.parametrize(
    ("lat", "lon", "true_area"),
    [
        (lat, lon, 0.5 * np.pi),
    ],
)
def test_quad_area(lat, lon, true_area):
    assert np.isclose(np.sum(quad_area(lat, lon)), true_area)


# what to return when pytest passes
