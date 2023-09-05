import numpy as np
import pytest
from regional_mom6 import angle_between
from regional_mom6 import quadilateral_area
from regional_mom6 import quadilateral_areas
from regional_mom6 import rectangular_hgrid
import xarray as xr


@pytest.mark.parametrize(
    ("v1", "v2", "v3", "true_angle"),
    [
        ([1, 0, 0], [0, 1, 0], [0, 0, 1], np.pi / 2),
        ([1, 0, 0], [1, 1, 0], [0, 1, 1], np.pi / 4),
        ([1, 0, 0], [1, 1, 1], [0, 0, 1], np.pi / 4),
        ([1, 1, 1], [1, 1, 0], [0, 1, 1], 2 * np.pi / 3)
    ],
)
def test_angle_between(v1, v2, v3, true_angle):
    assert np.isclose(angle_between(v1, v2, v3), true_angle)


# create a lat-lon mesh that covers 1/4 of the North Hemisphere
lon1, lat1 = np.meshgrid(np.linspace(0, 90, 5), np.linspace(0, 90, 5))

# create a lat-lon mesh that covers 1/4 of the whole globe
lon2, lat2 = np.meshgrid(np.linspace(-45, 45, 5), np.linspace(-90, 90, 5))


@pytest.mark.parametrize(
    ("lat", "lon", "true_area"),
    [
        (lat1, lon1, 0.5 * np.pi),
        (lat2, lon2, np.pi),
    ],
)
def test_quadilateral_areas(lat, lon, true_area):
    assert np.isclose(np.sum(quadilateral_areas(lat, lon)), true_area)


# a simple test that rectangular_hgrid runs without erroring
@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (np.linspace(0, 10, 7), np.linspace(0, 10, 13)),
    ],
)
def test_rectangular_hgrid(lat, lon):
    assert isinstance(rectangular_hgrid(lat, lon), xr.Dataset)
