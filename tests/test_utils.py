import numpy as np
import pytest

from regional_mom6.utils import vecdot, angle_between, get_edge

V1 = np.zeros((2, 4, 3))
V2 = np.zeros((2, 4, 3))
true_V1dotV2 = np.zeros((2, 4))

for i in np.arange(2):
    for j in np.arange(4):
        sum = 0
        for k in np.arange(3):
            V1[i, j, k] = -1.0 * np.array(2 * i - 3 * j + 2 * k)
            V2[i, j, k] = 1.0 * np.array(i + j + k + 1)
            sum += -(2 * i - 3 * j + 2 * k) * (i + j + k + 1)
            true_V1dotV2[i, j] = float(sum)


@pytest.mark.parametrize(
    ("v1", "v2", "true_v1dotv2"),
    [
        (np.array((1.0, 2.0, 3.0)), np.array((-2.0, -3.0, 4.0)), 4.0),
        (V1, V2, true_V1dotV2),
    ],
)
def test_vecdot(v1, v2, true_v1dotv2):
    assert np.isclose(vecdot(v1, v2), true_v1dotv2, atol=np.finfo(v1.dtype).eps).all()


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
    assert np.isclose(angle_between(v1, v2, v3).item(), true_angle)


def test_get_edge(get_rectilinear_hgrid):
    hgrid = get_rectilinear_hgrid
    res = get_edge(hgrid, "north")
    assert (res.x == hgrid.x.isel(nyp=-1)).all()
    res = get_edge(hgrid, "south")
    assert (res.x == hgrid.x.isel(nyp=0)).all()
    res = get_edge(hgrid, "east")
    assert (res.x == hgrid.x.isel(nxp=-1)).all()
    res = get_edge(hgrid, "west")
    assert (res.x == hgrid.x.isel(nxp=0)).all()
