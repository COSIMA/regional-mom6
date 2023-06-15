import numpy as np
from mom6_regional import angle_between


# placeholder trivial test test
def test_angle_between():
    assert np.isclose(angle_between([1, 0, 0], [0, 1, 0], [0, 0, 1]), np.pi / 2)
