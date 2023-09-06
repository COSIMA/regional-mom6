import numpy as np


def vecdot(v1, v2):
    return np.sum(v1 * v2, axis=-1)
