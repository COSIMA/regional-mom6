import numpy as np
import pytest

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
