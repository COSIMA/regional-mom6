import pytest

import numpy as np
import os

import mom6_regional


@pytest.fixture
def tmp_chdir(tmp_path):
    """Change to tmp_path, saving current directory.

    This restores the current directory when the test finishes.
    """

    original_dir = os.getcwd()
    os.chdir(tmp_path)

    yield True

    os.chdir(original_dir)


@pytest.fixture(scope="session")
def simple_test():
    """This better returns true.
    """
    a = 1

    return a == 1
