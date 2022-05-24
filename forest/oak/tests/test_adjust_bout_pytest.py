import os

import numpy as np
import pandas as pd
import pytest
from scipy import interpolate

from forest.oak.base import adjust_bout


TEST_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def fs():
    return 10


@pytest.fixture(scope="session")
def signal_bout():
    data = pd.read_csv(os.path.join(TEST_DATA_DIR, "test_data_bout.csv"))
    timestamp = np.array(data["timestamp"], dtype="float64")
    timestamp = timestamp/1000
    x = np.array(data["x"], dtype="float64")
    return timestamp, x


def test_adjust_bout(signal_bout):
    timestamp, x = signal_bout
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    f = interpolate.interp1d(timestamp, x)
    x_interp = f(t_interp)

    x_interp_adjust = adjust_bout(x_interp, fs)

    num_seconds = np.floor(len(x_interp)/fs)
    num_seconds_adjust = np.floor(len(x_interp_adjust)/fs)
    assert len(x_interp) == 98
    assert len(x_interp_adjust) == 100
    assert x_interp_adjust[-1] == x_interp[-1]
    assert int(num_seconds) == 9
    assert int(num_seconds_adjust) == 10
