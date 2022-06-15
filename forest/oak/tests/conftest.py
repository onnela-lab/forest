from datetime import datetime
import os

import numpy as np
import pandas as pd
import pytest


TEST_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def fs():
    return 10


@pytest.fixture(scope="module")
def wavelet():
    return 'gmw', {'beta': 90, 'gamma': 3}


@pytest.fixture(scope="module")
def min_amp():
    return 0.3


@pytest.fixture(scope="module")
def step_freq():
    return 1.4, 2.3


@pytest.fixture(scope="module")
def alpha():
    return 0.6


@pytest.fixture(scope="module")
def beta():
    return 2.5


@pytest.fixture(scope="module")
def delta():
    return 20


@pytest.fixture(scope="module")
def min_t():
    return 3


@pytest.fixture(scope="module")
def signal_bout():
    data = pd.read_csv(os.path.join(TEST_DATA_DIR, "test_data_bout.csv"))
    timestamp = np.array(data["timestamp"], dtype="float64") / 1000
    t = data["UTC time"].tolist()
    x = np.array(data["x"], dtype="float64")
    y = np.array(data["y"], dtype="float64")
    z = np.array(data["z"], dtype="float64")

    t = [t_ind.replace("T", " ") for t_ind in t]
    t = [datetime.strptime(t_ind, '%Y-%m-%d %H:%M:%S.%f') for t_ind in t]
    return timestamp, t, x, y, z
