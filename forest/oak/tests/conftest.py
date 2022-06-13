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
def signal_bout():
    data = pd.read_csv(os.path.join(TEST_DATA_DIR, "test_data_bout.csv"))
    timestamp = np.array(data["timestamp"], dtype="float64")
    t = data["UTC time"].tolist()
    x = np.array(data["x"], dtype="float64")
    y = np.array(data["y"], dtype="float64")
    z = np.array(data["z"], dtype="float64")

    timestamp = timestamp/1000
    t = [t_ind.replace("T", " ") for t_ind in t]
    t = [datetime.strptime(t_ind, '%Y-%m-%d %H:%M:%S.%f') for t_ind in t]
    return timestamp, t, x, y, z
