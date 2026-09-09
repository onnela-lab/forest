import sys
from datetime import datetime
from os.path import abspath, dirname, join as pathjoin

import numpy as np
import pandas as pd
import pytest


# insert the root of the repo folder into the python path so we can just import the codebase without
# relative import issues or needing to install the beiwe-forest package.
repo_root = abspath(dirname(dirname(abspath(__file__))))
src_root = abspath(pathjoin(repo_root, "src"))
forest_root = abspath(pathjoin(repo_root, "src", "forest"))

if src_root not in sys.path:
    sys.path.insert(0, src_root)


# FIXME: this is purely a migration artifact, oak expects this folder, probably just from
# signal_bout below. Merge and and all the test configuration into this file.
TEST_DATA_DIR = pathjoin(dirname(abspath(__file__)), "oak")


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
    data = pd.read_csv(pathjoin(TEST_DATA_DIR, "test_data_bout.csv"))
    timestamp = np.array(data["timestamp"], dtype="float64") / 1000
    t = data["UTC time"].tolist()
    x = np.array(data["x"], dtype="float64")
    y = np.array(data["y"], dtype="float64")
    z = np.array(data["z"], dtype="float64")

    t = [t_ind.replace("T", " ") for t_ind in t]
    t = [datetime.strptime(t_ind, '%Y-%m-%d %H:%M:%S.%f') for t_ind in t]
    return timestamp, t, x, y, z
