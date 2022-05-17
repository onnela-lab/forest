import math
from sklearn.metrics import mean_squared_error

from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from scipy import interpolate

from forest.oak.main import preprocess_bout
from forest.oak.main import adjust_bout


@pytest.fixture(scope="session")
def fs():
    fs = 10
    return fs


@pytest.fixture(scope="session")
def gravity():
    gravity = 9.80665
    return gravity


@pytest.fixture(scope="session")
def signal():
    data = pd.read_csv("test_data.csv")
    timestamp, t, _, x, y, z = data.T.to_numpy()
    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    timestamp = timestamp/1000
    t = [t_ind.replace("T", " ") for t_ind in t]
    t = [datetime.strptime(t_ind, '%Y-%m-%d %H:%M:%S.%f')
         for t_ind in t]
    return timestamp, t, x, y, z


def test_np_arange(timestamp, fs):
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    # check if new sampling fs is within error
    close_to = [math.isclose(d, 1/fs, abs_tol=1e-5) for d in np.diff(t_interp)]
    assert len(t_interp) == 98 and all(close_to)


def test_interpolate(timestamp, x, fs):
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    f = interpolate.interp1d(timestamp, x)
    x_interp = f(t_interp)
    rms = np.sqrt(mean_squared_error(x, x_interp))
    assert (len(x_interp) == 98 and
            rms > 0 and rms < 0.05 and
            np.round(np.mean(x_interp), 3) == -0.761 and
            np.round(np.std(x_interp), 3) == 0.156)


def test_vm_bout(timestamp, x, y, z, fs):
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    f = interpolate.interp1d(timestamp, x)
    x_interp = f(t_interp)

    f = interpolate.interp1d(timestamp, y)
    y_interp = f(t_interp)

    f = interpolate.interp1d(timestamp, z)
    z_interp = f(t_interp)

    x_interp_adjust = adjust_bout(x_interp, fs)
    y_interp_adjust = adjust_bout(y_interp, fs)
    z_interp_adjust = adjust_bout(z_interp, fs)

    num_seconds = np.floor(len(x_interp_adjust)/fs)

    x_interp_adjust = x_interp_adjust[:int(num_seconds*fs)].astype(float)
    y_interp_adjust = y_interp_adjust[:int(num_seconds*fs)].astype(float)
    z_interp_adjust = z_interp_adjust[:int(num_seconds*fs)].astype(float)

    vm_interp = np.sqrt(x_interp_adjust**2 +
                        y_interp_adjust**2 +
                        z_interp_adjust**2) - 1

    assert (len(vm_interp) == 100 and
            np.round(np.mean(vm_interp), 3) == 0.036 and
            np.round(np.std(vm_interp), 3) == 0.229)


def test_num_seconds(timestamp, x, fs):
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    f = interpolate.interp1d(timestamp, x)
    x_interp = f(t_interp)

    x_interp_adjust = adjust_bout(x_interp, fs)

    num_seconds = np.floor(len(x_interp)/fs)
    num_seconds_adjust = np.floor(len(x_interp_adjust)/fs)
    assert int(num_seconds) == 9 and int(num_seconds_adjust) == 10
