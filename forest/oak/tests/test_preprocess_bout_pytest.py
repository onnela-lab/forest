import math

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
    data = pd.read_csv("test_data_bout.csv")
    timestamp = np.array(data["timestamp"], dtype="float64")
    t = data["UTC time"].tolist()
    x = np.array(data["x"], dtype="float64")  # x-axis acc.
    y = np.array(data["y"], dtype="float64")  # y-axis acc.
    z = np.array(data["z"], dtype="float64")  # z-axis acc.

    timestamp = timestamp/1000
    t = [t_ind.replace("T", " ") for t_ind in t]
    t = [datetime.strptime(t_ind, '%Y-%m-%d %H:%M:%S.%f')
         for t_ind in t]
    return timestamp, t, x, y, z


def test_np_arange(signal, fs):
    timestamp, _, _, _, _ = signal()
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    # check if new sampling fs is within error
    close_to = [math.isclose(d, 1/fs, abs_tol=1e-5) for d in np.diff(t_interp)]
    assert len(t_interp) == 98 and all(close_to)


def test_interpolate(signal, fs):
    timestamp, _, x, _, _ = signal()
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    f = interpolate.interp1d(timestamp, x)
    x_interp = f(t_interp)
    assert (len(x_interp) == 98 and
            np.round(np.mean(x_interp), 3) == -0.761 and
            np.round(np.std(x_interp), 3) == 0.156)


def test_num_seconds(signal, fs):
    timestamp, _, x, _, _ = signal()
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    f = interpolate.interp1d(timestamp, x)
    x_interp = f(t_interp)

    x_interp_adjust = adjust_bout(x_interp, fs)

    num_seconds = np.floor(len(x_interp)/fs)
    num_seconds_adjust = np.floor(len(x_interp_adjust)/fs)
    assert int(num_seconds) == 9 and int(num_seconds_adjust) == 10


def test_vm_bout(signal, fs):
    timestamp, _, x, y, z = signal()
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

    x_interp_adjust = x_interp_adjust[:int(num_seconds*fs)]
    y_interp_adjust = y_interp_adjust[:int(num_seconds*fs)]
    z_interp_adjust = z_interp_adjust[:int(num_seconds*fs)]

    vm_interp = np.sqrt(x_interp_adjust**2 +
                        y_interp_adjust**2 +
                        z_interp_adjust**2) - 1

    assert (len(vm_interp) == 100 and
            np.round(np.mean(vm_interp), 3) == 0.036 and
            np.round(np.std(vm_interp), 3) == 0.229)


def test_preprocess_bout(signal):
    timestamp, _, x, y, z = signal()
    x_bout, y_bout, z_bout, vm_bout = preprocess_bout(timestamp, x, y, z)
    vm_test = np.sqrt(x_bout**2 + y_bout**2 + z_bout**2) - 1
    assert (len(x_bout) == 100 and len(y_bout) == 100 and len(z_bout) == 100
            and len(vm_bout) == 100 and
            np.array_equal(vm_bout, vm_test))
