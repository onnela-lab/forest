import numpy as np
from scipy import interpolate

from forest.oak.base import adjust_bout


def test_adjust_bout(signal_bout, fs):
    timestamp, _, x = signal_bout
    t_interp = np.arange(timestamp[0], timestamp[-1], (1/fs))
    f = interpolate.interp1d(timestamp, x)
    x_interp = f(t_interp)
    x_interp_adjust = adjust_bout(x_interp, fs)
    num_seconds_adjust = np.floor(len(x_interp_adjust)/fs)
    assert len(x_interp_adjust) == 100
    assert x_interp_adjust[-1] == x_interp[-1]
    assert int(num_seconds_adjust) == 10
