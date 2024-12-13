import numpy as np
from scipy import interpolate

from forest.oak.base import adjust_bout


def test_adjust_bout(signal_bout, fs):
    timestamp, _, x, y, z = signal_bout
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

    assert len(vm_interp) == 100
    assert np.round(np.mean(vm_interp), 3) == 0.036
    assert np.round(np.std(vm_interp), 3) == 0.229
