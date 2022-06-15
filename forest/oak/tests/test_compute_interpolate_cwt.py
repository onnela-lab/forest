import math

import numpy as np

from forest.oak.base import preprocess_bout, compute_interpolate_cwt


def test_compute_interpolate_cwt(signal_bout, fs, wavelet):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[3]

    freqs_interp, coefs_interp = compute_interpolate_cwt(vm_bout, fs, wavelet)

    expected_output_amp = 0.000652
    expected_output_freqs = np.arange(0.5, 4.5, 0.05)
    assert math.isclose(np.max(coefs_interp), expected_output_amp,
                        abs_tol=1e-4)
    assert np.array_equal(freqs_interp, expected_output_freqs)
