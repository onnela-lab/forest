import numpy as np

from forest.oak.base import (preprocess_bout, compute_interpolate_cwt,
                             identify_peaks_in_cwt)


def test_identify_peaks_in_cwt(signal_bout, fs, min_amp, step_freq, alpha,
                               beta, wavelet):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[1]

    freqs_interp, coefs_interp = compute_interpolate_cwt(vm_bout, fs, wavelet)

    dp = identify_peaks_in_cwt(freqs_interp, coefs_interp, fs, step_freq,
                               alpha, beta)

    expected_output_val = np.ones(10)
    expected_output_ind = np.array([23, 22, 21, 22, 21, 27, 26, 25, 25, 24])
    assert np.array_equal(np.argmax(dp, axis=0), expected_output_ind)
    assert np.array_equal(np.max(dp, axis=0), expected_output_val)
