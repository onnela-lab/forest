import numpy as np


from forest.oak.base import preprocess_bout, find_walking


def test_find_walking(signal_bout, fs, min_amp, step_freq, alpha, beta,
                      min_t, delta):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[1]
    cadence_bout = find_walking(vm_bout, fs, min_amp, step_freq, alpha, beta,
                                min_t, delta)
    expected_output = np.array([1.65, 1.6, 1.55, 1.6, 1.55, 1.85, 1.8, 1.75,
                                1.75, 1.7])
    assert len(cadence_bout) == 10
    assert np.array_equal(np.round(cadence_bout, 2), expected_output)
