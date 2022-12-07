import numpy as np

from forest.oak.base import preprocess_bout


def test_get_pp(signal_bout, fs):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[1]
    vm_res_sec = vm_bout.reshape((fs, -1), order="F")
    pp = np.array([max(vm_res_sec[:, i])-min(vm_res_sec[:, i])
                   for i in range(vm_res_sec.shape[1])])
    expected_output = np.array([0.64, 0.71, 1.11, 0.79, 0.37, 0.70, 1.20, 0.61,
                                0.66, 0.48])
    assert np.array_equal(np.round(pp, 2), expected_output)
