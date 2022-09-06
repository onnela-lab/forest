import numpy as np

from forest.oak.base import preprocess_bout


def test_preprocess_bout(signal_bout):
    timestamp, _, x, y, z = signal_bout
    x_bout, y_bout, z_bout, vm_bout = preprocess_bout(timestamp, x, y, z)
    vm_test = np.sqrt(x_bout**2 + y_bout**2 + z_bout**2) - 1
    assert len(x_bout) == 100
    assert len(y_bout) == 100
    assert len(z_bout) == 100
    assert len(vm_bout) == 100
    assert np.array_equal(vm_bout, vm_test)
