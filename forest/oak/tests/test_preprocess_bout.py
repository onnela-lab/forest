from forest.oak.base import preprocess_bout


def test_preprocess_bout(signal_bout):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[1]
    assert len(vm_bout) == 100
