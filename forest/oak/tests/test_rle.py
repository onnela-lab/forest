import numpy as np
import pytest

from forest.oak.base import rle


@pytest.fixture(scope="session")
def signal():
    return np.array(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])


def test_empty():
    run_length, start_ind, val = rle(np.array([]))
    assert run_length is None and start_ind is None and val is None


def test_length(signal):
    run_length, start_ind, val = rle(signal)
    assert len(run_length) == 5 and len(start_ind) == 5 and len(val) == 5


def test_return_run_length(signal):
    run_length = rle(signal)[0]
    expected_output = np.array([3, 11, 1, 5, 1])
    assert np.array_equal(run_length, expected_output)


def test_return_start_ind(signal):
    start_ind = rle(signal)[1]
    expected_output = np.array([0, 3, 14, 15, 20])
    assert np.array_equal(start_ind, expected_output)


def test_return_val(signal):
    val = rle(signal)[2]
    expected_output = np.array([0, 1, 0, 1, 0])
    assert np.array_equal(val, expected_output)
