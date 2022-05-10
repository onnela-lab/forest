# test_rle_pytest.py

import sys

import numpy as np
import pytest

sys.path.insert(0, 'C:/Users/mstra/Documents/Python/forest-private/forest/oak')
from oak_main import rle


@pytest.fixture(scope="session")
def signal_empty():
    return np.array([])


@pytest.fixture(scope="session")
def signal():
    return np.array(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])


def test_empty(signal_empty):
    out1, out2, out3 = rle(signal_empty)
    assert out1 is None and out2 is None and out3 is None


def test_length(signal):
    out1, out2, out3 = rle(signal)
    assert len(out1) == 5 and len(out2) == 5 and len(out3) == 5


def test_return0(signal):
    out1 = rle(signal)[0]
    expected_output = np.array([3, 11, 1, 5, 1])
    assert (out1 == expected_output).all() is True


def test_return1(signal):
    out2 = rle(signal)[1]
    expected_output = np.array([0, 3, 14, 15, 20])
    assert (out2 == expected_output).all() is True


def test_return2(signal):
    out3 = rle(signal)[2]
    expected_output = np.array([0, 1, 0, 1, 0])
    assert (out3 == expected_output).all() is True
