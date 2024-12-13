import numpy as np
import pytest

from forest.oak.base import find_continuous_dominant_peaks


@pytest.fixture(scope="session")
def test_input():
    return (np.array([[0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                      [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).T)


def test_find_continuous_dominant_peaks_params1(test_input):
    test_output = find_continuous_dominant_peaks(test_input, 5, 0)
    expected_output = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    assert np.array_equal(test_output, expected_output)


def test_find_continuous_dominant_peaks_params2(test_input):
    test_output = find_continuous_dominant_peaks(test_input, 10, 1)
    expected_output = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).T
    assert np.array_equal(test_output, expected_output)


def test_find_continuous_dominant_peaks_params3(test_input):
    test_output = find_continuous_dominant_peaks(test_input, 3, 20)
    expected_output = np.array([[0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                                [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).T
    assert np.array_equal(test_output, expected_output)
