import numpy as np
import pytest

from forest.oak.main import find_continuous_dominant_peaks


@pytest.fixture(scope="session")
def test_params1():
    return (5, 0)


@pytest.fixture(scope="session")
def test_params2():
    return (10, 1)


@pytest.fixture(scope="session")
def test_params3():
    return (3, 20)


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


@pytest.fixture(scope="session")
def expected_output1():
    return (np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T)


@pytest.fixture(scope="session")
def expected_output2():
    return (np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).T)


@pytest.fixture(scope="session")
def expected_output3():
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


def test_find_continuous_dominant_peaks_params1(test_input, expected_output1,
                                                test_params1):
    epsilon, delta = test_params1
    test_output = find_continuous_dominant_peaks(test_input, epsilon, delta)
    assert np.array_equal(test_output, expected_output1)


def test_find_continuous_dominant_peaks_params2(test_input, expected_output2,
                                                test_params2):
    epsilon, delta = test_params2
    test_output = find_continuous_dominant_peaks(test_input, epsilon, delta)
    assert np.array_equal(test_output, expected_output2)


def test_find_continuous_dominant_peaks_params3(test_input, expected_output3,
                                                test_params3):
    epsilon, delta = test_params3
    test_output = find_continuous_dominant_peaks(test_input, epsilon, delta)
    assert np.array_equal(test_output, expected_output3)
