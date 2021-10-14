"""Tests for simulate_gps_data module"""

import pytest

from forest.bonsai.simulate_gps_data import get_path


@pytest.fixture(scope="session")
def coords1():
    return 51.458498, -2.59638


@pytest.fixture(scope="session")
def coords2():
    return 51.457619, -2.608466


@pytest.fixture(scope="session")
def coords3():
    return 51.458492, -2.59635


@pytest.fixture(scope="session")
def api_key():
    return "5b3ce3597851110001cf6248551c505f7c61488a887356ff5ea924d5"


def test_get_path_starting_lattitude(coords1, coords2, api_key):
    assert get_path(coords1, coords2, "car", api_key)[0][0][0] == 51.458498


def test_get_path_ending_longitude(coords1, coords2, api_key):
    assert get_path(coords1, coords2, "car", api_key)[0][-1][1] == -2.608466


def test_get_path_distance(coords1, coords2, api_key):
    assert get_path(coords1, coords2, "car", api_key)[1] == 843.0532531565476


def test_get_path_close_locations(coords1, coords3, api_key):
    """Tests case distance of locations is less than 250 meters."""
    assert len(get_path(coords1, coords3, "foot", api_key)[0]) == 2
