"""
Tests for simulate_gps_data
module
"""

import pytest

from forest.bonsai.simulate_gps_data import get_path


@pytest.fixture(scope="session")
def path1():
    """Random path"""

    lat1 = 51.458498
    lon1 = -2.59638
    lat2 = 51.457619
    lon2 = -2.608466
    path, dist = get_path(
        lat1,
        lon1,
        lat2,
        lon2,
        "car",
        "5b3ce3597851110001cf6248551c505f7c61488a887356ff5ea924d5",
    )

    return path, dist

def test_get_path_simple_case1(path1):
    """Tests ending lattitude."""

    assert path1[0][0][1] == 51.458498

def test_get_path_simple_case2(path1):
    """Tests ending longitude of path."""

    assert path1[0][-1][0] == -2.608466

def test_get_path_simple_case3(path1):
    """Tests distance of path."""

    assert path1[1] == 843.0532531565476


def test_get_path_close_locations():
    """Tests case distance of locations is less than 250 meters."""
    lat1 = 51.458498
    lon1 = -2.59638
    lat2 = 51.458492
    lon2 = -2.59635
    path, dist = get_path(
        lat1,
        lon1,
        lat2,
        lon2,
        "foot",
        "5b3ce3597851110001cf6248551c505f7c61488a887356ff5ea924d5",
    )
    assert len(path) == 2
