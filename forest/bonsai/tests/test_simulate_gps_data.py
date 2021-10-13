"""Tests for simulate_gps_data module"""

import pytest

from forest.bonsai.simulate_gps_data import get_path

coords1 = (51.458498, -2.59638)
coords2 = (51.457619, -2.608466)
coords3 = (51.458492, -2.59635)
api_key = "5b3ce3597851110001cf6248551c505f7c61488a887356ff5ea924d5"


def test_get_path_starting_lattitude():
    assert get_path(coords1, coords2, "car", api_key)[0][0][1] == 51.458498


def test_get_path_ending_longitude():
    assert get_path(coords1, coords2, "car", api_key)[0][-1][0] == -2.608466


def test_get_path_distance():
    assert get_path(coords1, coords2, "car", api_key)[1] == 843.0532531565476


def test_get_path_close_locations():
    """Tests case distance of locations is less than 250 meters."""
    assert len(get_path(coords1, coords3, "foot", api_key)[0]) == 2
