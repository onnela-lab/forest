"""
Tests for simulate_gps_data
module
"""

import tempfile
import shutil
import sys
import time
import datetime
import requests
import pytest
import numpy as np
from forest.bonsai.simulate_gps_data import get_path
from forest.jasmine.data2mobmat import great_circle_dist


def test_get_path_simple_case():
    """
    Tests simple case of getting a path by car
    """
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
    assert (
        path[0][1] == lat1 and path[-1][0] == lon2 and dist == 843.0532531565476
    )


def test_get_path_close_locations():
    """
    Tests case distance of locations is less than 250 meters
    """
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
