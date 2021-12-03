"""Tests for traj2stats summary statistics in Jasmine"""

import pytest
from shapely.geometry import Point

from forest.jasmine.data2mobmat import great_circle_dist
from forest.jasmine.traj2stats import transform_point_to_circle


@pytest.fixture()
def coords1():
    return 51.457183, -2.597960


@pytest.fixture()
def coords2():
    return 51.457267, -2.598045


def test_transform_point_to_circle_simple_case(coords1, coords2):
    """Testing creating a circle
    from a center point in coordinates
    """
    circle1 = transform_point_to_circle(coords1, 15)
    point2 = Point(coords2)
    assert circle1.contains(point2)


def test_transform_point_to_circle_zero_radius(coords1):
    """Testing creating a circle from
    a center point in coordinates with zero radius
    """
    circle1 = transform_point_to_circle(coords1, 0)
    assert len(circle1.exterior.coords) == 0


def test_transform_point_to_circle_radius(coords1):
    """Testing creating a circle from a center point
    in coordinates and checking radius is approximately correct
    """

    circle1 = transform_point_to_circle(coords1, 5)
    point_in_edge = [
        circle1.exterior.coords.xy[0][2], circle1.exterior.coords.xy[1][2]
        ]

    distance = great_circle_dist(
        *coords1, *point_in_edge
    )
    assert distance >= 4 and distance <= 5
