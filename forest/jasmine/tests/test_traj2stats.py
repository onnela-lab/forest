"""Tests for traj2stats summary statistics in Jasmine"""

import numpy as np
import pytest
from shapely.geometry import Point

from forest.jasmine.data2mobmat import great_circle_dist
from forest.jasmine.traj2stats import (
    Frequency, gps_summaries, Hyperparameters, transform_point_to_circle
)


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
    circle1 = transform_point_to_circle(*coords1, 15)
    point2 = Point(coords2)
    assert circle1.contains(point2)


def test_transform_point_to_circle_zero_radius(coords1):
    """Testing creating a circle from
    a center point in coordinates with zero radius
    """
    circle1 = transform_point_to_circle(*coords1, 0)
    assert len(circle1.exterior.coords) == 0


def test_transform_point_to_circle_radius(coords1):
    """Testing creating a circle from a center point
    in coordinates and checking radius is approximately correct
    """

    circle1 = transform_point_to_circle(*coords1, 5)
    point_in_edge = [
        circle1.exterior.coords.xy[0][2],
        circle1.exterior.coords.xy[1][2],
    ]

    distance = great_circle_dist(*coords1, *point_in_edge)[0]
    assert distance >= 4 and distance <= 5


@pytest.fixture()
def sample_trajectory():
    """16 minutes of a random trajectory"""
    return np.array(
        [
            [
                2,
                51.45435654,
                -2.58555554,
                1633042800,
                51.45435654,
                -2.58555554,
                1633082400,
                1,
            ],
            [
                1,
                51.45435654,
                -2.58555554,
                1633082400,
                51.4539284,
                -2.5861665,
                1633083300,
                1,
            ],
            [
                2,
                51.4539284,
                -2.5861665,
                1633083300,
                51.4539284,
                -2.5861665,
                1633086900,
                1,
            ],
            [
                1,
                51.4539284,
                -2.5861665,
                1633086900,
                51.45435654,
                -2.58555554,
                1633087800,
                1,
            ],
            [
                2,
                51.45435654,
                -2.58555554,
                1633087800,
                51.45435654,
                -2.58555554,
                1633105800,
                1,
            ],
            [
                1,
                51.45435654,
                -2.58555554,
                1633105800,
                51.4542473,
                -2.5861116,
                1633107600,
                1,
            ],
            [
                2,
                51.4542473,
                -2.5861116,
                1633107600,
                51.4542473,
                -2.5861116,
                1633118400,
                1,
            ],
            [
                1,
                51.4542473,
                -2.5861116,
                1633118400,
                51.45435654,
                -2.58555554,
                1633120300,
                1,
            ],
            [
                2,
                51.45435654,
                -2.58555554,
                1633120300,
                51.45435654,
                -2.58555554,
                1633129500,
                1,
            ],
        ]
    )


@pytest.fixture()
def sample_nearby_locations():
    ids = {
        "post_box": [560374554],
        "vending_machine": [6239576723, 6239576724],
        "bicycle_parking": [6868356189, 8942254228],
        "music_school": [8942254270],
        "studio": [8942254271],
        "parking_entrance": [8942254275],
        "pub": [126489283],
        "fast_food": [290604903],
    }
    locations = {
        560374554: [[51.4541068, -2.5850407]],
        6239576723: [[51.4545961, -2.5854302]],
        6239576724: [[51.4546918, -2.585846]],
        6868356189: [[51.4542648, -2.5858638]],
        8942254228: [[51.4539812, -2.585332]],
        8942254270: [[51.4539735, -2.5859814]],
        8942254271: [[51.4539284, -2.5861665]],
        8942254275: [[51.4540391, -2.5855091]],
        126489283: [
            [51.4542965, -2.5861303],
            [51.4542656, -2.5862657],
            [51.454227, -2.5862296],
            [51.4542473, -2.5861116],
            [51.4542651, -2.5861036],
            [51.4542965, -2.5861303],
        ],
        290604903: [
            [51.4542878, -2.5863848],
            [51.4542109, -2.5863147],
            [51.454227, -2.5862296],
            [51.4542656, -2.5862657],
            [51.4543119, -2.5863116],
            [51.4542878, -2.5863848],
        ],
    }
    tags = {
        560374554: {
            "amenity": "post_box",
            "collection_times": "Mo-Fr 09:00; Sa 07:00",
            "operator": "Royal Mail",
            "post_box:type": "pillar",
            "postal_code": "BS2",
            "ref": "BS2 231D",
            "royal_cypher": "EIIR",
        },
        6239576723: {
            "amenity": "vending_machine",
            "payment:coins": "yes",
            "payment:notes": "no",
            "ref:ringgo": "2695",
            "vending": "parking_tickets",
        },
        6239576724: {
            "amenity": "vending_machine",
            "payment:coins": "yes",
            "payment:notes": "no",
            "ref:ringgo": "2787",
            "vending": "parking_tickets",
        },
        6868356189: {"amenity": "bicycle_parking", "capacity": "8"},
        8942254228: {"amenity": "bicycle_parking", "capacity": "20"},
        8942254270: {
            "addr:city": "Bristol",
            "addr:country": "GB",
            "addr:housenumber": "1",
            "addr:postcode": "BS2 0JF",
            "addr:street": "Passage Street",
            "amenity": "music_school",
            "name": "BIMM Institute",
        },
        8942254271: {
            "addr:city": "Bristol",
            "addr:country": "GB",
            "addr:housenumber": "1",
            "addr:postcode": "BS2 0JF",
            "addr:street": "Passage Street",
            "amenity": "studio",
            "name": "Heart FM Bristol",
            "studio": "radio",
        },
        8942254275: {
            "access": "private",
            "amenity": "parking_entrance",
            "parking": "surface",
        },
        126489283: {
            "addr:city": "Bristol",
            "addr:housenumber": "16",
            "addr:postcode": "BS2 0JF",
            "addr:street": "Passage Street",
            "amenity": "pub",
            "building": "yes",
            "fhrs:id": "312586",
            "name": "The Bridge Inn",
            "wheelchair": "no",
        },
        290604903: {
            "addr:city": "Bristol",
            "addr:housenumber": "18",
            "addr:postcode": "BS2 0JF",
            "addr:street": "Passage Street",
            "amenity": "fast_food",
            "building": "yes",
            "cuisine": "sandwich",
            "fhrs:id": "310927",
            "name": "Sandwich Box",
            "wheelchair": "no",
        },
    }
    return ids, locations, tags


def test_gps_summaries_shape(
    coords1, sample_trajectory, sample_nearby_locations, mocker
):
    mocker.patch(
        "forest.jasmine.traj2stats.get_nearby_locations",
        return_value=sample_nearby_locations,
    )
    mocker.patch("forest.jasmine.traj2stats.locate_home", return_value=coords1)

    parameters = Hyperparameters()
    parameters.save_osm_log = True

    summary, _ = gps_summaries(
        traj=sample_trajectory,
        tz_str="Europe/London",
        frequency=Frequency.HOURLY,
        parameters=parameters,
        places_of_interest=["pub", "fast_food"],
    )
    assert summary.shape == (24, 21)


def test_gps_summaries_places_of_interest(
    coords1, sample_trajectory, sample_nearby_locations, mocker
):
    """Testing amount of time spent in places is valid"""
    mocker.patch(
        "forest.jasmine.traj2stats.get_nearby_locations",
        return_value=sample_nearby_locations,
    )
    mocker.patch("forest.jasmine.traj2stats.locate_home", return_value=coords1)

    parameters = Hyperparameters()
    parameters.save_osm_log = True

    summary, _ = gps_summaries(
        traj=sample_trajectory,
        tz_str="Europe/London",
        frequency=Frequency.HOURLY,
        parameters=parameters,
        places_of_interest=["pub", "fast_food"],
    )
    time_in_places_of_interest = (
        summary["pub"] + summary["fast_food"] + summary["other"]
    )
    assert np.all(time_in_places_of_interest <= summary["total_pause_time"])


def test_gps_summaries_obs_day_night(
    coords1, sample_trajectory, sample_nearby_locations, mocker
):
    """Testing total observation time is same
    as day observation plus night observation times
    """
    mocker.patch(
        "forest.jasmine.traj2stats.get_nearby_locations",
        return_value=sample_nearby_locations,
    )
    mocker.patch("forest.jasmine.traj2stats.locate_home", return_value=coords1)

    parameters = Hyperparameters()
    parameters.save_osm_log = True

    summary, _ = gps_summaries(
        traj=sample_trajectory,
        tz_str="Europe/London",
        frequency=Frequency.DAILY,
        parameters=parameters,
        places_of_interest=["pub", "fast_food"],
    )
    total_obs = summary["obs_day"] + summary["obs_night"]
    assert np.all(round(total_obs, 4) == round(summary["obs_duration"], 4))


def test_gps_summaries_datetime_nighttime_shape(
    coords1, sample_trajectory, sample_nearby_locations, mocker
):
    """Testing shape of datetime nighttime summary stats"""
    mocker.patch(
        "forest.jasmine.traj2stats.get_nearby_locations",
        return_value=sample_nearby_locations,
    )
    mocker.patch("forest.jasmine.traj2stats.locate_home", return_value=coords1)

    parameters = Hyperparameters()
    parameters.save_osm_log = True
    parameters.split_day_night = True

    summary, _ = gps_summaries(
        traj=sample_trajectory,
        tz_str="Europe/London",
        frequency=Frequency.DAILY,
        parameters=parameters,
        places_of_interest=["pub", "fast_food"],
    )
    assert summary.shape == (2, 46)


def test_gps_summaries_log_format(
    coords1, sample_trajectory, sample_nearby_locations, mocker
):
    """Testing json logs contain all
    dates from summary stats
    """
    mocker.patch(
        "forest.jasmine.traj2stats.get_nearby_locations",
        return_value=sample_nearby_locations,
    )
    mocker.patch("forest.jasmine.traj2stats.locate_home", return_value=coords1)

    parameters = Hyperparameters()
    parameters.save_osm_log = True

    summary, log = gps_summaries(
        traj=sample_trajectory,
        tz_str="Europe/London",
        frequency=Frequency.DAILY,
        parameters=parameters,
        places_of_interest=["pub", "fast_food"],
    )
    dates_stats = (
        summary["day"].astype(int).astype(str)
        + "/"
        + summary["month"].astype(int).astype(str)
        + "/"
        + summary["year"].astype(int).astype(str)
    )
    dates_log = np.array(list(log.keys()))
    assert np.all(dates_stats == dates_log)
