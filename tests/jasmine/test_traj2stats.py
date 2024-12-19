"""Tests for traj2stats summary statistics in Jasmine"""

import numpy as np
import pytest
from shapely.geometry import Point
import sys
sys.path.append('C:/Users/gioef/Desktop/onnela_lab/forest/src')

from forest.jasmine.data2mobmat import great_circle_dist
from forest.jasmine.traj2stats import (
    Frequency, gps_summaries, Hyperparameters, transform_point_to_circle,
    avg_mobility_trace_difference, create_mobility_trace, get_pause_array,
    extract_pause_from_row, compute_window_and_count
)


@pytest.fixture
def coords1():
    return 51.457183, -2.597960


@pytest.fixture
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
    assert 4 <= distance <= 5


@pytest.fixture
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


@pytest.fixture
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


def test_gps_summaries_summary_vals(
    coords1, sample_trajectory, sample_nearby_locations, mocker
):
    """Testing gps summaries summary values are correct"""
    mocker.patch(
        "forest.jasmine.traj2stats.get_nearby_locations",
        return_value=sample_nearby_locations,
    )
    mocker.patch("forest.jasmine.traj2stats.locate_home", return_value=coords1)

    parameters = Hyperparameters()

    summary, _ = gps_summaries(
        traj=sample_trajectory,
        tz_str="Europe/London",
        frequency=Frequency.DAILY,
        parameters=parameters,
    )

    assert summary["obs_duration"].iloc[0] == 24
    assert summary["obs_day"].iloc[0] == 10
    assert summary["obs_night"].iloc[0] == 14
    assert summary["obs_day"].iloc[1] == 0
    assert summary["obs_night"].iloc[1] == 0
    assert summary["home_time"].iloc[0] == 0
    assert summary["dist_traveled"].iloc[0] == 0.208
    assert np.round(summary["max_dist_home"].iloc[0], 3) == 0.915
    assert np.round(summary["radius"].iloc[0], 3) == 0.013
    assert np.round(summary["diameter"].iloc[0], 3) == 0.064
    assert summary["num_sig_places"].iloc[0] == 2
    assert np.round(summary["entropy"].iloc[0], 3) == 0.468
    assert round(summary["total_flight_time"].iloc[0], 3) == 1.528
    assert round(summary["av_flight_length"].iloc[0], 3) == 0.052
    assert round(summary["sd_flight_length"].iloc[0], 3) == 0.012
    assert round(summary["av_flight_duration"].iloc[0], 3) == 0.382
    assert round(summary["sd_flight_duration"].iloc[0], 3) == 0.132
    assert round(summary["total_pause_time"].iloc[0], 3) == 22.472
    assert round(summary["av_pause_duration"].iloc[0], 3) == 4.494
    assert round(summary["sd_pause_duration"].iloc[0], 3) == 3.496


def test_gps_summaries_pcr(
    coords1, sample_trajectory, sample_nearby_locations, mocker
):
    """Testing gps summaries pcr"""
    mocker.patch(
        "forest.jasmine.traj2stats.get_nearby_locations",
        return_value=sample_nearby_locations,
    )
    mocker.patch("forest.jasmine.traj2stats.locate_home", return_value=coords1)

    parameters = Hyperparameters()
    parameters.pcr_bool = True

    summary, _ = gps_summaries(
        traj=sample_trajectory,
        tz_str="Europe/London",
        frequency=Frequency.DAILY,
        parameters=parameters,
    )

    assert summary["physical_circadian_rhythm"].iloc[0] == 0
    assert summary["physical_circadian_rhythm_stratified"].iloc[0] == 0


@pytest.fixture
def mobmat1():
    """mobility matrix 1"""
    return np.array(
        [
            [16.49835, -142.72462, 1],
            [16.49521, -142.72461, 2],
            [51.45435654, -2.58555554, 3],
            [51.45435621, -2.58555524, 4],
            [51.45435632, -2.58555544, 5]
        ]
    )


@pytest.fixture
def mobmat2():
    """mobility matrix 2"""
    return np.array(
        [
            [51.45435654, -2.58555554, 1],
            [51.45435654, -2.58555554, 2],
            [51.45435654, -2.58555554, 3],
            [51.45435654, -2.58555554, 4],
            [51.45435654, -2.58555554, 5]
        ]
    )


@pytest.fixture
def mobmat3():
    """mobility matrix 3"""
    return np.array(
        [
            [51.45435654, -2.58555554, 7],
            [51.45435654, -2.58555554, 8],
            [51.45435654, -2.58555554, 9],
            [51.45435654, -2.58555554, 10],
            [51.45435654, -2.58555554, 11]
        ]
    )


def test_avg_mobility_trace_difference_common_timestamps(
    mobmat1, mobmat2
):
    """Testing avg mobility trace difference
    when there are common timestamps and all points are close
    """

    time_range = (3, 5)
    res = avg_mobility_trace_difference(
        time_range, mobmat1, mobmat2
    )

    assert res == 1


def test_avg_mobility_trace_difference_common_timestamps2(
    mobmat1, mobmat2
):
    """Testing avg mobility trace difference
    when there are common timestamps and some points are close
    """

    time_range = (1, 5)
    res = avg_mobility_trace_difference(
        time_range, mobmat1, mobmat2
    )

    assert res == 0.6


def test_avg_mobility_trace_difference_no_common_timestamps(
    mobmat1, mobmat3
):
    """Testing avg mobility trace difference
    when there are no common timestamps
    """

    time_range = (1, 5)
    res = avg_mobility_trace_difference(
        time_range, mobmat1, mobmat3
    )

    assert res == 0


def test_create_mobility_trace_shape(sample_trajectory):
    """Testing shape of mobility trace"""

    res = create_mobility_trace(sample_trajectory)

    assert res.shape == (81200, 3)


def test_create_mobility_trace_start_end_times(sample_trajectory):
    """Testing start and end times of mobility trace"""

    res = create_mobility_trace(sample_trajectory)

    assert res[0, 2] == 1633042800.0
    assert res[-1, 2] == 1633129499.0


def test_get_pause_array_shape(sample_trajectory, coords2):
    """Testing shape of pause array"""

    parameters = Hyperparameters()

    pause_array = get_pause_array(
        sample_trajectory[sample_trajectory[:, 0] == 2, :],
        *coords2,
        parameters
    )

    assert pause_array.shape == (3, 3)


def test_get_pause_array_times(sample_trajectory, coords2):
    """Testing times spent in places of pause array"""

    parameters = Hyperparameters()

    pause_array = get_pause_array(
        sample_trajectory[sample_trajectory[:, 0] == 2, :],
        *coords2,
        parameters
    )

    assert pause_array[0, 2] == 1113.3333333333333
    assert pause_array[-1, 2] == 180


def test_get_pause_array_house(sample_trajectory):
    """Testing case where house is in pause array"""

    house_coords = (51.45435654, -2.58555554)
    parameters = Hyperparameters()

    pause_array = get_pause_array(
        sample_trajectory[sample_trajectory[:, 0] == 2, :],
        *house_coords,
        parameters
    )

    assert pause_array.shape == (2, 3)


def test_extract_pause_from_row_shape(sample_trajectory):
    """Testing shape of pause array"""

    pause_list = extract_pause_from_row(
        sample_trajectory[0, :]
    )

    assert len(pause_list) == 3


def test_extract_pause_from_row_time(sample_trajectory):
    """Testing pause time of row"""

    pause_list = extract_pause_from_row(
        sample_trajectory[0, :]
    )

    true_val = sample_trajectory[0, 6] - sample_trajectory[0, 3]

    assert pause_list[2] == true_val / 60


def test_compute_window_size(sample_trajectory):
    """Testing window size is correct"""

    window, _ = compute_window_and_count(
        sample_trajectory[0, 3], sample_trajectory[-1, 6], 60
    )

    assert window == 3600


def test_compute_window_count(sample_trajectory):
    """Testing number of windows is correct"""

    _, num_windows = compute_window_and_count(
        sample_trajectory[0, 3], sample_trajectory[-1, 6], 60
    )

    assert num_windows == 24


def test_compute_window_size_6_hour(sample_trajectory):
    """Testing window size is correct 6 hour window"""

    window, _ = compute_window_and_count(
        sample_trajectory[0, 3], sample_trajectory[-1, 6], 360
    )

    assert window == 3600 * 6


def test_compute_window_count_6_hour(sample_trajectory):
    """Testing number of windows is correct 6 hour window"""

    _, num_windows = compute_window_and_count(
        sample_trajectory[0, 3], sample_trajectory[-1, 6], 360
    )

    assert num_windows == 4
