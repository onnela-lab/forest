"""Tests for simulate_gps_data module"""

import numpy as np
import pytest

from forest.bonsai.simulate_gps_data import (bounding_box,
    get_basic_path, get_path, Attributes, Person)
from forest.jasmine.data2mobmat import great_circle_dist


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
def directions1():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "bbox": [-2.608912, 51.456932, -2.595808, 51.461801],
                "type": "Feature",
                "properties": {
                    "segments": [
                        {
                            "distance": 1687.9,
                            "duration": 287.0,
                            "steps": [
                                {
                                    "distance": 42.7,
                                    "duration": 13.5,
                                    "type": 11,
                                    "instruction": "Head northeast",
                                    "name": "-",
                                    "way_points": [0, 2],
                                },
                                {
                                    "distance": 10.4,
                                    "duration": 7.5,
                                    "type": 1,
                                    "instruction": "Turn right",
                                    "name": "-",
                                    "way_points": [2, 3],
                                },
                                {
                                    "distance": 246.5,
                                    "duration": 63.9,
                                    "type": 1,
                                    "instruction": "Turn right onto Upper Maudlin Street, B4051",
                                    "name": "Upper Maudlin Street, B4051",
                                    "way_points": [3, 26],
                                },
                                {
                                    "distance": 62.3,
                                    "duration": 9.0,
                                    "type": 1,
                                    "instruction": "Turn right onto Saint Michaels Hill",
                                    "name": "Saint Michaels Hill",
                                    "way_points": [26, 31],
                                },
                                {
                                    "distance": 603.8,
                                    "duration": 85.9,
                                    "type": 12,
                                    "instruction": "Keep left onto Saint Michaels Hill",
                                    "name": "Saint Michaels Hill",
                                    "way_points": [31, 63],
                                },
                                {
                                    "distance": 466.5,
                                    "duration": 60.2,
                                    "type": 0,
                                    "instruction": "Turn left onto Tyndalls Park Road",
                                    "name": "Tyndalls Park Road",
                                    "way_points": [63, 81],
                                },
                                {
                                    "distance": 244.4,
                                    "duration": 44.2,
                                    "type": 0,
                                    "instruction": "Turn left onto Whiteladies Road, A4018",
                                    "name": "Whiteladies Road, A4018",
                                    "way_points": [81, 99],
                                },
                                {
                                    "distance": 11.2,
                                    "duration": 2.7,
                                    "type": 0,
                                    "instruction": "Turn left onto Queen's Avenue",
                                    "name": "Queen's Avenue",
                                    "way_points": [99, 100],
                                },
                                {
                                    "distance": 0.0,
                                    "duration": 0.0,
                                    "type": 10,
                                    "instruction": "Arrive at Queen's Avenue, on the right",
                                    "name": "-",
                                    "way_points": [100, 100],
                                },
                            ],
                        }
                    ],
                    "summary": {"distance": 1687.9, "duration": 287.0},
                    "way_points": [0, 100],
                },
                "geometry": {
                    "coordinates": [
                        [-2.596325, 51.458538],
                        [-2.596069, 51.458726],
                        [-2.595929, 51.458832],
                        [-2.595808, 51.458777],
                        [-2.595906, 51.458695],
                        [-2.595986, 51.45861],
                        [-2.596307, 51.458234],
                        [-2.596294, 51.458191],
                        [-2.596342, 51.458133],
                        [-2.596401, 51.458124],
                        [-2.59648, 51.458046],
                        [-2.596535, 51.457985],
                        [-2.596569, 51.45795],
                        [-2.596552, 51.457916],
                        [-2.596573, 51.457891],
                        [-2.596632, 51.457873],
                        [-2.596701, 51.457797],
                        [-2.597007, 51.45749],
                        [-2.596998, 51.457463],
                        [-2.597031, 51.457425],
                        [-2.597081, 51.457408],
                        [-2.597129, 51.457365],
                        [-2.597153, 51.457344],
                        [-2.597137, 51.457307],
                        [-2.597435, 51.456996],
                        [-2.597508, 51.456975],
                        [-2.597554, 51.456932],
                        [-2.597647, 51.456976],
                        [-2.597762, 51.457035],
                        [-2.597896, 51.457118],
                        [-2.598092, 51.457262],
                        [-2.598156, 51.457341],
                        [-2.59826, 51.457383],
                        [-2.598454, 51.457491],
                        [-2.598592, 51.457577],
                        [-2.599095, 51.45792],
                        [-2.59919, 51.457999],
                        [-2.599377, 51.458261],
                        [-2.599648, 51.458579],
                        [-2.599808, 51.458707],
                        [-2.600055, 51.458861],
                        [-2.600244, 51.45898],
                        [-2.600318, 51.459026],
                        [-2.600391, 51.459073],
                        [-2.600423, 51.459094],
                        [-2.600503, 51.459168],
                        [-2.600851, 51.459522],
                        [-2.600973, 51.459674],
                        [-2.601096, 51.459859],
                        [-2.601129, 51.459944],
                        [-2.60116, 51.460035],
                        [-2.601169, 51.460074],
                        [-2.601238, 51.46026],
                        [-2.601371, 51.460478],
                        [-2.601482, 51.460583],
                        [-2.601494, 51.460594],
                        [-2.601714, 51.460765],
                        [-2.601963, 51.460948],
                        [-2.602227, 51.461136],
                        [-2.602438, 51.461307],
                        [-2.602487, 51.461346],
                        [-2.602601, 51.46146],
                        [-2.602684, 51.461567],
                        [-2.602886, 51.461801],
                        [-2.602995, 51.461758],
                        [-2.603032, 51.461744],
                        [-2.603109, 51.461726],
                        [-2.603453, 51.461572],
                        [-2.604003, 51.461335],
                        [-2.604018, 51.461332],
                        [-2.604935, 51.461083],
                        [-2.6055, 51.460912],
                        [-2.606136, 51.460659],
                        [-2.60642, 51.460527],
                        [-2.607213, 51.460158],
                        [-2.607755, 51.459923],
                        [-2.607856, 51.459884],
                        [-2.60795, 51.459857],
                        [-2.607969, 51.459851],
                        [-2.608203, 51.459818],
                        [-2.608571, 51.459798],
                        [-2.6087, 51.459784],
                        [-2.608716, 51.459657],
                        [-2.608732, 51.459535],
                        [-2.608757, 51.459388],
                        [-2.608763, 51.459334],
                        [-2.608792, 51.459076],
                        [-2.608822, 51.458869],
                        [-2.608858, 51.458631],
                        [-2.608873, 51.458534],
                        [-2.60888, 51.458488],
                        [-2.6089, 51.458325],
                        [-2.608904, 51.458284],
                        [-2.60891, 51.458231],
                        [-2.608912, 51.45821],
                        [-2.60887, 51.458135],
                        [-2.608852, 51.458],
                        [-2.608812, 51.457897],
                        [-2.60878, 51.457783],
                        [-2.60863, 51.457627],
                        [-2.60847, 51.457639],
                    ],
                    "type": "LineString",
                },
            }
        ],
        "bbox": [-2.608912, 51.456932, -2.595808, 51.461801],
        "metadata": {
            "attribution": "openrouteservice.org | OpenStreetMap contributors",
            "service": "routing",
            "timestamp": 1634297702219,
            "query": {
                "coordinates": [[-2.59638, 51.458498], [-2.608466, 51.457619]],
                "profile": "driving-car",
                "format": "geojson",
            },
            "engine": {
                "version": "6.6.1",
                "build_date": "2021-07-05T10:57:48Z",
                "graph_date": "2021-10-03T10:50:45Z",
            },
        },
    }


def test_get_path_starting_latitude(coords1, coords2, directions1, mocker):
    mocker.patch(
        "openrouteservice.Client.directions", return_value=directions1
    )
    assert (
        get_path(coords1, coords2, "car", "mock_api_key")[0][0][0] == 51.458498
    )


def test_get_path_ending_longitude(coords1, coords2, directions1, mocker):
    mocker.patch(
        "openrouteservice.Client.directions", return_value=directions1
    )
    assert (
        get_path(coords1, coords2, "car", "mock_api_key")[0][-1][1]
        == -2.608466
    )


def test_get_path_distance(coords1, coords2, directions1, mocker):
    mocker.patch(
        "openrouteservice.Client.directions", return_value=directions1
    )
    assert (
        get_path(coords1, coords2, "car", "mock_api_key")[1]
        == 843.0532531565476
    )


def test_get_path_close_locations(coords1, coords3):
    """Tests case distance of locations is less than 250 meters."""
    assert len(get_path(coords1, coords3, "foot", "mock_api_key")[0]) == 2


@pytest.fixture
def random_path(directions1, coords1, coords2):
    lat1, lon1 = coords1
    lat2, lon2 = coords2
    # path returned from running get_path function
    # starting from coords1 and ending on coords2
    # with car as transport
    coordinates = directions1["features"][0]["geometry"]["coordinates"]
    path_coordinates = [[coord[1], coord[0]] for coord in coordinates]

    # sometimes if exact coordinates of location are not in a road
    # the starting or ending coordinates of route will be returned
    # in the nearer road which can be slightly different than
    # the ones provided
    if path_coordinates[0] != [lat1, lon1]:
        path_coordinates[0] = [lat1, lon1]
    if path_coordinates[-1] != [lat2, lon2]:
        path_coordinates[-1] = [lat2, lon2]

    return np.array(path_coordinates)


def test_get_basic_path_simple_case(random_path):
    """Test simple case of getting basic path"""
    basic_random_path = get_basic_path(random_path, "car")
    boolean_matrix = basic_random_path == np.array(
        [
            [51.458498, -2.59638],
            [51.456975, -2.597508],
            [51.460035, -2.60116],
            [51.459923, -2.607755],
            [51.457619, -2.608466],
        ]
    )
    assert np.sum(boolean_matrix) == 10


def test_get_basic_path_length_by_bicycle(random_path):
    basic_random_path_bicycle = get_basic_path(random_path, "bicycle")
    assert len(basic_random_path_bicycle) == 8


def test_get_basic_path_length_by_car(random_path):
    basic_random_path_car = get_basic_path(random_path, "car")
    assert len(basic_random_path_car) == 5


def test_get_basic_path_length_by_bus(random_path):
    basic_random_path_bus = get_basic_path(random_path, "bus")
    assert len(basic_random_path_bus) == 7


@pytest.fixture(scope="session")
def sample_coordinates():
    return 51.458726, -2.596069


def test_bounding_box_simple_case(sample_coordinates):
    """Test bounding box simple case distance"""
    bbox = bounding_box(sample_coordinates, 500)
    # distance of bbox corner from center
    distance = np.round(
        great_circle_dist(
            sample_coordinates[0], sample_coordinates[1], bbox[0], bbox[1]
        )
    )
    # 707 comes from Pythagorean theorem
    assert distance == 707


def test_zero_meters_bounding_box(sample_coordinates):
    bbox = bounding_box(sample_coordinates, 0)
    assert bbox[0] == bbox[2] and bbox[1] == bbox[3]


@pytest.fixture(scope="session")
def sample_locations():
    """All nodes of preferred categories around sample_coordinates"""
    return {
        "cafe": [
            (51.4730335, -2.5868174),
            (51.4503187, -2.6000677),
            (51.4542379, -2.5957399),
            (51.4566122, -2.6049184),
            (51.4491962, -2.597486),
            (51.4513174, -2.5778401),
        ],
        "bar": [
            (51.4592495, -2.6111314),
            (51.4588478, -2.6106816),
            (51.4531018, -2.5981952),
            (51.4499107, -2.5808118),
            (51.4511935, -2.5924017),
            (51.4497689, -2.5973878),
            (51.4532564, -2.592465),
        ],
        "restaurant": [
            (51.4587343, -2.6110204),
            (51.4555718, -2.6198581),
            (51.4515447, -2.5852328),
            (51.4564397, -2.589893),
            (51.4564159, -2.5899766),
            (51.4593627, -2.5916202),
            (51.4544069, -2.5940264),
            (51.4590668, -2.5915499),
        ],
        "cinema": [
            (51.4589416, -2.5860582),
            (51.4513069, -2.5982655),
            (51.4611435, -2.5933154),
            (51.4561455, -2.5967294),
            (51.4629542, -2.6096249),
            (51.4517717, -2.598108),
            (51.4566839, -2.5915285),
        ],
        "park": [
            (51.4609571, -2.5855459),
            (51.4679187, -2.598686),
            (51.4426429, -2.6014811),
            (51.4405893, -2.5954815),
            (51.4721951, -2.576308),
        ],
        "dance": [(51.4573642, -2.6177539)],
        "fitness": [
            (51.4541587, -2.5899168),
            (51.4559791, -2.5917542),
            (51.4516719, -2.5799672),
            (51.4566176, -2.5695463),
        ],
        "office": [
            (51.4566176, -2.5695463),
            (51.4561455, -2.5967294),
        ],
        "university": [],
    }


def test_person_main_employment(sample_coordinates, sample_locations):
    attributes = Attributes(vehicle=Vehicle.NONE,
                            main_occupation=Occupation.WORK,
                            active_status=6,
                            travelling_status=8,
                            preferred_places=["cinema", "bar", "park"])
    random_person = Person(sample_coordinates, attributes, sample_locations)
    assert random_person.main_occupation == 1


def test_person_cafe_places(sample_coordinates, sample_locations):
    """Test one place from cafe_places attribute is actual cafe"""
    attributes = Attributes(vehicle=Vehicle.NONE,
                            main_occupation=Occupation.NONE,
                            active_status=2,
                            travelling_status=7,
                            preferred_places=["cafe", "cinema", "park"])
    random_person = Person(sample_coordinates, attributes, sample_locations)
    cafe_place = (
        random_person.cafe_places[0][0],
        random_person.cafe_places[0][1],
    )
    assert cafe_place in sample_locations["cafe"]


def test_person_office_address(sample_coordinates, sample_locations):
    """Test person going to work office_address"""
    attributes = Attributes(vehicle=Vehicle.NONE,
                            main_occupation=Occupation.WORK,
                            active_status=6,
                            travelling_status=7,
                            preferred_places=["cafe", "cinema", "park"])
    random_person = Person(sample_coordinates, attributes, sample_locations)
    office_address = (
        random_person.office_address[0],
        random_person.office_address[1],
    )
    assert office_address in sample_locations["office"]


def test_person_office_days(sample_coordinates, sample_locations):
    """Test person going to work office_address"""
    attributes = Attributes(vehicle=Vehicle.NONE,
                            main_occupation=Occupation.WORK,
                            active_status=6,
                            travelling_status=7,
                            preferred_places=["cafe", "bar", "park"])
    random_person = Person(sample_coordinates, attributes, sample_locations)
    assert len(random_person.office_days) <= 5
