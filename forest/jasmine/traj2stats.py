"""Module used to impute missing data, by combining functions defined in other
modules and calculate summary statistics of imputed trajectories.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
import requests
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import transform

from forest.bonsai.simulate_gps_data import bounding_box
from forest.constants import Frequency, OSM_OVERPASS_URL, OSMTags
from forest.jasmine.data2mobmat import (gps_to_mobmat, infer_mobmat,
                                        great_circle_dist,
                                        pairwise_great_circle_dist)
from forest.jasmine.mobmat2traj import (imp_to_traj, impute_gps, locate_home,
                                        num_sig_places)
from forest.jasmine.sogp_gps import bv_select
from forest.poplar.legacy.common_funcs import (datetime2stamp, read_data,
                                               stamp2datetime,
                                               write_all_summaries)
from forest.utils import get_ids


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@dataclass
class Hyperparameters:
    """Class containing hyperparemeters for gps imputation and trajectory
     summary statistics calculation.

    Args:
        itrvl, accuracylim, r, w, h: hyperparameters for the
            gps_to_mobmat function.
        itrvl, r: hyperparameters for the infer_mobmat function.
        l1, l2, l3, a1, a2, b1, b2, b3, sigma2, tol, d: hyperparameters
            for the bv_select function.
        l1, l2, a1, a2, b1, b2, b3, g, method, switch, num, linearity:
            hyperparameters for the impute_gps function.
        itrvl, r, w, h: hyperparameters for the imp_to_traj function.
        log_threshold: int, time spent in a pause needs to exceed the
            log_threshold to be placed in the log
            only if save_osm_log True, in minutes
        split_day_night: bool, True if you want to split all metrics to
            datetime and nighttime patterns
            only for daily frequency
        person_point_radius: float, radius of the person's circle when
            discovering places near him in pauses
        place_point_radius: float, radius of place's circle
            when place is returned as centre coordinates from osm
        save_osm_log: bool, True if you want to output a log of locations
            visited and their tags
        quality_threshold: float, a percentage value of the fraction of data
            required for a summary to be created
        pcr_bool: bool, True if you want to calculate the physical
            circadian rhythm
        pcr_window: int, number of days to look back and forward
            for calculating the physical circadian rhythm
        pcr_sample_rate: int, number of seconds between each sample
            for calculating the physical circadian rhythm
    """
    # imputation hyperparameters
    l1: int = 60 * 60 * 24 * 10
    l2: int = 60 * 60 * 24 * 30
    l3: float = 0.002
    g: int = 200
    a1: int = 5
    a2: int = 1
    b1: float = 0.3
    b2: float = 0.2
    b3: float = 0.5
    d: int = 100
    sigma2: float = 0.01
    tol: float = 0.05
    switch: int = 3
    num: int = 10
    linearity: int = 2
    method: str = "GLC"
    itrvl: int = 10
    accuracylim: int = 51
    r: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None

    # summary statistics hyperparameters
    save_osm_log: bool = False
    log_threshold: int = 60
    split_day_night: bool = False
    person_point_radius: float = 2
    place_point_radius: float = 7.5
    quality_threshold: float = 0.05
    pcr_bool: bool = False
    pcr_window: int = 14
    pcr_sample_rate: int = 30


def transform_point_to_circle(lat: float, lon: float, radius: float
                              ) -> Polygon:
    """This function transforms a set of cooordinates to a shapely
    circle with a provided radius.

    Args:
        lat: float, latitude of the center of the circle
        lon: float, longitude of the center of the circle
        radius: float, in meters
    Returns:
        shapely polygon of a circle
    """

    local_azimuthal_projection = (
        f"+proj=aeqd +R=6371000 +units=m +lat_0={lat} +lon_0={lon}"
    )
    wgs84_to_aeqd = Transformer.from_crs(
        "+proj=longlat +datum=WGS84 +no_defs",
        local_azimuthal_projection,
    ).transform
    aeqd_to_wgs84 = Transformer.from_crs(
        local_azimuthal_projection,
        "+proj=longlat +datum=WGS84 +no_defs",
    ).transform

    center = Point(lat, lon)
    point_transformed = transform(wgs84_to_aeqd, center)
    buffer = point_transformed.buffer(radius)
    return transform(aeqd_to_wgs84, buffer)


def get_nearby_locations(
    traj: np.ndarray, osm_tags: Optional[List[OSMTags]] = None
) -> Tuple[dict, dict, dict]:
    """This function returns a dictionary of nearby locations,
    a dictionary of nearby locations' names, and a dictionary of
    nearby locations' coordinates.

    Args:
        traj: numpy array, trajectory
        osm_tags: list of OSMTags (in constants),
            types of nearby locations supported by Overpass API
            defaults to [OSMTags.AMENITY, OSMTags.LEISURE]
    Returns:
        A tuple of:
         dictionary, contains nearby locations' ids
         dictionary, contains nearby locations' coordinates
         dictionary, contains nearby locations' tags
    Raises:
        RuntimeError: if the query to Overpass API fails
    """

    if osm_tags is None:
        osm_tags = [OSMTags.AMENITY, OSMTags.LEISURE]
    pause_vec = traj[traj[:, 0] == 2]
    latitudes: List[float] = [pause_vec[0, 1]]
    longitudes: List[float] = [pause_vec[0, 2]]
    for row in pause_vec:
        minimum_distance = np.min([
            great_circle_dist(row[1], row[2], lat, lon)[0]
            for lat, lon in zip(latitudes, longitudes)
            ])
        # only add coordinates to the list if they are not too close
        # with the other coordinates in the list
        if minimum_distance > 1000:
            latitudes.append(row[1])
            longitudes.append(row[2])

    query = "[out:json];\n("

    for lat, lon in zip(latitudes, longitudes):
        bbox = bounding_box((lat, lon), 1000)

        for tag in osm_tags:
            if tag == OSMTags.BUILDING:
                query += f"""
                \tnode{bbox}['building'='residential'];
                \tway{bbox}['building'='residential'];
                \tnode{bbox}['building'='office'];
                \tway{bbox}['building'='office'];
                \tnode{bbox}['building'='commercial'];
                \tway{bbox}['building'='commercial'];
                \tnode{bbox}['building'='supermarket'];
                \tway{bbox}['building'='supermarket'];
                \tnode{bbox}['building'='stadium'];
                \tway{bbox}['building'='stadium'];"""
            elif tag == OSMTags.HIGHWAY:
                query += f"""
                \tnode{bbox}['highway'='motorway'];
                \tway{bbox}['highway'='motorway'];
                \tnode{bbox}['highway'='trunk'];
                \tway{bbox}['highway'='trunk'];
                \tnode{bbox}['highway'='primary'];
                \tway{bbox}['highway'='primary'];
                \tnode{bbox}['highway'='secondary'];
                \tway{bbox}['highway'='secondary'];
                \tnode{bbox}['highway'='tertiary'];
                \tway{bbox}['highway'='tertiary'];
                \tnode{bbox}['highway'='road'];
                \tway{bbox}['highway'='road'];"""
            else:
                query += f"""
                \tnode{bbox}['{tag.value}'];
                \tway{bbox}['{tag.value}'];"""

    query += "\n);\nout geom qt;"

    response = requests.post(OSM_OVERPASS_URL,
                             data={"data": query}, timeout=60)
    try:
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError,
        requests.exceptions.ReadTimeout
    ) as err:
        raise RuntimeError(
            f"Timeout error: {err} \n"
            "OpenStreetMap query is too large. "
            "Do not use save_osm_log or places_of_interest "
            "unless you need them. \n"
            "Query to Overpass API failed to return data in alloted time"
        )

    res = response.json()
    ids: Dict[str, List[int]] = {}
    locations: Dict[int, List[List[float]]] = {}
    tags: Dict[int, Dict[str, str]] = {}

    for element in res["elements"]:

        element_id = element["id"]

        for tag in osm_tags:
            if tag.value in element["tags"]:
                if element["tags"][tag.value] not in ids.keys():
                    ids[element["tags"][tag.value]] = [element_id]
                else:
                    ids[element["tags"][tag.value]].append(element_id)
                continue

        if element["type"] == "node":
            locations[element_id] = [[element["lat"], element["lon"]]]
        elif element["type"] == "way":
            locations[element_id] = [
                [x["lat"], x["lon"]] for x in element["geometry"]
            ]

        tags[element_id] = element["tags"]

    return ids, locations, tags


def avg_mobility_trace_difference(
    time_range: Tuple[int, int], mobility_trace1: np.ndarray,
    mobility_trace2: np.ndarray
) -> float:
    """This function calculates the average mobility trace difference

    Args:
        time_range: tuple of two ints, time range of mobility_trace
        mobility_trace1: numpy array, mobility trace 1
            contains 3 columns: [x, y, t]
        mobility_trace2: numpy array, mobility trace 2
            contains 3 columns: [x, y, t]
    Returns:
        float, average mobility trace difference
    Raises:
        ValueError: if the calculation fails
    """

    # Create masks for timestamps that lie within the specified time range
    mask1 = (
        (mobility_trace1[:, 2] >= time_range[0])
        & (mobility_trace1[:, 2] <= time_range[1])
    )
    mask2 = (
        (mobility_trace2[:, 2] >= time_range[0])
        & (mobility_trace2[:, 2] <= time_range[1])
    )

    # Create a set of common timestamps for efficient lookup
    common_times = (
        set(mobility_trace1[mask1, 2]) & set(mobility_trace2[mask2, 2])
    )

    # Create masks for the common timestamps
    mask1_common = np.isin(mobility_trace1[:, 2], list(common_times))
    mask2_common = np.isin(mobility_trace2[:, 2], list(common_times))

    if not any(mask1_common) or not any(mask2_common):
        return 0

    # Calculate distances using the common timestamp masks
    dists = great_circle_dist(
        mobility_trace1[mask1_common, 0], mobility_trace1[mask1_common, 1],
        mobility_trace2[mask2_common, 0], mobility_trace2[mask2_common, 1]
    )

    dist_flag = dists <= 10
    res = np.mean(dist_flag)
    if np.isnan(res):
        raise ValueError("PCR calculation failed")

    return float(res)


def routine_index(
    time_range: Tuple[int, int], mobility_trace: np.ndarray,
    pcr_window: int = 14, pcr_sample_rate: int = 30,
    stratified: bool = False, timezone: str = "US/Eastern",
) -> float:
    """This function calculates the routine index of a trajectory

    Description of routine index can be found in the paper:
    Canzian and Musolesi's 2015 paper in the Proceedings of the 2015
    ACM International Joint Conference on Pervasive and Ubiquitous Computing,
    titled “Trajectories of depression: unobtrusive monitoring of depressive
    states by means of smartphone mobility traces analysis.”

    Args:
        time_range: tuple of two ints, time range of mobility_trace
        mobility_trace: numpy array, trajectory
            contains 3 columns: [x, y, t]
        pcr_window: int, number of days to look back and forward
            for calculating the physical circadian rhythm
        pcr_sample_rate: int, number of seconds between each sample
            for calculating the physical circadian rhythm
        stratified: bool, True if you want to calculate the routine index
            for weekdays and weekends separately
        timezone: str, timezone of the mobility trace
    Returns:
        float, routine index
    """

    t_1, t_2 = time_range

    t_init = mobility_trace[:, 2].min()
    t_fin = mobility_trace[:, 2].max()

    t_1 = max(t_1, t_init)
    t_2 = min(t_2, t_fin)

    # n1, n2 are the number of days before and after the time range
    n1 = int(round((t_1 - t_init) / (24 * 60 * 60)))
    n2 = int(round((t_fin - t_2) / (24 * 60 * 60)))

    # to avoid long computational times
    # only look at the last window days and next window days
    n1 = min(n1, pcr_window)
    n2 = min(n2, pcr_window)

    if max(n1, n2) == 0:
        return 0

    shifts = list(range(1, n1 + 1)) + list(range(-n2, 0))
    if stratified:
        time_mid = int((t_1 + t_2) / 2)
        weekend_today = datetime(
            *stamp2datetime(time_mid, timezone)
        ).weekday() >= 5
        if weekend_today:
            shifts = [
                s for s in shifts
                if datetime(
                    *stamp2datetime(
                        time_mid - s * 24 * 60 * 60, timezone
                    )
                ).weekday() >= 5
            ]
        else:
            shifts = [
                s for s in shifts
                if datetime(
                    *stamp2datetime(
                        time_mid - s * 24 * 60 * 60, timezone
                    )
                ).weekday() < 5
            ]

    res = sum(
        avg_mobility_trace_difference(
            time_range, mobility_trace[::pcr_sample_rate],
            np.column_stack(
                [
                    mobility_trace[:, :2],
                    mobility_trace[:, 2] + i * 24 * 60 * 60
                ]
            )
        )
        for i in shifts
    )

    return res / (n1 + n2)


def create_mobility_trace(traj: np.ndarray) -> np.ndarray:
    """This function creates a mobility trace from a trajectory

    Args:
        traj: numpy array, trajectory
            contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
    Returns:
        numpy array, mobility trace
            contains 3 columns: [x, y, t]
    """

    pause_vec = traj[traj[:, 0] == 2]

    # Calculate the time ranges for all pauses
    start_times = pause_vec[:, 3].astype(int)
    end_times = pause_vec[:, 6].astype(int)
    time_ranges = [np.arange(s, e) for s, e in zip(start_times, end_times)]

    # Flatten time_ranges and get the corresponding locations
    flat_time_ranges = np.concatenate(time_ranges)
    repeats = [len(r) for r in time_ranges]
    locs = np.repeat(pause_vec[:, 1:3], repeats, axis=0)

    # Stack locations and time_ranges to get the mobility trace
    mobility_trace = np.column_stack([locs, flat_time_ranges])

    # check if duplicate timestamps exist
    _, unique_indices = np.unique(mobility_trace[:, 2], return_index=True)

    return mobility_trace[unique_indices]


def get_day_night_indices(
    traj: np.ndarray, tz_str: str, index: int, start_time: int, end_time: int,
    current_time_list: List[int]
) -> Tuple[np.ndarray, int, int, int, int]:
    """This function returns the indices of the rows in the trajectory
     if the trajectory is split into day and night.

    Args:
        traj: numpy array, trajectory
            contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
        tz_str: str, timezone
        index: int, index of the window
        start_time: int, starting time of the window
        end_time: int, ending time of the window
        current_time_list: list of int, current time
    Returns:
        A tuple of:
         numpy array, indices of the rows in the trajectory
            if the trajectory is split into day and night
         int, index of the row in the trajectory
            where the first part of the trajectory ends
         int, index of the row in the trajectory
            where the second part of the trajectory starts
         int, starting time of the second part of the trajectory
         int, ending time of the second part of the trajectory
    """

    current_time_list2 = current_time_list.copy()
    current_time_list3 = current_time_list.copy()
    current_time_list2[3] = 8
    current_time_list3[3] = 20
    start_time2 = datetime2stamp(current_time_list2, tz_str)
    end_time2 = datetime2stamp(current_time_list3, tz_str)
    if index % 2 == 0:
        # daytime
        index_rows = (traj[:, 3] <= end_time2) * (traj[:, 6] >= start_time2)

        return index_rows, 0, 0, start_time2, end_time2

    # nighttime
    index1 = (
        (traj[:, 6] < start_time2)
        * (traj[:, 3] < end_time)
        * (traj[:, 6] > start_time)
    )
    index2 = (
        (traj[:, 3] > end_time2)
        * (traj[:, 3] < end_time)
        * (traj[:, 6] > start_time)
    )
    stop1 = sum(index1) - 1
    stop2 = sum(index1)
    index_rows = index1 + index2

    return index_rows, stop1, stop2, start_time2, end_time2


def smooth_temp_ends(
    temp: np.ndarray, index_rows: np.ndarray, t0_temp: float,
    t1_temp: float, parameters: Hyperparameters, i: int, start_time: int,
    end_time2: int, start_time2: int, end_time: int, stop1: int, stop2: int
) -> np.ndarray:
    """This function smooths the starting and ending points of the
    trajectory.

    Args:
        temp: numpy array, trajectory
            contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
        index_rows: numpy array, indices of the rows in the trajectory
            if the trajectory is split into day and night
        t0_temp: float, starting time of the trajectory
        t1_temp: float, ending time of the trajectory
        parameters: Hyperparameters, hyperparameters in functions
            recommend to set it to default
        i: int, index of the window
        start_time: int, starting time of the window
        end_time2: int, ending time of the second part of the trajectory
        start_time2: int, starting time of the second part of the trajectory
        end_time: int, ending time of the window
        stop1: int, index of the row in the trajectory
            where the first part of the trajectory ends
        stop2: int, index of the row in the trajectory
            where the second part of the trajectory starts
    Returns:
        temp: numpy array, trajectory
            contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
    """
    if sum(index_rows) == 1:
        p0 = (t0_temp - temp[0, 3]) / (temp[0, 6] - temp[0, 3])
        p1 = (t1_temp - temp[0, 3]) / (temp[0, 6] - temp[0, 3])
        x0, y0 = temp[0, [1, 2]]
        x1, y1 = temp[0, [4, 5]]
        temp[0, 1] = (1 - p0) * x0 + p0 * x1
        temp[0, 2] = (1 - p0) * y0 + p0 * y1
        temp[0, 3] = t0_temp
        temp[0, 4] = (1 - p1) * x0 + p1 * x1
        temp[0, 5] = (1 - p1) * y0 + p1 * y1
        temp[0, 6] = t1_temp
    else:
        if parameters.split_day_night and i % 2 != 0:
            t0_temp_l = [start_time, end_time2]
            t1_temp_l = [start_time2, end_time]
            start_temp = [0, stop2]
            end_temp = [stop1, -1]
            for j in range(2):
                p0 = (temp[start_temp[j], 6] - t0_temp_l[j]) / (
                    temp[start_temp[j], 6] - temp[start_temp[j], 3]
                )
                p1 = (t1_temp_l[j] - temp[end_temp[j], 3]) / (
                    temp[end_temp[j], 6] - temp[end_temp[j], 3]
                )
                temp[start_temp[j], 1] = (1 - p0) * temp[
                    start_temp[j], 4
                ] + p0 * temp[start_temp[j], 1]
                temp[start_temp[j], 2] = (1 - p0) * temp[
                    start_temp[j], 5
                ] + p0 * temp[start_temp[j], 2]
                temp[start_temp[j], 3] = t0_temp_l[j]
                temp[end_temp[j], 4] = (1 - p1) * temp[
                    end_temp[j], 1
                ] + p1 * temp[end_temp[j], 4]
                temp[end_temp[j], 5] = (1 - p1) * temp[
                    end_temp[j], 2
                ] + p1 * temp[end_temp[j], 5]
                temp[end_temp[j], 6] = t1_temp_l[j]
        else:
            p0 = (temp[0, 6] - t0_temp) / (temp[0, 6] - temp[0, 3])
            p1 = (
                (t1_temp - temp[-1, 3])
                / (temp[-1, 6] - temp[-1, 3])
                )
            temp[0, 1] = (1 - p0) * temp[0, 4] + p0 * temp[0, 1]
            temp[0, 2] = (1 - p0) * temp[0, 5] + p0 * temp[0, 2]
            temp[0, 3] = t0_temp
            temp[-1, 4] = (1 - p1) * temp[-1, 1] + p1 * temp[-1, 4]
            temp[-1, 5] = (1 - p1) * temp[-1, 2] + p1 * temp[-1, 5]
            temp[-1, 6] = t1_temp

    return temp


def get_pause_array(pause_vec: np.ndarray, home_lat: float, home_lon: float,
                    parameters: Hyperparameters) -> np.ndarray:
    """This function returns a numpy array of pauses.

    Args:
        pause_vec: numpy array, contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
        home_lat: float, latitude of the home
        home_lon: float, longitude of the home
        parameters: Hyperparameters, hyperparameters in functions
    Returns:
        pause_array: numpy array, contains 3 columns: [x, y, t]
    """
    pause_array: np.ndarray = np.array([])
    for row in pause_vec:
        if (
            great_circle_dist(row[1], row[2], home_lat, home_lon)[0]
            > 2*parameters.place_point_radius
        ):
            if len(pause_array) == 0:
                pause_array = np.array(
                    [extract_pause_from_row(row)]
                )
            elif (
                np.min(
                    great_circle_dist(
                        row[1], row[2],
                        pause_array[:, 0], pause_array[:, 1],
                    )
                )
                > 2*parameters.place_point_radius
            ):
                pause_array = np.append(
                    pause_array,
                    [extract_pause_from_row(row)],
                    axis=0,
                )
            else:
                pause_array[
                    np.argmin(
                        great_circle_dist(
                            row[1], row[2],
                            pause_array[:, 0], pause_array[:, 1],
                        )
                    ),
                    -1,
                ] += (row[6] - row[3]) / 60

    return pause_array


def extract_pause_from_row(row: np.ndarray) -> list:
    """This function extracts the pause from a row in a trajectory.

    Args:
        row: numpy array, contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
    Returns:
        list, pause
    """
    return [row[1], row[2], (row[6] - row[3]) / 60]


def get_polygon(saved_polygons: dict, lat: float, lon: float, label: str,
                radius: float) -> Tuple[Polygon, dict]:
    """This function returns a saved polygon if it exists,
    otherwise it computes a polygon and saves it.

    Args:
        saved_polygons: dict, contains saved polygons
        lat: float, latitude of the center of the circle
        lon: float, longitude of the center of the circle
        label: str, label of the location
        radius: float, radius of the circle
    Returns:
        A tuple with the following elements:
         shapely polygon
         dict, contains saved polygons
    """
    loc_str = f"{lat}, {lon} - {label}"
    if loc_str in saved_polygons.keys():
        return saved_polygons[loc_str], saved_polygons

    circle = transform_point_to_circle(lat, lon, radius)
    saved_polygons[loc_str] = circle
    return circle, saved_polygons


def intersect_with_places_of_interest(
    pause: list, places_of_interest: list, saved_polygons: dict,
    parameters: Hyperparameters, ids: dict, locations: dict,
    ids_keys_list: list
) -> Tuple[list, bool]:
    """This function computes the intersection between a pause and
    places of interest.

    Args:
        pause: list, pause
        places_of_interest: list of str, places of interest
        saved_polygons: dict, contains saved polygons
        parameters: Hyperparameters, hyperparameters in functions
        ids: dict, contains nearby locations' ids
        locations: dict, contains nearby locations' coordinates
        ids_keys_list: list of str, keys of ids
    Returns:
        A tuple with the following elements:
         list of float, intersection between a pause and
            places of interest
         bool, True if the pause is not intersected with
            any place of interest
    """
    all_place_probs = [0] * len(places_of_interest)
    pause_circle, saved_polygons = get_polygon(
        saved_polygons, pause[0], pause[1], "person",
        parameters.person_point_radius
    )
    add_to_other = True
    for j, place in enumerate(places_of_interest):
        if place not in ids_keys_list:
            continue
        for element_id in ids[place]:
            intersection_area = 0

            if len(locations[element_id]) == 1:
                loc_lat, loc_lon = locations[element_id][0]

                loc_circle = get_polygon(
                    saved_polygons, loc_lat, loc_lon, "place",
                    parameters.place_point_radius
                )

                intersection_area = pause_circle.intersection(
                    loc_circle
                ).area
            elif len(locations[element_id]) >= 3:
                polygon = Polygon(locations[element_id])

                intersection_area = pause_circle.intersection(
                    polygon
                ).area

            if intersection_area > 0:
                all_place_probs[j] += intersection_area
                add_to_other = False

    return all_place_probs, add_to_other


def compute_flight_pause_stats(
    flight_d_vec: np.ndarray, flight_t_vec: np.ndarray,
    pause_t_vec: np.ndarray,
) -> list:
    """This function computes the flight and pause statistics.

    Args:
        flight_d_vec: numpy array, contains flight distances
        flight_t_vec: numpy array, contains flight durations
        pause_t_vec: numpy array, contains pause durations
    Returns:
        list with the following elements:
            av_f_len: float, average flight length
            sd_f_len: float, standard deviation of flight length
            av_f_dur: float, average flight duration
            sd_f_dur: float, standard deviation of flight duration
            av_p_dur: float, average pause duration
            sd_p_dur: float, standard deviation of pause duration
    """
    if len(flight_d_vec) > 0:
        av_f_len = np.mean(flight_d_vec)
        sd_f_len = np.std(flight_d_vec)
        av_f_dur = np.mean(flight_t_vec)
        sd_f_dur = np.std(flight_t_vec)
    else:
        av_f_len = 0
        sd_f_len = 0
        av_f_dur = 0
        sd_f_dur = 0

    if len(pause_t_vec) > 0:
        av_p_dur = np.mean(pause_t_vec)
        sd_p_dur = np.std(pause_t_vec)
    else:
        av_p_dur = 0
        sd_p_dur = 0

    return [av_f_len, sd_f_len, av_f_dur, sd_f_dur, av_p_dur, sd_p_dur]


def final_hourly_prep(
    obs_dur: float, time_at_home: float, dist_traveled: float,
    max_dist_home: float, total_flight_time: float, total_pause_time: float,
    flight_pause_stats: list, all_place_times: list,
    all_place_times_adjusted: list, summary_stats: list, log_tags: dict,
    log_tags_temp: list, datetime_list: List[int],
    places_of_interest: Optional[List[str]]
) -> Tuple[list, dict]:
    """This function prepares the final hourly summary statistics.

    Args:
        obs_dur: float, observed duration
        time_at_home: float, time at home
        dist_traveled: float, distance traveled
        max_dist_home: float, maximum distance from home
        total_flight_time: float, total flight time
        total_pause_time: float, total pause time
        flight_pause_stats: list, flight and pause statistics
        all_place_times: list of float, time spent at places of interest
        all_place_times_adjusted: list of float, adjusted time spent at
            places of interest
        summary_stats: list, summary statistics
        log_tags: dict, contains log of tags of all locations visited
            from openstreetmap
        log_tags_temp: list, log of tags of all locations visited
            from openstreetmap
        datetime_list: list of int, current time
        places_of_interest: list of str, places of interest
    Returns:
        A tuple of:
         a list, summary statistics
         a dict, contains log of tags of all locations visited
            from openstreetmap
    """

    year, month, day, hour = datetime_list[:4]
    (
        av_f_len, sd_f_len, av_f_dur, sd_f_dur, av_p_dur, sd_p_dur
    ) = flight_pause_stats

    if obs_dur == 0:
        res = [
            year,
            month,
            day,
            hour,
            0,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
        ]
        if places_of_interest is not None:
            for place_int in range(2 * len(places_of_interest) + 1):
                res.append(pd.NA)
        summary_stats.append(res)
        log_tags[f"{day}/{month}/{year} {hour}:00"] = []
    else:
        res = [
            year,
            month,
            day,
            hour,
            obs_dur / 60,
            time_at_home / 60,
            dist_traveled / 1000,
            max_dist_home / 1000,
            total_flight_time / 60,
            av_f_len,
            sd_f_len,
            av_f_dur / 60,
            sd_f_dur / 60,
            total_pause_time / 60,
            av_p_dur / 60,
            sd_p_dur / 60,
        ]
        if places_of_interest is not None:
            res += all_place_times
            res += all_place_times_adjusted
        log_tags[f"{day}/{month}/{year} {hour}:00"] = log_tags_temp

        summary_stats.append(res)

    return summary_stats, log_tags


def final_daily_prep(
    obs_dur: float, obs_day: float, obs_night: float, time_at_home: float,
    dist_traveled: float, max_dist_home: float, radius: float,
    diameter: float, num_sig: int, entropy: float, total_flight_time: float,
    total_pause_time: float, flight_pause_stats: list,
    all_place_times: list, all_place_times_adjusted: list,
    summary_stats: list, log_tags: dict, log_tags_temp: list,
    datetime_list: List[int], places_of_interest: Optional[List[str]],
    parameters: Hyperparameters, pcr: float, pcr_stratified: float, i: int
) -> Tuple[list, dict]:
    """This function prepares the final daily summary statistics.

    Args:
        obs_dur: float, observed duration
        obs_day: float, observed duration during the day
        obs_night: float, observed duration during the night
        time_at_home: float, time at home
        dist_traveled: float, distance traveled
        max_dist_home: float, maximum distance from home
        radius: float, radius of gyration
        diameter: float, diameter of gyration
        num_sig: int, number of significant places
        entropy: float, entropy of the trajectory
        total_flight_time: float, total flight time
        total_pause_time: float, total pause time
        flight_pause_stats: list, flight and pause statistics
        all_place_times: list of float, time spent at places of interest
        all_place_times_adjusted: list of float, adjusted time spent at
            places of interest
        summary_stats: list, summary statistics
        log_tags: dict, contains log of tags of all locations visited
            from openstreetmap
        log_tags_temp: list, log of tags of all locations visited
            from openstreetmap
        datetime_list: list of int, current time
        places_of_interest: list of str, places of interest
        parameters: Hyperparameters, hyperparameters in functions
        pcr: float, physical circadian rhythm
        pcr_stratified: float, physical circadian rhythm stratified
        i: int, index of the window
    Returns:
        A tuple of:
         a list, summary statistics
         a dict, contains log of tags of all locations visited
            from openstreetmap
    """

    year, month, day = datetime_list[:3]
    (
        av_f_len, sd_f_len, av_f_dur, sd_f_dur, av_p_dur, sd_p_dur
    ) = flight_pause_stats

    if obs_dur == 0:
        res = [
            year,
            month,
            day,
            0,
            0,
            0,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
        ]
        if parameters.pcr_bool:
            res += [pcr, pcr_stratified]
        if places_of_interest is not None:
            for place_int in range(2 * len(places_of_interest) + 1):
                res.append(pd.NA)
        summary_stats.append(res)
        log_tags[f"{day}/{month}/{year}"] = []
    else:
        res = [
            year,
            month,
            day,
            obs_dur / 3600,
            obs_day / 3600,
            obs_night / 3600,
            time_at_home / 3600,
            dist_traveled / 1000,
            max_dist_home / 1000,
            radius / 1000,
            diameter / 1000,
            num_sig,
            entropy,
            total_flight_time / 3600,
            av_f_len / 1000,
            sd_f_len / 1000,
            av_f_dur / 3600,
            sd_f_dur / 3600,
            total_pause_time / 3600,
            av_p_dur / 3600,
            sd_p_dur / 3600,
        ]
        if parameters.pcr_bool:
            res += [pcr, pcr_stratified]
        if places_of_interest is not None:
            res += all_place_times
            res += all_place_times_adjusted
        summary_stats.append(res)
        if parameters.split_day_night:
            if i % 2 == 0:
                time_cat = "daytime"
            else:
                time_cat = "nighttime"
            log_tags[f"{day}/{month}/{year}, {time_cat}"] = (
                log_tags_temp
            )
        else:
            log_tags[f"{day}/{month}/{year}"] = log_tags_temp

    return summary_stats, log_tags


def format_summary_stats(
    summary_stats: list, log_tags: dict, frequency: Frequency,
    parameters: Hyperparameters, places_of_interest: Optional[List[str]]
) -> Tuple[pd.DataFrame, dict]:
    """This function formats the summary statistics.

    Args:
        summary_stats: list, summary statistics
        log_tags: dict, contains log of tags of all locations visited
            from openstreetmap
        frequency: Frequency, the time windows of the summary statistics
        parameters: Hyperparameters, hyperparameters in functions
            recommend to set it to default
        places_of_interest: list of str, places of interest
    Returns:
        A tuple of:
         a pd dataframe, summary statistics
         a dict, contains log of tags of all locations visited
            from openstreetmap
    """

    summary_stats_df = pd.DataFrame(summary_stats)

    if places_of_interest is None:
        places_of_interest2 = []
        places_of_interest3 = []
    else:
        places_of_interest2 = places_of_interest.copy()
        places_of_interest2.append("other")
        places_of_interest3 = [f"{pl}_adjusted" for pl in places_of_interest]

    if parameters.pcr_bool:
        pcr_cols = [
            "physical_circadian_rhythm",
            "physical_circadian_rhythm_stratified",
        ]
    else:
        pcr_cols = []

    if frequency != Frequency.DAILY:
        summary_stats_df.columns = (
            [
                "year",
                "month",
                "day",
                "hour",
                "obs_duration",
                "home_time",
                "dist_traveled",
                "max_dist_home",
                "total_flight_time",
                "av_flight_length",
                "sd_flight_length",
                "av_flight_duration",
                "sd_flight_duration",
                "total_pause_time",
                "av_pause_duration",
                "sd_pause_duration",
            ]
            + places_of_interest2
            + places_of_interest3
        )
    else:
        summary_stats_df.columns = (
            [
                "year",
                "month",
                "day",
                "obs_duration",
                "obs_day",
                "obs_night",
                "home_time",
                "dist_traveled",
                "max_dist_home",
                "radius",
                "diameter",
                "num_sig_places",
                "entropy",
                "total_flight_time",
                "av_flight_length",
                "sd_flight_length",
                "av_flight_duration",
                "sd_flight_duration",
                "total_pause_time",
                "av_pause_duration",
                "sd_pause_duration",
            ]
            + pcr_cols
            + places_of_interest2
            + places_of_interest3
        )

    if parameters.split_day_night:
        summary_stats_df2 = split_day_night_cols(summary_stats_df)
    else:
        summary_stats_df2 = summary_stats_df

    return summary_stats_df2, log_tags


def gps_summaries(
    traj: np.ndarray,
    tz_str: str,
    frequency: Frequency,
    parameters: Hyperparameters,
    places_of_interest: Optional[List[str]] = None,
    osm_tags: Optional[List[OSMTags]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """This function derives summary statistics from the imputed trajectories

    If the frequency is hourly, it returns
    ["year","month","day","hour","obs_duration","pause_time","flight_time","home_time",
    "max_dist_home", "dist_traveled","av_flight_length","sd_flight_length",
    "av_flight_duration","sd_flight_duration"]
    if the frequency is daily, it additionally returns
    ["obs_day","obs_night","radius","diameter""num_sig_places","entropy",
    "physical_circadian_rhythm","physical_circadian_rhythm_stratified"]

    Args:
        traj: 2d array, output from imp_to_traj(), which is a n by 8 mat,
            with headers as [s,x0,y0,t0,x1,y1,t1,obs]
            where s means status (1 as flight and 0 as pause),
            x0,y0,t0: starting lat,lon,timestamp,
            x1,y1,t1: ending lat,lon,timestamp,
            obs (1 as observed and 0 as imputed)
        tz_str: timezone
        frequency: Frequency, the time windows of the summary statistics
        parameters: Hyperparameters, hyperparameters in functions
            recommend to set it to default
        places_of_interest: list of "osm_tags" places to watch,
            keywords as used in openstreetmaps
            e.g. ["cafe", "hospital", "restaurant"]
        osm_tags: list of tags to search for in openstreetmaps
            avoid using a lot of them if large area is covered
    Returns:
        A tuple of:
         a pd dataframe, with each row as an hour/day,
            and each col as a feature/stat
         a dictionary, contains log of tags of all locations visited
            from openstreetmap
    Raises:
        RuntimeError: if the query to Overpass API fails
        ValueError: Frequency is not valid
    """

    if frequency in [Frequency.HOURLY_AND_DAILY, Frequency.MINUTE]:
        raise ValueError(f"Frequency cannot be {frequency.name.lower()}.")

    if frequency != Frequency.DAILY:
        parameters.split_day_night = False

    ids: Dict[str, List[int]] = {}
    locations: Dict[int, List[List[float]]] = {}
    tags: Dict[int, Dict[str, str]] = {}
    if places_of_interest is not None or parameters.save_osm_log:
        ids, locations, tags = get_nearby_locations(traj, osm_tags)
        ids_keys_list = list(ids.keys())
    else:
        ids_keys_list = []

    obs_traj = traj[traj[:, 7] == 1, :]
    home_lat, home_lon = locate_home(obs_traj, tz_str)
    summary_stats: List[List[float]] = []
    log_tags: Dict[str, List[dict]] = {}
    saved_polygons: Dict[str, Polygon] = {}
    if frequency != Frequency.DAILY:
        # find starting and ending time
        logger.info("Calculating the hourly summary stats...")
        start_stamp, end_stamp = get_time_range(
            traj, [4, 5], tz_str
        )
        window, num_windows = compute_window_and_count(
            start_stamp, end_stamp, frequency.value
        )
    else:
        # find starting and ending time
        logger.info("Calculating the daily summary stats...")
        start_stamp, end_stamp = get_time_range(
            traj, [3, 4, 5], tz_str, 3600*24
        )
        window, num_windows = compute_window_and_count(
            start_stamp, end_stamp, 24*60, parameters.split_day_night
        )

    if num_windows <= 0:
        raise ValueError("start time and end time are not correct")

    for i in range(num_windows):
        if parameters.split_day_night:
            i2 = i // 2
        else:
            i2 = i
        start_time = start_stamp + i2 * window
        end_time = start_stamp + (i2 + 1) * window
        start_time2 = 0
        end_time2 = 0

        current_time_list = stamp2datetime(start_time, tz_str)
        year, month, day, hour = current_time_list[:4]
        # take a subset, the starting point of the last traj <end_time
        # and the ending point of the first traj >start_time
        index_rows = (traj[:, 3] < end_time) * (traj[:, 6] > start_time)

        stop1 = 0
        stop2 = 0
        if parameters.split_day_night:
            index_rows, stop1, stop2, start_time2, end_time2 = (
                get_day_night_indices(
                    traj, tz_str, i, start_time, end_time, current_time_list
                )
            )

        if sum(index_rows) == 0 and parameters.split_day_night:
            # if there is no data in the day, then we need to
            # to add empty rows to the dataframe with 21 columns
            res = [year, month, day] + [0] * 18
            if places_of_interest is not None:
                # add empty data for places of interest
                # for daytime/nighttime + other
                res += [0] * (2 * len(places_of_interest) + 1)
            summary_stats.append(res)
            continue
        elif sum(index_rows) == 0 and not parameters.split_day_night:
            # There is no data and it is daily data, so we need to add empty
            # rows
            res = [year, month, day] + [0] * 3 + [pd.NA] * 15

            if places_of_interest is not None:
                # add empty data for places of interest
                # for daytime/nighttime + other
                res += [0] * (2 * len(places_of_interest) + 1)
            summary_stats.append(res)
            continue

        temp = traj[index_rows, :]
        # take a subset which is exactly one hour/day,
        # cut the trajs at two ends proportionally
        if parameters.split_day_night and i % 2 == 0:
            t0_temp = start_time2
            t1_temp = end_time2
        else:
            t0_temp = start_time
            t1_temp = end_time

        temp = smooth_temp_ends(
            temp, index_rows, t0_temp, t1_temp, parameters, i, start_time,
            end_time2, start_time2, end_time, stop1, stop2
        )

        obs_dur = sum((temp[:, 6] - temp[:, 3])[temp[:, 7] == 1])
        d_home_1 = great_circle_dist(
            home_lat, home_lon, temp[:, 1], temp[:, 2]
            )
        d_home_2 = great_circle_dist(
            home_lat, home_lon, temp[:, 4], temp[:, 5]
            )
        d_home = (d_home_1 + d_home_2) / 2
        max_dist_home = max(np.concatenate((d_home_1, d_home_2)))
        time_at_home = sum((temp[:, 6] - temp[:, 3])[d_home <= 50])
        mov_vec = np.round(
            great_circle_dist(
                temp[:, 4], temp[:, 5], temp[:, 1], temp[:, 2]
            ),
            0,
        )
        flight_d_vec = mov_vec[temp[:, 0] == 1]
        flight_t_vec = (temp[:, 6] - temp[:, 3])[temp[:, 0] == 1]
        pause_t_vec = (temp[:, 6] - temp[:, 3])[temp[:, 0] == 2]
        total_pause_time = sum(pause_t_vec)
        total_flight_time = sum(flight_t_vec)
        dist_traveled = sum(mov_vec)
        # Locations of importance
        all_place_times = []
        all_place_times_adjusted = []
        log_tags_temp = []
        if places_of_interest is not None or parameters.save_osm_log:
            pause_vec = temp[temp[:, 0] == 2]
            pause_array = get_pause_array(
                pause_vec, home_lat, home_lon, parameters
            )

            if places_of_interest is not None:
                all_place_times = [0] * (len(places_of_interest) + 1)
                all_place_times_adjusted = all_place_times[:-1]

            for pause in pause_array:
                if places_of_interest is not None:
                    all_place_probs, add_to_other = (
                        intersect_with_places_of_interest(
                            pause, places_of_interest, saved_polygons,
                            parameters, ids, locations, ids_keys_list
                        )
                    )

                    # in case of pause not in places of interest
                    if add_to_other:
                        all_place_times[-1] += pause[2] / 60
                    else:
                        all_place_probs2 = np.array(all_place_probs) / sum(
                            all_place_probs
                        )
                        chosen_type = np.argmax(all_place_probs2)
                        all_place_times[chosen_type] += pause[2] / 60
                        for h, prob in enumerate(all_place_probs2):
                            all_place_times_adjusted[h] += (
                                prob * pause[2] / 60
                            )

                if parameters.save_osm_log:
                    if pause[2] >= parameters.log_threshold:
                        for place_id, place_coordinates in locations.items():
                            if len(place_coordinates) == 1:
                                if (
                                    great_circle_dist(
                                        pause[0], pause[1],
                                        place_coordinates[0][0],
                                        place_coordinates[0][1],
                                    )[0]
                                    < parameters.place_point_radius
                                ):
                                    log_tags_temp.append(tags[place_id])
                            elif len(place_coordinates) >= 3:
                                polygon = Polygon(place_coordinates)
                                point = Point(pause[0], pause[1])
                                if polygon.contains(point):
                                    log_tags_temp.append(tags[place_id])

        flight_pause_stats = compute_flight_pause_stats(
            flight_d_vec, flight_t_vec, pause_t_vec
        )
        datetime_list = [year, month, day, hour, 0, 0]

        if frequency != Frequency.DAILY:
            summary_stats, log_tags = final_hourly_prep(
                obs_dur, time_at_home, dist_traveled, max_dist_home,
                total_flight_time, total_pause_time, flight_pause_stats,
                all_place_times, all_place_times_adjusted, summary_stats,
                log_tags, log_tags_temp, datetime_list, places_of_interest
            )
        else:
            hours = []
            for j in range(temp.shape[0]):
                time_list = stamp2datetime(
                    (temp[j, 3] + temp[j, 6]) / 2,
                    tz_str,
                    )
                hours.append(time_list[3])
            hours_array = np.array(hours)
            day_index = (hours_array >= 8) * (hours_array <= 19)
            night_index = np.logical_not(day_index)
            day_part = temp[day_index, :]
            night_part = temp[night_index, :]
            obs_day = sum(
                (day_part[:, 6] - day_part[:, 3])[day_part[:, 7] == 1]
            )
            obs_night = sum(
                (night_part[:, 6] - night_part[:, 3])[night_part[:, 7] == 1]
            )
            temp_pause = temp[temp[:, 0] == 2, :]
            centroid_x = np.dot(
                (temp_pause[:, 6] - temp_pause[:, 3]) / total_pause_time,
                temp_pause[:, 1],
            )
            centroid_y = np.dot(
                (temp_pause[:, 6] - temp_pause[:, 3]) / total_pause_time,
                temp_pause[:, 2],
            )
            r_vec = great_circle_dist(
                centroid_x, centroid_y, temp_pause[:, 1], temp_pause[:, 2]
            )
            radius = np.dot(
                (temp_pause[:, 6] - temp_pause[:, 3]) / total_pause_time, r_vec
            )
            _, _, _, t_xy = num_sig_places(temp_pause, 50)
            num_sig = sum(np.array(t_xy) / 60 > 15)
            t_sig = np.array(t_xy)[np.array(t_xy) / 60 > 15]
            p = t_sig / sum(t_sig)
            entropy = -sum(p * np.log(p + 0.00001))
            # physical circadian rhythm
            if obs_dur != 0 and parameters.pcr_bool:
                mobility_trace = create_mobility_trace(traj)
                pcr = routine_index(
                    (start_time, end_time), mobility_trace,
                    parameters.pcr_window, parameters.pcr_sample_rate
                )
                pcr_stratified = routine_index(
                    (start_time, end_time), mobility_trace,
                    parameters.pcr_window, parameters.pcr_sample_rate,
                    True, tz_str
                )
            else:
                pcr = pd.NA
                pcr_stratified = pd.NA

            # if there is only one significant place, the entropy is zero
            # but here it is -log(1.00001) < 0
            # but the small value is added to avoid log(0)
            # this is a bit of a hack, but it works
            if num_sig == 1:
                entropy = 0
            if temp.shape[0] == 1:
                diameter = 0.
            else:
                diameters = pairwise_great_circle_dist(temp[:, [1, 2]])
                diameter = max(diameters)

            summary_stats, log_tags = final_daily_prep(
                obs_dur, obs_day, obs_night, time_at_home, dist_traveled,
                max_dist_home, radius, diameter, num_sig, entropy,
                total_flight_time, total_pause_time, flight_pause_stats,
                all_place_times, all_place_times_adjusted, summary_stats,
                log_tags, log_tags_temp, datetime_list, places_of_interest,
                parameters, pcr, pcr_stratified, i
            )

    summary_stats_df2, log_tags = format_summary_stats(
        summary_stats, log_tags, frequency, parameters, places_of_interest
    )

    return summary_stats_df2, log_tags


def split_day_night_cols(summary_stats_df: pd.DataFrame) -> pd.DataFrame:
    """This function splits the summary statistics dataframe
    into daytime and nighttime columns.

    Args:
        summary_stats_df: pandas dataframe with summary statistics
    Returns:
        pandas dataframe with summary statistics
         split into daytime and nighttime columns
    """

    summary_stats_df_daytime = summary_stats_df[::2].reset_index(drop=True)
    summary_stats_df_nighttime = summary_stats_df[1::2].reset_index(drop=True)

    summary_stats_df2 = pd.concat(
        [
            summary_stats_df_daytime,
            summary_stats_df_nighttime.iloc[:, 3:],
        ],
        axis=1,
    )
    summary_stats_df2.columns = (
        list(summary_stats_df.columns)[:3]
        + [
            f"{cname}_daytime"
            for cname in list(summary_stats_df.columns)[3:]
        ]
        + [
            f"{cname}_nighttime"
            for cname in list(summary_stats_df.columns)[3:]
        ]
    )
    summary_stats_df2 = summary_stats_df2.drop(
        [
            "obs_day_daytime",
            "obs_night_daytime",
            "obs_day_nighttime",
            "obs_night_nighttime",
        ],
        axis=1,
    )
    summary_stats_df2.insert(
        3,
        "obs_duration",
        summary_stats_df2["obs_duration_daytime"]
        + summary_stats_df2["obs_duration_nighttime"],
    )

    return summary_stats_df2


def get_time_range(
    traj: np.ndarray, time_reset_indices: list,
    tz_str: str, offset_seconds: int = 0,
) -> Tuple[int, int]:
    """Computes the starting and ending time stamps
     based on given trajectory and indices.

    Args:
        traj: numpy array of trajectory
        time_reset_indices: list of indices to reset time
        offset_seconds: int, offset in seconds
        tz_str: str, timezone
    Returns:
        A tuple of two integers (start_stamp, end_stamp):
            start_stamp: int, starting time stamp
            end_stamp: int, ending time stamp
    """
    time_list = stamp2datetime(traj[0, 3], tz_str)
    for idx in time_reset_indices:
        time_list[idx] = 0
    start_stamp = datetime2stamp(time_list, tz_str)

    time_list = stamp2datetime(traj[-1, 6], tz_str)
    for idx in time_reset_indices:
        time_list[idx] = 0
    end_stamp = datetime2stamp(time_list, tz_str) + offset_seconds

    return start_stamp, end_stamp


def compute_window_and_count(
    start_stamp: int, end_stamp: int, window_minutes: int,
    split_day_night: bool = False
) -> Tuple[int, int]:
    """Computes the window and number of windows based on given time stamps.

    Args:
        start_stamp: int, starting time stamp
        end_stamp: int, ending time stamp
        window_minutes: int, window in minutes
        split_day_night: bool, True if split day and night
    Returns:
        A tuple of two integers (window, num_windows):
            window: int, window in seconds
            num_windows: int, number of windows
    """

    window = window_minutes * 60
    num_windows = (end_stamp - start_stamp) // window
    if split_day_night:
        num_windows *= 2
    return window, num_windows


def gps_quality_check(study_folder: str, study_id: str) -> float:
    """The function checks the gps data quality.

    Args:
        study_folder (str): The path to the study folder.
        study_id (str): The id code of the study.
    Returns:
        a scalar between 0 and 1, bigger means better data quality
            (percentage of data which meet the criterion)
    """
    gps_path = f"{study_folder}/{study_id}/gps"
    if not os.path.exists(gps_path):
        quality_check = 0.
    else:
        file_list = os.listdir(gps_path)
        for i, _ in enumerate(file_list):
            if file_list[i][0] == ".":
                file_list[i] = file_list[i][2:]
        file_path = [
            f"{gps_path }/{file_list[j]}"
            for j, _ in enumerate(file_list)
        ]
        file_path_array = np.sort(np.array(file_path))
        # check if there are enough data for the following algorithm
        quality_yes = 0.
        for i, _ in enumerate(file_path_array):
            df = pd.read_csv(file_path_array[i])
            if df.shape[0] > 60:
                quality_yes = quality_yes + 1.
        quality_check = quality_yes / (len(file_path_array) + 0.0001)
    return quality_check


def gps_stats_main(
    study_folder: str,
    output_folder: str,
    tz_str: str,
    frequency: Frequency,
    save_traj: bool,
    places_of_interest: Optional[list] = None,
    osm_tags: Optional[List[OSMTags]] = None,
    time_start: Optional[list] = None,
    time_end: Optional[list] = None,
    participant_ids: Optional[list] = None,
    parameters: Optional[Hyperparameters] = None,
    all_memory_dict: Optional[dict] = None,
    all_bv_set: Optional[dict] = None,
):
    """This the main function to do the GPS imputation.
    It calls every function defined before.

    Args:
        study_folder: str, the path of the study folder
        output_folder: str, the path of the folder
            where you want to save results
        tz_str: str, timezone
        frequency: Frequency, the frequency of the summary stats
            (resolution for summary statistics)
        save_traj: bool, True if you want to save the trajectories as a
            csv file, False if you don't
        places_of_interest: list of places to watch,
            keywords as used in openstreetmaps
        osm_tags: list of tags to search for in openstreetmaps
            avoid using a lot of them if large area is covered
        time_start: list, starting time of window of interest
        time_end: list ending time of the window of interest
            time should be a list of integers with format
            [year, month, day, hour, minute, second]
            if time_start is None and time_end is None: then it reads all
            the available files
            if time_start is None and time_end is given, then it reads all
            the files before the given time
            if time_start is given and time_end is None, then it reads all
            the files after the given time
        participant_ids: a list of beiwe IDs
        parameters: Hyperparameters, hyperparameters in functions
            recommend to set it to default
        all_memory_dict: dict, from previous run (none if it's the first time)
        all_bv_set: dict, from previous run (none if it's the first time)
    Returns:
        write summary stats as csv for each user during the specified
            period
        and a log of all locations visited as a json file if required
        and imputed trajectory if required
        and memory objects (all_memory_dict and all_bv_set)
            as pickle files for future use
        and a record csv file to show which users are processed
        and logger csv file to show warnings and bugs during the run
    Raises:
        ValueError: Frequency is not valid
    """

    # no minutely analysis on GPS data
    if frequency == Frequency.MINUTE:
        raise ValueError("Frequency cannot be minutely.")

    os.makedirs(output_folder, exist_ok=True)

    if parameters is None:
        parameters = Hyperparameters()

    pars0 = [
        parameters.l1, parameters.l2, parameters.l3, parameters.a1,
        parameters.a2, parameters.b1, parameters.b2, parameters.b3
    ]
    pars1 = [
        parameters.l1, parameters.l2, parameters.a1, parameters.a2,
        parameters.b1, parameters.b2, parameters.b3, parameters.g
    ]

    # participant_ids should be a list of str
    if participant_ids is None:
        participant_ids = get_ids(study_folder)
    # create a record of processed user participant_id and starting/ending time

    if all_memory_dict is None:
        all_memory_dict = {}
        for participant_id in participant_ids:
            all_memory_dict[str(participant_id)] = None

    if all_bv_set is None:
        all_bv_set = {}
        for participant_id in participant_ids:
            all_bv_set[str(participant_id)] = None

    if frequency == Frequency.HOURLY_AND_DAILY:
        os.makedirs(f"{output_folder}/hourly", exist_ok=True)
        os.makedirs(f"{output_folder}/daily", exist_ok=True)
    if save_traj:
        os.makedirs(f"{output_folder}/trajectory", exist_ok=True)

    for participant_id in participant_ids:
        logger.info("User: %s", participant_id)
        # data quality check
        quality = gps_quality_check(study_folder, participant_id)
        if quality > parameters.quality_threshold:
            # read data
            logger.info("Read in the csv files ...")
            data, _, _ = read_data(
                participant_id, study_folder, "gps",
                tz_str, time_start, time_end,
            )
            if data.shape == (0, 0):
                logger.info("No data available.")
                continue
            if parameters.r is None:
                params_r = float(parameters.itrvl)
            else:
                params_r = parameters.r
            if parameters.h is None:
                params_h = params_r
            else:
                params_h = parameters.h
            if parameters.w is None:
                params_w = np.mean(data.accuracy)
            else:
                params_w = parameters.w
            # process data
            mobmat1 = gps_to_mobmat(
                data, parameters.itrvl, parameters.accuracylim,
                params_r, params_w, params_h
            )
            mobmat2 = infer_mobmat(mobmat1, parameters.itrvl, params_r)
            out_dict = bv_select(
                mobmat2,
                parameters.sigma2,
                parameters.tol,
                parameters.d,
                pars0,
                all_memory_dict[str(participant_id)],
                all_bv_set[str(participant_id)],
            )
            all_bv_set[str(participant_id)] = bv_set = out_dict["BV_set"]
            all_memory_dict[str(participant_id)] = out_dict["memory_dict"]
            try:
                imp_table = impute_gps(
                    mobmat2, bv_set, parameters.method,
                    parameters.switch, parameters.num,
                    parameters.linearity, tz_str, pars1
                )
            except RuntimeError as e:
                logger.error("Error: %s", e)
                continue
            traj = imp_to_traj(imp_table, mobmat2, params_w)
            # raise error if traj coordinates are not in the range of
            # [-90, 90] and [-180, 180]
            if traj.shape[0] > 0:
                if (
                    np.max(traj[:, 1]) > 90
                    or np.min(traj[:, 1]) < -90
                    or np.max(traj[:, 2]) > 180
                    or np.min(traj[:, 2]) < -180
                    or np.max(traj[:, 4]) > 90
                    or np.min(traj[:, 4]) < -90
                    or np.max(traj[:, 5]) > 180
                    or np.min(traj[:, 5]) < -180
                ):
                    raise ValueError(
                        "Trajectory coordinates are not in the range of "
                        "[-90, 90] and [-180, 180]."
                    )
            # save all_memory_dict and all_bv_set
            with open(f"{output_folder}/all_memory_dict.pkl", "wb") as f:
                pickle.dump(all_memory_dict, f)
            with open(f"{output_folder}/all_bv_set.pkl", "wb") as f:
                pickle.dump(all_bv_set, f)
            if save_traj is True:
                pd_traj = pd.DataFrame(traj)
                pd_traj.columns = ["status", "x0", "y0", "t0", "x1", "y1",
                                   "t1", "obs"]
                pd_traj.to_csv(
                    f"{output_folder}/trajectory/{participant_id}.csv",
                    index=False
                )
            if frequency == Frequency.HOURLY_AND_DAILY:
                summary_stats1, logs1 = gps_summaries(
                    traj,
                    tz_str,
                    Frequency.HOURLY,
                    parameters,
                    places_of_interest,
                    osm_tags,
                )
                write_all_summaries(participant_id, summary_stats1,
                                    f"{output_folder}/hourly")
                summary_stats2, logs2 = gps_summaries(
                    traj,
                    tz_str,
                    Frequency.DAILY,
                    parameters,
                    places_of_interest,
                    osm_tags,
                )
                write_all_summaries(participant_id, summary_stats2,
                                    f"{output_folder}/daily")
                if parameters.save_osm_log:
                    os.makedirs(f"{output_folder}/logs", exist_ok=True)
                    with open(
                        f"{output_folder}/logs/locations_logs_hourly.json",
                        "w",
                    ) as hourly:
                        json.dump(logs1, hourly, indent=4)
                    with open(
                        f"{output_folder}/logs/locations_logs_daily.json",
                        "w",
                    ) as daily:
                        json.dump(logs2, daily, indent=4)
            else:
                summary_stats, logs = gps_summaries(
                    traj,
                    tz_str,
                    frequency,
                    parameters,
                    places_of_interest,
                    osm_tags,
                )
                write_all_summaries(
                    participant_id, summary_stats, output_folder
                )
                if parameters.save_osm_log:
                    os.makedirs(f"{output_folder}/logs", exist_ok=True)
                    with open(
                        f"{output_folder}/logs/locations_logs.json",
                        "w",
                    ) as loc:
                        json.dump(logs, loc, indent=4)
        else:
            logger.info(
                "GPS data are not collected"
                " or the data quality is too low"
            )
