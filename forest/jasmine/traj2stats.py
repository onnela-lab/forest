"""Module used to impute missing data, by combining functions defined in other
modules and calculate summary statistics of imputed trajectories.
"""

from enum import Enum
from functools import partial
import os
import pickle
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
import requests
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import transform

from forest.bonsai.simulate_gps_data import bounding_box
from forest.jasmine.data2mobmat import (GPS2MobMat, InferMobMat,
                                        great_circle_dist,
                                        pairwise_great_circle_dist)
from forest.jasmine.mobmat2traj import (Imp2traj, ImputeGPS, locate_home,
                                        num_sig_places)
from forest.jasmine.sogp_gps import BV_select
from forest.poplar.legacy.common_funcs import (datetime2stamp, read_data,
                                               stamp2datetime,
                                               write_all_summaries)

OVERPASS_URL = "http://overpass-api.de/api/interpreter"


class Frequency(Enum):
    """This class enumerates possible frequencies for summary data."""
    HOURLY = "hourly"
    DAILY = "daily"


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
    wgs84_to_aeqd = partial(
        pyproj.transform,
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection),
    )
    aeqd_to_wgs84 = partial(
        pyproj.transform,
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
    )

    center = Point(lat, lon)
    point_transformed = transform(wgs84_to_aeqd, center)
    buffer = point_transformed.buffer(radius)
    return transform(aeqd_to_wgs84, buffer)


def get_nearby_locations(traj: np.ndarray) -> Tuple[dict, dict, dict]:
    """This function returns a dictionary of nearby locations,
    a dictionary of nearby locations' names, and a dictionary of
    nearby locations' coordinates.

    Args:
        traj: numpy array, trajectory
    Returns:
        ids: dictionary, contains nearby locations' ids
        locations: dictionary, contains nearby locations' coordinates
        tags: dictionary, contains nearby locations' tags
    Raises:
        RuntimeError: if the query to Overpass API fails
    """

    pause_vec = traj[traj[:, 0] == 2]
    latitudes: List[float] = [pause_vec[0, 1]]
    longitudes: List[float] = [pause_vec[0, 2]]
    for row in pause_vec:
        minimum_distance = np.min([
            great_circle_dist(row[1], row[2], lat, lon)
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

        query += f"""
        \tnode{bbox}['leisure'];
        \tway{bbox}['leisure'];
        \tnode{bbox}['amenity'];
        \tway{bbox}['amenity'];"""

    query += "\n);\nout geom qt;"

    response = requests.get(OVERPASS_URL, params={"data": query}, timeout=60)
    response.raise_for_status()

    res = response.json()
    ids: Dict[str, List[int]] = {}
    locations: Dict[int, List[List[float]]] = {}
    tags: Dict[int, Dict[str, str]] = {}

    for element in res["elements"]:

        element_id = element["id"]

        if "amenity" in element["tags"]:
            if element["tags"]["amenity"] not in ids.keys():
                ids[element["tags"]["amenity"]] = [element_id]
            else:
                ids[element["tags"]["amenity"]].append(element_id)
        elif "leisure" in element["tags"]:
            if element["tags"]["leisure"] not in ids.keys():
                ids[element["tags"]["leisure"]] = [element_id]
            else:
                ids[element["tags"]["leisure"]].append(element_id)

        if element["type"] == "node":
            locations[element_id] = [[element["lat"], element["lon"]]]
        elif element["type"] == "way":
            locations[element_id] = [
                [x["lat"], x["lon"]] for x in element["geometry"]
            ]

        tags[element_id] = element["tags"]

    return ids, locations, tags


def gps_summaries(
    traj: np.ndarray,
    tz_str: str,
    frequency: Frequency,
    places_of_interest: Union[List[str], None] = None,
    save_log: bool = False,
    threshold: Union[int, None] = None,
    split_day_night: bool = False,
    person_point_radius: float = 2,
    place_point_radius: float = 7.5,
) -> Tuple[pd.DataFrame, dict]:
    """This function derives summary statistics from the imputed trajectories

    If the frequency is hourly, it returns
    ["year","month","day","hour","obs_duration","pause_time","flight_time","home_time",
    "max_dist_home", "dist_traveled","av_flight_length","sd_flight_length",
    "av_flight_duration","sd_flight_duration"]
    if the frequency is daily, it additionally returns
    ["obs_day","obs_night","radius","diameter","num_sig_places","entropy"]

    Args:
        traj: 2d array, output from Imp2traj(), which is a n by 8 mat,
            with headers as [s,x0,y0,t0,x1,y1,t1,obs]
            where s means status (1 as flight and 0 as pause),
            x0,y0,t0: starting lat,lon,timestamp,
            x1,y1,t1: ending lat,lon,timestamp,
            obs (1 as observed and 0 as imputed)
        tz_str: timezone
        frequency: Frequency, the time windows of the summary statistics
        places_of_interest: list of amenities or leisure places to watch,
            keywords as used in openstreetmaps
        save_log: bool, True if you want to output a log of locations
            visited and their tags
        threshold: int, time spent in a pause needs to exceed the threshold
            to be placed in the log
            only if save_log True, in minutes
        split_day_night: bool, True if you want to split all metrics to
            daytime and nighttime patterns
            only for daily frequency
        person_point_radius: float, radius of the person's circle when
            discovering places near him in pauses
        place_point_radius: float, radius of place's circle
            when place is returned as centre coordinates from osm
    Returns:
        a pd dataframe, with each row as an hour/day,
            and each col as a feature/stat
        a dictionary, contains log of tags of all locations visited
            from openstreetmap
    Raises:
        RuntimeError: if the query to Overpass API fails
    """

    if frequency == Frequency.HOURLY:
        split_day_night = False

    ids: Dict[str, List[int]] = {}
    locations: Dict[int, List[List[float]]] = {}
    tags: Dict[int, Dict[str, str]] = {}
    if places_of_interest is not None or save_log:
        ids, locations, tags = get_nearby_locations(traj)

    obs_traj = traj[traj[:, 7] == 1, :]
    home_lat, home_lon = locate_home(obs_traj, tz_str)
    summary_stats: List[List[float]] = []
    log_tags: Dict[str, List[dict]] = {}
    if frequency == Frequency.HOURLY:
        # find starting and ending time
        sys.stdout.write("Calculating the hourly summary stats...\n")
        time_list = stamp2datetime(traj[0, 3], tz_str)
        time_list[4:6] = [0, 0]
        start_stamp = datetime2stamp(time_list, tz_str)
        time_list = stamp2datetime(traj[-1, 6], tz_str)
        time_list[4:6] = [0, 0]
        end_stamp = datetime2stamp(time_list, tz_str)
        # start_time, end_time are exact points
        # (if it ends at 2019-3-8 11 o'clock, then 11 shouldn't be included)
        window = 60 * 60
        no_windows = (end_stamp - start_stamp) // window
    else:
        # find starting and ending time
        sys.stdout.write("Calculating the daily summary stats...\n")
        time_list = stamp2datetime(traj[0, 3], tz_str)
        time_list[3:6] = [0, 0, 0]
        start_stamp = datetime2stamp(time_list, tz_str)
        time_list = stamp2datetime(traj[-1, 6], tz_str)
        time_list[3:6] = [0, 0, 0]
        end_stamp = datetime2stamp(time_list, tz_str) + 3600 * 24
        # if it starts from 2019-3-8 11 o'clock,
        # then our daily summary starts from 2019-3-9)
        window = 60 * 60 * 24
        no_windows = (end_stamp - start_stamp) // window
        if split_day_night:
            no_windows *= 2

    if no_windows <= 0:
        raise ValueError("start time and end time are not correct")

    summary_stats_df = pd.DataFrame([])
    for i in range(no_windows):
        if split_day_night:
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
        if split_day_night:
            current_time_list2 = current_time_list.copy()
            current_time_list3 = current_time_list.copy()
            current_time_list2[3] = 8
            current_time_list3[3] = 20
            start_time2 = datetime2stamp(current_time_list2, tz_str)
            end_time2 = datetime2stamp(current_time_list3, tz_str)
            if i % 2 == 0:
                # daytime
                index_rows = (
                    (traj[:, 3] <= end_time2)
                    * (traj[:, 6] >= start_time2)
                    )
            else:
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

        if sum(index_rows) == 0 and split_day_night:
            # if there is no data in the day, then we need to
            # to add empty rows to the dataframe with 21 columns
            res = [year, month, day] + [0] * 18
            if places_of_interest is not None:
                # add empty data for places of interest
                # for daytime/nighttime + other
                res += [0] * (2 * len(places_of_interest) + 1)
            summary_stats.append(res)
            continue

        temp = traj[index_rows, :]
        # take a subset which is exactly one hour/day,
        # cut the trajs at two ends proportionally
        if split_day_night and i % 2 == 0:
            t0_temp = start_time2
            t1_temp = end_time2
        else:
            t0_temp = start_time
            t1_temp = end_time

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
            if split_day_night and i % 2 != 0:
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
        if places_of_interest is not None or save_log:
            pause_vec = temp[temp[:, 0] == 2]
            pause_array: np.ndarray = np.array([])
            for row in pause_vec:
                if (
                    great_circle_dist(row[1], row[2], home_lat, home_lon)
                    > 2*place_point_radius
                ):
                    if len(pause_array) == 0:
                        pause_array = np.array(
                            [[row[1], row[2], (row[6] - row[3]) / 60]]
                        )
                    elif (
                        np.min(
                            great_circle_dist(
                                row[1], row[2],
                                pause_array[:, 0], pause_array[:, 1],
                            )
                        )
                        > 2*place_point_radius
                    ):
                        pause_array = np.append(
                            pause_array,
                            [[row[1], row[2], (row[6] - row[3]) / 60]],
                            axis=0,
                        )
                    else:
                        pause_array[
                            great_circle_dist(
                                row[1], row[2],
                                pause_array[:, 0], pause_array[:, 1],
                            )
                            <= 2*place_point_radius,
                            -1,
                        ] += (row[6] - row[3]) / 60

            if places_of_interest is not None:
                all_place_times = [0] * (len(places_of_interest) + 1)
                all_place_times_adjusted = all_place_times[:-1]

            for pause in pause_array:
                if places_of_interest is not None:
                    all_place_probs = [0] * len(places_of_interest)
                    pause_circle = transform_point_to_circle(
                        pause[0], pause[1], person_point_radius
                    )
                    add_to_other = True
                    for j, place in enumerate(places_of_interest):
                        for element_id in ids[place]:
                            if len(locations[element_id]) == 1:
                                location_circle = transform_point_to_circle(
                                    locations[element_id][0][0],
                                    locations[element_id][0][1],
                                    place_point_radius,
                                )

                                intersection_area = pause_circle.intersection(
                                    location_circle
                                ).area
                                if intersection_area > 0:
                                    all_place_probs[j] += intersection_area
                                    add_to_other = False

                            elif len(locations[element_id]) >= 3:
                                polygon = Polygon(locations[element_id])

                                intersection_area = pause_circle.intersection(
                                    polygon
                                ).area
                                if intersection_area > 0:
                                    all_place_probs[j] += intersection_area
                                    add_to_other = False

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

                if save_log:
                    if threshold is None:
                        threshold = 60
                        sys.stdout.write(
                            "threshold parameter set to None,"
                            + " automatically converted to 60min."
                            + "\n"
                        )
                    if pause[2] >= threshold:
                        for place_id, place_coordinates in locations.items():
                            if len(place_coordinates) == 1:
                                if (
                                    great_circle_dist(
                                        pause[0], pause[1],
                                        place_coordinates[0][0],
                                        place_coordinates[0][1],
                                    )
                                    < place_point_radius
                                ):
                                    log_tags_temp.append(tags[place_id])
                            elif len(place_coordinates) >= 3:
                                polygon = Polygon(place_coordinates)
                                point = Point(pause[0], pause[1])
                                if polygon.contains(point):
                                    log_tags_temp.append(tags[place_id])

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
        if frequency == Frequency.HOURLY:
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
                    for p in range(2 * len(places_of_interest) + 1):
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
                    dist_traveled,
                    max_dist_home,
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
            if temp.shape[0] == 1:
                diameter = 0
            else:
                diameters = pairwise_great_circle_dist(temp[:, [1, 2]])
                diameter = max(diameters)
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
                if places_of_interest is not None:
                    for p in range(2 * len(places_of_interest) + 1):
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
                if places_of_interest is not None:
                    res += all_place_times
                    res += all_place_times_adjusted
                summary_stats.append(res)
                if split_day_night:
                    if i % 2 == 0:
                        time_cat = "daytime"
                    else:
                        time_cat = "nighttime"
                    log_tags[f"{day}/{month}/{year}, {time_cat}"] = (
                        log_tags_temp
                    )
                else:
                    log_tags[f"{day}/{month}/{year}"] = log_tags_temp
        summary_stats_df = pd.DataFrame(summary_stats)
        if places_of_interest is None:
            places_of_interest2 = []
            places_of_interest3 = []
        else:
            places_of_interest2 = places_of_interest.copy()
            places_of_interest2.append("other")
            places_of_interest3 = [
                f"{pl}_adjusted" for pl in places_of_interest
            ]
        if frequency == Frequency.HOURLY:
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
                + places_of_interest2
                + places_of_interest3
            )

    if split_day_night:
        summary_stats_df_daytime = summary_stats_df[::2].reset_index(
            drop=True
            )
        summary_stats_df_nighttime = summary_stats_df[1::2].reset_index(
            drop=True
            )

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
    else:
        summary_stats_df2 = summary_stats_df

    return summary_stats_df2, log_tags


def gps_quality_check(study_folder: str, id_code: str) -> float:
    """The function checks the gps data quality.

    Args:
        study_folder (str): The path to the study folder.
        id_code (str): The id code of the study.
    Returns:
        a scalar between 0 and 1, bigger means better data quality
            (percentage of data which meet the criterion)
    """
    gps_path = f"{study_folder}/{id_code}/gps"
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


def gps_stats_main(study_folder, output_folder, tz_str, option, save_traj, time_start = None, time_end = None, beiwe_id = None,
    parameters = None, all_memory_dict = None, all_BV_set=None, quality_threshold=None):
    """
    This the main function to do the GPS imputation. It calls every function defined before.
    Args:   study_folder, string, the path of the study folder
            output_folder, string, the path of the folder where you want to save results
            tz_str, string, timezone
            option, 'daily' or 'hourly' or 'both' (resolution for summary statistics)
            save_traj, bool, True if you want to save the trajectories as a csv file, False if you don't
            time_start, time_end are starting time and ending time of the window of interest
            time should be a list of integers with format [year, month, day, hour, minute, second]
            if time_start is None and time_end is None: then it reads all the available files
            if time_start is None and time_end is given, then it reads all the files before the given time
            if time_start is given and time_end is None, then it reads all the files after the given time
            beiwe_id: a list of beiwe IDs
            parameters: hyperparameters in functions, recommend to set it to none (by default)
            all_memory_dict and all_BV_set are dictionaries from previous run (none if it's the first time)
            quality_threshold: more-or-less a percentage value expressed as a floating point of the 
            fraction of data required for a summary to be created.
    Return: write summary stats as csv for each user during the specified period
            and imputed trajectory if required
            and memory objects (all_memory_dict and all_BV_set) as pickle files for future use
            and a record csv file to show which users are processed, from when to when
            and logger csv file to show warnings and bugs during the run
    """
    
    quality_threshold = quality_threshold if quality_threshold is not None else 0.05
    
    if os.path.exists(output_folder)==False:
        os.mkdir(output_folder)

    if parameters == None:
        parameters = [60*60*24*10,60*60*24*30,0.002,200,5,1,0.3,0.2,0.5,100,0.01,0.05,3,10,2,'GLC',10,51,None,None,None]
    [l1,l2,l3,g,a1,a2,b1,b2,b3,d,sigma2,tol,switch,num,linearity,method,itrvl,accuracylim,r,w,h] = parameters
    pars0 = [l1,l2,l3,a1,a2,b1,b2,b3]
    pars1 = [l1,l2,a1,a2,b1,b2,b3,g]

    if r == None:
        orig_r = None
    if w == None:
        orig_w = None
    if h == None:
        orig_h = None

    ## beiwe_id should be a list of str
    if beiwe_id == None:
        beiwe_id = os.listdir(study_folder)
    ## create a record of processed user ID and starting/ending time

    if all_memory_dict == None:
        all_memory_dict = {}
        for ID in beiwe_id:
            all_memory_dict[str(ID)] = None

    if all_BV_set == None:
        all_BV_set = {}
        for ID in beiwe_id:
            all_BV_set[str(ID)] = None

    if option == 'both':
        if os.path.exists(output_folder+"/hourly")==False:
            os.mkdir(output_folder+"/hourly")
        if os.path.exists(output_folder+"/daily")==False:
            os.mkdir(output_folder+"/daily")

    if save_traj == True:
        if os.path.exists(output_folder+"/trajectory")==False:
            os.mkdir(output_folder+"/trajectory")

    if len(beiwe_id)>0:
        for ID in beiwe_id:
            sys.stdout.write('User: '+ ID  + '\n')
            try:
                ## data quality check
                quality = gps_quality_check(study_folder, ID)
                if quality > quality_threshold:
                    ## read data
                    sys.stdout.write("Read in the csv files ..." + '\n')
                    data, stamp_start, stamp_end = read_data(ID, study_folder, "gps", tz_str, time_start, time_end)
                    if orig_r is None:
                        r = itrvl
                    if orig_h is None:
                        h = r
                    if orig_w is None:
                        w = np.mean(data.accuracy)
                    ## process data
                    mobmat1 = GPS2MobMat(data,itrvl,accuracylim,r,w,h)
                    mobmat2 = InferMobMat(mobmat1,itrvl,r)
                    out_dict = BV_select(mobmat2,sigma2,tol,d,pars0,all_memory_dict[str(ID)],all_BV_set[str(ID)])
                    all_BV_set[str(ID)] = BV_set = out_dict["BV_set"]
                    all_memory_dict[str(ID)] = out_dict["memory_dict"]
                    imp_table = ImputeGPS(mobmat2,BV_set,method,switch,num,linearity,tz_str,pars1)
                    traj = Imp2traj(imp_table,mobmat2,itrvl,r,w,h)
                    ## save all_memory_dict and all_BV_set
                    f = open(output_folder + "/all_memory_dict.pkl","wb")
                    pickle.dump(all_memory_dict,f)
                    f.close()
                    f = open(output_folder + "/all_BV_set.pkl","wb")
                    pickle.dump(all_BV_set,f)
                    f.close()
                    if save_traj == True:
                        pd_traj = pd.DataFrame(traj)
                        pd_traj.columns = ["status","x0","y0","t0","x1","y1","t1","obs"]
                        dest_path = output_folder +"/trajectory/" + str(ID) + ".csv"
                        pd_traj.to_csv(dest_path,index=False)
                    if option == 'both':
                        summary_stats1 = gps_summaries(traj,tz_str,'hourly')
                        write_all_summaries(ID, summary_stats1, output_folder + "/hourly")
                        summary_stats2 = gps_summaries(traj,tz_str,'daily')
                        write_all_summaries(ID, summary_stats2, output_folder + "/daily")
                    else:
                        summary_stats = gps_summaries(traj,tz_str,option)
                        write_all_summaries(ID, summary_stats, output_folder)
                else:
                    sys.stdout.write("GPS data are not collected or the data quality is too low." + '\n')
            except:
                sys.stdout.write("An error occured when processing the data." + '\n')
                pass
