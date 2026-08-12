"""
Module used to impute missing data, by combining functions defined in other modules and
calculate summary statistics of imputed trajectories.

Original Authors: Georgios Efstathiadis, Zachary Clement, Josh Barback
Optimization work: Eli Jones

"""
import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from pandas._libs.missing import NAType
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import transform

from forest.bonsai.simulate_gps_data import bounding_box
from forest.constants import Frequency, Frequency as Freq, OSMTags
from forest.jasmine.data2mobmat import (gps_to_mobmat, great_circle_dist, great_circle_dist_opt,
    great_circle_dist_specialized, infer_mobmat, pairwise_great_circle_dist)
from forest.jasmine.mobmat2traj import imp_to_traj, impute_gps, locate_home, num_sig_places
from forest.jasmine.sogp_gps import bv_select
from forest.poplar.legacy.common_funcs import (datetime2stamp, read_data, stamp2datetime,
    write_all_summaries)
from forest.utils import get_ids, overpass_request_json


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PD_TRAJ_COLUMNS = ("status", "x0", "y0", "t0", "x1", "y1", "t1", "obs")

FP64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
SECONDS_IN_DAY = 60 * 60 * 24

PARS0 = tuple[int, int, float, int, int, float, float, float]
PARS1 = tuple[int, int, int, int, float, float, float, int]

TRACE_CACHE = dict[tuple[int, int, int] | str, tuple[FP64Array, FP64Array, FP64Array] | float]

COORDS_OUT_OF_RANGE = "Trajectory coordinates are not in the range of [-90, 90] and [-180, 180]."

@dataclass
class Hyperparameters:
    """Class containing hyperparemeters for gps imputation and trajectory
     summary statistics calculation.
    
    Args:
        itrvl, accuracylim, r, w, h: hyperparameters for the gps_to_mobmat function.
        
        itrvl, r: hyperparameters for the infer_mobmat function.
        
        l1, l2, l3, a1, a2, b1, b2, b3, sigma2, tol, d: hyperparameters for the bv_select function.
        
        l1, l2, a1, a2, b1, b2, b3, g, method, switch, num, linearity: hyperparameters for the
            impute_gps function.
        
        itrvl, r, w, h: hyperparameters for the imp_to_traj function.
        
        log_threshold: int, time spent in a pause needs to exceed the
            log_threshold to be placed in the log only if save_osm_log True, in minutes
        
        split_day_night: bool, True if you want to split all metrics to datetime and nighttime
            patterns only for daily frequency
        
        person_point_radius: float, radius of the person's circle when discovering places near him
            in pauses
        
        place_point_radius: float, radius of place's circle when place is returned as centre
            coordinates from osm
            
        save_osm_log: bool, True if you want to output a log of locations visited and their tags
        
        quality_threshold: float, a percentage value of the fraction of data
            required for a summary to be created
        
        pcr_bool: bool, True if you want to calculate the physical circadian rhythm
        
        pcr_window: int, number of days to look back and forward for calculating the physical
            circadian rhythm
        
        pcr_sample_rate: int, number of seconds between each sample for calculating the physical
            circadian rhythm
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
    r: float | None = None
    w: float | None = None
    h: float | None = None
    
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


def transform_point_to_circle(lat: float, lon: float, radius: float) -> Polygon:
    """ This function transforms a set of cooordinates to a shapely circle with a provided radius.
    
    Args:
        lat: float, latitude of the center of the circle
        lon: float, longitude of the center of the circle
        radius: float, in meters
    Returns:
        shapely polygon of a circle
    """
    
    local_azimuthal_projection = (f"+proj=aeqd +R=6371000 +units=m +lat_0={lat} +lon_0={lon}")
    wgs84_to_aeqd = Transformer.from_crs(
        "+proj=longlat +datum=WGS84 +no_defs", local_azimuthal_projection
    ).transform
    aeqd_to_wgs84 = Transformer.from_crs(
        local_azimuthal_projection, "+proj=longlat +datum=WGS84 +no_defs"
    ).transform
    
    center = Point(lat, lon)
    point_transformed = transform(wgs84_to_aeqd, center)
    buffer = point_transformed.buffer(radius)
    return transform(aeqd_to_wgs84, buffer)


def get_nearby_locations(
    traj: np.ndarray,
    osm_tags: list[OSMTags] | None = None,
) -> tuple[dict[str, list[int]], dict[int, list[list[float]]], dict[int, dict[str, str]]]:
    """ This function returns a dictionary of nearby locations, a dictionary of nearby locations'
    names, and a dictionary of nearby locations' coordinates.
    
    Args:
        traj: numpy array, trajectory osm_tags: list of OSMTags (in constants), types of nearby
        locations supported by Overpass
            API defaults to [OSMTags.AMENITY, OSMTags.LEISURE]
    Returns:
        A tuple of:
            dictionary, contains nearby locations' ids dictionary, contains nearby locations'
            coordinates dictionary, contains nearby locations' tags
    Raises:
        RuntimeError: if the query to Overpass API fails
    """
    
    if osm_tags is None:
        osm_tags = [OSMTags.AMENITY, OSMTags.LEISURE]
    pause_vec = traj[traj[:, 0] == 2]
    latitudes: list[float] = [pause_vec[0, 1]]
    longitudes: list[float] = [pause_vec[0, 2]]
    for row in pause_vec:
        minimum_distance = np.min([
            great_circle_dist(row[1], row[2], lat, lon)[0]
            for lat, lon in zip(latitudes, longitudes)
        ])
        # only add to the list if they are not too close with the other coordinates in the list
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
                \tway{bbox}[ 'building'='residential'];
                \tnode{bbox}['building'='office'];
                \tway{bbox}[ 'building'='office'];
                \tnode{bbox}['building'='commercial'];
                \tway{bbox}[ 'building'='commercial'];
                \tnode{bbox}['building'='supermarket'];
                \tway{bbox}[ 'building'='supermarket'];
                \tnode{bbox}['building'='stadium'];
                \tway{bbox}[ 'building'='stadium'];"""
            elif tag == OSMTags.HIGHWAY:
                query += f"""
                \tnode{bbox}['highway'='motorway'];
                \tway{bbox}[ 'highway'='motorway'];
                \tnode{bbox}['highway'='trunk'];
                \tway{bbox}[ 'highway'='trunk'];
                \tnode{bbox}['highway'='primary'];
                \tway{bbox}[ 'highway'='primary'];
                \tnode{bbox}['highway'='secondary'];
                \tway{bbox}[ 'highway'='secondary'];
                \tnode{bbox}['highway'='tertiary'];
                \tway{bbox}[ 'highway'='tertiary'];
                \tnode{bbox}['highway'='road'];
                \tway{bbox}[ 'highway'='road'];"""
            else:
                query += f"""
                \tnode{bbox}['{tag.value}'];
                \tway{bbox}[ '{tag.value}'];"""
    
    query += "\n);\nout geom qt;"
    
    try:
        res = overpass_request_json(query, method="POST")
    except requests.exceptions.Timeout as err:
        raise RuntimeError(
            "Query to Overpass API timed out. The OpenStreetMap query may be too large. "
            "(save_osm_log and places_of_interest should be ommitted when not required.)"
        ) from err
    except requests.exceptions.HTTPError as err:
        raise RuntimeError(f"Query to Overpass API failed: {err}") from err
    
    ids: dict[str, list[int]] = {}
    locations: dict[int, list[list[float]]] = {}
    tags: dict[int, dict[str, str]] = {}
    
    for element in res["elements"]:
        element_id = element["id"]
        
        for tag in osm_tags:
            if tag.value in element["tags"]:
                if element["tags"][tag.value] not in ids:
                    ids[element["tags"][tag.value]] = [element_id]
                else:
                    ids[element["tags"][tag.value]].append(element_id)
                continue
        
        if element["type"] == "node":
            locations[element_id] = [[element["lat"], element["lon"]]]
        elif element["type"] == "way":
            locations[element_id] = [[x["lat"], x["lon"]] for x in element["geometry"]]
        
        tags[element_id] = element["tags"]
    
    return ids, locations, tags


## Routine Index


def routine_index(
    time_range: tuple[int, int],
    mobility_trace: np.ndarray,
    cache: TRACE_CACHE,
    pcr_window: int = 14,
    pcr_sample_rate: int = 30,
    stratified: bool = False,
    timezone: str = "US/Eastern",
) -> np.float64:
    """ This function calculates the routine index of a trajectory
    
    Description of routine index can be found in the paper:
    
    Canzian and Musolesi's 2015 paper in the Proceedings of the 2015 ACM International Joint
    Conference on Pervasive and Ubiquitous Computing, titled “Trajectories of depression:
    unobtrusive monitoring of depressive states by means of smartphone mobility traces analysis.”
    
    Args:
        time_range: tuple
            tuple of two ints, time range of mobility_trace
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
    
    # We have several variables that we can save a bunch of computation on through caching
    time_col, sampled_trace, prealloc_array = _get_inner_params(mobility_trace, pcr_sample_rate, cache)
    
    t_1, t_2, n_days_1, n_days_2 = _get_time_components(time_range, time_col, pcr_window)
    
    if max(n_days_1, n_days_2) == 0:
        return np.float64(0.0)
    
    shifts = list(range(1, n_days_1 + 1)) + list(range(-n_days_2, 0))
    if stratified:
        time_mid = int((t_1 + t_2) / 2)
        weekend_today = datetime(*stamp2datetime(time_mid, timezone)).weekday() >= 5
        if weekend_today:
            shifts = [
                s for s in shifts
                if datetime(*stamp2datetime(time_mid - s * SECONDS_IN_DAY, timezone)).weekday() >= 5
            ]
        else:
            shifts = [
                s for s in shifts
                if datetime(*stamp2datetime(time_mid - s * SECONDS_IN_DAY, timezone)).weekday() < 5
            ]
    
    # The original code used these un-mutated time_range values. It seems to me that t_1 and t_2
    # are slightly more constrained and probably cause no changes in the output, but the difference
    # is well below something I can benchmark, so we will keep the original values.
    average_traces_sum = _innermost_loop(
        time_range[0], time_range[1], shifts, cache, time_col, sampled_trace, prealloc_array
    )
    return average_traces_sum / (n_days_1 + n_days_2)


@numba.njit(cache=True, fastmath=True)
def _get_time_components(time_range, time_col, pcr_window):
    """ There are some very small performance wins if we compile this, reduces mess. """
    
    t_1, t_2 = time_range
    t_init = time_col.min()
    t_fin = time_col.max()
    t_1 = max(t_1, t_init)
    t_2 = min(t_2, t_fin)
    
    # n1, n2 are the number of days before and after the time range
    n1 = int(round((t_1 - t_init) / (SECONDS_IN_DAY)))
    n2 = int(round((t_fin - t_2) / (SECONDS_IN_DAY)))
    
    # to avoid long computational times only look at the last window days and next window days
    n1 = min(n1, pcr_window)
    n2 = min(n2, pcr_window)
    return t_1, t_2, n1, n2


def _get_inner_params(
    mobility_trace: FP64Array,
    pcr_sample_rate: int,
    cache: TRACE_CACHE
) -> tuple[FP64Array, FP64Array, FP64Array]:
    # These items are memory heavy, but are based on the value of an invariant - the mobility trace.
    # caching them instead of recomputing them is a 20% overall speedup.
    if params := cache.get("mobility_trace"):
        # this is a critical code fast path, this object type is correct, ignore the type warning.
        return params  # type: ignore
    
    # This code runs substantially slower when compiled with numba.
    time_col = np.asfortranarray(mobility_trace[:, 2])  # much more performant as fortran array.
    
    # minutely better to declare with order F
    preallocated_array = np.empty((mobility_trace.shape[0], 3), order="F")
    preallocated_array[:, :2] = mobility_trace[:, :2]
    
    # minutely but consistently better as fortran array
    sampled_trace = np.asfortranarray(mobility_trace[::pcr_sample_rate])
    
    cache["mobility_trace"] = time_col, sampled_trace, preallocated_array  # cache and return
    return time_col, sampled_trace, preallocated_array


def _innermost_loop(
    time_start: int,
    time_end: int,
    shifts: list[int],
    cache: TRACE_CACHE,
    time_col: FP64Array,
    sampled_trace: FP64Array,
    preallocated_array: FP64Array,
) -> float:
    # This function is hit twice with calls that repeat work, we can save on that by caching answers
    # on the already known parameters. There is virtually no overhead, performance gain depends on
    # the number of shifts per call, which changes with pcl_window (I think).
    
    prealloc_view = preallocated_array.view()[:, 2]  # using views directly is also slightly faster
    time_col = time_col.view()
    
    the_sum = 0.0
    for i in shifts:
        # preallocated_array[:, 2] = time_col + i * SECONDS_IN_DAY  # numba chokes on this operation
        # This construction at least avoids creating an intermediate array, numba still chokes.
        np.add(time_col, i * SECONDS_IN_DAY, out=prealloc_view)
        cache_key = (time_start, time_end, i)
        
        if hit := cache.get(cache_key):  # there is _no_ overhead to updating this cache
            # this is a critical fast path, this object type is correct, ignore the type warning.
            the_sum += hit  # type: ignore
            continue
        
        # time for the real math
        avg = avg_mobility_trace_difference(time_start, time_end, sampled_trace, preallocated_array)
        cache[cache_key] = avg
        the_sum += avg
    
    return the_sum


# "float64(float64, float64, float64[:,:], float64[:,:])"
@numba.njit(cache=True, fastmath=True)
def avg_mobility_trace_difference(
    time_start: int, time_end: int, mobility_trace1: FP64Array, mobility_trace2: FP64Array
) -> float:
    """ This function calculates the average mobility trace difference

    Args:
        time_start: int, starting time of the window
        time_end: int, ending time of the window
        mobility_trace1: numpy array, mobility trace 1
            contains 3 columns: [x, y, t]
        mobility_trace2: numpy array, mobility trace 2
            contains 3 columns: [x, y, t]
    Returns:
        float, average mobility trace difference
    Raises:
        ValueError: if the calculation fails
    """
    
    common_times, tr1, tr2 = _masks_and_common(time_start, time_end, mobility_trace1, mobility_trace2)
    if len(common_times) == 0:
        return 0
    
    mask2_common = _trace_diff_and_mask(common_times, mobility_trace2, tr2)
    mask1_common = _trace_diff_and_mask(common_times, mobility_trace1, tr1)
    
    dists = great_circle_dist_specialized(
        mobility_trace1, mask1_common, mobility_trace2, mask2_common
    )
    
    res, is_nan = _dist_flag_compute(dists)
    if is_nan:
        raise ValueError("PCR calculation failed")
    return res


# "Tuple((float64[:], float64[:], float64[:]))(int64, int64, float64[:,:], float64[:,:])"
@numba.njit(cache=True, fastmath=True)
def _masks_and_common(
    time_start: int, time_end: int, trace1: FP64Array, trace2: FP64Array,
) -> tuple[FP64Array, np.ndarray, np.ndarray]:
    """  """
    # wrapping this with numba jit yields about a 10% speedup overall improvement
    
    tr1 = trace1[:, 2]
    tr2 = trace2[:, 2]
    mask1 = (tr1 >= time_start) & (tr1 <= time_end)
    mask2 = (tr2 >= time_start) & (tr2 <= time_end)
    
    # Find common timestamps using array intersec - return is unique.
    # Original code: `common_times = list(set(mt1[mask1, 2]) & set(mt2[mask2, 2]))`
    return np.intersect1d(trace1[mask1, 2], trace2[mask2, 2]), tr1, tr2


# "boolean[:](float64[:], float64[:,:], float64[:])"
@numba.njit(cache=True, fastmath=True)
def _trace_diff_and_mask(
    common: FP64Array, mobility_trace: FP64Array, times: FP64Array
) -> BoolArray:
    """ Two-pointer merge: requires times be sorted ascending, which is guaranteed by
    create_mobility_trace's np.unique dedup/sort. """
    
    mask_size = mobility_trace.shape[0]
    common_size = common.shape[0]
    mask = np.zeros(mask_size, dtype=np.bool_)
    
    common_idx = 0
    for times_idx in range(mask_size):
        t = times[times_idx]
        while common_idx < common_size and common[common_idx] < t:
            common_idx += 1
        if common_idx < common_size and common[common_idx] == t:
            mask[times_idx] = True
    return mask


# "Tuple((float64, boolean))(float64[:])",
@numba.njit(cache=True, fastmath=True)
def _dist_flag_compute(dists: FP64Array) -> tuple[float, bool]:
    # gets a moderate speedup from numba
    dist_flag = dists <= 10
    res = np.mean(dist_flag)
    return float(res), np.isnan(res)


## End Routine Index

## create_mobility_trace


def create_mobility_trace(traj: np.ndarray) -> FP64Array:
    """ This function creates a mobility trace from a trajectory

    Args:
        traj: numpy array, trajectory
            contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
    Returns:
        numpy array, mobility trace
            contains 3 columns: [x, y, t]
    """
    
    pause_vec: np.ndarray = traj[traj[:, 0] == 2]
    
    # Calculate the time ranges for all pauses
    start_times: np.ndarray = pause_vec[:, 3].astype(np.int_)
    end_times: np.ndarray = pause_vec[:, 6].astype(np.int_)
    time_ranges = [np.arange(s, e) for s, e in zip(start_times, end_times)]
    
    # Flatten time_ranges and get the corresponding locations
    
    flat_time_ranges = np.concatenate(time_ranges, dtype=np.float64)
    repeats = [len(r) for r in time_ranges]
    locs = np.repeat(pause_vec[:, 1:3], repeats, axis=0)
    
    # Stack locations and time_ranges to get the mobility trace
    mobility_trace = _optimized_column_stack(locs, flat_time_ranges)
    
    # With an optimized O(n) sort-check we can claw out about a 8% speedup on real data.
    filtered_trace = mobility_trace[:, 2]
    if _is_sorted_unique(filtered_trace):
        return mobility_trace
    
    ## np.unique is slow and unjittable. This numba-compatible change to build locs and
    ## flat_time_ranges ends up being slightly slower by about 20% even with compilation.
    # l = sum([len(r) for r in time_ranges])  # precompute total length for allocation
    # flat_time_ranges = np.empty(l, dtype=np.float64)
    # offset = 0
    # for r in time_ranges:
    #     flat_time_ranges[offset:offset+len(r)] = r
    #     offset += len(r)
    # locs = np.empty((l, 2), dtype=np.float64)
    # offset = 0
    # for i, r in enumerate(time_ranges):
    #     n = len(r)
    #     locs[offset:offset+n, 0] = pause_vec[i, 1]
    #     locs[offset:offset+n, 1] = pause_vec[i, 2]
    #     offset += n
    
    _, unique_indices = np.unique(filtered_trace, return_index=True)
    
    return mobility_trace[unique_indices]


@numba.njit(cache=True, fastmath=True)
def _is_sorted_unique(times: NDArray[np.float64]) -> bool:
    for i in range(1, len(times)):  # noqa - ruff with SIM enabled will mark this as simplifyable
        if times[i] <= times[i - 1]:  # but the replacement is not numba compatible.
            return False
    return True


## end create_mobility_trace



@numba.njit(cache=True, fastmath=True)
def _optimized_column_stack(arr1: NDArray[np.float64], arr2: NDArray[np.float64]) -> NDArray[np.float64]:
    # np.column_stack always benefits a little from numba
    return np.column_stack((arr1, arr2))


def get_day_night_indices(
    traj: np.ndarray,
    tz_str: str,
    index: int,
    start_time: int,
    end_time: int,
    current_time_list: list[int],
) -> tuple[np.ndarray, int, int, int, int]:
    """ This function returns the indices of the rows in the trajectory
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
    index1 = ((traj[:, 6] < start_time2) * (traj[:, 3] < end_time) * (traj[:, 6] > start_time))
    index2 = ((traj[:, 3] > end_time2) * (traj[:, 3] < end_time) * (traj[:, 6] > start_time))
    stop1 = sum(index1) - 1
    stop2 = sum(index1)
    index_rows = index1 + index2
    
    return index_rows, stop1, stop2, start_time2, end_time2


def smooth_temp_ends(
    temp: np.ndarray,
    index_rows: np.ndarray,
    t0_temp: float,
    t1_temp: float,
    parameters: Hyperparameters,
    i: int,
    start_time: int,
    end_time2: int,
    start_time2: int,
    end_time: int,
    stop1: int,
    stop2: int,
) -> np.ndarray:
    """ This function smooths the starting and ending points of the
    trajectory.
    
    Args:
        temp: numpy array, trajectory
            contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
        index_rows: numpy array, indices of the rows in the trajectory
            if the trajectory is split into day and night
        t0_temp: float, starting time of the trajectory
        t1_temp: float, ending time of the trajectory
            parameters: Hyperparameters, hyperparameters in functions recommend to set it to default
        i: int, index of the window
        start_time: int, starting time of the window
        end_time2: int, ending time of the second part of the trajectory
        start_time2: int, starting time of the second part of the trajectory
        end_time: int, ending time of the window
        stop1: int,
            index of the row in the trajectorywhere the first part of the trajectory ends
        stop2: int,
            index of the row in the trajectory where the second part of the trajectory starts
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
        
        # if expanding range, then convert to imputed status and not observed
        if max(p0, p1) > 1:
            temp[0, 7] = 0
    else:
        if parameters.split_day_night and i % 2 != 0:
            t0_temp_l = [start_time, end_time2]
            t1_temp_l = [start_time2, end_time]
            start_temp = [0, stop2]
            end_temp = [stop1, -1]
            for j in range(2):
                p0 = (
                    (temp[start_temp[j], 6] - t0_temp_l[j]) /
                    (temp[start_temp[j], 6] - temp[start_temp[j], 3])
                )
                p1 = (
                    (t1_temp_l[j] - temp[end_temp[j], 3]) /
                    (temp[end_temp[j], 6] - temp[end_temp[j], 3])
                )
                temp[start_temp[j], 1] = (
                    (1 - p0) * temp[start_temp[j], 4] + p0 * temp[start_temp[j], 1]
                )
                temp[start_temp[j], 2] = (
                    (1 - p0) * temp[start_temp[j], 5] + p0 * temp[start_temp[j], 2]
                )
                
                temp[start_temp[j], 3] = t0_temp_l[j]
                temp[end_temp[j], 4] = (1 - p1) * temp[end_temp[j], 1] + p1 * temp[end_temp[j], 4]
                temp[end_temp[j], 5] = (1 - p1) * temp[end_temp[j], 2] + p1 * temp[end_temp[j], 5]
                temp[end_temp[j], 6] = t1_temp_l[j]
                
                if p0 > 1:
                    temp[start_temp[j], 7] = 0
                if p1 > 1:
                    temp[end_temp[j], 7] = 0
        
        else:  # (this is a for-else block, not an if-else block)
            p0 = (temp[0, 6] - t0_temp) / (temp[0, 6] - temp[0, 3])
            p1 = (t1_temp - temp[-1, 3]) / (temp[-1, 6] - temp[-1, 3])
            temp[0, 1] = (1 - p0) * temp[0, 4] + p0 * temp[0, 1]
            temp[0, 2] = (1 - p0) * temp[0, 5] + p0 * temp[0, 2]
            temp[0, 3] = t0_temp
            temp[-1, 4] = (1 - p1) * temp[-1, 1] + p1 * temp[-1, 4]
            temp[-1, 5] = (1 - p1) * temp[-1, 2] + p1 * temp[-1, 5]
            temp[-1, 6] = t1_temp
            
            if p0 > 1:
                temp[0, 7] = 0
            if p1 > 1:
                temp[-1, 7] = 0
    
    return temp


def get_pause_array(
    pause_vec: np.ndarray, home_lat: float, home_lon: float, parameters: Hyperparameters
) -> np.ndarray:
    """ This function returns a numpy array of pauses.
    
    Args:
        pause_vec: numpy array, contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
        home_lat: float, latitude of the home
        home_lon: float, longitude of the home
        parameters: Hyperparameters, hyperparameters in functions
    Returns:
        pause_array: numpy array, contains 3 columns: [x, y, t]
    """
    
    array: np.ndarray = np.array([])
    ppr = parameters.place_point_radius
    for row in pause_vec:
        if (great_circle_dist(row[1], row[2], home_lat, home_lon)[0] > 2 * ppr):
            
            if len(array) == 0:
                array = np.array([extract_pause_from_row(row)])
            
            elif np.min(great_circle_dist(row[1], row[2], array[:, 0], array[:, 1])) > 2 * ppr:
                array = np.append(array, [extract_pause_from_row(row)], axis=0)
            
            else:
                argmin = np.argmin(great_circle_dist(row[1], row[2], array[:, 0], array[:, 1]))
                array[argmin, -1] += (row[6] - row[3]) / 60
    
    return array


def extract_pause_from_row(row: np.ndarray) -> list:
    """ This function extracts the pause from a row in a trajectory.
    
    Args:
        row: numpy array, contains 8 columns: [s,x0,y0,t0,x1,y1,t1,obs]
    Returns:
        list, pause
    """
    return [row[1], row[2], (row[6] - row[3]) / 60]


def get_polygon(
    saved_polygons: dict,
    lat: float,
    lon: float,
    label: str,
    radius: float,
) -> tuple[Polygon, dict]:
    """ This function returns a saved polygon if it exists, or computes a polygon and saves it.
    
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
    if loc_str in saved_polygons:
        return saved_polygons[loc_str], saved_polygons
    
    circle = transform_point_to_circle(lat, lon, radius)
    saved_polygons[loc_str] = circle
    return circle, saved_polygons


def intersect_with_places_of_interest(
    pause: list,
    places_of_interest: list,
    saved_polygons: dict,
    parameters: Hyperparameters,
    ids: dict,
    locations: dict,
    ids_keys_list: list,
) -> tuple[list, bool]:
    """ This function computes the intersection between a pause and
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
        saved_polygons, pause[0], pause[1], "person", parameters.person_point_radius
    )
    add_to_other = True
    for j, place in enumerate(places_of_interest):
        if place not in ids_keys_list:
            continue
        for element_id in ids[place]:
            intersection_area = 0
            
            if len(locations[element_id]) == 1:
                # TODO: this branch is not covered by a test
                loc_lat, loc_lon = locations[element_id][0]

                # `_` is second reference to the unmodified saved_polygons dictionary
                loc_circle, _ = get_polygon(
                    saved_polygons, loc_lat, loc_lon, "place", parameters.place_point_radius
                )
                
                intersection_area = pause_circle.intersection(loc_circle).area
            elif len(locations[element_id]) >= 3:
                polygon = Polygon(locations[element_id])
                
                intersection_area = pause_circle.intersection(polygon).area
            
            if intersection_area > 0:
                all_place_probs[j] += intersection_area
                add_to_other = False
    
    return all_place_probs, add_to_other


def compute_flight_pause_stats(
    flight_d_vec: np.ndarray, flight_t_vec: np.ndarray, pause_t_vec: np.ndarray
) -> list:
    """ This function computes the flight and pause statistics.
    
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
    obs_dur: float,
    time_at_home: float,
    dist_traveled: float,
    max_dist_home: float,
    total_flight_time: float,
    total_pause_time: float,
    flight_pause_stats: list,
    all_place_times: list,
    all_place_times_adjusted: list,
    summary_stats: list,
    log_tags: dict,
    log_tags_temp: list,
    datetime_list: list[int],
    places_of_interest: list[str] | None,
) -> tuple[list, dict]:
    """ This function prepares the final hourly summary statistics.
    
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
    dt = datetime(year, month, day).strftime("%Y-%m-%d")
    av_f_len, sd_f_len, av_f_dur, sd_f_dur, av_p_dur, sd_p_dur = flight_pause_stats
    
    if obs_dur == 0:
        res = [
            dt, hour, 0, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
        ]
        if places_of_interest is not None:
            for _place_int in range(2 * len(places_of_interest) + 1):
                res.append(pd.NA)
        summary_stats.append(res)
        log_tags[f"{day}/{month}/{year} {hour}:00"] = []
    else:
        res = [
            dt,  # year, month, day
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
    obs_dur: float,
    obs_day: float,
    obs_night: float,
    time_at_home: float,
    dist_traveled: float,
    max_dist_home: float,
    radius: float,
    diameter: float,
    num_sig: int,
    entropy: float,
    total_flight_time: float,
    total_pause_time: float,
    flight_pause_stats: list,
    all_place_times: list,
    all_place_times_adjusted: list,
    summary_stats: list,
    log_tags: dict,
    log_tags_temp: list,
    datetime_list: list[int],
    places_of_interest: list[str] | None,
    parameters: Hyperparameters,
    pcr: float,
    pcr_stratified: float,
    i: int,
) -> tuple[list, dict]:
    """ This function prepares the final daily summary statistics.

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
        all_place_times_adjusted: list of float, adjusted time spent at places of interest
        summary_stats: list, summary statistics
        log_tags: dict, contains log of tags of all locations visited from openstreetmap
        log_tags_temp: list, log of tags of all locations visited from openstreetmap
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
    NA = pd.NA
    
    yr, mo, day = datetime_list[:3]
    date = datetime(yr, mo, day).strftime("%Y-%m-%d")
    av_f_len, sd_f_len, av_f_dur, sd_f_dur, av_p_dur, sd_p_dur = flight_pause_stats
    if parameters.split_day_night:
        if obs_dur == 0:
            res = [yr, mo, day, 0, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA]
            if parameters.pcr_bool:
                res += [pcr, pcr_stratified]
            if places_of_interest is not None:
                for _place_int in range(2 * len(places_of_interest) + 1):
                    res.append(pd.NA)
            summary_stats.append(res)
            log_tags[f"{day}/{mo}/{yr}"] = []
        else:
            res = [
                yr,
                mo,
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
            
            time_cat = "daytime" if i % 2 == 0 else "nighttime"
            log_tags[f"{day}/{mo}/{yr}, {time_cat}"] = (log_tags_temp)
    else:
        if obs_dur == 0:
            res = [date, 0, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA]
            if parameters.pcr_bool:
                res += [pcr, pcr_stratified]
            if places_of_interest is not None:
                for place_int in range(2 * len(places_of_interest) + 1):
                    res.append(pd.NA)
            summary_stats.append(res)
            log_tags[f"{day}/{mo}/{yr}"] = []
        else:
            res = [
                date,  # year, month, day
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
            log_tags[f"{day}/{mo}/{yr}"] = log_tags_temp
    
    return summary_stats, log_tags


def format_summary_stats(
    summary_stats: list,
    log_tags: dict,
    frequency: Frequency,
    parameters: Hyperparameters,
    places_of_interest: list[str] | None,
) -> tuple[pd.DataFrame, dict]:
    """ This function formats the summary statistics.
    
    Args:
        summary_stats: list, summary statistics
        log_tags: dict, contains log of tags of all locations visited from openstreetmap
        frequency: Frequency, the time windows of the summary statistics
        parameters: Hyperparameters, hyperparameters in functions, recommend to set it to default.
        places_of_interest: list of str, places of interest
    Returns:
        A tuple of:
         a pd dataframe, summary statistics
         a dict, contains log of tags of all locations visited
            from openstreetmap
    """
    
    summary_stats_df = pd.DataFrame(summary_stats)
    if parameters.split_day_night:
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
                ] + places_of_interest2 + places_of_interest3
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
                ] + pcr_cols + places_of_interest2 + places_of_interest3
            )
        summary_stats_df2 = split_day_night_cols(summary_stats_df)
    else:
        if places_of_interest is None:
            places_of_interest2 = []
            places_of_interest3 = []
        else:
            places_of_interest2 = places_of_interest.copy()
            places_of_interest2.append("Other")
            places_of_interest3 = [f"{pl} Adjusted" for pl in places_of_interest]
        
        if parameters.pcr_bool:
            pcr_cols = [
                "Physical Circadian Rhythm",
                "Physical Circadian Rhythm Stratified",
            ]
        else:
            pcr_cols = []
        
        if frequency != Frequency.DAILY:
            summary_stats_df.columns = (
                [
                    "Date",
                    "Hour",
                    "Obs Duration",
                    "Home Duration",
                    "Distance Traveled",
                    "Distance From Home",
                    "Total Flight Time",
                    "Flight Distance Average",
                    "Flight Distance Stddev",
                    "Flight Duration Average",
                    "Flight Duration Stddev",
                    "Pause Time",
                    "Av Pause Duration",
                    "Sd Pause Duration",
                ] + pcr_cols + places_of_interest2 + places_of_interest3
            )
        else:
            summary_stats_df.columns = (
                [
                    "Date",
                    "Obs Duration",
                    "Obs Day",
                    "Obs Night",
                    "Home Duration",
                    "Distance Traveled",
                    "Distance From Home",
                    "Gyration Radius",
                    "Distance Diameter",
                    "Significant Location Count",
                    "Significant Location Entropy",
                    "Total Flight Time",
                    "Flight Distance Average",
                    "Flight Distance Stddev",
                    "Flight Duration Average",
                    "Flight Duration Stddev",
                    "Pause Time",
                    "Av Pause Duration",
                    "Sd Pause Duration",
                ] + pcr_cols + places_of_interest2 + places_of_interest3
            )
        
        if frequency != Frequency.DAILY:
            new_column_order = [
                "Date",
                "Hour",
                "Distance From Home",
                "Distance Traveled",
                "Flight Distance Average",
                "Flight Distance Stddev",
                "Flight Duration Average",
                "Flight Duration Stddev",
                "Home Duration",
                "Pause Time",
                "Obs Duration",
                "Total Flight Time",
                "Av Pause Duration",
                "Sd Pause Duration",
            ]
        else:
            new_column_order = [
                "Date",
                "Distance Diameter",
                "Distance From Home",
                "Distance Traveled",
                "Flight Distance Average",
                "Flight Distance Stddev",
                "Flight Duration Average",
                "Flight Duration Stddev",
                "Home Duration",
                "Gyration Radius",
                "Significant Location Count",
                "Significant Location Entropy",
                "Pause Time",
                "Obs Duration",
                "Obs Day",
                "Obs Night",
                "Total Flight Time",
                "Av Pause Duration",
                "Sd Pause Duration",
            ]
        
        full_column_order = new_column_order + [
            col for col in summary_stats_df.columns if col not in new_column_order
        ]
        summary_stats_df = summary_stats_df[full_column_order]
        summary_stats_df2 = summary_stats_df
    
    return summary_stats_df2, log_tags

## GPS Summaries

def gps_summaries(
    traj: np.ndarray,
    tz_str: str,
    frequency: Frequency,
    parameters: Hyperparameters,
    places_of_interest: list[str] | None = None,
    osm_tags: list[OSMTags] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """ This function derives summary statistics from the imputed trajectories
    
    If the frequency is hourly, it returns
    [
     "year", "month", "day", "hour",
     "obs_duration",
     "pause_time",
     "flight_time",
     "home_time",
     "max_dist_home",
     "dist_traveled",
     "av_flight_length",
     "sd_flight_length",
     "av_flight_duration"
     "sd_flight_duration"
    ]
    
    if the frequency is daily, it additionally returns
    [
      "obs_day",
      "obs_night",
      "radius",
      "diameter",
      "num_sig_places",
      "entropy",
      "physical_circadian_rhythm",
      "physical_circadian_rhythm_stratified",
    ]
    
    Args:
        traj: 2d array, output from imp_to_traj(), which is an N by 8 matrix with headers as
            `[s, x0, y0, t0, x1, y1, t1, obs]`
            where
            - s means status (1 as flight and 0 as pause).
            - x0, y0, t0: starting latitude, longitude, and timestamp
            - x1, y1, t1: ending latitude, longitude, and timestamp
            - obs: observation flag (1 as observed and 0 as imputed)
        
        tz_str: timezone string
        
        frequency: forest.constants.Frequency
            The time windows of the summary statistics
        
        parameters: Hyperparameters
            Hyperparameter settings passed in to further functions, see the Hyperparameters class.
            Note that enabling pcr_bool can be very computationally intensive.
        
        places_of_interest: list
            list of "osm_tags" places to watch, keywords as used in openstreetmaps
            e.g. ["cafe", "hospital", "restaurant"]
        
        osm_tags: list
            list of tags to search for in openstreetmaps
            (may be computationally intensive, particularly in large areas)
        
    Returns:
        A tuple of:
         a pd dataframe, with each row as an hour/day, and each col as a feature/stat
         
         a dictionary, contains log of tags of all locations visited from openstreetmap
    
    Raises:
        RuntimeError: if the query to Overpass API fails
        ValueError: Frequency is not valid
    """
    
    window, num_windows, start_stamp = _gps_frequency_init(traj, tz_str, frequency, parameters)
    
    ids, ids_keys_list, locations, tags = _gps_ids_locations_and_tags(
        traj, parameters, places_of_interest, osm_tags
    )
    
    obs_traj = traj[traj[:, 7] == 1, :]
    home_lat, home_lon = locate_home(obs_traj, tz_str)
    
    summary_stats: list[list[float | NAType]] = []
    log_tags: dict[str, list[dict]] = {}
    saved_polygons: dict[str, Polygon] = {}
    
    mobility_trace = None
    cache: TRACE_CACHE = {}
    
    for i in range(num_windows):
        start_time2 = 0
        end_time2 = 0
        stop1 = 0
        stop2 = 0
        
        i2 = i // 2 if parameters.split_day_night else i
        start_time = start_stamp + i2 * window
        end_time = start_stamp + (i2 + 1) * window
        
        current_time_list = stamp2datetime(start_time, tz_str)
        
        # take a subset, the starting point of the last traj <end_time and the ending point of the
        # first traj >start_time
        index_rows: BoolArray = (traj[:, 3] < end_time) * (traj[:, 6] > start_time)
        
        if parameters.split_day_night:
            index_rows, stop1, stop2, start_time2, end_time2 = (
                get_day_night_indices(traj, tz_str, i, start_time, end_time, current_time_list)
            )
        
        # if there is an empty row move on to the next window
        if not _gps_handle_empty_rows(
            index_rows, frequency, parameters, places_of_interest, current_time_list, summary_stats
        ):
            continue
        
        temp = traj[index_rows, :]
        # take a subset which is exactly one hour/day, cut the trajs at two ends proportionally
        if parameters.split_day_night and i % 2 == 0:
            t0_temp, t1_temp = start_time2, end_time2
        else:
            t0_temp, t1_temp = start_time, end_time
        
        temp = smooth_temp_ends(
            temp, index_rows, t0_temp, t1_temp, parameters, i, start_time, end_time2, start_time2,
            end_time, stop1, stop2
        )
        
        obs_dur = sum((temp[:, 6] - temp[:, 3])[temp[:, 7] == 1])
        
        # physical circadian rhythm  -  only logically compatible with daily
        if obs_dur != 0 and parameters.pcr_bool and frequency == Frequency.DAILY:
            
            if mobility_trace is None:
                # traj never mutates, we only need to calculate the mobility trace once
                mobility_trace = create_mobility_trace(traj)
            
            # using a cache on these sequential calls drops a benchmark where I ran 100 of these
            # on live data from 30s to 26s
            pcr = routine_index(
                (start_time, end_time),
                mobility_trace,
                cache,
                parameters.pcr_window,
                parameters.pcr_sample_rate,
                # False,
                # tz_str,
            )
            pcr_stratified = routine_index(
                (start_time, end_time),
                mobility_trace,
                cache,
                parameters.pcr_window,
                parameters.pcr_sample_rate,
                True,
                tz_str,
            )
        else:
            # pd.NAType is a duck-typed object compatible with float/float64. Ignore type warning
            pcr = pd.NA  # type: ignore
            pcr_stratified = pd.NA  # type: ignore
        
        # Locations of importance
        all_place_times = []
        all_place_times_adjusted = []
        log_tags_temp = []
        if places_of_interest is not None or parameters.save_osm_log:
            pause_vec = temp[temp[:, 0] == 2]
            pause_array = get_pause_array(pause_vec, home_lat, home_lon, parameters)
            
            if places_of_interest is not None:
                all_place_times = [0] * (len(places_of_interest) + 1)
                all_place_times_adjusted = all_place_times[:-1]
            
            for pause in pause_array:
                if places_of_interest is not None:
                    all_place_probs, add_to_other = (
                        intersect_with_places_of_interest(
                            pause, places_of_interest, saved_polygons, parameters, ids, locations,
                            ids_keys_list
                        )
                    )
                    
                    # in case of pause not in places of interest
                    if add_to_other:
                        all_place_times[-1] += pause[2] / 60
                    else:
                        all_place_probs2 = np.array(all_place_probs) / sum(all_place_probs)
                        chosen_type = np.argmax(all_place_probs2)
                        all_place_times[chosen_type] += pause[2] / 60
                        for h, prob in enumerate(all_place_probs2):
                            all_place_times_adjusted[h] += (prob * pause[2] / 60)
                
                if parameters.save_osm_log and pause[2] >= parameters.log_threshold:
                    for place_id, place_coordinates in locations.items():
                        
                        if len(place_coordinates) == 1:
                            if great_circle_dist(
                                pause[0], pause[1], place_coordinates[0][0], place_coordinates[0][1]
                            )[0] < parameters.place_point_radius:
                                log_tags_temp.append(tags[place_id])
                        
                        elif len(place_coordinates) >= 3:
                            polygon = Polygon(place_coordinates)
                            point = Point(pause[0], pause[1])
                            if polygon.contains(point):
                                log_tags_temp.append(tags[place_id])
        
        # distances etc
        d_home_1 = great_circle_dist_opt(home_lat, home_lon, temp[:, 1], temp[:, 2])
        d_home_2 = great_circle_dist_opt(home_lat, home_lon, temp[:, 4], temp[:, 5])
        d_home = (d_home_1 + d_home_2) / 2
        max_dist_home = max(np.concatenate((d_home_1, d_home_2)))
        time_at_home = sum((temp[:, 6] - temp[:, 3])[d_home <= 50])
        mov_vec = np.round(great_circle_dist_opt(temp[:, 4], temp[:, 5], temp[:, 1], temp[:, 2]), 0)
        flight_d_vec = mov_vec[temp[:, 0] == 1]
        flight_t_vec = (temp[:, 6] - temp[:, 3])[temp[:, 0] == 1]
        pause_t_vec = (temp[:, 6] - temp[:, 3])[temp[:, 0] == 2]
        total_pause_time = sum(pause_t_vec)
        total_flight_time = sum(flight_t_vec)
        dist_traveled = sum(mov_vec)
        flight_pause_stats = compute_flight_pause_stats(flight_d_vec, flight_t_vec, pause_t_vec)
        datetime_list = current_time_list[:4] + [0, 0]
        
        if frequency != Frequency.DAILY:
            summary_stats, log_tags = final_hourly_prep(
                obs_dur,
                time_at_home,
                dist_traveled,
                max_dist_home,
                total_flight_time,
                total_pause_time,
                flight_pause_stats,
                all_place_times,
                all_place_times_adjusted,
                summary_stats,
                log_tags,
                log_tags_temp,
                datetime_list,
                places_of_interest,
            )
        else:
            hours = []
            for j in range(temp.shape[0]):
                time_list = stamp2datetime((temp[j, 3] + temp[j, 6]) / 2, tz_str)
                hours.append(time_list[3])
            
            hours_array = np.array(hours)
            day_index = (hours_array >= 8) * (hours_array <= 19)
            night_index = np.logical_not(day_index)
            day_part = temp[day_index, :]
            night_part = temp[night_index, :]
            obs_day = sum((day_part[:, 6] - day_part[:, 3])[day_part[:, 7] == 1])
            obs_night = sum((night_part[:, 6] - night_part[:, 3])[night_part[:, 7] == 1])
            temp_pause = temp[temp[:, 0] == 2, :]
            
            centroid_x = np.dot(
                (temp_pause[:, 6] - temp_pause[:, 3]) / total_pause_time, temp_pause[:, 1]
            )
            centroid_y = np.dot(
                (temp_pause[:, 6] - temp_pause[:, 3]) / total_pause_time, temp_pause[:, 2]
            )
            
            r_vec = great_circle_dist(centroid_x, centroid_y, temp_pause[:, 1], temp_pause[:, 2])
            radius = np.dot((temp_pause[:, 6] - temp_pause[:, 3]) / total_pause_time, r_vec)
            _, _, _, t_xy = num_sig_places(temp_pause, 50)
            num_sig = sum(np.array(t_xy) / 60 > 15)
            t_sig = np.array(t_xy)[np.array(t_xy) / 60 > 15]
            p = t_sig / sum(t_sig)
            
            entropy = -sum(p * np.log(p + 0.00001))
            # if there is only one significant place, the entropy is zero but here it is
            # -log(1.00001) < 0 but the small value is added to avoid log(0)
            if num_sig == 1:
                entropy = 0
            diameter = 0.0 if temp.shape[0] == 1 else max(pairwise_great_circle_dist(temp[:, [1, 2]]))
            
            summary_stats, log_tags = final_daily_prep(
                obs_dur,
                obs_day,
                obs_night,
                time_at_home,
                dist_traveled,
                max_dist_home,
                radius,
                diameter,
                num_sig,
                entropy,
                total_flight_time,
                total_pause_time,
                flight_pause_stats,
                all_place_times,
                all_place_times_adjusted,
                summary_stats,
                log_tags,
                log_tags_temp,
                datetime_list,
                places_of_interest,
                parameters,
                pcr,
                pcr_stratified,
                i,
            )
    
    summary_stats_df2, log_tags = format_summary_stats(
        summary_stats, log_tags, frequency, parameters, places_of_interest
    )
    return summary_stats_df2, log_tags


def _gps_ids_locations_and_tags(
    traj: np.ndarray,
    parameters: Hyperparameters,
    places_of_interest: list[str] | None,
    osm_tags: list[OSMTags] | None,
) -> tuple[dict[str, list[int]], list[str], dict[int, list[list[float]]], dict[int, dict[str, str]]]:
    """ Helper function for gps_summaries, handles osm tags and places of interest variables. """
    
    if places_of_interest is not None or parameters.save_osm_log:
        ids, locations, tags = get_nearby_locations(traj, osm_tags)
        return ids, list(ids), locations, tags
    
    return {}, [], {}, {}


def _gps_frequency_init(
    traj: np.ndarray, tz_str: str, frequency: Frequency, parameters: Hyperparameters
) -> tuple[int, int, int]:
    """ Helper function for gps_summaries, handles frequency related initializations. """
    
    if frequency in [Frequency.HOURLY_AND_DAILY, Frequency.MINUTE]:
        raise ValueError(f"Frequency cannot be {frequency.name.lower()}.")
    
    if frequency != Frequency.DAILY:
        parameters.split_day_night = False  # force valid configuration
    
    if frequency != Frequency.DAILY:
        # find starting and ending time
        logger.info("Calculating the hourly summary stats...")
        start_stamp, end_stamp = get_time_range(traj, [4, 5], tz_str)
        window, num_windows = compute_window_and_count(start_stamp, end_stamp, frequency.value)
    else:
        # find starting and ending time
        logger.info("Calculating the daily summary stats...")
        start_stamp, end_stamp = get_time_range(traj, [3, 4, 5], tz_str, 3600 * 24)
        window, num_windows = compute_window_and_count(
            start_stamp, end_stamp, 24 * 60, parameters.split_day_night
        )
    
    if num_windows <= 0:
        raise ValueError(f"start time {start_stamp} and end time {end_stamp} are not correct.")
    
    return window, num_windows, start_stamp


def _gps_handle_empty_rows(
    index_rows: BoolArray,
    frequency: Frequency,
    parameters: Hyperparameters,
    places_of_interest: list[str] | None,
    current_time_list: list[int | float | NAType],
    summary_stats: list[list[int | float | NAType]],
) -> bool:
    """ Helper function for gps_summaries, handles empty row values. """
    
    if sum(index_rows) != 0:
        return True
    
    # year, month, day  --  mypy doesn't like lists with multiple types
    row= current_time_list[:3]
    
    # cases with no data in the day
    if parameters.split_day_night:
        # if there is no data in the day add empty rows to the dataframe with 21 columns
        row += [0]*18
    else:
        if frequency == Frequency.DAILY:
            row += [0]*3 + [pd.NA]*15  # pad it out
        else:
            # TODO: there is no test for this case
            row += [current_time_list[4], 0] + [pd.NA] * 11  # lead with the hour...
    
    if parameters.pcr_bool and frequency == Frequency.DAILY:  # Frequency.DAILY is implied True...
        row += [pd.NA] * 2  # circadian rhythm columns
    
    # add columns for places of interest...
    if places_of_interest is not None:
        row += [0] * (2 * len(places_of_interest) + 1)
    
    summary_stats.append(row)
    return False


## End GPS Summaries


def split_day_night_cols(summary_stats_df: pd.DataFrame) -> pd.DataFrame:
    """ This function splits the summary statistics dataframe
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
        list(summary_stats_df.columns)[:3] +
        [f"{cname}_daytime" for cname in list(summary_stats_df.columns)[3:]] +
        [f"{cname}_nighttime" for cname in list(summary_stats_df.columns)[3:]]
    )
    summary_stats_df2 = summary_stats_df2.drop(
        ["obs_day_daytime", "obs_night_daytime", "obs_day_nighttime", "obs_night_nighttime"],
        axis=1,
    )
    summary_stats_df2.insert(
        3,
        "obs_duration",
        summary_stats_df2["obs_duration_daytime"] + summary_stats_df2["obs_duration_nighttime"],
    )
    
    return summary_stats_df2


def get_time_range(
    traj: np.ndarray,
    time_reset_indices: list,
    tz_str: str,
    offset_seconds: int = 0,
) -> tuple[int, int]:
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
    start_stamp: int,
    end_stamp: int,
    window_minutes: int,
    split_day_night: bool = False
) -> tuple[int, int]:
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
        file_path = [f"{gps_path}/{file_list[j]}" for j, _ in enumerate(file_list)]
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
    places_of_interest: list | None = None,
    osm_tags: list[OSMTags] | None = None,
    time_start: list | None = None,
    time_end: list | None = None,
    participant_ids: list | None = None,
    parameters: Hyperparameters | None = None,
    all_memory_dict: dict | None = None,
    all_bv_set: dict | None = None,
):
    """ This the main function to do the GPS imputation.
    It calls every function defined before.
    
    Args:
        study_folder: str
            the path of the study folder
        output_folder: str
            the path of the folder where you want to save results. A folder named jasmine will be
            created containing all output.
        tz_str: str | timezone
            The desired timezone to use.
        frequency: constants.Frequency
            The frequency of the summary stats (resolution for summary statistics)
        save_traj: bool
            True if you want to save the trajectories as a csv file, False if you don't
        places_of_interest: list | None
            list of places to watch, keywords as used in openstreetmaps
        osm_tags: list | None
            list of tags to search for in openstreetmaps.
            Avoid using a lot of them if large area is covered.
        time_start: list
            Starting time of window of interest.
        time_end: list
            Ending time of the window of interest time should be a list of integers with format
                [year, month, day, hour, minute, second]
            if time_start is None and time_end is None, it reads all the available files.
            if time_start is None and time_end is given, it reads all files before the given time.
            if time_start is given and time_end is None, it reads all files after the given time.
        participant_ids: list
            A list of Beiwe Platform Participant IDs
        parameters: traj2stats.Hyperparameters
            Hyperparameters may require substantial computation.
        all_memory_dict: dict
            The all_memory_dict from previous run (None if it's the first time).
            Will be written as output to a clearly named file during the run.
        all_bv_set: dict
            The all_bv_set from a previous run (None if it's the first time).
            Will be written as output to a clearly named file during the run.
            
    Returns:
        Writes summary stats as csv for each user during the specified period.
        Optional output:
        - A log of all locations visited as a json file.
        - Imputed trajectory to a csv file.
        - Memory objects (all_memory_dict and all_bv_set) as pickle files for future use.
        - A record csv file to show which users were processed.
        - A logger csv file to show warnings and bugs during the run
    Raises:
        ValueError: Frequency is not valid
    """
    # no minutely analysis on GPS data
    if frequency == Frequency.MINUTE:
        raise ValueError("Frequency cannot be minutely.")
    
    parameters = parameters or Hyperparameters()
    frequencies = [Freq.HOURLY, Freq.DAILY] if frequency == Freq.HOURLY_AND_DAILY else [frequency]
    
    # Ensure that the correct output folder structures exist, centralize folder names
    trajectory_folder = f"{output_folder}/trajectory"
    logs_folder = f"{output_folder}/logs"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    
    # Do the same for frequencies and optional trajectory output
    for freq in frequencies:
        os.makedirs(f"{output_folder}/{freq.name.lower()}", exist_ok=True)
    if save_traj:
        os.makedirs(trajectory_folder, exist_ok=True)
    
    pars0: PARS0 = (
        parameters.l1, parameters.l2, parameters.l3, parameters.a1, parameters.a2, parameters.b1,
        parameters.b2, parameters.b3
    )
    pars1: PARS1 = (
        parameters.l1, parameters.l2, parameters.a1, parameters.a2, parameters.b1, parameters.b2,
        parameters.b3, parameters.g
    )
    
    # participant_ids should be a list of str
    participant_ids = participant_ids or get_ids(study_folder)
    
    # Create a record of processed participant_id and starting/ending time.
    # These are updated and saved to disk after each participant is processed.
    all_memory_dict_file = f"{output_folder}/all_memory_dict.pkl"
    all_bv_set_file = f"{output_folder}/all_bv_set.pkl"
    
    all_memory_dict = all_memory_dict or {p_id: None for p_id in participant_ids}
    all_bv_set = all_bv_set or {p_id: None for p_id in participant_ids}
    
    for participant_id in participant_ids:
        logger.info("User: %s", participant_id)
        
        # data quality check...
        quality = gps_quality_check(study_folder, participant_id)
        if quality <= parameters.quality_threshold:
            logger.info("GPS data are not collected or the data quality is too low")
            continue
        
        logger.info("Read in the csv files ...")  # read data...
        data, _, _ = read_data(
            participant_id,
            study_folder,
            "gps",
            tz_str,
            time_start,
            time_end,
        )
        assert isinstance(data, pd.DataFrame), "Data should be a pandas dataframe."
        
        # If the data comes from a study thata hada GPS fuzzing, and the study was prior to March
        # 2023, the longitude coordinates may be outside of the required range of (-180, 180). This
        # chunk of code wraps out of range coordinates to be in that range
        if (
            ("longitude" in data.columns) and
            ((data["longitude"].max() > 180) or (data["longitude"].min() < -180))
        ):
            logger.info("Reconciled bad longitude data for user %s", participant_id)
            data["longitude"] = (data["longitude"] + 180) % 360 - 180
            
            if ((places_of_interest is not None) or (osm_tags is not None)):
                logger.warning(
                    "Warning: user %s had longitude values outside the valid range [-180, 180] "
                    "but OSM location summaries were requested. Longitude values outside the "
                    "valid range may signify that GPS fuzzing was directed to be used in the "
                    "study setup file. If GPS coordinates were fuzzed, OSM location summaries "
                    "are meaningless", participant_id
                )
        
        if data.shape == (0, 0):
            logger.info("No data available.")
            continue
        
        # finally done with most checks and setup
        params_r = float(parameters.itrvl) if parameters.r is None else parameters.r
        params_h = params_r if parameters.h is None else parameters.h
        params_w = np.mean(data.accuracy) if parameters.w is None else parameters.w
        
        # process data
        mobmat1 = gps_to_mobmat(
            data, parameters.itrvl, parameters.accuracylim, params_r, params_w, params_h
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
        
        # impute_gps can fail, if so we skip this participant.
        try:
            imp_table = impute_gps(
                mobmat2, bv_set, parameters.method, parameters.switch, parameters.num,
                parameters.linearity, tz_str, pars1
            )
        except RuntimeError as e:
            logger.error("Error: %s", e)
            continue
        
        traj = imp_to_traj(imp_table, mobmat2, params_w)
        
        # raise error if traj coordinates are not in the range of [-90, 90] and [-180, 180]
        if traj.shape[0] > 0 and (
            np.max(traj[:, 1]) > 90  or np.min(traj[:, 1]) < -90  or
            np.max(traj[:, 2]) > 180 or np.min(traj[:, 2]) < -180 or
            np.max(traj[:, 4]) > 90  or np.min(traj[:, 4]) < -90  or
            np.max(traj[:, 5]) > 180 or np.min(traj[:, 5]) < -180
        ):
            raise ValueError(COORDS_OUT_OF_RANGE)
        
        # save all_memory_dict and all_bv_set
        with open(all_memory_dict_file, "wb") as f1, open(all_bv_set_file, "wb") as f2:
            pickle.dump(all_memory_dict, f1)
            pickle.dump(all_bv_set, f2)
        
        if save_traj is True:
            pd_traj = pd.DataFrame(traj)
            pd_traj.columns = ["status", "x0", "y0", "t0", "x1", "y1", "t1", "obs"]
            pd_traj.to_csv(f"{trajectory_folder}/{participant_id}.csv", index=False)
        
        # generate summary stats. (variable "frequency" is already declared in signature)
        for freq in frequencies:
            gps_stats_generate_summary(
                traj=traj,
                tz_str=tz_str,
                frequency=freq,
                participant_id=participant_id,
                output_folder=f"{output_folder}/{freq.name.lower()}",
                logs_folder=logs_folder,
                parameters=parameters,
                places_of_interest=places_of_interest,
                osm_tags=osm_tags,
            )


def gps_stats_generate_summary(
    traj: np.ndarray,
    tz_str: str,
    frequency: Frequency,
    participant_id: str,
    output_folder: str,
    logs_folder: str,
    parameters: Hyperparameters,
    places_of_interest: list | None = None,
    osm_tags: list[OSMTags] | None = None,
):
    """ This is simply the inner functionality of gps_stats_main.
    Runs summaries code, writes to disk, saves logs if required. """
    
    summary_stats, logs = gps_summaries(
        traj,
        tz_str,
        frequency,
        parameters,
        places_of_interest,
        osm_tags
    )
    
    write_all_summaries(participant_id, summary_stats, output_folder)
    
    if parameters.save_osm_log:
        with open(f"{logs_folder}/locations_logs_{frequency.name.lower()}.json", "a") as loc:
            json.dump(logs, loc, indent=4)


def handle_out_of_range_coodinate_bug(
    data: pd.DataFrame,
    participant_id: str,
    places_of_interest: list | None,
    osm_tags: list[OSMTags] | None,
):
    """ If the data comes from a study thata had a GPS fuzzing enabled, and the study was prior to
    March 2023 (and probably just iOS devices), the longitude coordinates may be outside of the
    required range of (-180, 180). This chunk of code wraps out of range coordinates to be in that
    range. """
    if (
        ("longitude" in data.columns) and
        ((data["longitude"].max() > 180) or (data["longitude"].min() < -180))
    ):
        logger.info("Reconciled bad longitude data for user %s", participant_id)
        data["longitude"] = (data["longitude"] + 180) % 360 - 180
        
        if ((places_of_interest is not None) or (osm_tags is not None)):
            logger.warning(
                "Warning: user %s had longitude values outside the valid range [-180, 180] "
                "but OSM location summaries were requested. Longitude values outside the "
                "valid range may signify that GPS fuzzing was directed to be used in the "
                "study setup file. If GPS coordinates were fuzzed, OSM location summaries "
                "are meaningless", participant_id
            )

##
## old versions of mobility_trace_difference, preserved for reference
##

# def avg_mobility_trace_difference(
#     time_range: tuple[int, int], mobility_trace1: FP64Array, mobility_trace2: FP64Array
# ) -> float:
#     """ This function calculates the average mobility trace difference

#     Args:
#         time_range:
#             tuple of two ints, time range of mobility_trace
#         mobility_trace1: numpy array, mobility trace 1
#             contains 3 columns: [x, y, t]
#         mobility_trace2: numpy array, mobility trace 2
#             contains 3 columns: [x, y, t]
#     Returns:
#         float, average mobility trace difference
#     Raises:
#         ValueError: if the calculation fails
#     """
#     # Create masks for timestamps that lie within the specified time range
#     # mask1 = ((mobility_trace1[:, 2] >= time_range[0]) & (mobility_trace1[:, 2] <= time_range[1]))
#     # mask2 = ((mobility_trace2[:, 2] >= time_range[0]) & (mobility_trace2[:, 2] <= time_range[1]))


#     # Original was slower, required more lists again later:
#     #    common_times = list(set(mt1[mask1, 2]) & set(mt2[mask2, 2]))
#     #    common_times = np.intersect1d(mobility_trace1[mask1, 2], mobility_trace2[mask2, 2])

#     # Find common timestamps using an optimized array intersection and staying in ndarrays
#     common_times = _masks_and_common(mobility_trace1, mobility_trace2, time_range)
#     if len(common_times) == 0:  # short circuit on no common times
#         return 0

#     # Create masks for the common timestamps
#     # The unique guarantee allows us to use assume_unique=True, about
#     # 3x faster according to docs. (Numba makes isin much slower, v0.66)
#     # mask1_common = np.isin(mobility_trace1[:, 2], common_times, assume_unique=True)
#     # mask2_common = np.isin(mobility_trace2[:, 2], common_times, assume_unique=True)
#     # We have created _isin() below, which is a reduced-scope copy
#     # of np.isin(), keeping only what we need (~10% boost)

#     # Though these appear independent (no shared writes) threading did not provide a speedup;
#     # it slows down. Suspect does not release the GIL. Slowdown at the time was about 1.2x.
#     mask2_common = _isin(mobility_trace2[:, 2], common_times)
#     mask1_common = _isin(mobility_trace1[:, 2], common_times)

#     # The existitng great circle distance code had several slow spots and accepted extra input
#     # types. Rewriting it with several different optimizations and merging in the mask operation
#     # itself was a substantial speedup, probably 30%
#     dists = _great_circle_dist_specialized(
#         mobility_trace1, mask1_common, mobility_trace2, mask2_common
#     )

#     # small function with just enough compute on our inputs to be useful to compile.
#     res, is_nan = _dist_flag_compute(dists)

#     if is_nan:
#         raise ValueError("PCR calculation failed")
#     return res


# this function was part of optimization work, it pulls out the relevant logic from np.isin(),
# and resulted in a small speedup. It was later superseded.
# def _isin(ar1: FP64Array, ar2: FP64Array) -> BoolArray:
#     """ This function was copied out of the numpy source and reduced in scope in order to optimize
#     it. Numba makes multiple parts in here slower and was not workable. """

#     ar1_shape = np.asarray(ar1).shape  # stash a reference to the original shape
#     ar1 = np.asarray(ar1).ravel()  # ravel (kind of a weak copy) both arrays,
#     ar2 = np.asarray(ar2).ravel()  # old comment: "behavior for the first array could be different"

#     # This code is run when
#     # a) the first condition is true, making the code significantly faster
#     # b) the second condition is true (i.e. `ar1` or `ar2` may contain arbitrary objects), since
#     #    then sorting is not guaranteed to work
#     if len(ar2) < 10 * len(ar1) ** 0.145:
#         mask = np.zeros(len(ar1), dtype=np.bool_)
#         for a in ar2:
#             mask |= (ar1 == a)
#         return mask

#     # ar = np.concatenate((ar1, ar2))  # replacing the concatenate may be slightly faster
#     ar = np.empty(ar1.shape[0] + ar2.shape[0], dtype=ar1.dtype)
#     ar[:ar1.shape[0]] = ar1
#     ar[ar1.shape[0]:] = ar2

#     # Must be a stable sort. Values from array 1 must come before those from array 2 when sorted.
#     order = ar.argsort(stable=True)
#     sar = ar[order]
#     bool_ar = (sar[1:] == sar[:-1])

#     flag = np.empty(bool_ar.shape[0] + 1, dtype=np.bool_)
#     flag[:-1] = bool_ar
#     flag[-1] = False

#     ret = np.empty(ar.shape, dtype=np.bool_)
#     ret[order] = flag
#     r = ret[:len(ar1)]
#     return r.reshape(ar1_shape)
