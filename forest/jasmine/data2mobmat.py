"""This module contains functions to convert raw GPS data to mobility matrix
"""

import logging
import math
from typing import Tuple, Union, Optional, List
from itertools import groupby

import numpy as np
import pandas as pd

from forest.constants import EARTH_RADIUS_METERS

# a threshold
TOLERANCE = 1e-6


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cartesian(
    lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]
) -> Union[
    Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """This function converts latitude and longitude to cartesian coordinates.

    Args:
        lat: float or 1d np.array, latitude, range[-180, 180]
        lon: float or 1d np.array, longitude, range[-180, 180]
            should be of same length(data type) as lat
    Returns:
        the corresponding cartesian coordiantes ((0,0,0) as geocenter)
    """
    lat = lat / 180 * math.pi
    lon = lon / 180 * math.pi
    z_coord = EARTH_RADIUS_METERS * np.sin(lat)
    u_var = EARTH_RADIUS_METERS * np.cos(lat)
    x_coord = u_var * np.cos(lon)
    y_coord = u_var * np.sin(lon)
    return x_coord, y_coord, z_coord


def great_circle_dist(
    lat1: Union[float, np.ndarray], lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray], lon2: Union[float, np.ndarray]
) -> np.ndarray:
    """This function calculates the great circle distance
     between various pairs of locations.

    Args:
        lat1: Union[float, np.ndarray], latitude of location1,
            range[-180, 180]
        lon1: Union[float, np.ndarray], longitude of location1,
            range[-180, 180]
        lat2: Union[float, np.ndarray], latitude of location2,
            range[-180, 180]
        lon2: Union[float, np.ndarray], longitude of location2,
            range[-180, 180]
    Returns:
        the great circle distance between location1 and location2
    """
    lat1 = lat1 / 180 * math.pi
    lon1 = lon1 / 180 * math.pi
    lat2 = lat2 / 180 * math.pi
    lon2 = lon2 / 180 * math.pi
    temp = (
        np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
        + np.sin(lat1) * np.sin(lat2)
    )

    # due to measurement errors, temp may be out of the domain of "arccos"
    if not isinstance(temp, np.ndarray):
        temp = np.array([temp])

    temp[temp > 1] = 1
    temp[temp < -1] = -1

    theta = np.arccos(temp)
    return theta * EARTH_RADIUS_METERS


def shortest_dist_to_great_circle(
    location1: np.ndarray,
    location2: np.ndarray,
    location3: np.ndarray
) -> np.ndarray:
    """This function calculates the shortest distance from location 1
        to the great circle determined by location 2 and 3.

    Args:
        location1: 2d np.array with latitudes and longitudes
         of locations, range[-180, 180]
        location2: np.ndarray, latitude and longitude of location 2
        location3: np.ndarray, latitude and longitude of location 3
    Returns:
        the shortest distance from location 1 to the great circle
            determined by location 2 and 3
            (the path which is perpendicular to the great circle)
            unit is meter, data type matches the input
    """
    lat_start, lon_start = location2
    lat_end, lon_end = location3
    lat, lon = location1[:, 0], location1[:, 1]
    # if loc2 and 3 are too close to determine a great circle, return 0
    if (
        abs(lat_start - lat_end) < TOLERANCE
        and abs(lon_start - lon_end) < TOLERANCE
    ):
        return np.zeros(len(lat))
    x_coord, y_coord, z_coord = cartesian(lat, lon)
    x_start, y_start, z_start = cartesian(lat_start, lon_start)
    x_end, y_end, z_end = cartesian(lat_end, lon_end)
    cross_product = np.cross(
        np.array([x_start, y_start, z_start]),
        np.array([x_end, y_end, z_end])
    )
    N = cross_product / (np.linalg.norm(cross_product) + TOLERANCE)
    C = np.array([x_coord, y_coord, z_coord]) / EARTH_RADIUS_METERS
    temp = np.dot(N, C)
    # make temp fall into the domain of "arccos"
    if isinstance(temp, np.ndarray):
        temp[temp > 1] = 1
        temp[temp < -1] = -1
    else:
        temp = min(temp, 1)
        temp = max(temp, -1)
    noc = np.arccos(temp)
    d = abs(math.pi / 2 - noc) * EARTH_RADIUS_METERS
    return d


def pairwise_great_circle_dist(latlon_array: np.ndarray) -> List[float]:
    """This function calculates the pairwise great circle distance
        between any pair of locations.

    Args:
        latlon_array: 2d np.array, should be a n by 2 array.
            The first column is latitude and the second is longitude.
            each element should be within [-180, 180]
    Returns:
        a list of length n*(n-1)/2,
            the pairwise great circle distance between any pair of locations
    """
    dist = []
    k = np.shape(latlon_array)[0]
    for i in range(k - 1):
        for j in np.arange(i + 1, k):
            distance_float = great_circle_dist(
                latlon_array[i, 0],
                latlon_array[i, 1],
                latlon_array[j, 0],
                latlon_array[j, 1],
            )[0]
            dist.append(distance_float)
    return dist


def collapse_data(
    data: pd.DataFrame, interval: float, accuracy_limit: float
) -> np.ndarray:
    """This function collapses the raw data into a 2d numpy array,
        with each row as a chunk of data.

    Args:
        data: pd.DataFrame, the pd dataframe from read_data()
        interval: float, the window size of moving average in seconds
        accuracy_limit: float, a threshold to filter out GPS record
            with accuracy higher than this limit.
    Returns:
        a 2d numpy array, average over the measures every interval seconds
            with first column as an indicator:
            - if it is 1, it is observed, and the subsequent columns
                are timestamp, latitude, longitude
            - if it is 4, it is a missing interval and the subsequent columns
                are starting timestamp, ending timestamp, None
    """
    # Filter out rows where the GPS accuracy is beyond
    # the provided accuracy_limit
    data = data[data.accuracy < accuracy_limit]
    if data.shape[0] == 0:
        raise ValueError(
            f"No GPS record with accuracy less than {accuracy_limit}."
        )

    # Get the start and end timestamps in seconds
    t_start = sorted(np.array(data.timestamp))[0] / 1000
    t_end = sorted(np.array(data.timestamp))[-1] / 1000

    # Initialize an empty 2D numpy array for the collapsed data
    avgmat = np.empty([int(np.ceil((t_end - t_start) / interval)) + 2, 4])

    logger.info(
        "Collapse data within %s second intervals ...", interval
    )

    idx_avgmat: int = 0
    count: int = 0
    num_interval: int = 1

    # Initialize the first row of the output matrix
    # [1, timestamp, latitude, longitude]
    nextline = [1, t_start + interval / 2, data.iloc[0, 2], data.iloc[0, 3]]

    for i in np.arange(1, data.shape[0]):
        # If the timestamp of the current row is within the current interval
        if data.iloc[i, 0] / 1000 < t_start + interval:
            # Accumulate latitude and longitude for averaging later
            nextline[2] += data.iloc[i, 2]
            nextline[3] += data.iloc[i, 3]
            num_interval += 1
        else:
            # When the current row's timestamp exceeds the current interval,
            # we compute the average for latitude and longitude
            nextline[2] /= num_interval
            nextline[3] /= num_interval

            # Store the averaged data in the output matrix
            avgmat[idx_avgmat, :] = nextline

            count += 1
            idx_avgmat += 1

            # Compute the number of missing intervals
            num_miss = int(
                np.floor(
                    (data.iloc[i, 0] / 1000 - (t_start + interval)) / interval
                    )
            )

            # If there are missing intervals
            if num_miss > 0:
                # Insert a row of missing interval into the output matrix
                avgmat[idx_avgmat, :] = [
                    4, t_start + interval,
                    t_start + interval * (num_miss + 1), None,
                ]
                count += 1
                idx_avgmat += 1

            # Move the start time to the end
            # of the last missing interval or the current interval
            t_start += interval * (num_miss + 1)

            # Initialize the next row of the output matrix
            nextline = [
                1, t_start + interval / 2, data.iloc[i, 2], data.iloc[i, 3]
            ]
            num_interval = 1

    # Trim the output matrix to remove unused rows
    avgmat = avgmat[0:count, :]

    return avgmat


def exist_knot(
    avg_mat: np.ndarray, distance_threshold: float
) -> Tuple[int, Optional[int]]:
    """This function checks if there is a knot in the observed data chunk.

    Args:
        avg_mat: np.ndarray, avgmat from collapse_data()
        distance_threshold : float,
            The distance threshold for detecting a knot.
            If the distance to the great circle is greater than
            this threshold, it is considered a knot.

    Returns:
        Tuple[int, Optional[int]]: A tuple containing two elements:
            - The first element is an indicator,
             which is 1 if a knot is found, otherwise 0.
            - The second element is the index of the knot
             if it exists, otherwise None.
    """
    # Get the number of rows in the observed_data array
    num_rows = avg_mat.shape[0]

    # If there is more than one row of data
    if num_rows > 1:

        # Get the latitude and longitude at the start and end of the data
        location2 = avg_mat[0, [2, 3]]
        location3 = avg_mat[num_rows - 1, [2, 3]]

        # Get the entire latitude and longitude data columns
        location1 = avg_mat[:, [2, 3]]

        # Calculate the shortest distance from each point
        # to the great circle defined by the start and end points
        shortest_distances = shortest_dist_to_great_circle(
            location1, location2, location3
        )

        # If the maximum distance is less than the threshold,
        # return 0 and None (indicating no knot found)
        if max(shortest_distances) < distance_threshold:
            return 0, None

        # If a knot was found, return 1 and the index of the knot
        return 1, int(np.argmax(shortest_distances))

    # If there is only one row of data, return 0 and None
    # (indicating no knot found)
    return 0, None


def mark_single_measure(
    input_matrix: np.ndarray, interval: float
) -> np.ndarray:
    """Marks a single measure as status "3" (unknown).

    Args:
        input_matrix: np.array, avgmat from collapse_data(),
            just one observed chunk without missing intervals
        interval: float, the window size of moving average,  unit is second

    Returns:
        a 2d numpy array of trajectories, with structure as
        [status, lat_start, lon_start,
        stamp_start, lat_end, lon_end, stamp_end]
    """
    return np.array(
        [
            3, input_matrix[2], input_matrix[3],
            input_matrix[1] - interval / 2,
            None, None, input_matrix[1] + interval / 2
        ]
    )


def mark_complete_pause(
    input_matrix: np.ndarray, interval: float, nrows: int,
) -> np.ndarray:
    """Marks a complete pause as status "2" (pause) if
     all points are within the maximum pause radius.

    Args:
        input_matrix: np.array, avgmat from collapse_data(),
            just one observed chunk without missing intervals
        interval: float, the window size of moving average,  unit is second
        nrows: int, the number of rows in the input_matrix

    Returns:
        a 2d numpy array of trajectories, with structure as
        [status, lat_start, lon_start,
        stamp_start, lat_end, lon_end, stamp_end]
    """
    mean_lon = (input_matrix[0, 2] + input_matrix[nrows - 1, 2]) / 2
    mean_lat = (input_matrix[0, 3] + input_matrix[nrows - 1, 3]) / 2
    return np.array(
        [
            2,
            mean_lon,
            mean_lat,
            input_matrix[0, 1] - interval / 2,
            mean_lon,
            mean_lat,
            input_matrix[nrows - 1, 1] + interval / 2,
        ]
    )


def detect_knots(
    input_matrix: np.ndarray, nrows: int, w: float, h: float
) -> list:
    """Detects knots in the data.

    Args:
        input_matrix: np.array, avgmat from collapse_data(),
            just one observed chunk without missing intervals
        nrows: int, the number of rows in the input_matrix
        w: float, a threshold for distance,
            if the distance to the great circle is greater than
            this threshold, we consider there is a knot
        h: float, a threshold of distance, if the movement
            between two timestamps is less than h,
            consider it as a pause and a knot

    Returns:
        a list of indices of knots
    """
    knot_indices = [0, nrows - 1]
    movement_distances = np.array(
        [
            great_circle_dist(
                input_matrix[i, 2], input_matrix[i, 3],
                input_matrix[i + 1, 2], input_matrix[i + 1, 3]
            )[0]
            for i in range(nrows - 1)
        ]
    )
    # Indices where the movement is less than the pause threshold,
    # considered as pauses
    pause_indices = np.arange(0, nrows - 1)[movement_distances < h]

    temp_indices = []
    for j in range(len(pause_indices) - 1):
        if pause_indices[j + 1] - pause_indices[j] == 1:
            temp_indices.extend([pause_indices[j], pause_indices[j + 1]])

    # all the consequential numbers in between are inserted twice,
    # but start and end are inserted once
    long_pause_indices = np.unique(temp_indices)[
        np.array(
            [len(list(group)) for key, group in groupby(temp_indices)]
        ) == 1
    ]
    # pause 0,1,2, correspond to point [0,1,2,3],
    # so the end number should plus 1
    # Adjust pause indices to represent points, not movements
    long_pause_indices[np.arange(1, len(long_pause_indices), 2)] += 1
    # the key is to update the knot list and sort them
    knot_indices.extend(long_pause_indices.tolist())
    knot_indices.sort()
    knot_indices = list(set(knot_indices))

    # While loop to continue process until no more knot is detected
    knot_detection_complete = False
    while not knot_detection_complete:
        sub_matrices = []
        for i in range(len(knot_indices) - 1):
            knot_start = knot_indices[i]
            knot_end = min(knot_indices[i + 1] + 1, nrows - 1)
            sub_matrices.append(
                input_matrix[knot_start:knot_end, :]
            )

        knot_exists = np.empty(len(sub_matrices))
        knot_position = np.empty(len(sub_matrices))
        for i, sub_matrix in enumerate(sub_matrices):
            knot_exists[i], knot_position[i] = exist_knot(sub_matrix, w)

        # If no knots detected, stop the while loop
        if sum(knot_exists) == 0:
            knot_detection_complete = True
        else:
            for i, sub_matrix in enumerate(sub_matrices):
                if knot_exists[i] == 1:
                    knot_indices.append(
                        int(sub_matrix[int(knot_position[i]), 4])
                    )
            knot_indices.sort()
    return knot_indices


def prepare_output_data(
    input_matrix: np.ndarray, knot_indices: list, h: float
) -> np.ndarray:
    """Prepares the output data by detecting flights and pauses.

    Args:
        input_matrix: np.array, avgmat from collapse_data(),
            just one observed chunk without missing intervals
        knot_indices: list, a list of indices of knots
        h: float, a threshold of distance, if the movement
            between two timestamps is less than h,
            consider it as a pause and a knot

    Returns:
        a 2d numpy array of trajectories, with structure as
            [status, lat_start, lon_start,
            stamp_start, lat_end, lon_end, stamp_end]
    """
    flight_and_pause_data = []
    for j in range(len(knot_indices) - 1):
        start_index = knot_indices[j]
        end_index = knot_indices[j + 1]
        # calculate the movement distance between two timestamps
        movement_distances = np.array(
            [
                great_circle_dist(
                    input_matrix[i, 2], input_matrix[i, 3],
                    input_matrix[i + 1, 2], input_matrix[i + 1, 3]
                )
                for i in np.arange(start_index, end_index)
            ]
        )
        # if there is no movement, consider it as a pause
        if sum(movement_distances >= h) == 0:
            mean_lon = (
                input_matrix[start_index, 2] + input_matrix[end_index, 2]
            ) / 2
            mean_lat = (
                input_matrix[start_index, 3] + input_matrix[end_index, 3]
            ) / 2
            flight_and_pause_data.append([
                2,
                mean_lon,
                mean_lat,
                input_matrix[start_index, 1],
                mean_lon,
                mean_lat,
                input_matrix[end_index, 1],
            ])
        # if there is movement, consider it as a flight
        else:
            flight_and_pause_data.append([
                1,
                input_matrix[start_index, 2],
                input_matrix[start_index, 3],
                input_matrix[start_index, 1],
                input_matrix[end_index, 2],
                input_matrix[end_index, 3],
                input_matrix[end_index, 1],
            ])

    return np.array(flight_and_pause_data)


def extract_flights(
    input_matrix: np.ndarray, interval: float, r: float, w: float, h: float
) -> np.ndarray:
    """This function extracts flights and pauses from one observed chunk.

    If there is only one measure in this chunk, mark it as status "3" (unknown)
    If there is no pause, mark it as status "1" (flight)
    If there is pause, mark it as status "2" (pause)

    Args:
        input_matrix: np.array, avgmat from collapse_data(),
            just one observed chunk without missing intervals
        interval: float, the window size of moving average,  unit is second
        r: float, the maximam radius of a pause
        w: float, a threshold for distance,
            if the distance to the great circle is greater than
            this threshold, we consider there is a knot
        h: float, a threshold of distance, if the movement
            between two timestamps is less than h,
            consider it as a pause and a knot
    Returns:
        a 2d numpy array of trajectories, with structure as
            [status, lat_start, lon_start,
            stamp_start, lat_end, lon_end, stamp_end]
    """
    # sometimes input_matrix is a 1d array and sometimes it's 2d array
    # which correspond to if and elif below
    # Check if the input_matrix is a single measure (1D array)
    if len(input_matrix.shape) == 1:
        return mark_single_measure(input_matrix, interval)
    # Check if the input_matrix has only one row (one measure in a 2D array)
    if len(input_matrix.shape) == 2 and input_matrix.shape[0] == 1:
        return mark_single_measure(input_matrix[0, :], interval)

    nrows = input_matrix.shape[0]
    # Add a new column for indices
    input_matrix = np.hstack(
        (input_matrix, np.arange(nrows).reshape((nrows, 1)))
    )

    # Check if all points are within the maximum pause radius
    # indicating a pause
    if nrows > 1 and max(pairwise_great_circle_dist(input_matrix[:, 2:4])) < r:
        return mark_complete_pause(input_matrix, interval, nrows)

    # Not a simple pause, contains at least one flight
    # Detect knots in the data
    knot_indices = detect_knots(input_matrix, nrows, w, h)
    return prepare_output_data(input_matrix, knot_indices, h)


def gps_to_mobmat(
    raw_gps_data: pd.DataFrame, interval: float, accuracy_limit: float,
    r: float, w: float, h: float
) -> np.ndarray:
    """This function transforms raw GPS data
     to a matrix of first-step trajectories.

    It calls collapse_data() and extract_flights().
    Additionally, it divides the trajectory mat
    into obs - mis - obs - mis, then does extract_flights()
    on each observed chunk and stack them over

    Args:
        raw_gps_data: pd.DataFrame, dataframe from read_data()
        interval: float, the window size of moving average,  unit is second
        accuracy_limit: float, a threshold.
            We filter out GPS record with accuracy higher than this threshold.
        r: float, the maximum radius of a pause
        w: float, a threshold for distance,
            if the distance to the great circle is greater than
            this threshold, we consider there is a knot
        h: float, a threshold of distance, if the movemoent
            between two timestamps is less than h,
            consider it as a pause and a knot
    Returns:
        A 2D numpy array of observed trajectories (first-step) with columns as
            [status, lat_start, lon_start, stamp_start,
            lat_end, lon_end, stamp_end]
    """
    avgmat = collapse_data(raw_gps_data, interval, accuracy_limit)
    trajectory_matrix = np.zeros(7)
    current_index = 0
    logger.info("Extract flights and pauses ...")

    for i in range(avgmat.shape[0]):
        # if the status of the data is 4 (missing data)
        # divide the continuous data
        # into chunks and extract flights and pauses from each observed chunk
        if avgmat[i, 0] == 4:
            temp = extract_flights(
                avgmat[np.arange(current_index, i), :], interval, r, w, h
            )
            trajectory_matrix = np.vstack((trajectory_matrix, temp))
            current_index = i + 1

    # handle remaining data after the last missing data
    if current_index < avgmat.shape[0]:
        temp = extract_flights(
            avgmat[np.arange(current_index, avgmat.shape[0]), :],
            interval, r, w, h
        )
        trajectory_matrix = np.vstack((trajectory_matrix, temp))

    # remove the first row of zeros from the trajectory matrix
    mobmat = np.delete(trajectory_matrix, 0, 0)

    return mobmat


def force_valid_longitude(longitude: float) -> float:
    """Forces a longitude coordinate to be within -180 and 180

    In some cases, the imputation code seems to yield out-of-range
    GPS coordinates. This function wrps longitude coordinates to be back
    in the correct range so an error isn't thrown. 

    For example, 190 would get transformed into -170.
    
    Args:
        longitude: float. The longitude to be coerced
    """
    return (longitude + 180) % 360 - 180


def compute_flight_positions(
    index: int, mobmat: np.ndarray, interval: float
) -> np.ndarray:
    """Computes the flight positions of the given index in the trajectory.

    Args:
        index: int, the current index in the trajectory.
        mobmat: np.array, a 2d numpy array
            (the mobility matrix output from gps_to_mobmat())
        interval: float, the window size of moving average,  unit is second

    Returns:
        The updated mobility matrix.
    """

    # Calculate the time difference between
    # the current point and the previous point
    time_diff = mobmat[index, 3] - mobmat[index - 1, 6]

    # Calculate the change in x and y positions between
    # the current point and the previous point
    delta_x = mobmat[index, 1] - mobmat[index - 1, 4]
    delta_y = mobmat[index, 2] - mobmat[index - 1, 5]

    # Calculate the rate of change for x and y positions over time
    rate_of_change_x = delta_x / time_diff
    rate_of_change_y = delta_y / time_diff

    # Calculate the adjustment factor for the start and end points
    adjustment_start = interval / 2 * rate_of_change_x
    adjustment_end = interval / 2 * rate_of_change_y

    # Adjust the start and end x and y positions using the adjustment factors
    start_x = mobmat[index, 1] - adjustment_start
    start_y = mobmat[index, 2] - adjustment_end
    end_x = mobmat[index, 1] + adjustment_start
    end_y = mobmat[index, 2] + adjustment_end

    # Update the mobility matrix with the new start and end positions
    mobmat[index, 1] = start_x
    mobmat[index, 4] = end_x
    mobmat[index, 2] = force_valid_longitude(start_y)
    mobmat[index, 5] = force_valid_longitude(end_y)

    return mobmat


def compute_future_flight_positions(
    index: int, mobmat: np.ndarray, interval: float
) -> np.ndarray:
    """Computes the flight positions of the given index in the trajectory.
     using the next point instead of the previous point

    Args:
        index: int, the current index in the trajectory.
        mobmat: np.array, a 2d numpy array
            (the mobility matrix output from gps_to_mobmat())
        interval: float, the window size of moving average,  unit is second

    Returns:
        The updated mobility matrix.
    """

    # Calculate the time difference between
    # the current point and the next point
    time_diff = mobmat[index + 1, 3] - mobmat[index, 6]

    # Calculate the change in x and y positions between
    # the current point and the next point
    delta_x = mobmat[index + 1, 1] - mobmat[index, 1]
    delta_y = mobmat[index + 1, 2] - mobmat[index, 2]

    # Calculate the rate of change for x and y positions over time
    rate_of_change_x = delta_x / time_diff
    rate_of_change_y = delta_y / time_diff

    # Calculate the adjustment factor for the start and end points
    adjustment_start = interval / 2 * rate_of_change_x
    adjustment_end = interval / 2 * rate_of_change_y

    # Adjust the start and end x and y positions using the adjustment factors
    start_x = mobmat[index, 1] - adjustment_start
    start_y = mobmat[index, 2] - adjustment_end
    end_x = mobmat[index, 1] + adjustment_start
    end_y = mobmat[index, 2] + adjustment_end

    # Update the mobility matrix with the new start and end positions
    mobmat[index, 1] = start_x
    mobmat[index, 4] = end_x
    mobmat[index, 2] = force_valid_longitude(start_y)
    mobmat[index, 5] = force_valid_longitude(end_y)

    return mobmat


def infer_status_and_positions(
    index: int, mobmat: np.ndarray,
    interval: float, r: float
) -> np.ndarray:
    """Infers the status and positions of the given index in the trajectory.

    Args:
        index: int, the current index in the trajectory.
        mobmat: np.array, a 2d numpy array
         (the mobility matrix output from gps_to_mobmat())
        interval: float, the window size of moving average,  unit is second
        r: float, the maximam radius of a pause

    Returns:
        The updated mobility matrix.

    Raises:
        ValueError: If the index is less than 0.
    """

    if index < 0:
        raise ValueError("Index must be greater than or equal to 0")
    # If the status is unknown and it's the first index
    # set the status to 2 (pause)
    elif index == 0:
        status = 2
        mobmat[index, [4, 5]] = mobmat[index, [1, 2]]
    # If the status is unknown and it's not the first index,
    # infer the status based on its distance to the previous
    # point and the time interval
    else:
        # calculate the distance to the previous point
        distance = great_circle_dist(
            mobmat[index, 1], mobmat[index, 2],
            mobmat[index - 1, 4], mobmat[index - 1, 5]
        )[0]
        # if the time difference to the previous point is
        # less than or equal to 3 times the time interval
        if mobmat[index, 3] - mobmat[index - 1, 6] <= interval * 3:
            # if the distance is less than the maximum radius of a pause
            # assign the status as 'pause' (status code 2)
            if distance < r:
                status = 2  # pause
                mobmat[index, [4, 5]] = mobmat[index, [1, 2]]
            # if the distance is greater than the maximum radius of a pause
            # assign the status as 'flight' (status code 1)
            else:
                status = 1  # flight
                mobmat = compute_flight_positions(index, mobmat, interval)
        # if the time difference to the previous point
        # is more than 3 times the time interval
        else:
            # if there is a next point
            if (index + 1) < mobmat.shape[0]:
                # calculate the distance to the next point
                future_distance = great_circle_dist(
                    mobmat[index, 1], mobmat[index, 2],
                    mobmat[index + 1, 1], mobmat[index + 1, 2]
                )[0]
                # if the time difference to the next point is
                # less than or equal to 3 times the time interval
                if mobmat[index + 1, 3] - mobmat[index, 6] <= interval * 3:
                    # if the distance is less than
                    # the maximum radius of a pause
                    # assign the status as 'pause' (status code 2)
                    if future_distance < r:
                        status = 2  # pause
                        mobmat[index, [4, 5]] = mobmat[index, [1, 2]]
                    # if the distance is greater
                    # than the maximum radius of a pause
                    # assign the status as 'flight' (status code 1)
                    else:
                        status = 1  # flight
                        mobmat = compute_future_flight_positions(
                            index, mobmat, interval
                        )
                # if the time difference to the next point is
                # more than 3 times the time interval
                # assign the status as 'pause' (status code 2)
                else:
                    status = 2  # pause
                    mobmat[index, [4, 5]] = mobmat[index, [1, 2]]
            # if there is no next point
            # assign the status as 'pause' (status code 2)
            else:
                status = 2  # pause
                mobmat[index, [4, 5]] = mobmat[index, [1, 2]]

    mobmat[index, 0] = status

    return mobmat


def merge_pauses_and_bridge_gaps(mobmat: np.ndarray) -> np.ndarray:
    """Merge consecutive pauses and bridge any gaps in mobility matrix.

    Args:
        mobmat: np.array, a 2d numpy array

    Returns:
        Updated mobility matrix with merged pauses and bridged gaps.
    """

    # Initialize a list to hold indices of consecutive pauses
    consecutive_pause_indices = []

    # Iterate over the mobility matrix, starting from the second row
    for j in range(1, mobmat.shape[0]):
        # If current and previous status are 'pause' and their times are equal,
        # add their indices to the list
        if (
            mobmat[j, 0] == 2 and mobmat[j - 1, 0] == 2
            and mobmat[j, 3] == mobmat[j - 1, 6]
        ):
            consecutive_pause_indices.extend([j - 1, j])

    # Find unique indices where start and end of pauses occur
    # These indices are those that are inserted only once in the list
    pause_boundaries = np.unique(consecutive_pause_indices)[
        np.array(
            [
                len(list(group)) for key, group
                in groupby(consecutive_pause_indices)
            ]
        ) == 1
    ]

    # Iterate over each pair of start and end indices
    for j in range(len(pause_boundaries) // 2):
        start = pause_boundaries[2 * j]
        end = pause_boundaries[2 * j + 1]

        # Compute mean x and y coordinates over the pause interval
        mean_x = np.mean(mobmat[start:(end+1), 1])
        mean_y = np.mean(mobmat[start:(end+1), 2])

        # Replace the start of the pause interval
        # with mean coordinates and appropriate times
        mobmat[start, :] = [
            2, mean_x, mean_y, mobmat[start, 3],
            mean_x, mean_y, mobmat[end, 6]
        ]

        # Set status of bridged gaps to 5
        mobmat[(start + 1):(end + 1), 0] = 5

    # Remove bridged gaps from the mobility matrix
    mobmat = mobmat[mobmat[:, 0] != 5, :]

    return mobmat


def correct_missing_intervals(
    mobmat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Corrects missing intervals in the mobility matrix.

    Args:
        mobmat: np.array, a 2d numpy array

    Returns:
        Tuple of corrected mobility matrix and array of new pauses.
    """

    # Initialize a list to hold new pauses
    new_pauses = []

    # Iterate over the mobility matrix, starting from the second row
    for j in range(1, mobmat.shape[0]):
        # Check if there's a gap in time between current and previous rows
        if mobmat[j, 3] > mobmat[j - 1, 6]:

            # Compute the distance between current and previous positions
            distance = great_circle_dist(
                mobmat[j, 1], mobmat[j, 2], mobmat[j - 1, 4], mobmat[j - 1, 5]
            )[0]

            # If the distance is less than 10 units, start correcting positions
            if distance < 10:

                # For consecutive pauses, make them stay at the same position
                if mobmat[j, 0] == 2 and mobmat[j - 1, 0] == 2:
                    initial_x = mobmat[j - 1, 4]
                    initial_y = mobmat[j - 1, 5]

                    # Make sure both pause positions
                    # re at the same coordinates
                    mobmat[j, [1, 4]] = mobmat[j - 1, [1, 4]] = initial_x
                    mobmat[j, [2, 5]] = mobmat[j - 1, [2, 5]] = initial_y

                # If moving state follows a pause,
                # make it start from where the pause ended
                if mobmat[j, 0] == 1 and mobmat[j - 1, 0] == 2:
                    mobmat[j, [1, 2]] = mobmat[j - 1, [4, 5]]

                # If pause follows a moving state,
                # make the movement end where the pause starts
                if mobmat[j, 0] == 2 and mobmat[j - 1, 0] == 1:
                    mobmat[j - 1, [4, 5]] = mobmat[j, [1, 2]]

                # For consecutive moving states, make them meet at the midpoint
                if mobmat[j, 0] == 1 and mobmat[j - 1, 0] == 1:
                    mean_x = (mobmat[j, 1] + mobmat[j - 1, 4]) / 2
                    mean_y = (mobmat[j, 2] + mobmat[j - 1, 5]) / 2
                    mobmat[j - 1, 4] = mobmat[j, 1] = mean_x
                    mobmat[j - 1, 5] = mobmat[j, 2] = mean_y

                # Add the corrected interval as a new pause
                new_pause = [
                    2,
                    mobmat[j, 1],
                    mobmat[j, 2],
                    mobmat[j - 1, 6],
                    mobmat[j, 1],
                    mobmat[j, 2],
                    mobmat[j, 3],
                    0,
                ]
                new_pauses.append(new_pause)

    # Convert the list of new pauses into a numpy array
    new_pauses_array = np.array(new_pauses)

    return new_pauses_array, mobmat


def infer_mobmat(mobmat: np.ndarray, interval: float, r: float) -> np.ndarray:
    """This function takes the first-step trajectory matrix
        as input and return the final trajectory matrix as output.

    It carries out the following steps:
        1) Infers the unknown status for observations in the matrix.
        2) Combines close pauses into single pause events.
        3) Joins flights (sequences of observations) together
            to form a continuous trajectory.
    In addition, it adds a column to the trajectory matrix indicating
     whether an observation is 'observed' (1) or 'imputed' (0).

    Args:
        mobmat: np.array, a 2d numpy array (output from gps_to_mobmat())
        interval: float, the window size of moving average,  unit is second
        r: float, the maximam radius of a pause

    Returns:
        a 2d numpy array as a final trajectory matrix
    """
    logger.info("Infer unclassified windows ...")

    # Infer unknown status
    # The 'unknown' status (code 3)
    # is inferred based on neighbouring data points
    # and specific conditions about time intervals and distances.
    for i, status in enumerate(mobmat[:, 0]):
        if status == 3:
            mobmat = infer_status_and_positions(i, mobmat, interval, r)

    # merge consecutive pauses
    logger.info("Merge consecutive pauses and bridge gaps ...")
    mobmat = merge_pauses_and_bridge_gaps(mobmat)

    # check for missing intervals and correct them
    new_pauses_array, mobmat = correct_missing_intervals(mobmat)

    n_rows = mobmat.shape[0]

    # connect flights and pauses
    for j in np.arange(1, n_rows):
        # If the current and previous states form a flight-pause pair
        # and their times are contiguous
        # (their end and start times are the same),
        # then we need to make sure the position
        # of the flight and pause match up.
        if (
            mobmat[j, 0] * mobmat[j - 1, 0] == 2
            and mobmat[j, 3] == mobmat[j - 1, 6]
        ):
            # If the current state is a flight, update its starting position
            # to be the same as the ending position of the preceding pause.
            if mobmat[j, 0] == 1:
                mobmat[j, [1, 2]] = mobmat[j - 1, [4, 5]]
            # If the previous state is a flight, update its ending position
            # to be the same as the starting position of the following pause.
            if mobmat[j - 1, 0] == 1:
                mobmat[j - 1, [4, 5]] = mobmat[j, [1, 2]]

    # Add a column of 1s to indicate that all observations are observed
    mobmat = np.hstack(
        (mobmat, np.ones(n_rows).reshape(n_rows, 1))
    )
    # Append new pauses to the trajectory matrix
    if new_pauses_array.shape[0] > 0:
        mobmat = np.vstack((mobmat, new_pauses_array))
    # Sort the matrix by start time
    mobmat = mobmat[mobmat[:, 3].argsort()].astype(float)

    return mobmat
