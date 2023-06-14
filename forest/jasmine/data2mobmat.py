"""This module contains functions to convert raw GPS data to mobility matrix
"""

import sys
import math
from typing import Tuple, Union, Optional, List
from itertools import groupby

import numpy as np
import pandas as pd

# the radius of the earth
R = 6.371 * 10 ** 6
# a threshold
TOLERANCE = 1e-6


def unique(anylist: list) -> list:
    """This function returns a list of unique elements in anylist.

    Args:
        anylist: list, could be a list of any type
    Returns:
        a list of unique elements in anylist
    """
    return list(set(anylist))


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
    z_coord = R * np.sin(lat)
    u_var = R * np.cos(lat)
    x_coord = u_var * np.cos(lon)
    y_coord = u_var * np.sin(lon)
    return x_coord, y_coord, z_coord


def great_circle_dist(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """This function calculates the great circle distance
        between two locations.

    Args:
        lat1: float, latitude of location1,
            range[-180, 180]
        lon1: float, longitude of location1,
            range[-180, 180]
        lat2: float, latitude of location2,
            range[-180, 180]
        lon2: float, longitude of location2,
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
    # lat, lon sometimes could be np.array,
    # so there are two version of this correction
    if isinstance(temp, np.ndarray):
        temp[temp > 1] = 1
        temp[temp < -1] = -1
    else:
        temp = min(temp, 1)
        temp = max(temp, -1)
    theta = np.arccos(temp)
    distance = theta * R
    return distance


def shortest_dist_to_great_circle(
    location1: np.ndarray,
    location2: Tuple[float, float],
    location3: Tuple[float, float]
) -> np.ndarray:
    """This function calculates the shortest distance from location 1
        to the great circle determined by location 2 and 3.

    Args:
        location1: 2d np.array with latitudes and longitudes of locations, range[-180, 180]
        location2: Tuple[float, float], latitude and longitude of location 2
        location3: Tuple[float, float], latitude and longitude of location 3
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
    C = np.array([x_coord, y_coord, z_coord]) / R
    temp = np.dot(N, C)
    # make temp fall into the domain of "arccos"
    if isinstance(temp, np.ndarray):
        temp[temp > 1] = 1
        temp[temp < -1] = -1
    else:
        temp = min(temp, 1)
        temp = max(temp, -1)
    noc = np.arccos(temp)
    d = abs(math.pi / 2 - noc) * R
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
            dist.append(
                great_circle_dist(
                    latlon_array[i, 0],
                    latlon_array[i, 1],
                    latlon_array[j, 0],
                    latlon_array[j, 1],
                )
            )
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

    # Get the start and end timestamps in seconds
    t_start = sorted(np.array(data.timestamp))[0] / 1000
    t_end = sorted(np.array(data.timestamp))[-1] / 1000

    # Initialize an empty 2D numpy array for the collapsed data
    avgmat = np.empty([int(np.ceil((t_end - t_start) / interval)) + 2, 4])

    sys.stdout.write(
        f"Collapse data within {interval} second intervals ...\n"
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
    input_matrix: np.ndarray, interval: float, nrows: int, r: float
) -> np.ndarray:
    """Marks a complete pause as status "2" (pause) if
     all points are within the maximum pause radius.

    Args:
        input_matrix: np.array, avgmat from collapse_data(),
            just one observed chunk without missing intervals
        interval: float, the window size of moving average,  unit is second
        nrows: int, the number of rows in the input_matrix
        r: float, the maximum radius of a pause

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
            )
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
    knot_indices = unique(knot_indices)

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
        return mark_complete_pause(input_matrix, interval, nrows, r)

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
    sys.stdout.write("Extract flights and pauses ...\n")

    for i in range(avgmat.shape[0]):
        # if the status of the data is 4 (missing data), divide the continuous data
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
            avgmat[np.arange(current_index, avgmat.shape[0]), :], interval, r, w, h
        )
        trajectory_matrix = np.vstack((trajectory_matrix, temp))

    # remove the first row of zeros from the trajectory matrix
    mobmat = np.delete(trajectory_matrix, 0, 0)

    return mobmat


def infer_mobmat(mobmat: np.ndarray, interval: float, r: float) -> np.ndarray:
    """This function takes the first-step trajectory mat
        as input and return the final trajectory mat as output.

    This function is the second step of the trajectory imputation.
    It calls gps_to_mobmat() and does the following:
        (1) infer the unknown status ("code==3" in the script below)
        (2) combine close pauses
        (3) join all flights
            (the end of first flight and the start of next flight)
            and pauses to form a continuous trajectory
    it also stacks one more column on the mobmat from gps_to_mobmat(),
    which is a col of 1, indicating
    they are observed intsead of imputed, for future use.

    Args:
        mobmat: np.array, a 2d numpy array (output from gps_to_mobmat())
        interval: float, the window size of moving average,  unit is second
        r: float, the maximam radius of a pause
    Returns:
        a 2d numpy array as a final trajectory mat
    """
    sys.stdout.write("Infer unclassified windows ..." + "\n")
    code = mobmat[:, 0]
    x0 = mobmat[:, 1]
    y0 = mobmat[:, 2]
    t0 = mobmat[:, 3]
    x1 = mobmat[:, 4]
    y1 = mobmat[:, 5]
    t1 = mobmat[:, 6]

    for i, status in enumerate(code):
        if status == 3 and i == 0:
            status = 2
            x1[i] = x0[i]
            y1[i] = y0[i]
        if status == 3 and i > 0:
            d = great_circle_dist(x0[i], y0[i], x1[i - 1], y1[i - 1])
            if t0[i] - t1[i - 1] <= interval * 3:
                if d < r:
                    status = 2
                    x1[i] = x0[i]
                    y1[i] = y0[i]
                else:
                    status = 1
                    s_x = (
                        x0[i] - interval / 2 /
                        (t0[i] - t1[i - 1]) * (x0[i] - x1[i - 1])
                    )
                    s_y = (
                        y0[i] - interval / 2 /
                        (t0[i] - t1[i - 1]) * (y0[i] - y1[i - 1])
                    )
                    e_x = (
                        x0[i] + interval / 2 /
                        (t0[i] - t1[i - 1]) * (x0[i] - x1[i - 1])
                    )
                    e_y = (
                        y0[i] + interval / 2 /
                        (t0[i] - t1[i - 1]) * (y0[i] - y1[i - 1])
                    )
                    x0[i] = s_x
                    x1[i] = e_x
                    y0[i] = s_y
                    y1[i] = e_y
            if t0[i] - t1[i - 1] > interval * 3:
                if (i + 1) < len(code):
                    f = great_circle_dist(x0[i], y0[i], x0[i + 1], y0[i + 1])
                    if t0[i + 1] - t1[i] <= interval * 3:
                        if f < r:
                            status = 2
                            x1[i] = x0[i]
                            y1[i] = y0[i]
                        else:
                            status = 1
                            s_x = (
                                x0[i] - interval / 2 /
                                (t0[i + 1] - t1[i]) * (x0[i + 1] - x0[i])
                            )
                            s_y = (
                                y0[i] - interval / 2 /
                                (t0[i + 1] - t1[i]) * (y0[i + 1] - y0[i])
                            )
                            e_x = (
                                x0[i] + interval / 2 /
                                (t0[i + 1] - t1[i]) * (x0[i + 1] - x0[i])
                            )
                            e_y = (
                                y0[i] + interval / 2 /
                                (t0[i + 1] - t1[i]) * (y0[i + 1] - y0[i])
                            )
                            x0[i] = s_x
                            x1[i] = e_x
                            y0[i] = s_y
                            y1[i] = e_y
                    else:
                        status = 2
                        x1[i] = x0[i]
                        y1[i] = y0[i]
                else:
                    status = 2
                    x1[i] = x0[i]
                    y1[i] = y0[i]
        mobmat[i, :] = [status, x0[i], y0[i], t0[i], x1[i], y1[i], t1[i]]

    # merge consecutive pauses
    sys.stdout.write("Merge consecutive pauses and bridge gaps ..." + "\n")
    k = []
    for j in np.arange(1, len(code)):
        if code[j] == 2 and code[j - 1] == 2 and t0[j] == t1[j - 1]:
            k.append(j - 1)
            k.append(j)
    # all the consequential numbers in between are
    # inserted twice, but start and end are inserted once
    rk = np.unique(k)[
        np.array([len(list(group)) for key, group in groupby(k)]) == 1
    ]
    for j in range(int(len(rk) / 2)):
        start = rk[2 * j]
        end = rk[2 * j + 1]
        mx = np.mean(x0[np.arange(start, end + 1)])
        my = np.mean(y0[np.arange(start, end + 1)])
        mobmat[start, :] = [2, mx, my, t0[start], mx, my, t1[end]]
        mobmat[np.arange(start + 1, end + 1), 0] = 5
    mobmat = mobmat[mobmat[:, 0] != 5, :]

    # check missing intervals,
    # if starting and ending point are close, make them same
    new_pauses = []
    for j in np.arange(1, mobmat.shape[0]):
        if mobmat[j, 3] > mobmat[j - 1, 6]:
            d = great_circle_dist(
                mobmat[j, 1], mobmat[j, 2], mobmat[j - 1, 4], mobmat[j - 1, 5]
            )
            if d < 10:
                if mobmat[j, 0] == 2 and mobmat[j - 1, 0] == 2:
                    initial_x = mobmat[j - 1, 4]
                    initial_y = mobmat[j - 1, 5]
                    mobmat[j, 1] = mobmat[j, 4] = mobmat[j - 1, 1] = mobmat[
                        j - 1, 4
                    ] = initial_x
                    mobmat[j, 2] = mobmat[j, 5] = mobmat[j - 1, 2] = mobmat[
                        j - 1, 5
                    ] = initial_y
                if mobmat[j, 0] == 1 and mobmat[j - 1, 0] == 2:
                    mobmat[j, 1] = mobmat[j - 1, 4]
                    mobmat[j, 2] = mobmat[j - 1, 5]
                if mobmat[j, 0] == 2 and mobmat[j - 1, 0] == 1:
                    mobmat[j - 1, 4] = mobmat[j, 1]
                    mobmat[j - 1, 5] = mobmat[j, 2]
                if mobmat[j, 0] == 1 and mobmat[j - 1, 0] == 1:
                    mean_x = (mobmat[j, 1] + mobmat[j - 1, 4]) / 2
                    mean_y = (mobmat[j, 2] + mobmat[j - 1, 5]) / 2
                    mobmat[j - 1, 4] = mobmat[j, 1] = mean_x
                    mobmat[j - 1, 5] = mobmat[j, 2] = mean_y
                new_pauses.append(
                    [
                        2,
                        mobmat[j, 1],
                        mobmat[j, 2],
                        mobmat[j - 1, 6],
                        mobmat[j, 1],
                        mobmat[j, 2],
                        mobmat[j, 3],
                        0,
                    ]
                )
    new_pauses_arr = np.array(new_pauses)

    # connect flights and pauses
    for j in np.arange(1, mobmat.shape[0]):
        if (
            mobmat[j, 0] * mobmat[j - 1, 0] == 2
            and mobmat[j, 3] == mobmat[j - 1, 6]
        ):
            if mobmat[j, 0] == 1:
                mobmat[j, 1] = mobmat[j - 1, 4]
                mobmat[j, 2] = mobmat[j - 1, 5]
            if mobmat[j - 1, 0] == 1:
                mobmat[j - 1, 4] = mobmat[j, 1]
                mobmat[j - 1, 5] = mobmat[j, 2]

    mobmat = np.hstack(
        (mobmat, np.ones(mobmat.shape[0]).reshape(mobmat.shape[0], 1))
    )
    mobmat = np.vstack((mobmat, new_pauses_arr))
    mobmat = mobmat[mobmat[:, 3].argsort()].astype(float)
    return mobmat
