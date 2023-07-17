"""This module contains functions to convert the mobility matrix into
trajectories. It is part of the Jasmine package.
"""
import sys
import math

import numpy as np
import scipy.stats as stat

from ..poplar.legacy.common_funcs import stamp2datetime
from .data2mobmat import great_circle_dist, great_circle_dist_vec, exist_knot


# the details of the functions are in paper [Liu and Onnela (2020)]
def update_existing_place(loc_x, loc_y, num_xy, t_xy, data, index):
    """Update the existing significant place with the new data point.

    Args:
        loc_x: list, a list of latitudes of significant places
        loc_y: list, a list of longitudes of significant places
        num_xy: list, a list of frequency/counts
            (appear in the dataset) of significant places
        t_xy: list, a list of duration at those significant places
        data: 1d np.ndarray, (7)
        index: int, index of the existing significant place
    """
    loc_x[index] = (
        loc_x[index] * num_xy[index] + data[1]
    ) / (num_xy[index] + 1)
    loc_y[index] = (
        loc_y[index] * num_xy[index] + data[2]
    ) / (num_xy[index] + 1)
    num_xy[index] += 1
    t_xy[index] += data[6] - data[3]


def add_new_place(loc_x, loc_y, num_xy, t_xy, data):
    """Add a new significant place from the new data point.

    Args:
        loc_x: list, a list of latitudes of significant places
        loc_y: list, a list of longitudes of significant places
        num_xy: list, a list of frequency/counts
            (appear in the dataset) of significant places
        t_xy: list, a list of duration at those significant places
        data: 1d np.ndarray, (7)
    """
    loc_x.append(data[1])
    loc_y.append(data[2])
    num_xy.append(1)
    t_xy.append(data[6] - data[3])


def num_sig_places(data, dist_threshold):
    """This function is used to find significant places

    Args:
        data: 2d np.ndarray, (k*7)
        dist_threshold: float,
         radius of a significant place
    Returns:
        loc_x: list, a list of latitudes of significant places
        loc_y: list, a list of longitudes of significant places
        num_xy: list, a list of frequency/counts
         (appear in the dataset) of significant places
        t_xy: list, a list of duration at those significant places
    """
    loc_x = []
    loc_y = []
    num_xy = []
    t_xy = []

    for i in range(data.shape[0]):
        if not loc_x:
            add_new_place(loc_x, loc_y, num_xy, t_xy, data[i])
            continue

        distances = great_circle_dist_vec(
            np.full(len(loc_x), data[i, 1]),
            np.full(len(loc_x), data[i, 2]),
            np.array(loc_x),
            np.array(loc_y),
        )

        min_distance_index = np.argmin(distances)
        if distances[min_distance_index] > dist_threshold:
            # add a new significant place
            # if the distance is larger than dist_threshold
            add_new_place(loc_x, loc_y, num_xy, t_xy, data[i])
        else:
            # update the existing significant place
            update_existing_place(
                loc_x, loc_y, num_xy, t_xy, data[i], min_distance_index
            )

    return loc_x, loc_y, num_xy, t_xy


def locate_home(mob_mat, timezone):
    """This function is used to locate the home of a user

    Args:
        mob_mat: np.ndarray, a k*7 2d array
            output from infer_mobmat()
        timezone: str, timezone
    Returns:
        home_x, home_y: scalars,
            represent the latitude and longtitude of user's home
    Raises:
        RuntimeError: if not enough data to infer home location
    """
    # Extract the observed trajectories from the mobility matrix
    obs_traj = mob_mat[mob_mat[:, 0] == 2, :]

    # Initialize list to hold the hours of each observation
    hours = []

    # Convert timestamp to datetime and store the hour of each observation
    for i in range(obs_traj.shape[0]):
        time_list = stamp2datetime(
            (obs_traj[i, 3] + obs_traj[i, 6]) / 2, timezone
        )
        hours.append(time_list[3])

    # Convert list to numpy array
    hours_arr = np.array(hours)

    # If there are no observations between 19:00 and 09:00, raise an error
    if ((hours_arr >= 19) + (hours_arr <= 9)).sum() <= 0:
        raise RuntimeError(
            "No home location found: Too few observations at night"
        )

    # Extract the pauses (observed trajectories)
    # that occurred between 19:00 and 09:00
    home_time_pauses = obs_traj[
        ((hours_arr >= 19) + (hours_arr <= 9)) * obs_traj[:, 0] == 2, :
    ]

    # Identify significant places during home-time pauses
    locations_lat, locations_lon, frequency, _ = num_sig_places(
        home_time_pauses, 20
    )

    # Find the index of the most frequently visited location,
    # which is presumed to be home
    home_index = np.argmax(frequency)

    # Extract the latitude and longitude of the inferred home location
    home_x, home_y = locations_lat[home_index], locations_lon[home_index]

    return home_x, home_y


def calculate_k1(method, timestamp, x_coord, y_coord, bv_set, parameters):
    """Calculate the similarity measure between
     a given point and a set of base vectors,
     using one of three specified methods: 'TL', 'GL', or 'GLC'.

    Args:
        method: str,
            Method to use for the calculation. Should be 'TL', 'GL', or 'GLC'.
        timestamp: float,
            The timestamp of the point to compare.
        x_coord: float,
            The x-coordinate (e.g. longitude) of the point to compare.
        y_coord: float,
            The y-coordinate (e.g. lattitude) of the point to compare.
        bv_set: np.ndarray,
            A set of base vectors to compare with, typically
            a subset of output from BV_select().
        parameters: list,
            A list of parameters to use for the calculation.

    Returns:
        similarity_measures: np.ndarray,
            A 1d array of similarity measures
            between the given point and each base vector.

    Raises:
        ValueError: If an invalid method is specified.
    """
    [
        length_1, length_2, amplitude_1, amplitude_2,
        weight_1, weight_2, weight_3, spatial_scale
    ] = parameters

    mean_x = ((bv_set[:, 1] + bv_set[:, 4]) / 2).astype(float)
    mean_y = ((bv_set[:, 2] + bv_set[:, 5]) / 2).astype(float)
    mean_t = ((bv_set[:, 3] + bv_set[:, 6]) / 2).astype(float)

    if method not in ["TL", "GL", "GLC"]:
        raise ValueError(
            f"Invalid method: {method}. Expected 'TL', 'GL', or 'GLC'."
        )
    # 'TL' method
    if method == "TL":
        k1 = np.exp(-abs(timestamp - mean_t) / length_1) * np.exp(
            -((np.sin(abs(timestamp - mean_t) / 86400 * math.pi)) ** 2)
            / amplitude_1
        )
        k2 = np.exp(-abs(timestamp - mean_t) / length_2) * np.exp(
            -((np.sin(abs(timestamp - mean_t) / 604800 * math.pi)) ** 2)
            / amplitude_2
        )
        return (
            weight_1 / (weight_1 + weight_2) * k1
            + weight_2 / (weight_1 + weight_2) * k2
        )
    # 'GL' method
    if method == "GL":
        distance = great_circle_dist_vec(x_coord, y_coord, mean_x, mean_y)
        return np.exp(-distance / spatial_scale)
    # 'GLC' method
    if method == "GLC":
        k1 = np.exp(-abs(timestamp - mean_t) / length_1) * np.exp(
            -((np.sin(abs(timestamp - mean_t) / 86400 * math.pi)) ** 2)
            / amplitude_1
        )
        k2 = np.exp(-abs(timestamp - mean_t) / length_2) * np.exp(
            -((np.sin(abs(timestamp - mean_t) / 604800 * math.pi)) ** 2)
            / amplitude_2
        )
        distance = great_circle_dist_vec(x_coord, y_coord, mean_x, mean_y)
        k3 = np.exp(-distance / spatial_scale)
        return weight_1 * k1 + weight_2 * k2 + weight_3 * k3
    return None


def indicate_flight(
    method,
    current_t,
    current_x,
    current_y,
    dest_t,
    dest_x,
    dest_y,
    bv_set,
    switch,
    num,
    pars,
):
    """
    This function generates a binary variable to indicate
     whether a person is moving or not.

    Args:
        method: str, the method to be used for calculation,
         should be either 'TL', 'GL', or 'GLC'.
        current_t: float, the current time point.
        current_x: float, the current longitudinal position.
        current_y: float, the current latitudinal position.
        dest_t: float, the time point at the destination.
        dest_x: float, the longitudinal position of the destination.
        dest_y: float, the latitudinal position of the destination.
        bv_subset: np.ndarray,
         the subset of output from the BV_select() function.
         It is a 2D array.
        switch: int, the number of binary variables to be generated.
        num: int, checks the top k similarities.
         This helps to avoid the cumulative effect
         of many low probability trajectories.
        pars: list, the parameters that are required
         for the calculate_k1 function.

    Returns:
        numpy.ndarray: A 1D array of 0 and 1,
         indicating the status of an incoming flight.
    """
    # Calculate k1 using the specified method
    k1 = calculate_k1(method, current_t, current_x, current_y, bv_set, pars)

    # Select flight and pause indicators from the bv_subset
    flight_k = k1[bv_set[:, 0] == 1]
    pause_k = k1[bv_set[:, 0] == 2]

    # Sort the flight and pause indicators
    sorted_flight = np.sort(flight_k)[::-1]
    sorted_pause = np.sort(pause_k)[::-1]

    # Calculate the probability p0
    p0 = np.mean(sorted_flight[:num]) / (
        np.mean(sorted_flight[:num]) + np.mean(sorted_pause[:num]) + 1e-8
    )

    # Calculate the great circle distance to
    # the destination and the required speed
    distance_to_destination = great_circle_dist(
        current_x, current_y,
        dest_x, dest_y
    )
    speed_to_destination = distance_to_destination / (
        dest_t - current_t + 0.0001
    )

    # Adjust the probability based on the required speed
    # to reach the destination
    # design an exponential function here to adjust
    # the probability based on the speed needed
    # p = p0*exp(|v-2|+/s)  v=2--p=p0   v=14--p=1
    p0 = max(p0, 1e-5)
    p0 = min(p0, 1 - 1e-5)
    s = -12 / np.log(p0)
    p1 = min(1, p0 * np.exp(min(max(0, speed_to_destination - 2) / s, 1e2)))

    # Generate the binary variables
    movement_indicator = stat.bernoulli.rvs(p1, size=switch)

    return movement_indicator


def adjust_direction(
    linearity,
    delta_x,
    delta_y,
    start_x,
    start_y,
    end_x,
    end_y,
    origin_x,
    origin_y,
    dest_x,
    dest_y,
):
    """
    This function adjusts the trajectory direction based on
     a given linearity parameter.

    Args:
        linearity: float,
         Controls the smoothness of a trajectory.
         A larger value tends to result in a more linear trajectory
         from the starting point towards the destination,
         while a smaller value leads to more random directions.
        delta_x, delta_y: float,
         Initial displacements in the x and y axes.
        start_x, start_y: float,
         Coordinates of the start point.
        end_x, end_y: float,
         Coordinates of the end point.
        origin_x, origin_y: float,
         Coordinates of the origin point.
        dest_x, dest_y: float,
         Coordinates of the destination point.

    Returns:
        tuple: Two floats representing
         the adjusted displacement in the x and y axes.
    """
    # Calculate the norm of the vector from origin to destination
    origin_to_dest_norm = np.sqrt(
        (dest_x - origin_x) ** 2 + (dest_y - origin_y) ** 2
    )

    # Generate a random number with
    # uniform distribution within the range [0, linearity]
    smoothness_factor = np.random.uniform(low=0, high=linearity)

    # Adjust the displacement based on the smoothness factor
    adjusted_delta_x = delta_x + smoothness_factor * (
        dest_x - origin_x
    ) / origin_to_dest_norm
    adjusted_delta_y = delta_y + smoothness_factor * (
        dest_y - origin_y
    ) / origin_to_dest_norm

    # Calculate the norms of the initial and adjusted displacements
    initial_displacement_norm = np.sqrt(delta_x**2 + delta_y**2)
    adjusted_displacement_norm = np.sqrt(
        adjusted_delta_x**2 + adjusted_delta_y**2
    )

    # Normalize the adjusted displacement to maintain
    # the same magnitude as the initial displacement
    normalized_delta_x = (
        adjusted_delta_x
        * initial_displacement_norm / adjusted_displacement_norm
    )
    normalized_delta_y = (
        adjusted_delta_y
        * initial_displacement_norm / adjusted_displacement_norm
    )

    # Calculate the inner product of the vector
    # from start to end and the adjusted displacement vector
    inner_product = np.inner(
        np.array([end_x - start_x, end_y - start_y]),
        np.array([normalized_delta_x, normalized_delta_y])
    )

    # If the inner product is less than zero,
    # reverse the direction of the adjusted displacement
    if inner_product < 0:
        return -normalized_delta_x, -normalized_delta_y

    return normalized_delta_x, normalized_delta_y


def multiplier(t_diff):
    """This function generates a multiplier
     based on the time difference between two points.

    Args:
        t_diff, float, difference in time (unit in second)

    Returns:
        float, a multiplication coefficient
    """
    # If the time difference is less than 30 minutes,
    if t_diff <= 30 * 60:
        return 1
    # If the time difference is less than 3 hours,
    if t_diff <= 180 * 60:
        return 5
    # If the time difference is less than 18 hours,
    if t_diff <= 1080 * 60:
        return 10
    return 50


def checkbound(current_x, current_y, start_x, start_y, end_x, end_y):
    """This function checks whether a point is out of the boundary
     determined by the starting and ending points.

    Args:
        current_x, current_y: float,
            Coordinates of the current point.
        start_x, start_y: float,
            Coordinates of the start point.
        end_x, end_y: float,
            Coordinates of the end point.

    Returns:
        binary indicator, int, indicates whether (current_x, current_y)
            is out of the boundary determiend by
            starting and ending points
    """
    # Calculate the maximum and minimum x and y coordinates
    # of the starting and ending points
    # to determine the boundary box of the trajectory
    max_x = max(start_x, end_x)
    min_x = min(start_x, end_x)
    max_y = max(start_y, end_y)
    min_y = min(start_y, end_y)
    # If the current point is inside of the boundary box,
    # return 1, otherwise return 0
    if (
        current_x < max_x + 0.01
        and current_x > min_x - 0.01
        and current_y < max_y + 0.01
        and current_y > min_y - 0.01
    ):
        return 1
    return 0


def create_tables(mob_mat, bv_set):
    """This function creates three tables:
        one for observed flights, one for observed pauses,
        and one for missing intervals.

    Args:
        mob_mat: 2d np.ndarray, output from infer_mobmat()
        bv_set: np.ndarray, output from BV_select()

    Returns:
        3 2d np.ndarray, one for observed flights,
         one for observed pauses, one for missing interval
         (where the last two cols are the status
         of previous obs traj and next obs traj)
    """
    # Number of rows in the mobility matrix and bv_set
    mob_mat_rows = np.shape(mob_mat)[0]
    bv_set_rows = np.shape(bv_set)[0]

    # Create flight table: select rows from bv_set where first column equals 1
    flight_table = bv_set[[bv_set[i, 0] == 1 for i in range(bv_set_rows)], :]

    # Create pause table: select rows from bv_set where first column equals 2
    pause_table = bv_set[[bv_set[i, 0] == 2 for i in range(bv_set_rows)], :]

    # Initialize missing intervals table
    mis_table = np.zeros((1, 8))

    # Iterate over mobility matrix rows to find missing intervals
    for i in range(mob_mat_rows - 1):
        # Check if end of current interval doesn't match
        # the start of the next interval
        if mob_mat[i + 1, 3] != mob_mat[i, 6]:
            # Record missing interval along with its status
            # (flight/pause) before and after
            mov = np.array(
                [
                    mob_mat[i, 4],
                    mob_mat[i, 5],
                    mob_mat[i, 6],
                    mob_mat[i + 1, 1],
                    mob_mat[i + 1, 2],
                    mob_mat[i + 1, 3],
                    mob_mat[i, 0],
                    mob_mat[i + 1, 0],
                ]
            )
            # Append the missing interval to the table
            mis_table = np.vstack((mis_table, mov))

    # Remove the first row of missing_intervals_table
    # (which was initialized to zero)
    mis_table = np.delete(mis_table, 0, 0)

    return flight_table, pause_table, mis_table


def calculate_delta(flight_table, flight_index, backwards=False):
    """This function calculates the displacement
        and time difference between two points.

    Args:
        flight_table: 2d np.ndarray, output from create_tables()
        flight_index: int, index of the flight in flight_table
        backwards: bool, whether the flight is backwards

    Returns:
        delta_x, delta_y, delta_t: float,
            displacement and time difference
            between two points
    """
    delta_x = flight_table[flight_index, 4] - flight_table[flight_index, 1]
    delta_y = flight_table[flight_index, 5] - flight_table[flight_index, 2]
    delta_t = flight_table[flight_index, 6] - flight_table[flight_index, 3]
    if backwards:
        delta_x = -delta_x
        delta_y = -delta_y
    return delta_x, delta_y, delta_t


def adjust_delta_if_needed(start_t, delta_t, delta_x, delta_y, end_t):
    """This function adjusts the displacement and time difference
        between two points if start_t + delta_t > end_t.

    Args:
        start_t: float, start time
        delta_t: float, time difference
        delta_x, delta_y: float, displacement
        end_t: float, end time

    Returns:
        delta_x, delta_y, delta_t: float,
            adjusted displacement and time difference
            between two points
    """
    if start_t + delta_t > end_t:
        temp = delta_t
        delta_t = end_t - start_t
        delta_x = delta_x * delta_t / temp
        delta_y = delta_y * delta_t / temp
    return delta_x, delta_y, delta_t


def calculate_position(start_t, end_t, try_t, start_delta, end_delta):
    """This function calculates the position of a point
        at a given time.

    Args:
        start_t: float, start time
        end_t: float, end time
        try_t: float, time to be calculated
        start_delta, end_delta: float, displacement

    Returns:
        float, position at a given time
    """

    time_diff = end_t - start_t + 1e-5
    time_part1 = try_t - start_t
    time_part2 = end_t - try_t
    res = time_part2 / time_diff * start_delta + \
        time_part1 / time_diff * end_delta

    return res


def update_table(imp_table, array):
    return np.vstack((imp_table, np.array(array)))


def forward_impute(
    start_t, start_x, start_y, end_t, end_x, end_y,
    bv_set, switch, num, pars, flight_table, linearity,
    mis_row, pause_table, imp_table,
    start_s, method, counter
):
    """
    This function imputes a missing interval
        from the start point to the end point.

    Args:
        start_t: float, start time
        start_x, start_y: float, start position
        end_t: float, end time
        end_x, end_y: float, end position
        bv_set: np.ndarray, output from BV_select()
        switch: int, the number of binary variables to be generated
        num: int, checks the top k similarities
        pars: list, the parameters that are required
         for the calculate_k1 function
        flight_table: np.ndarray, output from create_tables()
        linearity: float, controls the smoothness of a trajectory
        mis_row: np.ndarray, a row of missing interval table
        pause_table: np.ndarray, output from create_tables()
        imp_table: np.ndarray, output from create_tables()
        start_s: int, status of the start point
        method: str, the method to be used for calculation,
         should be either 'TL', 'GL', or 'GLC'
        counter: int, number of imputed trajectories

    Returns:
        imp_table: updated imp_table
        tuple of 4 elements:
            start_s: int, status of the start point
            start_t: float, start time
            start_x, start_y: float, start position
        counter: updated counter
    """

    I0 = indicate_flight(
        method, start_t, start_x, start_y, end_t, end_x,
        end_y, bv_set, switch, num, pars
    )

    condition = (sum(I0 == 1) == switch and start_s == 2) or \
        (sum(I0 == 0) < switch and start_s == 1)

    if condition:
        weight = calculate_k1(
            method, start_t, start_x, start_y,
            flight_table, pars
        )

        normalize_w = (weight + 1e-5) / sum(weight + 1e-5)
        flight_index = np.random.choice(flight_table.shape[0], p=normalize_w)

        delta_x, delta_y, delta_t = calculate_delta(flight_table, flight_index)

        delta_x, delta_y, delta_t = adjust_delta_if_needed(
            start_t, delta_t,
            delta_x, delta_y, end_t
        )

        delta_x, delta_y = adjust_direction(
            linearity, delta_x, delta_y,
            start_x, start_y, end_x, end_y,
            *mis_row[[0, 1, 3, 4]],
        )

        try_t = start_t + delta_t
        try_x = calculate_position(
            start_t, end_t, try_t, start_x + delta_x, end_x
        )
        try_y = calculate_position(
            start_t, end_t, try_t, start_y + start_t, end_y
        )

        mov1 = great_circle_dist(try_x, try_y, start_x, start_y)
        mov2 = great_circle_dist(end_x, end_y, start_x, start_y)

        check1 = checkbound(
            try_x, try_y,
            *mis_row[[0, 1, 3, 4]],
        )
        check2 = int(mov1 < mov2)

        # conditions and actions
        if end_t > start_t and check1 == 1 and check2 == 1:
            current_t = start_t + delta_t
            current_x = calculate_position(
                start_t, end_t, current_t,
                start_x + delta_x, end_x
            )
            current_y = calculate_position(
                start_t, end_t, current_t,
                start_y + delta_y, end_y
            )
            imp_table = update_table(
                imp_table,
                [
                    1, start_x, start_y,
                    start_t, current_x,
                    current_y, current_t
                ]
            )

            start_x, start_y, start_t, start_s = (
                current_x, current_y, current_t, 1
            )
            counter += 1

        if end_t > start_t and check2 == 0:
            speed = mov1 / delta_t
            t_need = mov2 / speed
            current_t = start_t + t_need
            imp_table = update_table(
                imp_table,
                [
                    1, start_x, start_y,
                    start_t, end_x,
                    end_y, current_t
                ]
            )
            start_x, start_y, start_t, start_s = end_x, end_y, current_t, 1
            counter += 1
        else:
            weight = calculate_k1(
                method, start_t, start_x, start_y,
                pause_table, pars
            )
            normalize_w = (weight + 1e-5) / sum(weight + 1e-5)
            pause_index = np.random.choice(pause_table.shape[0], p=normalize_w)
            delta_t = (
                pause_table[pause_index, 6] - pause_table[pause_index, 3]
            ) * multiplier(end_t - start_t)

            if start_t + delta_t < end_t:
                current_t = start_t + delta_t
                imp_table = update_table(
                    imp_table,
                    [
                        2, start_x, start_y,
                        start_t, start_x,
                        start_y, current_t
                    ]
                )
                start_t, start_s = current_t, 2
                counter += 1
            else:
                imp_table = update_table(
                    imp_table,
                    [
                        1, start_x, start_y,
                        start_t, end_x,
                        end_y, end_t
                    ]
                )
                start_t = end_t
    return imp_table, (start_s, start_t, start_x, start_y), counter


def backward_impute(
    end_t,
    end_x,
    end_y,
    start_t,
    start_x,
    start_y,
    bv_set,
    switch,
    num,
    pars,
    flight_table,
    linearity,
    mis_row,
    pause_table,
    imp_table,
    end_s,
    method,
    counter,
):
    """
    This function imputes a missing interval
        from the end point to the start point.

    Args:
        end_t: float, end time
        end_x, end_y: float, end position
        start_t: float, start time
        start_x, start_y: float, start position
        bv_set: np.ndarray, output from BV_select()
        switch: int, the number of binary variables to be generated
        num: int, checks the top k similarities
        pars: list, the parameters that are required
         for the calculate_k1 function
        flight_table: np.ndarray, output from create_tables()
        linearity: float, controls the smoothness of a trajectory
        mis_row: np.ndarray, a row of missing interval table
        pause_table: np.ndarray, output from create_tables()
        imp_table: np.ndarray, output from create_tables()
        end_s: int, status of the end point
        method: str, the method to be used for calculation,
         should be either 'TL', 'GL', or 'GLC'
        counter: int, number of imputed trajectories

    Returns:
        imp_table: updated imp_table
        tuple of 4 elements:
            end_s: int, status of the end point
            end_t: float, end time
            end_x, end_y: float, end position
        counter: updated counter
    """

    I1 = indicate_flight(
        method, end_t, end_x, end_y, start_t, start_x,
        start_y, bv_set, switch, num, pars
    )

    condition = (sum(I1 == 1) == switch and end_s == 2) or \
        (sum(I1 == 0) < switch and end_s == 1)

    if condition:
        weight = calculate_k1(
            method, end_t, end_x, end_y,
            flight_table, pars
        )

        normalize_w = (weight + 1e-5) / sum(weight + 1e-5)
        flight_index = np.random.choice(flight_table.shape[0], p=normalize_w)

        delta_x, delta_y, delta_t = calculate_delta(
            flight_table, flight_index, backwards=True
        )

        delta_x, delta_y, delta_t = adjust_delta_if_needed(
            start_t, delta_t,
            delta_x, delta_y, end_t
        )

        delta_x, delta_y = adjust_direction(
            linearity, delta_x, delta_y,
            end_x, end_y, start_x, start_y,
            *mis_row[[3, 4, 0, 1]],
        )

        try_t = end_t - delta_t
        try_x = calculate_position(
            end_t, try_t, 0, start_x, start_t, end_x + delta_x
        )
        try_y = calculate_position(
            end_t, try_t, 0, start_y, start_t, end_y + delta_y
        )

        mov1 = great_circle_dist(try_x, try_y, end_x, end_y)
        mov2 = great_circle_dist(end_x, end_y, start_x, start_y)

        check1 = checkbound(
            try_x, try_y,
            *mis_row[[0, 1, 3, 4]],
        )
        check2 = int(mov1 < mov2)

        # conditions and actions
        if end_t > start_t and check1 == 1 and check2 == 1:
            current_t = end_t - delta_t
            current_x = calculate_position(
                end_t, current_t, 0,
                start_x, start_t, end_x + delta_x
            )
            current_y = calculate_position(
                end_t, current_t, 0,
                start_y, end_y + delta_y
            )
            imp_table = update_table(
                imp_table,
                [
                    1, current_x, current_y,
                    current_t, end_x, end_y, end_t
                ]
            )

            end_x, end_y, end_t, end_s = current_x, current_y, current_t, 1
            counter += 1

        if end_t > start_t and check2 == 0:
            speed = mov1 / delta_t
            t_need = mov2 / speed
            current_t = end_t - t_need
            imp_table = update_table(
                imp_table,
                [
                    1, start_x, start_y,
                    current_t, end_x, end_y, end_t
                ]
            )
            end_x, end_y, end_t, end_s = start_x, start_y, current_t, 1
            counter += 1
        else:
            weight = calculate_k1(
                method, end_t, end_x, end_y,
                pause_table, pars
            )
            normalize_w = (weight + 1e-5) / sum(weight + 1e-5)
            pause_index = np.random.choice(pause_table.shape[0], p=normalize_w)
            delta_t = (
                pause_table[pause_index, 6] - pause_table[pause_index, 3]
            ) * multiplier(end_t - start_t)

            if start_t + delta_t < end_t:
                current_t = end_t - delta_t
                imp_table = update_table(
                    imp_table,
                    [
                        2, end_x, end_y,
                        current_t, end_x, end_y, end_t
                    ]
                )
                end_t, end_s = current_t, 2
                counter += 1
            else:
                imp_table = update_table(
                    imp_table,
                    [
                        1, start_x, start_y,
                        start_t, end_x, end_y, end_t
                    ])
                end_t = start_t

    return imp_table, (end_s, end_t, end_x, end_y), counter


def impute_gps(mob_mat, bv_set, method, switch, num, linearity, tz_str, pars):
    """
    This is the algorithm for the bi-directional imputation in the paper

    Args:
        mob_mat: 2d np.ndarray, output from infer_mobmat()
        bv_set: np.ndarray, output from BV_select()
        method: str, the method to be used for calculation,
            should be either 'TL', 'GL', or 'GLC'
        switch: int, the number of binary variables to be generated
        num: int, checks the top k similarities
        linearity: float, controls the smoothness of a trajectory
        tz_str: str, time zone
        pars: list, the parameters that are required
            for the calculate_k1 function

    Returns:
        2d array simialr to mob_mat, but it
            is a complete imputed traj (first-step result)
            with headers [imp_s,imp_x0,imp_y0,imp_t0,imp_x1,imp_y1,imp_t1]
    """
    # identify home location
    home_coords = locate_home(mob_mat, tz_str)
    sys.stdout.write("Imputing missing trajectories ...\n")

    # create three tables
    # for observed flights, observed pauses, and missing intervals
    flight_table, pause_table, mis_table = create_tables(mob_mat, bv_set)

    # initialize the imputed trajectory table
    imp_table = np.zeros((1, 7))

    # iterate over missing intervals
    for i in range(mis_table.shape[0]):
        # Extract the start and end times of the missing interval
        mis_t0 = mis_table[i, 2]
        mis_t1 = mis_table[i, 5]

        # get the number of flights observed in the nearby 24 hours
        nearby_flight = sum(
            (flight_table[:, 6] > mis_t0 - 12 * 60 * 60)
            * (flight_table[:, 3] < mis_t1 + 12 * 60 * 60)
        )

        # get the distance difference between start and end
        # coordinates of the missing interval
        distance_difference = great_circle_dist(
            *mis_table[i, [0, 1]], *mis_table[i, [3, 4]]
        )

        # get the time difference between start and end
        # times of the missing interval
        time_difference = mis_t1 - mis_t0

        # get the distance between the start location
        # of the missing interval and the home location
        start_home_distance = great_circle_dist(
           *mis_table[i, [0, 1]], *home_coords
        )
        # get the distance between the end location
        # of the missing interval and the home location
        end_home_distance = great_circle_dist(
            *mis_table[i, [3, 4]], *home_coords
        )

        # if a person remains at the same place at the begining
        # and end of missing, just assume he satys there all the time
        if (
            mis_table[i, 0] == mis_table[i, 3]
            and mis_table[i, 1] == mis_table[i, 4]
        ):
            imp_table = update_table(
                imp_table,
                [2, *mis_table[i, 0:6]],
            )
        # if the distance difference is more than 300 km,
        # we assume the person takes a flight
        elif distance_difference > 300000:
            speed = distance_difference / time_difference
            # if the speed is more than 210 m/s, we assume it is a flight
            if speed > 210:
                imp_table = update_table(
                    imp_table,
                    [1, *mis_table[i, 0:6]],
                )
            else:
                # if the speed is less than 210 m/s,
                # generate a random speed between 244 and 258 m/s
                random_speed = np.random.uniform(low=244, high=258)
                # calculate the time needed to travel the distance
                t_need = distance_difference / random_speed
                # generate a random start time
                # from the start time of the missing interval
                # to the end time of the missing interval minus
                # the time needed to travel the distance
                t_s = np.random.uniform(
                    low=mis_t0, high=mis_t1 - t_need
                )
                t_e = t_s + t_need
                imp_table = update_table(
                    imp_table,
                    [
                        [2, 1, 2],
                        [mis_table[i, 0], mis_table[i, 0], mis_table[i, 3]],
                        [mis_table[i, 1], mis_table[i, 1], mis_table[i, 4]],
                        [mis_table[i, 2], t_s, t_e],
                        [mis_table[i, 0], mis_table[i, 3], mis_table[i, 3]],
                        [mis_table[i, 1], mis_table[i, 4], mis_table[i, 4]],
                        [t_s, t_e, mis_table[i, 5]],
                    ]
                )
        # add one more check about how many flights observed
        # in the nearby 24 hours
        elif (
            nearby_flight <= 5
            and time_difference > 6 * 60 * 60
            and min(start_home_distance, end_home_distance) > 50
        ):
            # if the distance difference is less than 3 km,
            # generate a random speed between 1 and 1.8 m/s
            if distance_difference < 3000:
                random_speed = np.random.uniform(low=1, high=1.8)
            # if the distance difference is more than 3 km,
            # generate a random speed between 13 and 32 m/s
            else:
                random_speed = np.random.uniform(low=13, high=32)

            t_need = min(distance_difference / random_speed, time_difference)
            # if the time needed to travel the distance
            # is less than the time difference,
            if t_need == time_difference:
                imp_table = update_table(
                    imp_table,
                    [1, *mis_table[i, 0:6]],
                )
            else:
                t_s = np.random.uniform(
                    low=mis_t0, high=mis_t1 - t_need
                )
                t_e = t_s + t_need
                imp_table = update_table(
                    imp_table,
                    [
                        [2, 1, 2],
                        [mis_table[i, 0], mis_table[i, 0], mis_table[i, 3]],
                        [mis_table[i, 1], mis_table[i, 1], mis_table[i, 4]],
                        [mis_table[i, 2], t_s, t_e],
                        [mis_table[i, 0], mis_table[i, 3], mis_table[i, 3]],
                        [mis_table[i, 1], mis_table[i, 4], mis_table[i, 4]],
                        [t_s, t_e, mis_table[i, 5]],
                    ]
                )

        else:
            # solve the problem that a person has a
            # trajectory like flight/pause/flight/pause/flight...
            # we want it more like flght/flight/flight
            # /pause/pause/pause/flight/flight...
            # start from two ends, we make it harder to change the current
            # pause/flight status by drawing multiple random
            # variables form bin(p0) and require them to be all 0/1
            # "switch" is the number of random variables
            start_x, start_y, start_t, end_x, end_y, end_t, start_s, end_s = \
                mis_table[i, :]
            if (
                time_difference > 4 * 60 * 60
                and min(start_home_distance, end_home_distance) <= 50
            ):
                t_need = min(distance_difference / 0.6, time_difference)
                # if distance from home to start/end is less than 50 km
                # and the time difference is more than 4 hours
                # set the start/end time to be the same
                # and the start/end location to be the same
                if start_home_distance <= 50:
                    imp_table = update_table(
                        imp_table,
                        [
                            2, start_x, start_y, start_t,
                            start_x, start_y, end_t - t_need,
                        ]
                    )
                    start_t = end_t - t_need
                else:
                    imp_table = update_table(
                        imp_table,
                        [
                            2, end_x, end_y, start_t + t_need,
                            end_x, end_y, end_t,
                        ]
                    )
                    end_t = start_t + t_need

            counter = 0

            while start_t < end_t:
                # if start and end location are different
                # and the time difference is less than 30 seconds
                if (
                    abs(start_x - end_x) + abs(start_y - end_y) > 0
                    and end_t - start_t < 30
                ):  # avoid extreme high speed
                    imp_table = update_table(
                        imp_table,
                        [
                            1, start_x, start_y, start_t,
                            end_x, end_y, end_t,
                        ]
                    )
                    start_t = end_t
                # if start and end location are the same
                elif start_x == end_x and start_y == end_y:
                    imp_table = update_table(
                        imp_table,
                        [
                            2, start_x, start_y, start_t,
                            end_x, end_y, end_t,
                        ]
                    )
                    start_t = end_t
                # if start and end location are different
                # and the time difference is more than 30 seconds
                else:
                    direction = "forward" if counter % 2 == 0 else "backward"

                    if direction == "forward":
                        imp_table, start, counter = forward_impute(
                            start_t, start_x, start_y, end_t, end_x, end_y,
                            bv_set, switch, num, pars, flight_table, linearity,
                            mis_table[i, :], pause_table, imp_table,
                            start_s, method, counter
                        )
                        start_s, start_t, start_x, start_y = start
                    elif direction == "backward":
                        imp_table, end, counter = backward_impute(
                            end_t, end_x, end_y, start_t, start_x, start_y,
                            bv_set, switch, num, pars, flight_table, linearity,
                            mis_table[i, :], pause_table, imp_table,
                            end_s, method, counter
                        )
                        end_s, end_t, end_x, end_y = end
    imp_table = imp_table[imp_table[:, 3].argsort()].astype(float)
    return imp_table


def imp_to_traj(imp_table, mob_mat, r, w, h):
    """This function tidies up the first-step imputed trajectory,
    such as combining pauses, flights shared by both observed
    and missing intervals, also combine consecutive flight
    with slightly different directions as one longer flight

    Args:
        imp_table: 2d array, the first-step imputed trajectory
        mob_mat: 2d array, output from infer_mobmat()
        r: float, the radius of the earth
        w: float, the weight for the distance
        h: float, the weight for the time

    Returns:
        2d array, the final imputed trajectory,
            with one more columm compared to imp_table
            which is an indicator showing if the piece
            of traj is imputed (0) or observed (1)
    """
    sys.stdout.write("Tidying up the trajectories...\n")

    # Create a table for missing values
    mis_table = np.zeros((1, 8))

    # Find missing values in mobility matrix and add them to the mis_table
    for i in range(mob_mat.shape[0] - 1):
        if mob_mat[i + 1, 3] != mob_mat[i, 6]:
            movement_data = np.array(
                [
                    *mob_mat[i, 4:7],
                    *mob_mat[i + 1, 1:4],
                    mob_mat[i, 0],
                    mob_mat[i + 1, 0],
                ]
            )
            mis_table = np.vstack((mis_table, movement_data))

    # Delete the first row of zeros
    mis_table = np.delete(mis_table, 0, 0)

    tidy_trajectory = []
    for k in range(mis_table.shape[0]):
        # Create an index to select values from the imputed_trajectory
        index = (imp_table[:, 3] >= mis_table[k, 2]) * (
            imp_table[:, 6] <= mis_table[k, 5]
        )
        temp = imp_table[index, :]

        # Iterate through the rows of the temporary trajectory
        start_idx = 0
        end_idx = 1
        while start_idx < temp.shape[0]:
            if end_idx < temp.shape[0]:
                if temp[end_idx, 0] == temp[start_idx, 0]:
                    end_idx += 1

            # Check if we have reached the end or found a different trajectory
            if (
                end_idx == temp.shape[0]
                or temp[
                    min(end_idx, temp.shape[0] - 1), 0
                ] != temp[start_idx, 0]
            ):
                start = start_idx
                end = end_idx - 1
                start_idx = end_idx
                end_idx += 1

                # If the trajectory is a flight, add it to the tidy_trajectory
                if temp[start, 0] == 2:
                    tidy_trajectory.append(
                        [
                            2,
                            *temp[start, 1:4],
                            *temp[end, 4:7],
                        ]
                    )
                # If the trajectory is a pause, add it to the tidy_trajectory
                elif end == start:
                    tidy_trajectory.append(
                        [
                            1,
                            *temp[start, 1:4],
                            *temp[end, 4:7],
                        ]
                    )
                else:
                    # More complex case
                    # when multiple pauses/trajectories are combined
                    mat = np.vstack(
                        (
                            temp[start, 1:4],
                            temp[np.arange(start, end + 1), 4:7]
                        )
                    )
                    mat = np.append(
                        mat,
                        np.arange(0, mat.shape[0]).reshape(mat.shape[0], 1),
                        1
                    )
                    complete = 0
                    knots = [0, mat.shape[0] - 1]
                    while complete == 0:
                        mat_list = []
                        for i in range(len(knots) - 1):
                            mat_list.append(
                                mat[
                                    knots[i]:min(
                                        knots[i + 1] + 1, mat.shape[0] - 1
                                    ),
                                    :,
                                ]
                            )
                        knot_yes = np.empty(len(mat_list))
                        knot_pos = np.empty(len(mat_list))
                        for i, mat_elem in enumerate(mat_list):
                            knot_yes[i], knot_pos[i] = exist_knot(
                                mat_elem, w
                            )
                        if sum(knot_yes) == 0:
                            complete = 1
                        else:
                            for i, mat_elem in enumerate(mat_list):
                                if knot_yes[i] == 1:
                                    knots.append(
                                        int((mat_elem)[int(knot_pos[i]), 3])
                                    )
                            knots.sort()
                    for j in range(len(knots) - 1):
                        tidy_trajectory.append(
                            [
                                1,
                                *mat[knots[j], 0:3],
                                *mat[knots[j + 1], 0:3],
                            ]
                        )

    traj = np.array(tidy_trajectory)
    if traj.shape[0] != 0:
        traj = np.hstack((traj, np.zeros((traj.shape[0], 1))))
        final_traj = np.vstack((traj, mob_mat))
    else:
        final_traj = mob_mat

    float_traj = final_traj[final_traj[:, 3].argsort()].astype(float)
    final_traj = float_traj[float_traj[:, 6] - float_traj[:, 3] > 0, :]

    return final_traj
