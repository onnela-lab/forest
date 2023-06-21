"""This module contains functions to convert the mobility matrix into
trajectories. It is part of the Jasmine package.
"""
import sys
import math

import numpy as np
import scipy.stats as stat

from ..poplar.legacy.common_funcs import stamp2datetime
from .data2mobmat import great_circle_dist, exist_knot


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

        distances = great_circle_dist(
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
            output from InferMobMat()
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
        distance = great_circle_dist(x_coord, y_coord, mean_x, mean_y)
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
        distance = great_circle_dist(x_coord, y_coord, mean_x, mean_y)
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
    Args: method, string, should be 'TL', or 'GL' or 'GLC'
          current_t, current_x, current_y, dest_t,dest_x,dest_y are scalars
          bv_set, 2d array, the (subset of) output from BV_select()
          switch: the number of binary variables we want to generate,
           this controls the difficulty to change
             the status from flight to pause or from pause to flight
          num: check top k similarities
           (avoid the cumulative effect of many low prob trajs)
    Return: 1d array of 0 and 1, of length switch,
     indicator of a incoming flight
    """
    k1 = calculate_k1(method, current_t, current_x, current_y, bv_set, pars)
    flight_k = k1[bv_set[:, 0] == 1]
    pause_k = k1[bv_set[:, 0] == 2]
    sorted_flight = np.sort(flight_k)[::-1]
    sorted_pause = np.sort(pause_k)[::-1]
    p0 = np.mean(sorted_flight[0:num]) / (
        np.mean(sorted_flight[0:num]) + np.mean(sorted_pause[0:num]) + 1e-8
    )
    d_dest = great_circle_dist(current_x, current_y, dest_x, dest_y)
    v_dest = d_dest / (dest_t - current_t + 0.0001)
    # design an exponential function here to adjust
    # the probability based on the speed needed
    # p = p0*exp(|v-2|+/s)  v=2--p=p0   v=14--p=1
    p0 = max(p0, 1e-5)
    p0 = min(p0, 1 - 1e-5)
    s = -12 / np.log(p0)
    p1 = min(1, p0 * np.exp(min(max(0, v_dest - 2) / s, 1e2)))
    out = stat.bernoulli.rvs(p1, size=switch)
    return out


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
    Args: linearity, a scalar that controls the smoothness of a trajectory
          a large linearity tends to have a more linear traj
          from starting point toward destination
          a small one tends to have more random directions

          delta_x,delta_y,start_x,start_y,end_x,end_y,origin_x
          ,origin_y,dest_x,dest_y are scalars
    Return: 2 scalars, represent the adjusted dispacement in two axises
    """
    norm1 = np.sqrt((dest_x - origin_x) ** 2 + (dest_y - origin_y) ** 2)
    k = np.random.uniform(
        low=0, high=linearity
    )  # this is another parameter which controls the smoothness
    new_x = delta_x + k * (dest_x - origin_x) / norm1
    new_y = delta_y + k * (dest_y - origin_y) / norm1
    norm2 = np.sqrt(delta_x**2 + delta_y**2)
    norm3 = np.sqrt(new_x**2 + new_y**2)
    norm_x = new_x * norm2 / norm3
    norm_y = new_y * norm2 / norm3
    inner = np.inner(
        np.array([end_x - start_x, end_y - start_y]),
        np.array([norm_x, norm_y])
    )
    if inner < 0:
        return -norm_x, -norm_y
    return norm_x, norm_y


def multiplier(t_diff):
    """
    Args: a scalar, difference in time (unit in second)
    Return: a scalar, a multiplication coefficient
    """
    if t_diff <= 30 * 60:
        return 1
    if t_diff <= 180 * 60:
        return 5
    if t_diff <= 1080 * 60:
        return 10
    return 50


def checkbound(current_x, current_y, start_x, start_y, end_x, end_y):
    """
    Args: all scalars
    Return: 1/0, indicates whether (current_x, current_y)
            is out of the boundary determiend by
            starting and ending points
    """
    max_x = max(start_x, end_x)
    min_x = min(start_x, end_x)
    max_y = max(start_y, end_y)
    min_y = min(start_y, end_y)
    if (
        current_x < max_x + 0.01
        and current_x > min_x - 0.01
        and current_y < max_y + 0.01
        and current_y > min_y - 0.01
    ):
        return 1
    return 0


def create_tables(mob_mat, bv_set):
    """
    Args: mob_mat, 2d array, output from InferMobMat()
          bv_set, 2d array, output from BV_select()
    Return: 3 2d arrays, one for observed flights,
     one for observed pauses, one for missing interval
     (where the last two cols are the status
     of previous obs traj and next obs traj)
    """
    mob_mat_rows = np.shape(mob_mat)[0]
    bv_set_rows = np.shape(bv_set)[0]
    index = [bv_set[i, 0] == 1 for i in range(bv_set_rows)]
    flight_table = bv_set[index, :]
    index = [bv_set[i, 0] == 2 for i in range(bv_set_rows)]
    pause_table = bv_set[index, :]
    mis_table = np.zeros((1, 8))
    for i in range(mob_mat_rows - 1):
        if mob_mat[i + 1, 3] != mob_mat[i, 6]:
            # also record if it's flight/pause
            # before and after the missing interval
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
            mis_table = np.vstack((mis_table, mov))
    mis_table = np.delete(mis_table, 0, 0)
    return flight_table, pause_table, mis_table


def impute_gps(mob_mat, bv_set, method, switch, num, linearity, tz_str, pars):
    """
    This is the algorithm for the bi-directional imputation in the paper
    Args: mob_mat, 2d array, output from InferMobMat()
          bv_set, 2d array, output from BV_select()
          method, string, should be 'TL', or 'GL' or 'GLC'
          switch, the number of binary variables we want to
             generate, this controls the difficulty to change
             the status from flight to pause or from pause to flight
          linearity, a scalar that controls the smoothness of a trajectory
               a large linearity tends to have
               a more linear traj from starting point toward destination
               a small one tends to have more random directions
          tz_str, timezone
    Return: 2d array simialr to mob_mat, but it
            is a complete imputed traj (first-step result)
            with headers [imp_s,imp_x0,imp_y0,imp_t0,imp_x1,imp_y1,imp_t1]
    """
    home_x, home_y = locate_home(mob_mat, tz_str)
    sys.stdout.write("Imputing missing trajectories ...\n")
    flight_table, pause_table, mis_table = create_tables(mob_mat, bv_set)
    imp_x0 = np.array([])
    imp_x1 = np.array([])
    imp_y0 = np.array([])
    imp_y1 = np.array([])
    imp_t0 = np.array([])
    imp_t1 = np.array([])
    imp_s = np.array([])
    for i in range(mis_table.shape[0]):
        mis_t0 = mis_table[i, 2]
        mis_t1 = mis_table[i, 5]
        nearby_flight = sum(
            (flight_table[:, 6] > mis_t0 - 12 * 60 * 60)
            * (flight_table[:, 3] < mis_t1 + 12 * 60 * 60)
        )
        d_diff = great_circle_dist(
            mis_table[i, 0], mis_table[i, 1], mis_table[i, 3], mis_table[i, 4]
        )
        t_diff = mis_table[i, 5] - mis_table[i, 2]
        distance1 = great_circle_dist(
            mis_table[i, 0], mis_table[i, 1], home_x, home_y
        )
        distance2 = great_circle_dist(
            mis_table[i, 3], mis_table[i, 4], home_x, home_y
        )
        # if a person remains at the same place at the begining
        # and end of missing, just assume he satys there all the time
        if (
            mis_table[i, 0] == mis_table[i, 3]
            and mis_table[i, 1] == mis_table[i, 4]
        ):
            imp_s = np.append(imp_s, 2)
            imp_x0 = np.append(imp_x0, mis_table[i, 0])
            imp_x1 = np.append(imp_x1, mis_table[i, 3])
            imp_y0 = np.append(imp_y0, mis_table[i, 1])
            imp_y1 = np.append(imp_y1, mis_table[i, 4])
            imp_t0 = np.append(imp_t0, mis_table[i, 2])
            imp_t1 = np.append(imp_t1, mis_table[i, 5])
        elif d_diff > 300000:
            v_diff = d_diff / t_diff
            if v_diff > 210:
                imp_s = np.append(imp_s, 1)
                imp_x0 = np.append(imp_x0, mis_table[i, 0])
                imp_x1 = np.append(imp_x1, mis_table[i, 3])
                imp_y0 = np.append(imp_y0, mis_table[i, 1])
                imp_y1 = np.append(imp_y1, mis_table[i, 4])
                imp_t0 = np.append(imp_t0, mis_table[i, 2])
                imp_t1 = np.append(imp_t1, mis_table[i, 5])
            else:
                v_random = np.random.uniform(low=244, high=258)
                t_need = d_diff / v_random
                t_s = np.random.uniform(
                    low=mis_table[i, 2], high=mis_table[i, 5] - t_need
                )
                t_e = t_s + t_need
                imp_s = np.append(imp_s, [2, 1, 2])
                imp_x0 = np.append(
                    imp_x0,
                    [mis_table[i, 0], mis_table[i, 0], mis_table[i, 3]]
                )
                imp_x1 = np.append(
                    imp_x1,
                    [mis_table[i, 0], mis_table[i, 3], mis_table[i, 3]]
                )
                imp_y0 = np.append(
                    imp_y0,
                    [mis_table[i, 1], mis_table[i, 1], mis_table[i, 4]]
                )
                imp_y1 = np.append(
                    imp_y1,
                    [mis_table[i, 1], mis_table[i, 4], mis_table[i, 4]]
                )
                imp_t0 = np.append(imp_t0, [mis_table[i, 2], t_s, t_e])
                imp_t1 = np.append(imp_t1, [t_s, t_e, mis_table[i, 5]])
        # add one more check about how many flights observed
        # in the nearby 24 hours
        elif (
            nearby_flight <= 5
            and t_diff > 6 * 60 * 60
            and min(distance1, distance2) > 50
        ):
            if d_diff < 3000:
                v_random = np.random.uniform(low=1, high=1.8)
                t_need = min(d_diff / v_random, t_diff)
            else:
                v_random = np.random.uniform(low=13, high=32)
                t_need = min(d_diff / v_random, t_diff)
            if t_need == t_diff:
                imp_s = np.append(imp_s, 1)
                imp_x0 = np.append(imp_x0, mis_table[i, 0])
                imp_x1 = np.append(imp_x1, mis_table[i, 3])
                imp_y0 = np.append(imp_y0, mis_table[i, 1])
                imp_y1 = np.append(imp_y1, mis_table[i, 4])
                imp_t0 = np.append(imp_t0, mis_table[i, 2])
                imp_t1 = np.append(imp_t1, mis_table[i, 5])
            else:
                t_s = np.random.uniform(
                    low=mis_table[i, 2], high=mis_table[i, 5] - t_need
                )
                t_e = t_s + t_need
                imp_s = np.append(imp_s, [2, 1, 2])
                imp_x0 = np.append(
                    imp_x0, [mis_table[i, 0], mis_table[i, 0], mis_table[i, 3]]
                )
                imp_x1 = np.append(
                    imp_x1, [mis_table[i, 0], mis_table[i, 3], mis_table[i, 3]]
                )
                imp_y0 = np.append(
                    imp_y0, [mis_table[i, 1], mis_table[i, 1], mis_table[i, 4]]
                )
                imp_y1 = np.append(
                    imp_y1, [mis_table[i, 1], mis_table[i, 4], mis_table[i, 4]]
                )
                imp_t0 = np.append(imp_t0, [mis_table[i, 2], t_s, t_e])
                imp_t1 = np.append(imp_t1, [t_s, t_e, mis_table[i, 5]])
        else:
            # solve the problem that a person has a
            # trajectory like flight/pause/flight/pause/flight...
            # we want it more like flght/flight/flight
            # /pause/pause/pause/flight/flight...
            # start from two ends, we make it harder to change the current
            # pause/flight status by drawing multiple random
            # variables form bin(p0) and require them to be all 0/1
            # "switch" is the number of random variables
            start_t = mis_table[i, 2]
            end_t = mis_table[i, 5]
            start_x = mis_table[i, 0]
            end_x = mis_table[i, 3]
            start_y = mis_table[i, 1]
            end_y = mis_table[i, 4]
            start_s = mis_table[i, 6]
            end_s = mis_table[i, 7]
            if t_diff > 4 * 60 * 60 and min(distance1, distance2) <= 50:
                t_need = min(d_diff / 0.6, t_diff)
                if distance1 <= 50:
                    imp_s = np.append(imp_s, 2)
                    imp_t0 = np.append(imp_t0, start_t)
                    imp_t1 = np.append(imp_t1, end_t - t_need)
                    imp_x0 = np.append(imp_x0, start_x)
                    imp_x1 = np.append(imp_x1, start_x)
                    imp_y0 = np.append(imp_y0, start_y)
                    imp_y1 = np.append(imp_y1, start_y)
                    start_t = end_t - t_need
                else:
                    imp_s = np.append(imp_s, 2)
                    imp_t0 = np.append(imp_t0, start_t + t_need)
                    imp_t1 = np.append(imp_t1, end_t)
                    imp_x0 = np.append(imp_x0, end_x)
                    imp_x1 = np.append(imp_x1, end_x)
                    imp_y0 = np.append(imp_y0, end_y)
                    imp_y1 = np.append(imp_y1, end_y)
                    end_t = start_t + t_need
            counter = 0
            while start_t < end_t:
                if (
                    abs(start_x - end_x) + abs(start_y - end_y) > 0
                    and end_t - start_t < 30
                ):  # avoid extreme high speed
                    imp_s = np.append(imp_s, 1)
                    imp_t0 = np.append(imp_t0, start_t)
                    imp_t1 = np.append(imp_t1, end_t)
                    imp_x0 = np.append(imp_x0, start_x)
                    imp_x1 = np.append(imp_x1, end_x)
                    imp_y0 = np.append(imp_y0, start_y)
                    imp_y1 = np.append(imp_y1, end_y)
                    start_t = end_t
                    # should check the missing legnth first, if it's less than
                    # 12 hours, do the following, otherewise,
                    # insert home location at night most visited
                    # places in the interval as known
                elif start_x == end_x and start_y == end_y:
                    imp_s = np.append(imp_s, 2)
                    imp_t0 = np.append(imp_t0, start_t)
                    imp_t1 = np.append(imp_t1, end_t)
                    imp_x0 = np.append(imp_x0, start_x)
                    imp_x1 = np.append(imp_x1, end_x)
                    imp_y0 = np.append(imp_y0, start_y)
                    imp_y1 = np.append(imp_y1, end_y)
                    start_t = end_t
                else:
                    if counter % 2 == 0:
                        direction = "forward"
                    else:
                        direction = "backward"

                    if direction == "forward":
                        direction = ""
                        I0 = indicate_flight(
                            method,
                            start_t,
                            start_x,
                            start_y,
                            end_t,
                            end_x,
                            end_y,
                            bv_set,
                            switch,
                            num,
                            pars,
                        )
                        if (sum(I0 == 1) == switch and start_s == 2) or (
                            sum(I0 == 0) < switch and start_s == 1
                        ):
                            weight = calculate_k1(
                                method, start_t, start_x, start_y,
                                flight_table, pars
                            )
                            normalize_w = (weight + 1e-5) / sum(weight + 1e-5)
                            flight_index = np.random.choice(
                                flight_table.shape[0], p=normalize_w
                            )
                            delta_x = (
                                flight_table[flight_index, 4]
                                - flight_table[flight_index, 1]
                            )
                            delta_y = (
                                flight_table[flight_index, 5]
                                - flight_table[flight_index, 2]
                            )
                            delta_t = (
                                flight_table[flight_index, 6]
                                - flight_table[flight_index, 3]
                            )
                            if start_t + delta_t > end_t:
                                temp = delta_t
                                delta_t = end_t - start_t
                                delta_x = delta_x * delta_t / temp
                                delta_y = delta_y * delta_t / temp
                            delta_x, delta_y = adjust_direction(
                                linearity,
                                delta_x,
                                delta_y,
                                start_x,
                                start_y,
                                end_x,
                                end_y,
                                mis_table[i, 0],
                                mis_table[i, 1],
                                mis_table[i, 3],
                                mis_table[i, 4],
                            )
                            try_t = start_t + delta_t
                            try_x = (end_t - try_t) / (
                                end_t - start_t + 1e-5
                            ) * (
                                start_x + delta_x
                            ) + (try_t - start_t + 1e-5) / (
                                end_t - start_t
                            ) * end_x
                            try_y = (end_t - try_t) / (
                                end_t - start_t + 1e-5
                            ) * (
                                start_y + delta_y
                            ) + (try_t - start_t + 1e-5) / (
                                end_t - start_t
                            ) * end_y
                            mov1 = great_circle_dist(
                                try_x, try_y, start_x, start_y
                            )
                            mov2 = great_circle_dist(
                                end_x, end_y, start_x, start_y
                            )
                            check1 = checkbound(
                                try_x,
                                try_y,
                                mis_table[i, 0],
                                mis_table[i, 1],
                                mis_table[i, 3],
                                mis_table[i, 4],
                            )
                            check2 = (mov1 < mov2) * 1
                            if end_t > start_t and check1 == 1 and check2 == 1:
                                imp_s = np.append(imp_s, 1)
                                imp_t0 = np.append(imp_t0, start_t)
                                current_t = start_t + delta_t
                                imp_t1 = np.append(imp_t1, current_t)
                                imp_x0 = np.append(imp_x0, start_x)
                                current_x = (end_t - current_t) / (
                                    end_t - start_t
                                ) * (
                                    start_x + delta_x
                                ) + (current_t - start_t) / (
                                    end_t - start_t
                                ) * end_x
                                imp_x1 = np.append(imp_x1, current_x)
                                imp_y0 = np.append(imp_y0, start_y)
                                current_y = (end_t - current_t) / (
                                    end_t - start_t
                                ) * (
                                    start_y + delta_y
                                ) + (current_t - start_t) / (
                                    end_t - start_t
                                ) * end_y
                                imp_y1 = np.append(imp_y1, current_y)
                                start_x = current_x
                                start_y = current_y
                                start_t = current_t
                                start_s = 1
                                counter = counter + 1
                            if end_t > start_t and check2 == 0:
                                speed = mov1 / delta_t
                                t_need = mov2 / speed
                                imp_s = np.append(imp_s, 1)
                                imp_t0 = np.append(imp_t0, start_t)
                                current_t = start_t + t_need
                                imp_t1 = np.append(imp_t1, current_t)
                                imp_x0 = np.append(imp_x0, start_x)
                                imp_x1 = np.append(imp_x1, end_x)
                                imp_y0 = np.append(imp_y0, start_y)
                                imp_y1 = np.append(imp_y1, end_y)
                                start_x = end_x
                                start_y = end_y
                                start_t = current_t
                                start_s = 1
                                counter = counter + 1
                            else:
                                weight = calculate_k1(
                                    method, start_t, start_x, start_y,
                                    pause_table, pars
                                )
                                normalize_w = (weight + 1e-5) / sum(
                                    weight + 1e-5
                                )
                                pause_index = np.random.choice(
                                    pause_table.shape[0], p=normalize_w
                                )
                                delta_t = (
                                    pause_table[pause_index, 6]
                                    - pause_table[pause_index, 3]
                                ) * multiplier(end_t - start_t)
                                if start_t + delta_t < end_t:
                                    imp_s = np.append(imp_s, 2)
                                    imp_t0 = np.append(imp_t0, start_t)
                                    current_t = start_t + delta_t
                                    imp_t1 = np.append(imp_t1, current_t)
                                    imp_x0 = np.append(imp_x0, start_x)
                                    imp_x1 = np.append(imp_x1, start_x)
                                    imp_y0 = np.append(imp_y0, start_y)
                                    imp_y1 = np.append(imp_y1, start_y)
                                    start_t = current_t
                                    start_s = 2
                                    counter = counter + 1
                                else:
                                    imp_s = np.append(imp_s, 1)
                                    imp_t0 = np.append(imp_t0, start_t)
                                    imp_t1 = np.append(imp_t1, end_t)
                                    imp_x0 = np.append(imp_x0, start_x)
                                    imp_x1 = np.append(imp_x1, end_x)
                                    imp_y0 = np.append(imp_y0, start_y)
                                    imp_y1 = np.append(imp_y1, end_y)
                                    start_t = end_t

                    if direction == "backward":
                        direction = ""
                        I1 = indicate_flight(
                            method,
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
                        )
                        if (sum(I1 == 1) == switch and end_s == 2) or (
                            sum(I1 == 0) < switch and end_s == 1
                        ):
                            weight = calculate_k1(
                                method, end_t, end_x, end_y,
                                flight_table, pars
                            )
                            normalize_w = (weight + 1e-5) / sum(
                                weight + 1e-5
                            )
                            flight_index = np.random.choice(
                                flight_table.shape[0], p=normalize_w
                            )
                            delta_x = -(
                                flight_table[flight_index, 4]
                                - flight_table[flight_index, 1]
                            )
                            delta_y = -(
                                flight_table[flight_index, 5]
                                - flight_table[flight_index, 2]
                            )
                            delta_t = (
                                flight_table[flight_index, 6]
                                - flight_table[flight_index, 3]
                            )
                            if start_t + delta_t > end_t:
                                temp = delta_t
                                delta_t = end_t - start_t
                                delta_x = delta_x * delta_t / temp
                                delta_y = delta_y * delta_t / temp
                            delta_x, delta_y = adjust_direction(
                                linearity,
                                delta_x,
                                delta_y,
                                end_x,
                                end_y,
                                start_x,
                                start_y,
                                mis_table[i, 3],
                                mis_table[i, 4],
                                mis_table[i, 0],
                                mis_table[i, 1],
                            )
                            try_t = end_t - delta_t
                            try_x = (end_t - try_t) / (
                                end_t - start_t + 1e-5
                            ) * start_x + (try_t - start_t) / (
                                end_t - start_t + 1e-5
                            ) * (
                                end_x + delta_x
                            )
                            try_y = (end_t - try_t) / (
                                end_t - start_t + 1e-5
                            ) * start_y + (try_t - start_t) / (
                                end_t - start_t + 1e-5
                            ) * (
                                end_y + delta_y
                            )
                            mov1 = great_circle_dist(
                                try_x, try_y, end_x, end_y
                            )
                            mov2 = great_circle_dist(
                                end_x, end_y, start_x, start_y
                            )
                            check1 = checkbound(
                                try_x,
                                try_y,
                                mis_table[i, 0],
                                mis_table[i, 1],
                                mis_table[i, 3],
                                mis_table[i, 4],
                            )
                            check2 = (mov1 < mov2) * 1
                            if end_t > start_t and check1 == 1 and check2 == 1:
                                imp_s = np.append(imp_s, 1)
                                imp_t1 = np.append(imp_t1, end_t)
                                current_t = end_t - delta_t
                                imp_t0 = np.append(imp_t0, current_t)
                                imp_x1 = np.append(imp_x1, end_x)
                                current_x = (end_t - current_t) / (
                                    end_t - start_t
                                ) * start_x + (current_t - start_t) / (
                                    end_t - start_t
                                ) * (
                                    end_x + delta_x
                                )
                                imp_x0 = np.append(imp_x0, current_x)
                                imp_y1 = np.append(imp_y1, end_y)
                                current_y = (end_t - current_t) / (
                                    end_t - start_t
                                ) * start_y + (current_t - start_t) / (
                                    end_t - start_t
                                ) * (
                                    end_y + delta_y
                                )
                                imp_y0 = np.append(imp_y0, current_y)
                                end_x = current_x
                                end_y = current_y
                                end_t = current_t
                                end_s = 1
                                counter = counter + 1
                            if end_t > start_t and check2 == 0:
                                speed = mov1 / delta_t
                                t_need = mov2 / speed
                                imp_s = np.append(imp_s, 1)
                                imp_t1 = np.append(imp_t1, end_t)
                                current_t = end_t - t_need
                                imp_t0 = np.append(imp_t0, current_t)
                                imp_x1 = np.append(imp_x1, end_x)
                                imp_x0 = np.append(imp_x0, start_x)
                                imp_y1 = np.append(imp_y1, end_y)
                                imp_y0 = np.append(imp_y0, start_y)
                                end_x = start_x
                                end_y = start_y
                                end_t = current_t
                                end_s = 1
                                counter = counter + 1
                            else:
                                weight = calculate_k1(
                                    method, end_t, end_x, end_y,
                                    pause_table, pars
                                )
                                normalize_w = (weight + 1e-5) / sum(
                                    weight + 1e-5
                                )
                                pause_index = np.random.choice(
                                    pause_table.shape[0], p=normalize_w
                                )
                                delta_t = (
                                    pause_table[pause_index, 6]
                                    - pause_table[pause_index, 3]
                                ) * multiplier(end_t - start_t)
                                if start_t + delta_t < end_t:
                                    imp_s = np.append(imp_s, 2)
                                    imp_t1 = np.append(imp_t1, end_t)
                                    current_t = end_t - delta_t
                                    imp_t0 = np.append(imp_t0, current_t)
                                    imp_x0 = np.append(imp_x0, end_x)
                                    imp_x1 = np.append(imp_x1, end_x)
                                    imp_y0 = np.append(imp_y0, end_y)
                                    imp_y1 = np.append(imp_y1, end_y)
                                    end_t = current_t
                                    end_s = 2
                                    counter = counter + 1
                                else:
                                    imp_s = np.append(imp_s, 1)
                                    imp_t1 = np.append(imp_t1, end_t)
                                    imp_t0 = np.append(imp_t0, start_t)
                                    imp_x0 = np.append(imp_x0, start_x)
                                    imp_x1 = np.append(imp_x1, end_x)
                                    imp_y0 = np.append(imp_y0, start_y)
                                    imp_y1 = np.append(imp_y1, end_y)
                                    end_t = start_t
    imp_table = np.stack(
        [imp_s, imp_x0, imp_y0, imp_t0, imp_x1, imp_y1, imp_t1], axis=1
    )
    imp_table = imp_table[imp_table[:, 3].argsort()].astype(float)
    return imp_table


def imp_to_traj(imp_table, mob_mat, r, w, h):
    """
    This function tidies up the first-step imputed trajectory,
    such as combining pauses, flights shared by
    both observed and missing intervals, also combine consecutive flight
    with slightly different directions
    as one longer flight
    Args: imp_table, 2d array, output from impute_gps()
          mob_mat, 2d array, output from InferMobMat()
              r: the maximum radius of a pause
              w: a threshold for distance, if the distance to the
                 great circle is greater than
                 this threshold, we consider there is a knot
              h: a threshold of distance, if the movemoent between
                 two timestamps is less than h,
                 consider it as a pause and a knot
    Return: 2d array, the final imputed trajectory,
    with one more columm compared to imp_table
            which is an indicator showing if the piece
            of traj is imputed (0) or observed (1)
    """
    sys.stdout.write("Tidying up the trajectories..." + "\n")
    mis_table = np.zeros((1, 8))
    for i in range(np.shape(mob_mat)[0] - 1):
        if mob_mat[i + 1, 3] != mob_mat[i, 6]:
            # also record if it's flight/pause
            # before and after the missing interval
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
            mis_table = np.vstack((mis_table, mov))
    mis_table = np.delete(mis_table, 0, 0)

    traj = []
    for k in range(mis_table.shape[0]):
        index = (imp_table[:, 3] >= mis_table[k, 2]) * (
            imp_table[:, 6] <= mis_table[k, 5]
        )
        temp = imp_table[index, :]
        a = 0
        b = 1
        while a < temp.shape[0]:
            if b < temp.shape[0]:
                if temp[b, 0] == temp[a, 0]:
                    b = b + 1
            if (
                b == temp.shape[0]
                or temp[min(b, temp.shape[0] - 1), 0] != temp[a, 0]
            ):
                start = a
                end = b - 1
                a = b
                b = b + 1
                if temp[start, 0] == 2:
                    traj.append(
                        [
                            2,
                            temp[start, 1],
                            temp[start, 2],
                            temp[start, 3],
                            temp[end, 4],
                            temp[end, 5],
                            temp[end, 6],
                        ]
                    )
                elif end == start:
                    traj.append(
                        [
                            1,
                            temp[start, 1],
                            temp[start, 2],
                            temp[start, 3],
                            temp[end, 4],
                            temp[end, 5],
                            temp[end, 6],
                        ]
                    )
                else:
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
                        traj.append(
                            [
                                1,
                                mat[knots[j], 0],
                                mat[knots[j], 1],
                                mat[knots[j], 2],
                                mat[knots[j + 1], 0],
                                mat[knots[j + 1], 1],
                                mat[knots[j + 1], 2],
                            ]
                        )
    traj = np.array(traj)
    if traj.shape[0] != 0:
        traj = np.hstack((traj, np.zeros((traj.shape[0], 1))))
        full_traj = np.vstack((traj, mob_mat))
    else:
        full_traj = mob_mat
    float_traj = full_traj[full_traj[:, 3].argsort()].astype(float)
    final_traj = float_traj[float_traj[:, 6] - float_traj[:, 3] > 0, :]
    return final_traj
