"""
Module to simulate realistic call/text data.
"""

import os
import random
import string

import numpy as np
import pandas as pd

from ..poplar.legacy.common_funcs import datetime2stamp, stamp2datetime


ORIG_TIME = datetime2stamp([2020, 8, 24, 0, 0, 0], "America/New_York")


def gen_status() -> int:
    """
    Generates a random status based on a probability distribution.

    This function generates a status for a user activity. With probability 80%, the function
    returns 0 (indicating inactivity). Otherwise, it returns 1 (indicating activity).

    Returns:
        int: 0 if generated random value is less than or equal to 0.8 (indicating inactivity), 
        1 otherwise (indicating activity).
    """
    if np.random.random() <= 0.8:
        return 0
    return 1


def exist_text_call(hour: int, status: str) -> int:
    """
    Determines whether a text or call exists.

    Given an hour and a status (active or inactive), this function determines 
    the probability of a text or call existing. Different hours have different 
    base probabilities, and if the status is active, the base probability is tripled.

    Args:
        hour (int): The hour at which the function checks for a text or call.
        status (str): The activity status, either "active" or "inactive".

    Returns:
        int: 1 if a random number is less than or equal to the determined probability
        (indicating a text or call exists), 0 otherwise.
    """
    if hour in [0, 1, 2, 3, 4, 5, 6]:
        prob = 0
    elif hour in [7, 8, 22, 23]:
        prob = 0.01
    else:
        prob = 0.05
    if status == "active":
        prob = 3 * prob
    if np.random.random() <= prob:
        return 1
    return 0


def gen_random_id(k: int) -> list:
    """Generates k random IDs, each of length 10.

    Args:
        k (int): The number of IDs to generate.

    Returns:
        list: A list of k unique random IDs.
    """
    letters = string.ascii_lowercase
    hashed_ids = []
    for i in range(k):
        hashed_ids.append("".join(random.choice(letters) for i in range(10)))
    return hashed_ids


def number_of_distinct_inds(stream: str) -> int:
    """
    Determines the number of distinct individuals in a stream.

    This function generates a random number and based on that, it determines 
    the number of distinct individuals in the given stream (either "texts" or "calls").

    Args:
        stream (str): The type of stream, either "texts" or "calls".

    Returns:
        int: The number of distinct individuals in the stream.
    """
    random_var = np.random.random()
    if stream == "texts":
        if random_var <= 0.4:
            num = 1
        elif random_var <= 0.7:
            num = 2
        elif random_var <= 0.9:
            num = 3
        else:
            num = 4
    else:
        if random_var <= 0.8:
            num = 1
        elif random_var <= 0.95:
            num = 2
        else:
            num = 3
    return num


def gen_round(stream: str) -> int:
    """
    Generates a round number for a given stream.

    This function generates a random number and based on that, it determines 
    the round number for the given stream (either "texts" or "calls").

    Args:
        stream (str): The type of stream, either "texts" or "calls".

    Returns:
        int: The round number for the stream.
    """
    random_var = np.random.random()
    if stream == "calls":
        if random_var <= 0.85:
            round_num = 1
        elif random_var <= 0.95:
            round_num = 2
        else:
            round_num = 3
    else:
        if random_var <= 0.25:
            round_num = 1
        elif random_var <= 0.75:
            round_num = 2
        elif random_var <= 0.85:
            round_num = 3
        elif random_var <= 0.95:
            round_num = 4
        else:
            round_num = 5
    return round_num


def gen_dir(round_num: int) -> list:
    """
    Generates a list of direction values.

    This function generates a list of round_num direction values (either 1 or 0)
    based on a certain probability distribution.

    Args:
        round_num (int): The number of direction values to generate.

    Returns:
        list: A list of round_num direction values.
    """
    direction = []
    random_var = np.random.random()
    if random_var <= 0.6:
        direction.append(1)
    else:
        direction.append(0)
    if round_num > 1:
        for _ in range(round_num - 1):
            random_var = np.random.random()
            if random_var <= 0.9:
                direction.append(1 - direction[-1])
            else:
                direction.append(direction[-1])
    return direction


def gen_text_len() -> int:
    """
    Generates a random length for a text.

    Returns:
        int: The length of a text.
    """
    random_var = np.random.random()
    if random_var <= 0.7:
        length = np.random.randint(10) + 1
    elif random_var <= 0.9:
        length = 10 + np.random.randint(20) + 1
    else:
        length = 30 + np.random.randint(20) + 1
    return length


def gen_call_dur() -> int:
    """
    Generates a random duration for a call.

    Returns:
        int: The duration of a call.
    """
    random_var = np.random.random()
    if random_var <= 0.2:
        dur = 0
    elif random_var <= 0.8:
        dur = np.random.randint(300) + 3
    else:
        dur = np.random.randint(300) + 300
    return dur


def gen_timestamp_call(dur: list) -> tuple:
    """
    Generates timestamps for a call.

    Given a duration, this function generates a list of timestamps for a call.

    Args:
        dur (int): The duration of the call.

    Returns:
        tuple: A tuple containing the modified duration and a list of timestamps for the call.
    """
    stamps = []
    if sum(dur) > 60 * 60:
        dur = dur / 2
    else:
        current_t = 0
        remain = 60 * 60 - sum(dur)
        for i, dur_i in enumerate(dur):
            t_now = np.random.randint(int(remain / (len(dur) - i + 1) * 2))
            stamps.append(current_t + t_now)
            current_t = current_t + t_now + dur_i
            remain = remain - t_now
    return dur, stamps


def gen_timestamp_text(round_num: int) -> list:
    """
    Generates timestamps for a text.

    Given a round number, this function generates a list of timestamps for a text.

    Args:
        round_num (int): The round number.

    Returns:
        list: A list of timestamps for the text.
    """
    stamps = [np.random.randint(3600) for _ in range(round_num)]
    stamps.sort()
    return stamps


def gen_text_files(output_folder: str):
    """
    Generates text files.

    Given an output folder, this function generates text files containing
    simulated data for two users over 14 days.

    Args:
        output_folder (str): The directory in which to create the text files.

    Raises:
        OSError: If the directory cannot be created.
    """
    os.makedirs(output_folder, exist_ok=True)
    for idx in ["user_1", "user_2"]:
        os.makedirs(f"{output_folder}/{idx}/texts", exist_ok=True)
        phone_nums = gen_random_id(20)
        for i in range(14):
            status = gen_status()
            for j in range(24):
                if exist_text_call(j, status) == 1:
                    start_t = ORIG_TIME + i * 3600 * 24 + j * 3600
                    [y, m, d, h, _, _] = stamp2datetime(start_t, "UTC")
                    filename = f"{y}-{m:02d}-{d:02d} {h:02d}_00_00.csv"
                    num = number_of_distinct_inds("texts")
                    contacts = np.random.choice(phone_nums, num, replace=False)
                    data = []
                    for g in range(num):
                        round_num = gen_round("texts")
                        directions = gen_dir(round_num)
                        stamps = gen_timestamp_text(round_num)
                        for k in range(round_num):
                            if directions[k] == 1:
                                sms = "sent SMS"
                            else:
                                sms = "received SMS"
                            new_line = [
                                (start_t + stamps[k]) * 1000,
                                "-",
                                contacts[g],
                                sms,
                                gen_text_len(),
                                (
                                  start_t + stamps[k] - np.random.randint(10)
                                ) * 1000,
                            ]
                            data.append(new_line)
                    data = pd.DataFrame(
                        data,
                        columns=[
                            "timestamp",
                            "UTC time",
                            "hashed phone number",
                            "sent vs received",
                            "message length",
                            "time sent",
                        ],
                    )
                    data.to_csv(
                        f"{output_folder}/{idx}/texts/{filename}",
                        index=False
                    )


def gen_call_files(output_folder: str):
    """
    Generates call files.

    Given an output folder, this function generates call files containing
    simulated data for two users over 14 days.

    Args:
        output_folder (str): The directory in which to create the call files.

    Raises:
        OSError: If the directory cannot be created.
    """
    os.makedirs(output_folder, exist_ok=True)
    for idx in ["user_1", "user_2"]:
        os.makedirs(f"{output_folder}/{idx}/calls", exist_ok=True)
        phone_nums = gen_random_id(20)
        for i in range(14):
            status = gen_status()
            for j in range(24):
                if exist_text_call(j, status) == 1:
                    start_t = ORIG_TIME + i * 3600 * 24 + j * 3600
                    [y, m, d, h, _, _] = stamp2datetime(start_t, "UTC")
                    filename = f"{y}-{m:02d}-{d:02d} {h:02d}_00_00.csv"
                    num = number_of_distinct_inds("texts")
                    contacts = np.random.choice(phone_nums, num, replace=False)
                    data = []
                    all_dur = []
                    all_dir = []
                    all_phone = []
                    for g in range(num):
                        round_num = gen_round("calls")
                        directions = gen_dir(round_num)
                        for k in range(round_num):
                            all_dur.append(gen_call_dur())
                            all_dir.append(directions[k])
                            all_phone.append(contacts[g])

                    all_dur = np.array(all_dur)
                    all_dur, all_stamps = gen_timestamp_call(all_dur)
                    for z in range(len(all_dur)):
                        if all_dir[z] == 1:
                            call_type = "Outgoing Call"
                        else:
                            call_type = "Incoming Call"
                        if all_dur[z] == 0:
                            call_type = "Missed Call"
                        new_line = [
                            start_t * 1000 + all_stamps[z] * 1000,
                            "-",
                            all_phone[z],
                            call_type,
                            all_dur[z],
                        ]
                        data.append(new_line)
                    data = pd.DataFrame(
                        data,
                        columns=[
                            "timestamp",
                            "UTC time",
                            "hashed phone number",
                            "call type",
                            "duration in seconds",
                        ],
                    )
                    data.to_csv(
                        f"{output_folder}/{idx}/calls/{filename}",
                        index=False
                    )


def sim_log_data(output_folder: str):
    """
    Simulates log data.

    Given an output folder, this function generates text and call files
    containing simulated data for two users over 14 days.

    Args:
        output_folder (str): The directory in which to create the log files.
    """
    gen_text_files(output_folder)
    gen_call_files(output_folder)
