import os
import random
import string

import numpy as np
import pandas as pd

from ..poplar.legacy.common_funcs import datetime2stamp, stamp2datetime


ORIG_TIME = datetime2stamp([2020, 8, 24, 0, 0, 0], "America/New_York")


def gen_status():
    rv = np.random.random()
    if rv <= 0.8:
        return 0
    else:
        return 1


def exist_text_call(hour, status):
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
    else:
        return 0


def gen_random_id(k):
    letters = string.ascii_lowercase
    hashed_ids = []
    for i in range(k):
        hashed_ids.append("".join(random.choice(letters) for i in range(10)))
    return hashed_ids


def number_of_distinct_inds(stream):
    rv = np.random.random()
    if stream == "texts":
        if rv <= 0.4:
            num = 1
        elif rv <= 0.7:
            num = 2
        elif rv <= 0.9:
            num = 3
        else:
            num = 4
    else:
        if rv <= 0.8:
            num = 1
        elif rv <= 0.95:
            num = 2
        else:
            num = 3
    return num


def gen_round(stream):
    rv = np.random.random()
    if stream == "calls":
        if rv <= 0.85:
            r = 1
        elif rv <= 0.95:
            r = 2
        else:
            r = 3
    else:
        if rv <= 0.25:
            r = 1
        elif rv <= 0.75:
            r = 2
        elif rv <= 0.85:
            r = 3
        elif rv <= 0.95:
            r = 4
        else:
            r = 5
    return r


def gen_dir(r):
    direction = []
    rv = np.random.random()
    if rv <= 0.6:
        direction.append(1)
    else:
        direction.append(0)
    if r > 1:
        for i in range(r - 1):
            rv = np.random.random()
            if rv <= 0.9:
                direction.append(1 - direction[-1])
            else:
                direction.append(direction[-1])
    return direction


def gen_text_len():
    rv = np.random.random()
    if rv <= 0.7:
        length = np.random.randint(10) + 1
    elif rv <= 0.9:
        length = 10 + np.random.randint(20) + 1
    else:
        length = 30 + np.random.randint(20) + 1
    return length


def gen_call_dur():
    rv = np.random.random()
    if rv <= 0.2:
        dur = 0
    elif rv <= 0.8:
        dur = np.random.randint(300) + 3
    else:
        dur = np.random.randint(300) + 300
    return dur


def gen_timestamp_call(dur):
    stamps = []
    if sum(dur) > 60 * 60:
        dur = dur / 2
    else:
        current_t = 0
        remain = 60 * 60 - sum(dur)
        for i in range(len(dur)):
            t = np.random.randint(int(remain / (len(dur) - i + 1) * 2))
            stamps.append(current_t + t)
            current_t = current_t + t + dur[i]
            remain = remain - t
    return dur, stamps


def gen_timestamp_text(r):
    stamps = []
    for i in range(r):
        stamps.append(np.random.randint(3600))
    stamps.sort()
    return stamps


def gen_text_files(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for ID in ["user_1", "user_2"]:
        os.makedirs(f"{output_folder}/{ID}/texts", exist_ok=True)
        phone_nums = gen_random_id(20)
        for i in range(14):
            status = gen_status()
            for j in range(24):
                if exist_text_call(j, status) == 1:
                    start_t = ORIG_TIME + i * 3600 * 24 + j * 3600
                    [y, m, d, h, mins, sec] = stamp2datetime(start_t, "UTC")
                    filename = f"{y}-{m:02d}-{d:02d} {h:02d}_00_00.csv"
                    num = number_of_distinct_inds("texts")
                    contacts = np.random.choice(phone_nums, num, replace=False)
                    data = []
                    for g in range(num):
                        r = gen_round("texts")
                        directions = gen_dir(r)
                        stamps = gen_timestamp_text(r)
                        for k in range(r):
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
                        f"{output_folder}/{ID}/texts/{filename}",
                        index=False
                    )


def gen_call_files(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for ID in ["user_1", "user_2"]:
        os.makedirs(f"{output_folder}/{ID}/calls", exist_ok=True)
        phone_nums = gen_random_id(20)
        for i in range(14):
            status = gen_status()
            for j in range(24):
                if exist_text_call(j, status) == 1:
                    start_t = ORIG_TIME + i * 3600 * 24 + j * 3600
                    [y, m, d, h, mins, sec] = stamp2datetime(start_t, "UTC")
                    filename = f"{y}-{m:02d}-{d:02d} {h:02d}_00_00.csv"
                    num = number_of_distinct_inds("texts")
                    contacts = np.random.choice(phone_nums, num, replace=False)
                    data = []
                    all_dur = []
                    all_dir = []
                    all_phone = []
                    for g in range(num):
                        r = gen_round("calls")
                        directions = gen_dir(r)
                        for k in range(r):
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
                        f"{output_folder}/{ID}/calls/{filename}",
                        index=False
                    )


def sim_log_data(output_folder):
    gen_text_files(output_folder)
    gen_call_files(output_folder)
