from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
import pytest

from forest.oak.main import rle


TEST_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def fs():
    return 10


@pytest.fixture(scope="session")
def signal_bout():
    data = pd.read_csv(os.path.join(TEST_DATA_DIR, "test_data_bout.csv"))
    timestamp = np.array(data["timestamp"], dtype="float64")
    t = data["UTC time"].tolist()
    x = np.array(data["x"], dtype="float64")
    y = np.array(data["y"], dtype="float64")
    z = np.array(data["z"], dtype="float64")

    timestamp = timestamp/1000
    t = [t_ind.replace("T", " ") for t_ind in t]
    t = [datetime.strptime(t_ind, '%Y-%m-%d %H:%M:%S.%f')
         for t_ind in t]
    return timestamp, t, x, y, z


def test_(signal_bout, fs):
    timestamp, t, x, y, z = signal_bout

    t_shifted = [t_i-timedelta(microseconds=t_i.microsecond)
                 for t_i in t]

    hour_start = t_shifted[0]
    hour_start = (hour_start -
                  timedelta(minutes=hour_start.minute) -
                  timedelta(seconds=hour_start.second))
    hour_end = hour_start + timedelta(hours=1)
    t_sec_bins = pd.date_range(hour_start,
                               hour_end, freq='S').tolist()
    samples_per_sec, t_sec_bins = np.histogram(t_shifted,
                                               t_sec_bins)
    samples_enough = samples_per_sec >= (fs - 1)
    expected_output = np.array([1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119,
                                1120], dtype="int64")
    assert np.array_equal(np.where(samples_enough)[0], expected_output)

    run_length, start_ind, val = rle(samples_enough)
    bout_start = start_ind[val & (run_length >= 5)]
    expected_output = np.array([1112], dtype="int64")
    assert np.array_equal(bout_start, expected_output)

    bout_duration = run_length[val & (run_length >= 5)]
    expected_output = np.array([9], dtype="int64")
    assert np.array_equal(bout_duration, expected_output)

    bout_time = pd.date_range(
        t_sec_bins[bout_start[0]],
        t_sec_bins[bout_start[0] +
                   bout_duration[0]],
        freq='S').tolist()
    bout_time = bout_time[:-1]
    bout_time = [t_i.to_pydatetime()
                 for t_i in bout_time]

    acc_ind = np.isin(t_shifted, bout_time)

    expected_output = pd.date_range(datetime(2020, 2, 25, 18, 18, 32),
                                    datetime(2020, 2, 25, 18, 18, 41),
                                    freq='S').to_pydatetime()
    assert np.array_equal(bout_time, expected_output)
    assert len(np.where(acc_ind)[0]) == 90
