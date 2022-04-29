import os

import numpy as np
import pandas as pd
import pytest
from tempfile import TemporaryDirectory

from forest.poplar.legacy.common_funcs import (
    read_data, datetime2stamp, stamp2datetime, write_all_summaries,
    get_files_timestamps, filename2stamp
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(CURRENT_DIR, "test_data")


def test_datetime2stamp():
    time_list = [2020, 11, 1, 12, 9, 50]
    assert datetime2stamp(time_list=time_list,
                          tz_str="America/New_York") == 1604250590
    assert datetime2stamp(time_list=time_list,
                          tz_str="America/Chicago") == 1604254190


def test_datetime2stamp_bad_values():
    second_error_caught = False
    minute_error_caught = False
    hour_error_caught = False
    day_error_caught = False
    month_error_caught = False
    try:
        datetime2stamp(time_list=[2020, 11, 1, 12, 9, 150],
                       tz_str="America/New_York")
    except ValueError:
        second_error_caught = True
    assert second_error_caught
    try:
        datetime2stamp(time_list=[2020, 11, 1, 12, 209, 50],
                       tz_str="America/New_York")
    except ValueError:
        minute_error_caught = True
    assert minute_error_caught
    try:
        datetime2stamp(time_list=[2020, 11, 1, 35, 20, 50],
                       tz_str="America/New_York")
    except ValueError:
        hour_error_caught = True
    assert hour_error_caught
    try:
        datetime2stamp(time_list=[2020, 11, 35, 5, 20, 50],
                       tz_str="America/New_York")
    except ValueError:
        day_error_caught = True
    assert day_error_caught
    try:
        datetime2stamp(time_list=[2020, 15, 20, 5, 20, 50],
                       tz_str="America/New_York")
    except ValueError:
        month_error_caught = True
    assert month_error_caught


def test_stamp2datetime():
    t_stamp = 1604254190
    assert stamp2datetime(t_stamp, "Africa/Cairo") == [2020, 11, 1, 20, 9, 50]
    assert stamp2datetime(t_stamp, "America/Thule") == [2020, 11, 1, 14, 9, 50]


def test_stamp2datetime_float():
    t_stamp = 1604254190.5
    assert stamp2datetime(t_stamp, "Africa/Cairo") == [2020, 11, 1, 20, 9, 50]
    assert stamp2datetime(t_stamp, "America/Thule") == [2020, 11, 1, 14, 9, 50]


def test_filename2stamp():
    filename = "2021-02-05 00_00_00+00_00.csv"
    assert filename2stamp(filename) == 1612483200


def test_read_data():
    output_data = read_data("idr8gqdh", TEST_DATA_DIR, "gps",
                            "America/New_York",
                            time_start=[2020, 11, 1, 20, 9, 50],
                            time_end=[2022, 11, 1, 20, 9, 50])
    assert output_data[0].shape[0] == 23
    assert output_data[0].shape[1] == 6
    assert output_data[1] == 1639530000
    assert output_data[2] == 1639699200


def test_read_data_restriction_front():
    output_data = read_data("idr8gqdh", TEST_DATA_DIR, "gps",
                            "America/New_York",
                            time_start=[2021, 12, 15, 20, 9, 50],
                            time_end=[2022, 11, 1, 20, 9, 50])
    assert output_data[0].shape[0] == 14
    assert output_data[0].shape[1] == 6
    assert output_data[1] == 1639616990
    assert output_data[2] == 1639699200


def test_read_data_restriction_back():
    output_data = read_data("idr8gqdh", TEST_DATA_DIR, "gps",
                            "America/New_York",
                            time_start=[2020, 11, 1, 20, 9, 50],
                            time_end=[2021, 12, 15, 20, 9, 50])
    assert output_data[0].shape[0] == 9
    assert output_data[0].shape[1] == 6
    assert output_data[1] == 1639530000
    assert output_data[2] == 1639616990


def test_write_all_summaries():
    df_to_write = pd.DataFrame({"x": [5, 8], "y": [4, 9]})
    with TemporaryDirectory() as tempdir:
        write_all_summaries("test_id", df_to_write, tempdir)
        assert os.listdir(tempdir) == ['test_id.csv']
