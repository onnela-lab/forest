import os

import numpy as np
import pandas as pd
import pytest
from tempfile import TemporaryDirectory


from forest.poplar.functions.helpers import (
    clean_dataframe, get_windows, directory_size, get_ids, sort_by, join_lists,
    sample_range, iqr, sample_std, sample_var
)
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


def test_datetime2stamp_bad_seconds():
    with pytest.raises(ValueError):
        datetime2stamp(time_list=[2020, 11, 1, 12, 9, 150],
                       tz_str="America/New_York")


def test_datetime2stamp_bad_minutes():
    with pytest.raises(ValueError):
        datetime2stamp(time_list=[2020, 11, 1, 12, 209, 50],
                       tz_str="America/New_York")


def test_datetime2stamp_bad_hours():
    with pytest.raises(ValueError):
        datetime2stamp(time_list=[2020, 11, 1, 35, 20, 50],
                       tz_str="America/New_York")


def test_datetime2stamp_bad_days():
    with pytest.raises(ValueError):
        datetime2stamp(time_list=[2020, 11, 35, 5, 20, 50],
                       tz_str="America/New_York")


def test_datetime2stamp_bad_months():
    with pytest.raises(ValueError):
        datetime2stamp(time_list=[2020, 15, 20, 5, 20, 50],
                       tz_str="America/New_York")


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


def test_get_files_timestamps():
    file_list, timestamp_list = get_files_timestamps(
        os.path.join(TEST_DATA_DIR, "idr8gqdh", "gps")
    )

    assert np.array_equal(file_list, np.array(
        ['2021-12-15 01_00_00+00_00.csv', '2021-12-16 21_00_00+00_00.csv',
         '2021-12-17 00_00_00+00_00.csv']
    ))
    assert np.array_equal(timestamp_list, np.array(
        [1639530000, 1639688400, 1639699200]
    ))

# Testing functions in forest.poplar.functions.helpers


@pytest.fixture
def gps_df():
    return read_data("idr8gqdh", TEST_DATA_DIR, "gps", "America/New_York",
                     time_start=[2021, 12, 15, 20, 9, 50],
                     time_end=[2022, 11, 1, 20, 9, 50])[0]


def test_clean_dataframe(gps_df):
    clean_dataframe(gps_df)
    assert gps_df.shape == (14, 6)


def test_clean_dataframe_sort(gps_df):
    clean_dataframe(gps_df, sort=True)
    assert gps_df.shape == (14, 6)


def test_clean_dataframe_drop(gps_df):
    double_df = pd.concat([gps_df, gps_df])
    clean_dataframe(double_df, drop_duplicates=True)
    assert double_df.shape == (14, 6)


def test_clean_dataframe_update_index(gps_df):
    altitude_index_df = gps_df.set_index("altitude")
    clean_dataframe(altitude_index_df, update_index=True)
    assert altitude_index_df.shape == (14, 5)
    assert (altitude_index_df.index == pd.Index(np.arange(14))).all()


def test_get_windows(gps_df):
    windows = get_windows(gps_df, start=1639688600000, end=1639688800000,
                          window_length_ms=50000)
    assert set(windows.keys()) == {1639688600000, 1639688650000, 1639688700000,
                                   1639688750000}


def test_directory_size():
    dir_size = directory_size(TEST_DATA_DIR)
    assert dir_size == (0.0, 6)


def test_sort_by():
    assert sort_by([1, 2, 3, 4], ["b", "c", "a", "d"]) == [3, 1, 2, 4]


def test_join_lists():
    list_of_lists = [[5, 4, 6], [2, 5, 6], [3, 51, 8]]
    full_list = join_lists(list_of_lists)
    assert len(full_list) == 9
    assert full_list == [5, 4, 6, 2, 5, 6, 3, 51, 8]


def test_sample_range(gps_df):
    assert np.abs(sample_range(gps_df.altitude) - 4.9622964) < 0.0001


def test_iqr(gps_df):
    assert np.abs(iqr(gps_df.altitude) - 0.1230552) < 0.0001


def test_sample_std(gps_df):
    assert np.abs(sample_std(gps_df.altitude) - 2.0817497821658235) < 0.0001


def test_sample_var(gps_df):
    assert np.abs(sample_var(gps_df.altitude) - 4.333682155547454) < 0.0001


def test_get_ids():
    users_list = get_ids(TEST_DATA_DIR)
    assert set(users_list) == {"idr8gqdh"}
