import datetime
from dateutil import tz
import pytest

from forest.oak.base import preprocess_dates


@pytest.fixture(scope="session")
def test_file_list():
    file_list = [
        [f"2023-11-{day:02} {hr:02}_00_00+00_00.csv" for hr in range(24)]
        for day in range(1, 7)
    ]
    file_list = [item for sublist in file_list for item in sublist]
    return file_list


def test_preprocess_dates_length(test_file_list):
    """Test preprocess_dates function for length of output list"""

    time_start = None
    time_end = None
    fmt = "%Y-%m-%d %H_%M_%S"
    from_zone = tz.gettz("UTC")
    to_zone = tz.gettz("America/New_York")

    dates_shifted, _, _ = preprocess_dates(
        test_file_list, time_start, time_end, fmt, from_zone, to_zone
    )
    assert len(dates_shifted) == 144


def test_preprocess_dates_start_end_dates(test_file_list):
    """Test preprocess_dates function for start/end date"""

    time_start = None
    time_end = None
    fmt = "%Y-%m-%d %H_%M_%S"
    from_zone = tz.gettz("UTC")
    to_zone = tz.gettz("America/New_York")

    _, date_start, date_end = preprocess_dates(
        test_file_list, time_start, time_end, fmt, from_zone, to_zone
    )
    assert date_start == datetime.datetime(2023, 10, 31, 0, 0, tzinfo=to_zone)
    assert date_end == datetime.datetime(2023, 11, 6, 0, 0, tzinfo=to_zone)


def test_preprocess_dates_start_end_dates_inputs(test_file_list):
    """Test preprocess_dates function for start/end date, with inputs"""

    time_start = "2023-10-31 00_00_00"
    time_end = "2023-11-03 00_00_00"
    fmt = "%Y-%m-%d %H_%M_%S"
    from_zone = tz.gettz("UTC")
    to_zone = tz.gettz("America/New_York")

    _, date_start, date_end = preprocess_dates(
        test_file_list, time_start, time_end, fmt, from_zone, to_zone
    )
    assert date_start == datetime.datetime(2023, 10, 30, 0, 0, tzinfo=to_zone)
    assert date_end == datetime.datetime(2023, 11, 2, 0, 0, tzinfo=to_zone)
