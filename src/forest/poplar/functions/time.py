"""Functions for working with Beiwe time formats."""
from logging import getLogger
import datetime
from typing import Union

import pytz

from ..constants.time import (
    DATE_FORMAT, NAIVE_DATETIME_FORMAT, DAY_S, MIN_MS, UTC
)


logger = getLogger(__name__)


def local_now(to_format: str = NAIVE_DATETIME_FORMAT) -> str:
    """Get the current local time.

    Args:
        to_format (str):  Time format, expressed using directives from
            the datetime package.

    Returns:
        local (str):  Formatted local time now.
    """
    now = datetime.datetime.now().astimezone()
    local = now.strftime(to_format)
    return local


def convert_seconds(second_of_day: int) -> Union[str, None]:
    """Convert second of day to clock time.
    Use this function when working with survey schedules.

    Args:
        second_of_day (int):  Second of the day.

    Returns:
        time (str):  Clock time formatted as '%H:%M'.
    """
    if second_of_day > DAY_S:
        logger.warning("Input must be less than 86400.")
        return None
    time = to_readable(second_of_day * 1000, to_format="%H:%M", to_tz=UTC)
    return time


def reformat_datetime(datetime_string: str, from_format: str, to_format: str,
                      from_tz: Union[str, pytz.BaseTzInfo, None] = None
                      ) -> Union[str, None]:
    """Change the format of a datetime string.

    Args:
        datetime_string (str): A human-readable datetime string.
        from_format (str): The format of time, expressed using directives
            from the datetime package.
        to_format (str): Convert to this time format.
        from_tz (timezone from pytz.tzfile): Optionally, localize time
            before reformatting.

    Returns:
        reformat (str): Datetime string in to_format.
    """
    try:
        datetime_date = datetime.datetime.strptime(
            datetime_string, from_format
        )
        if from_tz is not None and not isinstance(from_tz, str):
            datetime_date = from_tz.localize(datetime_date)
        reformat = datetime_date.strftime(to_format)
        return reformat
    except ValueError:
        logger.warning(
            "Unable to reformat datetime string: %s.", datetime_string
        )
        return None


def to_timestamp(
    datetime_string: str, from_format: str,
    from_tz: Union[str, pytz.BaseTzInfo] = UTC
) -> Union[int, None]:
    """Convert a datetime string to a timestamp.

    Args:
        datetime_string (str):  A human-readable datetime string.
        from_format (str):  The format of time, expressed using directives
            from the datetime package.
        from_tz (timezone from pytz.tzfile):  The timezone of time.

    Returns:
        timestamp (int): Timestamp in milliseconds.
    """
    try:
        datetime_date = datetime.datetime.strptime(
            datetime_string, from_format
        )
        if not isinstance(from_tz, str):
            utc_dt = from_tz.localize(datetime_date)
            timestamp = round(utc_dt.timestamp() * 1000)
            return timestamp
        return None
    except ValueError:
        logger.warning(
            "Unable to get timestamp for datetime string: %s.", datetime_string
        )
        return None


def to_readable(timestamp: int, to_format: str,
                to_tz: Union[str, pytz.BaseTzInfo] = UTC) -> Union[str, None]:
    """Convert a timestamp to a human-readable string localized to a
    particular timezone.

    Args:
        timestamp (int): Timestamp in milliseconds.
        to_format (str): The format of readable, expressed using directives
            from the datetime package.
        to_tz (str or timezone from pytz.tzfile):  The timezone of readable.

    Returns:
        readable (str):  A human-readable datetime string.
    """
    try:
        if isinstance(to_tz, str):
            to_tz = pytz.timezone(to_tz)
        utc_dt = datetime.datetime.fromtimestamp(timestamp / 1000, tz=pytz.UTC)
        local_dt = utc_dt.astimezone(to_tz)
        readable = local_dt.strftime(to_format)
        return readable
    except ValueError:
        logger.warning("Unable to convert timestamp: %s.", timestamp)
        return None


def next_day(date: str) -> str:
    """Given a date, get the next date.

    Args:
        date (str):  A date in DATE_FORMAT.

    Returns:
        next_date (str):  Date of the next day in DATE_FORMAT.
    """
    datetime_date = datetime.datetime.strptime(date, DATE_FORMAT)
    next_dt = datetime_date + datetime.timedelta(days=1)
    next_date = next_dt.strftime(DATE_FORMAT)
    return next_date


def between_days(start_date: str, end_date: str) -> list:
    """Get a list of dates given start and end dates.

    Args:
        start_date, end_date (str): Dates in DATE_FORMAT.

    Returns:
        date_list (list): List of dates from start_date to
            end_date, inclusive.
    """
    datetime0 = datetime.datetime.strptime(start_date, DATE_FORMAT)
    datetime1 = datetime.datetime.strptime(end_date, DATE_FORMAT)
    dt_list = [datetime0]
    while dt_list[-1] < datetime1:
        dt_list.append(dt_list[-1] + datetime.timedelta(days=1))
    date_list = [
        datetime_date.strftime(DATE_FORMAT) for datetime_date in dt_list
    ]
    return date_list


def round_timestamp(timestamp: int, unit: int = MIN_MS) -> tuple:
    """
    Given an arbitrary timestamp, get timestamps for the nearest previous and
    following UTC time units.

        Args:
                timestamp (int):  Timestamp in milliseconds.
        unit (int):  Usually something like MIN_MS, HOUR_MS, etc.

    Returns:
        rounded (tuple):  A pair of millisecond timestamps:
            (<previous UTC unit timestamp>, <following UTC unit timestamp>)
    """
    previous = timestamp - (timestamp % unit)
    following = previous + unit
    rounded = (previous, following)
    return rounded
