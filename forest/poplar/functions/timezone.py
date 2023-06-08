"""Tools for extracting timezone information from GPS data."""
from logging import getLogger
import datetime
from typing import Union, Optional

import pytz
from timezonefinder import TimezoneFinder

from ..constants.time import HOUR_S


logger = getLogger(__name__)


def get_timezone(
    latitude: float, longitude: float,
) -> Optional[str]:
    """Get timezone from latitude and longitude.

    Args:
        latitude, longitude (float): Coordinates.

    Returns:
        timezone (str): Timezone string that can be read by pytz.timezone().
    """
    tf_obj = TimezoneFinder()
    timezone = tf_obj.timezone_at(lng=longitude, lat=latitude)
    return timezone


def get_offset(timestamp: int, timezone: Union[str, pytz.BaseTzInfo]) -> float:
    """Get UTC offset, given timestamp and timezone.

    Args:
        timestamp (int):  Millisecond timestamp.
        timezone (str or timezone from pytz.tzfile): Timezone for which to
            calculate UTC offset.

    Returns:
        offset (float):  UTC offset in hours.
    """
    if isinstance(timezone, str):
        timezone = pytz.timezone(timezone)
    datetime_date = datetime.datetime.fromtimestamp(timestamp / 1000, timezone)
    offset_s = datetime_date.utcoffset().total_seconds()
    offset = offset_s / HOUR_S
    return offset
