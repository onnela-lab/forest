"""
Tools for extracting timezone information from GPS data.

Original Authors: Georgios Efstathiadis, Josh Barback
"""

from datetime import datetime, tzinfo
from logging import getLogger
from zoneinfo import ZoneInfo

from timezonefinder import TimezoneFinder

from forest.poplar.constants.time import HOUR_S


logger = getLogger(__name__)


def get_timezone(latitude: float, longitude: float) -> str | None:
    """ Get timezone from latitude and longitude.

    Args:
        latitude, longitude (float): Coordinates.

    Returns:
        timezone (str): Timezone string that can be read by zoneinfo.ZoneInfo.
    """
    tf_obj = TimezoneFinder()
    timezone = tf_obj.timezone_at(lng=longitude, lat=latitude)
    return timezone


def get_offset(timestamp: int, timezone: str | tzinfo) -> float | None:
    """ Get UTC offset, given timestamp and timezone.

    Args:
        timestamp (int):  Millisecond timestamp.

        timezone (str or tzinfo): Timezone for which to calculate UTC offset.

    Returns:
        offset (float):  UTC offset in hours.
    """
    if isinstance(timezone, str):
        timezone = ZoneInfo(timezone)
    
    datetime_date = datetime.fromtimestamp(timestamp / 1000, timezone)
    offset_utc = datetime_date.utcoffset()
    
    if offset_utc is None:
        return None
    
    offset_s = offset_utc.total_seconds()
    offset = offset_s / HOUR_S
    return offset
