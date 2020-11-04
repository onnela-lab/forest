'''Tools for extracting timezone information from GPS data.

'''
from logging import getLogger
import pytz
import datetime
from timezonefinder import TimezoneFinder
from ..constants.time import HOUR_S


logger = getLogger(__name__)


def get_timezone(latitude, longitude, try_closest = True):
    '''
    Get timezone from latitude and longitude.

    Args:
        latitude, longitude (float): Coordinates.
        try_closest (bool): If True and no timezone found, will try to 
            find closest timezone within +/- 1 degree latitude & longitude.

    Returns:
        tz (str): Timezone string that can be read by pytz.timezone().       
    '''    
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng = longitude, lat = latitude)
    if tz is None and try_closest:
        logger.warning('No timezone found for %s, %s.  Looking for closest\
                       timezone.' % (str(latitude), str(longitude)))
        tz = tf.closest_timezone_at(lat=latitude, lng=longitude)    
    return(tz)


def get_offset(timestamp, timezone):
    '''
    Get UTC offset, given timestamp and timezone.
    
    Args:
        timestamp (int):  Millisecond timestamp.
        timezone (str or timezone from pytz.tzfile): Timezone for which to
            calculate UTC offset.
    
    Returns:
        offset (float):  UTC offset in hours.
    '''
    if type(timezone) is str:
        timezone = pytz.timezone(timezone)
    dt = datetime.datetime.fromtimestamp(timestamp/1000, timezone)
    offset_s = dt.utcoffset().total_seconds()
    offset = offset_s/HOUR_S
    return(offset)