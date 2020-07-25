'''Functions for working with Beiwe time formats.

'''
import pytz
import logging
import datetime
import holidays
from timezonefinder import TimezoneFinder
from .time_constants import *


logger = logging.getLogger(__name__)


def is_US_holiday(date, date_format = date_only):
    '''
    Identify dates that are US holidays.
    There is probably a better way to do this with pandas.
    
    Args:
        date (str): Date string.
        date_format (str): Format of date.
        
    Returns:
        is_holiday (bool): True if the date is a US holiday.
    '''    
    us_holidays = holidays.UnitedStates()
    d = datetime.datetime.strptime(date, date_format)
    is_holiday = d in us_holidays
    return(is_holiday)
    

def local_now(to_format = local_time_format):
    '''
    Get the current local time.
    
    Args:
        to_format (str):  Time format, expressed using directives from the datetime package.
    
    Returns:
        local (str):  Formatted local time now.
    
    '''
    now = datetime.datetime.now().astimezone()
    local = now.strftime(to_format)
    return(local)


def convert_seconds(s):
	'''
	Convert second of day to clock time.

	Args:
		s (int):  Second of the day.

	Returns:
		time (str):  Clock time formatted as '%H:%M'.
	'''
	time = to_readable(s*1000, to_format = '%H:%M', to_tz = UTC)
	return(time)


def datatime_to_dt(time):
    '''
    Convert a UTC datetime string in data_time_format to a datetime.datetime object.

    Args:
        time (str): UTC datetime string in data_time_format.
    
    Returns:
        dt (datetime.datetime): A timezone-aware datetime.datetime object.
    '''
    dt = datetime.datetime.strptime(time, data_time_format)
    dt = UTC.localize(dt)
    return(dt)


def reformat_datetime(time, from_format, to_format, from_tz = None):
    '''
    Change the format of a data/time string.    

	Args:
		time (str):  A human-readable date/time string.
		from_format (str):  
            The format of time, expressed using directives from the datetime package.
        to_format (str): 
            Time format to convert to.
        from_tz (timezone from pytz.tzfile): 
            Optionally, localize time before reformatting.
            
    Returns:
        reformat (str): Date/time string in to_format.
    '''
    dt = datetime.datetime.strptime(time, from_format)
    if not from_tz is None:
        dt = from_tz.localize(dt)
    reformat = dt.strftime(to_format)
    return(reformat)


def to_timestamp(time, from_format, from_tz = UTC):
	'''
	Convert a date/time string to a timestamp.
	
	Args:
		time (str):  A human-readable date/time string.
		from_format (str):  The format of time, expressed using directives from the datetime package.
		from_tz (timezone from pytz.tzfile):  The timezone of time.

	Returns:
		ts (int): Timestamp in milliseconds.
	'''
	dt = datetime.datetime.strptime(time, from_format)
	utc_dt = from_tz.localize(dt)
	ts = round(utc_dt.timestamp() * 1000) 
	return(ts)


def to_readable(timestamp, to_format, to_tz):    
	'''
	Convert a timestamp to a human-readable string localized to a particular timezone.

	Args:
		timestamp (int):  Timestamp in milliseconds.
		to_format (str):  The format of readable, expressed using directives from the datetime package.
		to_tz (str or timezone from pytz.tzfile):  The timezone of readable.

	Returns:
		readable (str):  A human-readable date/time string.
	'''
	if type(to_tz) is str:
		to_tz = pytz.timezone(to_tz)
	dt = datetime.datetime.utcfromtimestamp(timestamp/1000)
	utc_dt = pytz.utc.localize(dt)
	local_dt = utc_dt.astimezone(to_tz)
	readable = local_dt.strftime(to_format)
	return(readable)    


def get_timezone(latitude, longitude, try_closest = True):
    '''
    Get timezone from latitude and longitude.

    Args:
        latitude, longitude (float): Coordinates.
        try_closest (bool): 
            If True and no timezone found, will try to find closest timezone within +/- 1 degree latitude & longitude.

    Returns:
        tz (str): Timezone string that can be read by pytz.timezone().       
    '''    
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng = longitude, lat = latitude)
    if tz is None and try_closest:
        logger.warning('No timezone found for %s, %s.  Looking for closest timezone.' % (str(latitude), str(longitude)))
        tz = tf.closest_timezone_at(lat=latitude, lng=longitude)    
    return(tz)


def summarize_UTC_range(UTC_range, 
                        from_format = filename_time_format, 
                        to_format   = local_time_format, 
                        unit = 'days', ndigits = 1):
    '''
    Reformat start/finish datetimes and get elapsed time.
    Mainly used when handling filenames from raw Beiwe data.
        
    Args:
        UTC_range (list): Ordered pair of UTC datetimes, [start, finish].
        from_format, to_format (str): Formats for datetimes.
        unit (str): A key from time_unit_ms.
        ndigits (int): Optionally, number of digits for rounding.
    '''
    try:
        reformat = [reformat_datetime(t, from_format, to_format, UTC) for t in UTC_range]
        ms_start = to_timestamp(UTC_range[0], from_format)
        # round end time up to nearest hour:
        temp = to_timestamp(UTC_range[1], from_format) + hour_ms
        ms_end = temp - temp%hour_ms
        ms = ms_end - ms_start
        elapsed = ms/time_unit_ms[unit]
        if not ndigits is None:
            elapsed = round(elapsed, ndigits)
        return(reformat, elapsed, unit)
    except:
        return([None, None], 0, unit)
    
    
def between_days(start_date, end_date):
    '''    
    Get a list of dates given start and end dates.
    
    Args:
        start_date, end_date (str):
            Dates in date_only format.
            
    Returns:
        date_list (list): List of dates from start_date to end_date, inclusive.        
    '''    
    d0 = datetime.datetime.strptime(start_date, date_only)    
    d1 = datetime.datetime.strptime(end_date, date_only)    
    dt_list = [d0]
    while dt_list[-1] < d1:
        dt_list.append(dt_list[-1] + datetime.timedelta(days = 1))
    date_list = [dt.strftime(date_only) for dt in dt_list]
    return(date_list)
