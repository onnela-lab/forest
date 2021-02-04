''' Functions for working with Beiwe time formats.

'''
import pytz
from logging import getLogger
import datetime
from ..constants.time import (DATE_FORMAT, NAIVE_DATETIME_FORMAT, 
                              DAY_S, MIN_MS, UTC)


logger = getLogger(__name__)
    

def local_now(to_format = NAIVE_DATETIME_FORMAT):
    '''
    Get the current local time.
    
    Args:
        to_format (str):  Time format, expressed using directives from 
            the datetime package.
    
    Returns:
        local (str):  Formatted local time now.
    '''
    now = datetime.datetime.now().astimezone()
    local = now.strftime(to_format)
    return(local)


def convert_seconds(s):
    '''
    Convert second of day to clock time.
    Use this function when working with survey schedules.

    Args:
        s (int):  Second of the day.

    Returns:
        time (str):  Clock time formatted as '%H:%M'.
    '''
    if s > DAY_S:
        logger.warning('Input must be less than 86400.')
    else:
        time = to_readable(s*1000, to_format = '%H:%M', to_tz = UTC)
        return(time)


def reformat_datetime(datetime_string, from_format, to_format, 
                      from_tz = None):
    '''
    Change the format of a datetime string.    

    Args:
        datetime_string (str): A human-readable datetime string.
        from_format (str): The format of time, expressed using directives 
            from the datetime package.
        to_format (str): Convert to this time format.
        from_tz (timezone from pytz.tzfile): Optionally, localize time 
            before reformatting.
            
    Returns:
        reformat (str): Datetime string in to_format.
    '''
    try:
        dt = datetime.datetime.strptime(datetime_string, from_format)
        if not from_tz is None:
            dt = from_tz.localize(dt)
        reformat = dt.strftime(to_format)
        return(reformat)
    except:
        logger.warning('Unable to reformat datetime string: %s.' 
                       % datetime_string)


def to_timestamp(datetime_string, from_format, from_tz = UTC):
    '''
    Convert a datetime string to a timestamp.
    
    Args:
        datetime_string (str):  A human-readable datetime string.
        from_format (str):  The format of time, expressed using directives 
            from the datetime package.
        from_tz (timezone from pytz.tzfile):  The timezone of time.

    Returns:
        ts (int): Timestamp in milliseconds.
    '''
    try:
        dt = datetime.datetime.strptime(datetime_string, from_format)
        utc_dt = from_tz.localize(dt)
        ts = round(utc_dt.timestamp() * 1000) 
        return(ts)
    except:
        logger.warning('Unable to get timestamp for datetime string: %s.'
                       % datetime_string)


def to_readable(timestamp, to_format, to_tz = UTC):    
    '''
    Convert a timestamp to a human-readable string localized to a 
    particular timezone.

    Args:
        timestamp (int): Timestamp in milliseconds.
        to_format (str): The format of readable, expressed using directives 
            from the datetime package.
        to_tz (str or timezone from pytz.tzfile):  The timezone of readable.

    Returns:
        readable (str):  A human-readable datetime string.
    '''
    try:
        if type(to_tz) is str:
            to_tz = pytz.timezone(to_tz)
        dt = datetime.datetime.utcfromtimestamp(timestamp/1000)
        utc_dt = pytz.utc.localize(dt)
        local_dt = utc_dt.astimezone(to_tz)
        readable = local_dt.strftime(to_format)
        return(readable)    
    except:
        logger.warning('Unable to convert timestamp: %s.' % timestamp)


def next_day(date):
    '''
    Given a date, get the next date.
    
    Args:
        date (str):  A date in DATE_FORMAT.
    
    Returns:
        next_date (str):  Date of the next day in DATE_FORMAT.
    '''
    dt = datetime.datetime.strptime(date, DATE_FORMAT)
    next_dt = dt + datetime.timedelta(days = 1)
    next_date = next_dt.strftime(DATE_FORMAT)
    return(next_date)

    
def between_days(start_date, end_date):
    '''    
    Get a list of dates given start and end dates.
    
    Args:
        start_date, end_date (str): Dates in DATE_FORMAT.
            
    Returns:
        date_list (list): List of dates from start_date to 
            end_date, inclusive.        
    '''    
    d0 = datetime.datetime.strptime(start_date, DATE_FORMAT)    
    d1 = datetime.datetime.strptime(end_date, DATE_FORMAT)    
    dt_list = [d0]
    while dt_list[-1] < d1:
        dt_list.append(dt_list[-1] + datetime.timedelta(days = 1))
    date_list = [dt.strftime(DATE_FORMAT) for dt in dt_list]
    return(date_list)


def round_timestamp(timestamp, unit = MIN_MS):
    '''
    Given an arbitrary timestamp, get timestamps for the nearest previous and 
    following UTC time units.

	Args:
		timestamp (int):  Timestamp in milliseconds.
        unit (int):  Usually something like MIN_MS, HOUR_MS, etc.

    Returns:
        rounded (tuple):  A pair of millisecond timestamps:
            (<previous UTC unit timestamp>, <following UTC unit timestamp>)    
    '''
    previous = timestamp - (timestamp%unit)
    following = previous + unit   
    rounded = (previous, following)
    return(rounded)