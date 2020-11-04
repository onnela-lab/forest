'''Constants for working with Beiwe time formats.

'''
from pytz import timezone

# seconds
MIN_S  = 60
HOUR_S = 60*MIN_S
DAY_S  = 24*HOUR_S

# milliseconds
SEC_MS  = 1000
MIN_MS  = 1000*MIN_S
HOUR_MS = 1000*HOUR_S
DAY_MS  = 1000*DAY_S
WEEK_MS =    7*DAY_MS
YEAR_MS =  365*DAY_MS
TIME_MS = dict(zip(['milliseconds', 'seconds', 'minutes', 'hours', 'days', 
                    'weeks', 'years'],
                   [1, SEC_MS, MIN_MS, HOUR_MS, DAY_MS, WEEK_MS, YEAR_MS]))

# Beiwe day order
DAY_ORDER = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
             'Friday', 'Saturday']

# Beiwe time formats
DATA_DATETIME_FORMAT     = '%Y-%m-%dT%H:%M:%S.%f' # used in raw Beiwe data
FILENAME_DATETIME_FORMAT = '%Y-%m-%d %H_%M_%S' # used in raw Beiwe file names

# human-readable time formats
DATE_FORMAT = '%Y-%m-%d' # ISO 8601 date
TIME_FORMAT = '%H:%M:%S' # ISO 8601 time
TIMEZONE_FORMAT = '%Z'   # timezone name
OFFSET_FORMAT   = '%z'   # UTC offset

# human-readable and RFC 3339-compliant
NAIVE_DATETIME_FORMAT  = ' '.join([DATE_FORMAT, TIME_FORMAT])                  
AWARE_DATETIME_FORMAT  = ' '.join([DATE_FORMAT, TIME_FORMAT, TIMEZONE_FORMAT])
OFFSET_DATETIME_FORMAT = ' '.join([DATE_FORMAT, TIME_FORMAT, OFFSET_FORMAT])

# commonly used time zones
UTC = timezone('UTC')