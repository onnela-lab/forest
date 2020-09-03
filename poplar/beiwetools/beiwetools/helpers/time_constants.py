'''Constants for working with Beiwe time formats.

'''
from pytz import timezone
from collections import OrderedDict

# seconds
min_s  = 60
hour_s = 60*min_s
day_s  = 24*hour_s

# milliseconds
sec_ms  = 1000
min_ms  = 1000*min_s
hour_ms = 1000*hour_s
day_ms  = 1000*day_s
week_ms =    7*day_ms
month_ms =  30*day_ms
year_ms =  365*day_ms
time_unit_ms = OrderedDict(zip(['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'],
                               [sec_ms, min_ms, hour_ms, day_ms, week_ms, month_ms, year_ms]))

# Beiwe day order
day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

# time formats
data_time_format     = '%Y-%m-%dT%H:%M:%S.%f' # used in raw Beiwe data
filename_time_format = '%Y-%m-%d %H_%M_%S'    # used in raw Beiwe file names
date_only = '%Y-%m-%d'
time_only = '%H:%M:%S'
timezone_only = '%Z'
date_time_format  = ' '.join([date_only, time_only]) 
local_time_format = ' '.join([date_only, time_only, timezone_only])

# commonly used time zones
UTC = timezone('UTC')
