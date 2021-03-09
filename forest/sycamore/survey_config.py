import os
import logging
import pandas as pd
import glob
import json
import numpy as np
from typing import List
import datetime
import pytz
import math
import datetime


def convert_time_to_date(submit_time, dow, day, time):
    '''
    Function that takes a submission time and the given day of week and returns the date of a requested day.
    https://stackoverflow.com/questions/17277002/how-to-get-all-datetime-instances-of-the-current-week-given-a-day
    ''' 
    day = day % 7
    days = [submit_time + datetime.timedelta(days=i) for i in range(0 - dow, 7 - dow)]
    
    time = [t.split(':') for t in time]
    time = [[int(p) for p in t] for t in time]
    
    # Get rid of timing
#     https://stackoverflow.com/questions/26882499/reset-time-part-of-a-pandas-timestamp
#     print(time)
    days = [d - pd.offsets.Micro(0) for d in days]
    days = [[d.replace(hour = t[0], minute = t[1], second = t[2], microsecond = 0) for t in time] for d in days]
    
    return days[day] 



def generate_survey_times(time_start, time_end, survey_id = '', timings = [], survey_type = 'weekly'):
    '''
    Takes a start time and end time and generates a schedule of all sent surveys in time frame for the given survey type
    ''' 
    if survey_type not in ['weekly', 'absolute', 'relative']:
        raise ValueError('Incorrect type of survey. Ensure this is weekly, absolute, or relative.')  
        
    
    # Get the number of weeks between start and end time
    t_start = pd.Timestamp(time_start)
    t_end = pd.Timestamp(time_end)
    
    weeks = pd.Timedelta(t_end - t_start).days
    # Get ceiling number of weeks
    weeks = math.ceil(weeks/7.0)
    
    # for each week, generate the survey times and append to a list
    start_dates = [time_start + datetime.timedelta(days = 7*(i)) for i in range(weeks)]
    
    surveys = []
    
    for s in start_dates:
        # Get the starting day of week 
        dow_s = s.weekday()
        for i, t in enumerate(timings):
            surveys.extend(convert_time_to_date(s, dow = dow_s, day = i, time = t))
    
    return surveys