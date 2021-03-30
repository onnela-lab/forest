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
import functions


def convert_time_to_date(submit_time, day, time):
    '''
    Function that takes a submission time and the given day of week and returns the date of a requested day.
    https://stackoverflow.com/questions/17277002/how-to-get-all-datetime-instances-of-the-current-week-given-a-day
    Takes a single array of timings and a single day
    ''' 
    
    # Convert inputted desired day into an integer between 0 and 6
    day = day % 7
    # Get the days of the given week using the dow of the given submit day
    dow = submit_time.weekday()
    days = [submit_time + datetime.timedelta(days=i) for i in range(0 - dow, 7 - dow)]

    time = [str(datetime.timedelta(seconds = t)) for t in time]
    time = [t.split(':') for t in time]
    time = [[int(p) for p in t] for t in time]

    # Get rid of timing
#     https://stackoverflow.com/questions/26882499/reset-time-part-of-a-pandas-timestamp
#     print(time)
    days = [d - pd.offsets.Micro(0) for d in days]
    days = [[d.replace(hour = t[0], minute = t[1], second = t[2], microsecond = 0) for t in time] for d in days]
    
    return days[day] 



def generate_survey_times(time_start, time_end,timings = [], survey_type = 'weekly'):
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
    
    # Roll dates
    t_lag = list(np.roll(np.array(timings, dtype = "object"), -1))
    
    # for each week, generate the survey times and append to a list
    start_dates = [time_start + datetime.timedelta(days = 7*(i)) for i in range(weeks)]
    
    surveys = []
    
    for s in start_dates:
        # Get the starting day of week 
#         dow_s = s.weekday()
        for i, t in enumerate(t_lag):
            if len(t) > 0:
                surveys.extend(convert_time_to_date(s, day = i, time = t))
    
    return surveys


def gen_survey_schedule(config, time_start, time_end, beiwe_ids):
    # List of surveys
    surveys = functions.read_json(config)['surveys']
#     print(surveys)
    # For each survey create a list of survey times
    times_sur = []
    for u_id in beiwe_ids:
        for i,s in enumerate(surveys):
            if s['timings'] != []:
                s_times = generate_survey_times(time_start, time_end, timings = s['timings'])
                # Add in relative and absolute survey timings here
                ###
                tbl = pd.DataFrame(s_times, columns = ['delivery_time'])
                # Create the "next" time column too, which indicates the next time the survey will be deployed
                tbl['next_delivery_time'] = tbl.delivery_time.shift(-1)
                tbl['id'] = i
                tbl['beiwe_id'] = u_id
                # Get all question IDs for the survey
                qs = [q['question_id'] for q in s['content'] if 'question_id' in q.keys()]
                if len(qs) > 0:
                    q_ids = pd.DataFrame({'question_id': qs})
                    tbl = pd.merge(tbl, q_ids, how = 'cross')
                times_sur.append(tbl)
    
    times_sur = pd.concat(times_sur).reset_index(drop = True)
    return times_sur
        

def survey_submits_main(config_path, path, time_start, time_end, beiwe_ids, study_tz = None):
    # Generate aggregated survey data
    agg = functions.aggregate_surveys_config(path, config_path, study_tz)
    
    # Generate scheduled surveys data
    sched = gen_survey_schedule(config_path, time_start, time_end, beiwe_ids)
    
    # Merge survey submit lines onto the schedule data and identify submitted lines
    submit_lines = pd.merge(
        sched[['delivery_time', 'next_delivery_time', 'id', 'beiwe_id']].drop_duplicates(), 
        agg[['Local time', 'config_id', 'survey id' ,'user_id']].loc[agg.submit_line == 1].drop_duplicates(), 
        how = 'left', 
        left_on = ['id', 'beiwe_id'], 
        right_on = ['config_id', 'user_id'])
    
    submit_lines = submit_lines.loc[
        (submit_lines['Local time'] >= submit_lines['delivery_time']) & 
        (submit_lines['Local time'] < submit_lines['next_delivery_time'])]
    
    # Select appropriate columns and rename
    submit_lines = submit_lines[['survey id', 'delivery_time', 'beiwe_id', 'Local time']]
    
    submit_lines = submit_lines.rename(columns = {'Local time':'submit_time'})
    
    return submit_lines
    
    
    
    