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
from .functions import read_json, aggregate_surveys_no_config


def convert_time_to_date(submit_time, day, time):
    '''
    Takes a single array of timings and a single day
    Args: 
        submit_time(datetime):
            date in week for which we want to extract another date and time
        day(int):
            desired day of week
        time(list):
            List of timings times from the configuration surveys information
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
    Args:
        time_start(str):
            The first date for which we want to generate survey times
        time_end(str):
            The last date for which we want to generate survey times
        timings(list):
            list of survey timings, directly from the configuration file survey information
        survey_type(str):
            What type of survey schedule to generate times for
            NOTE: As of now this only works for weekly surveys
            
    Returns:
        surveys(list):
            A list of all survey times that occur between the time_start and time_end per the given survey timings schedule
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


def gen_survey_schedule(config_path, time_start, time_end, beiwe_ids):
    '''
    Args:
        config_path(str):
            File path to study configuration file
        time_start(str):
            The first date of the survey data
        time_end(str):
            The last date of the survey data
        beiwe_ids(list):
            List of users in study for which we are generating a survey schedule
        
    Returns:
        times_sur(DataFrame):
            DataFrame with a line for every survey deployed to every user in the study for the given time range
    '''
    # List of surveys
    surveys = read_json(config_path)['surveys']
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
        

def survey_submits(study_dir, config_path, time_start, time_end, beiwe_ids, agg, study_tz = None):
    '''
    Args:
        study_dir(str):
            File path to study data
        config_path(str):
            File path to study configuration file
        time_start(str):
            The first date of the survey data
        time_end(str):
            The last date of the survey data
        beiwe_ids(list):
            List of users in study for which we are generating a survey schedule        
        study_tz(str):
            Timezone of study. This defaults to 'America/New_York'
    
    Returns:
        A DataFrame with all surveys deployed in the given timeframe on the study to the users with completion times
    '''
    time_start = pd.Timestamp(time_start)
    time_end = pd.Timestamp(time_end)
    # Generate aggregated survey data
#     agg = functions.aggregate_surveys_config(study_dir, config_path, study_tz)
    
    # Generate scheduled surveys data
    sched = gen_survey_schedule(config_path, time_start, time_end, beiwe_ids)
    
    # Merge survey submit lines onto the schedule data and identify submitted lines
    submit_lines = pd.merge(
        sched[['delivery_time', 'next_delivery_time', 'id', 'beiwe_id']].drop_duplicates(), 
        agg[['Local time', 'config_id', 'survey id' ,'beiwe_id']].loc[agg.submit_line == 1].drop_duplicates(), 
        how = 'left', 
        left_on = ['id', 'beiwe_id'], 
        right_on = ['config_id', 'beiwe_id'])
    
    # Get the submigged survey line
    submit_lines['submit_flg'] = np.where(
    (submit_lines['Local time'] >= submit_lines['delivery_time']) & 
        (submit_lines['Local time'] < submit_lines['next_delivery_time']),
        1,0
    )
    
    # Take the maximum survey submit line
    submit_lines2 = submit_lines.groupby(['delivery_time', 'next_delivery_time', 'survey id', 'beiwe_id', 'config_id'])['submit_flg'].max().reset_index()
    
    # Merge on the times of the survey submission
    merge_cols = ['delivery_time', 'next_delivery_time', 'survey id', 'beiwe_id', 'config_id', 'submit_flg']
    submit_lines3 = pd.merge(submit_lines2, submit_lines[merge_cols + ['Local time']], how = 'left', left_on = merge_cols, right_on = merge_cols)

    submit_lines3['submit_time'] = np.where(submit_lines3.submit_flg == 1, submit_lines3['Local time'], np.array(0, dtype='datetime64[ns]'))
    
#     # Select appropriate columns
    submit_lines3 = submit_lines3[['survey id', 'delivery_time', 'beiwe_id', 'submit_flg', 'submit_time']]
    
#     submit_lines3['time_to_submit'] = np.where(submit_lines3['submit_flg'] == 1, submit_lines3['submit_time'] - submit_lines3['delivery_time'], np.array(0, dtype='datetime64[ns]'))
    submit_lines3['time_to_submit'] = submit_lines3['submit_time'] - submit_lines3['delivery_time']
    
    # Create a summary that has survey_id, beiwe_id, num_surveys, num submitted surveys, average time to submit
    summary_cols = ['survey id', 'beiwe_id']
    num_surveys = submit_lines3.groupby(summary_cols)['submit_flg'].count()
    num_complete_surveys = submit_lines3.groupby(summary_cols)['submit_flg'].sum()
    avg_time_to_submit = submit_lines3.loc[submit_lines3.submit_flg == 1].groupby(summary_cols)['time_to_submit'].apply(lambda x: sum(x, datetime.timedelta())/len(x))
#     avg_time_to_submit = submit_lines3.groupby(summary_cols)['time_to_submit'].apply(lambda x: sum(x, datetime.timedelta())/len(x))
    
    submit_lines_summary = pd.concat([num_surveys, num_complete_surveys, avg_time_to_submit], axis = 1).reset_index()
    submit_lines_summary.columns = ['survey id', 'beiwe_id', 'num_surveys', 'num_complete_surveys', 'avg_time_to_submit']
    
    return submit_lines3.sort_values(['survey id', 'beiwe_id']).drop_duplicates(), submit_lines_summary


def survey_submits_no_config(study_dir, study_tz = None):
    '''
    Alternative function for getting the survey completions (doesn't have expected times of surveys)
    Args:
        agg(DataFrame):
            Output of aggregate_surveys_no_config
    '''
    
    tmp = aggregate_surveys_no_config(study_dir, study_tz)
    
    def summarize_submits(df):
        tmp = {
            'min_time': df.min(),
            'max_time' : df.max()
        }
        return pd.Series(tmp, index = ['min_time', 'max_time'])

    tmp = tmp.groupby(['survey id', 'beiwe_id', 'surv_inst_flg'])['Local time'].apply(summarize_submits).reset_index()
    tmp = tmp.pivot(index = ['survey id', 'beiwe_id','surv_inst_flg'],columns='level_3', values = 'Local time').reset_index()
    tmp['time_to_complete'] = tmp['max_time'] - tmp['min_time']
    tmp['time_to_complete'] = [t.seconds for t in tmp['time_to_complete']]
    return tmp.sort_values(['beiwe_id', 'survey id'])
