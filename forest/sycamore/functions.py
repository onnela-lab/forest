import os
import logging
import pandas as pd
import glob
import json
import numpy as np
from typing import List
import datetime
import pytz

# import sys
# # add path to poplar dir
# sys.path.append('../')

# from poplar.legacy.common_funcs import stamp2datetime, datetime2stamp

# Explore use of logging function (TO DO: Read wiki)
logger = logging.getLogger(__name__)


# Modified from legacy beiwetools code
def read_json(path):
    '''
    Read a json file into a dictionary.
    Args:
        path (str):  Path to json file.
    Returns:
        dictionary (dict) 
    '''
    with open(path, 'r') as f:        
        dictionary = json.load(f)
    return(dictionary)



# load events & question types dictionary
# From Josh's legacy script
this_dir = os.path.dirname(__file__)
events = read_json(os.path.join(this_dir, 'events.json'))
question_type_names = read_json(os.path.join(this_dir, 'question_type_names.json'))

def make_lookup():
    lookup = {'iOS':{}, 'Android':{}}
    for k in question_type_names:    
        for opsys in ['iOS', 'Android']:
            opsys_name = question_type_names[k][opsys]
            lookup[opsys][opsys_name] = k
    return(lookup)

question_types_lookup = make_lookup()




def q_types_standardize(q, lkp = question_types_lookup):
    '''
    Standardizes question types using a lookup function

    Args:
        q (str):
            a single value for a question type
        lkp (dict):
            a lookup dictionary of question types and what they should map too. 
            Based on Josh's dictionary of question types. 

    Returns:
        s: string with the corrected question type
    '''
    # If it's an Android string, flip it from the key to the value
    if q in lkp['Android'].keys():
        return lkp['Android'][q]
    else:
        return q



def read_and_aggregate(path, user, data_stream):
    st_path = os.path.join(path, user, data_stream)
    if os.path.isdir(st_path):
        # get all survey timings files
        all_files = glob.glob(os.path.join(st_path, '*/*.csv'))
        # Sort file paths for when they're read in
        all_files = sorted(all_files)
        #Read in all files
        survey_data = [pd.read_csv(file) for file in all_files]
        survey_data = pd.concat(survey_data, axis = 0, ignore_index = False)
        survey_data['user_id'] = user
        survey_data['UTC time'] = survey_data['UTC time'].astype('datetime64[ns]')
        survey_data['DOW'] = survey_data['UTC time'].dt.dayofweek
        return survey_data
    else:
        logging.warning('No survey_timings for user %s.' % user)
        
        
        
def aggregate_surveys(path):
    # get a list of users (ignoring hidden files and registry file downloaded when using mano)
    
    # READ AND AGGREGATE DATA
    users = [u for u in os.listdir(path) if not u.startswith('.') and u != 'registry']
    
    if len(users) == 0:
        print('No users in directory')
        return  
    
    all_data = []
    for u in users:
        all_data.append(read_and_aggregate(path, u, 'survey_timings'))
        
    # Collapse all users into one file and drop duplicates
    all_data = pd.concat(all_data, axis = 0, ignore_index = False).drop_duplicates()        

    # FIX EVENT FIELDS    
    # Move Android evens from the question id field to the event field
    all_data.event = all_data.apply(lambda row: row['question id'] if row['question id'] in ['Survey first rendered and displayed to user', 'User hit submit'] else row['event'], axis = 1)
    all_data['question id'] = all_data.apply(lambda row: np.nan if row['question id'] == row['event'] else row['question id'], axis = 1)
    
    # Fix question types
    all_data['question type'] = all_data.apply(lambda row: q_types_standardize(row['question type'], question_types_lookup), axis = 1)
    
    # ADD A QUESTION INDEX (to track changed answers)
    all_data['question id lag'] = all_data['question id'].shift(1)
    all_data['question index']  = all_data.apply(lambda row: 1 if ((row['question id'] != row['question id lag'])) else 0, axis = 1)
    all_data['question index'] = all_data.groupby(['survey id', 'user_id'])['question index'].cumsum()
    
    del all_data['question id lag']
    # OUTPUT AGGREGATE
    return all_data

def check_survey_times(agg_data, config, tz_str= 'America/New_York'):
    local_tz = pytz.timezone(tz_str)
    agg_data['UTC time'] = agg_data['UTC time'].astimezone(timezone(tz_str))
#     agg_data['study_time'] = datetime2stamp(agg_data['timestamp'].astype('int'), tz_str )
#     agg_data['study_time'] = stamp2datetime(agg_data['study_time'], tz_str)
    return agg_data
    



def parse_timings(survey, survey_id):
    '''
    Args:
        survey (dict): 
            a survey dictionary from configuration file
        survey_id (int):
            a provided survey id, since the configuration file has no survey file
            
    Returns:
        A dataframe with a survey id, question_id, and weekly timings
    '''
    if 'timings' in survey.keys():
        # there are 7 elements 
        times = survey['timings']
        times_row = []
        # Loop through all days
        for i,t in enumerate(times):
            time_val = {}
            time_val['timings_day'] = 'timings_day_' + str(i)
            # If there is a timing for that day
            if len(t) > 0:
                # Convert seconds to datetime
                time_val['time'] = str(datetime.timedelta(seconds = t[0])) 
            else:
                time_val['time'] = float('NaN')
            times_row.append(time_val)
        output = pd.DataFrame(times_row)
        output['id'] = survey_id
       ## Add in absolute timings and relative timings        
    return output.pivot(columns = 'timings_day', values = 'time', index = 'id').reset_index()

def parse_surveys(config_file):
    '''
    Args:
        config_file(str):
            path to the study configuration file
    
    Returns:
        A dataframe with all surveys, question ids, question texts, question types and the expected timings of the survey
    '''
    # Read in configuration file
    with open(config_file) as f:
        data = json.load(f)
    surveys = data['surveys']
    
    # Create an array for surveys and one for timings
    output = []
    timings_arr = []
    for i,s in enumerate(surveys):
        # Pull out questions
        content = s['content']
        # Pull out timings
        timings = parse_timings(s, i)
        for q in s['content']:
            surv = {}
            surv['id'] = i
            surv['question_id'] = q['question_id']
            surv['question_text'] = q['question_text']
            surv['question_type'] = q['question_type']
            if 'text_field_type' in q.keys():
                surv['text_field_type'] = q['text_field_type']
            # Convert surv to data frame
            surv = pd.DataFrame([surv]).merge(timings, left_on = 'id', right_on = 'id')
#             surv['question_index'] = 1
#             surv['question_index'] = surv['question_index'].cumsum()
#             surv['question_index'] = np.cumsum(surv['question_index'])
#             print(np.cumsum(surv['question_index']))
            output.append(surv)
    output = pd.concat(output)
#     output['question_index'] = output.groupby('id')['question_index'].cumsum()
    
    return output



def get_survey_timings(person_ids: List[str], study_dir: str, survey_id: str):
    """
    Created on Thu Jan 28 11:34:23 2021

    @author: DEBEU
    
    Parameters
    ----------
    person_ids : list of beiwe_ids to .
    study_dir : raw data directory containing directories for each beiwe_id.
    survey_id : back_end id for survey (and name of folder within
    study_dir/beiwe_id/survey_timings).


    Returns
    -------
    Record with beiwe_id / phone_os / date_hour / start_time / end_time.
    For iOS users: 
        start_time = time of 'present' of first question
        end_time = time of 'submitted'
    For Android 
        start_time = ....
        end_time = ....
        
    Assumes
    -------
    Operating system-specific difference in registration of survey timings
    That a survey with survey_id is not triggered more than once an hour
    

    """
    record = np.array(['beiwe_id', 'phone_os', 'date_hour', 'start_time', 'end_time']).reshape(1,5)
    for pid in person_ids:
        print(pid)
        survey_dir = os.path.join(study_dir, pid, "survey_timings", survey_id)
        
        try:
            filepaths = os.listdir(survey_dir)
        except FileNotFoundError:
            continue
        
        # For each survey
        for fp in filepaths:
#             print(fp)
            try:
                f = pd.read_csv(os.path.join(survey_dir, fp))
            except:
                pass
            
            # Check whether participant uses iOS
            if 'event' in f.columns: # iOS: last columnname == 'event'
            # Note: this assumes that all files have headers (check!)
            ### Logic for iPhones ###
                
                # Here you could have a loop over pd.unique(f['survey id']) to do it in
                # one iteration for all surveys -->
                #for sid in survey_ids:
                # Note to Nellie: might be useful to have it iterate over surveys and store timings for each survey
           
                # select relevant rows and columns
                f = f.loc[(f['survey id'] == survey_id) & # only this survey 
                   ((f['event'] == 'present') | # only present / submit events
                    (f['event'] == 'submitted')), 
                  ['timestamp', 'UTC time','survey id', 'event']]
                
                # Extract time indicators
                # We assume participants enter only 1 survey per hour
                f['UTC time'] = f['UTC time'].astype('datetime64[ns]')
                f['date_hour'] = f['UTC time'].dt.strftime('%Y_%m_%d_%H')
    
                #sort by UTC_time
                f = f.sort_values(by='date_hour',ascending=True)
                
                f = f.drop_duplicates(subset=['date_hour', 'event'], keep='first')
    
                f = f.pivot(columns = 'event', values = 'UTC time', index = 'date_hour').reset_index()
                
                for timestamp in pd.unique(f['date_hour']):
                    try:                
                        present = f.loc[f['date_hour'] == timestamp, 'present'][0]
                    except KeyError:
                        present = None
                        
                    try:                
                        submitted = f.loc[f['date_hour'] == timestamp, 'submitted'][0]
                    except KeyError:
                        submitted = None
                   
                    record = np.vstack([record, 
                                        [pid,'iOs', timestamp,
                                         present, submitted]])
            else:
                #LOGIC FOR ANDROID USERS
                f = f.loc[(f['survey id'] == survey_id) & # only this survey 
                   ((f['question id'] == "Survey first rendered and displayed to user") | # only present / submit events
                    (f['question id'] == 'User hit submit')), 
                  ['timestamp', 'UTC time','question id']]
                
                # Extract time indicators
                # We assume participants enter only 1 survey per hour
                f['UTC time'] = f['UTC time'].astype('datetime64[ns]')
                f['date_hour'] = f['UTC time'].dt.strftime('%Y_%m_%d_%H')
                
                f = f.sort_values(by='date_hour',ascending=True)
                
                # Looks like if Androids have double events, you should take the   last
                f = f.drop_duplicates(subset=['date_hour', 'question id'], keep='last')
        
                f = f.pivot(columns = 'question id', values = 'UTC time', index = 'date_hour').rename(columns = {'Survey first rendered and displayed to user':'present', 'User hit submit':'submitted'}).reset_index()
                
                for timestamp in pd.unique(f['date_hour']):
                    try:                
                        present = f['present'][0]
                    except KeyError:
                        present = None
                        
                    try:                
                        submitted = f['submitted'][0]
                    except KeyError:
                        submitted = None
                   
                    record = np.vstack([record,
                                        [pid,'Android', timestamp,
                                         present, submitted]])
                    
    svtm = pd.DataFrame(record[1:,:],
                    columns = record[0])

    # Fix surveys that were completed over more than an hour
    svtm['day'] = pd.to_datetime(svtm['date_hour'].astype('str'), format='%Y_%m_%d_%H').dt.strftime('%Y-%m-%d')
    
    svtm = svtm.groupby(['beiwe_id', 'day', 'phone_os']).agg(
        {
         'start_time':min,    # Sum duration per group
          'end_time':max
         }).reset_index()
    
    svtm['duration'] = svtm['end_time'] - svtm['start_time']
    svtm['duration_in_sec'] = svtm['duration'].dt.seconds

    return svtm