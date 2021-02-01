import os
import logging
import pandas as pd
import glob
import json
import numpy as np

# Explore use of logging function
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
        return survey_data
    else:
        logging.warning('No survey_timings for user %s.' % user)



# Function to stack all survey timings files for all users given a single directory location. 
def aggregate_survey_timings(path):
    '''
    Args:
        path (str): 
            path to raw data
            
    Returns:
        all_data: Data frame with all question answers across all surveys from a study
        survey_begins: Data frame with all survey beginning times across all surveys from a study
        survey_submits: Data frame with all survey submit times across all surveys from a study
        survey_notifications: (iOS only) Data frame with the times a survey was delievered
        survey_question_times: (iOS only) Data frame with the times a question in a survey was presented 
        to the user and then not presented

    '''
    # get a list of users (ignoring hidden files and registry file downloaded when using mano)
    users = [u for u in os.listdir(path) if not u.startswith('.') and u != 'registry']
    
    if len(users) == 0:
        print('No users in directory')
        return
    # for each user, read and aggreagate all files
    all_data = []
    for u in users:
        all_data.append(read_and_aggregate(path, u, 'survey_timings'))
    
    # Collapse all users into one file and drop duplicates
    all_data = pd.concat(all_data, axis = 0, ignore_index = False).drop_duplicates()
    
    ### CREATE IDS:
    
    # Move Android evens from the question id field to the event field
    all_data.event = all_data.apply(lambda row: row['question id'] if row['question id'] in ['Survey first rendered and displayed to user', 'User hit submit'] else row['event'], axis = 1)
    all_data['question id'] = all_data.apply(lambda row: np.nan if row['question id'] == row['event'] else row['question id'], axis = 1)
    
    # Fix question types
    all_data['question type'] = all_data.apply(lambda row: q_types_standardize(row['question type'], question_types_lookup), axis = 1)
    
    # Create a single ID that indicates a unique survey/instance/user
    
    # 1. Mark beginning of surveys with a beg_flg
    # a. Android
    all_data['beg_flg'] = all_data.apply(lambda row: 1 if ((row['event'] == 'Survey first rendered and displayed to user')) else 0, axis = 1)
    # b. iOS
    all_data.loc[(all_data.event == 'notified') & (np.roll(all_data.event == 'present', -1)), 'beg_flg'] = 1
    
    # 2. Create an instance id
    all_data['instance_id'] = np.cumsum(all_data['beg_flg'])
    
    # Once instance_id is created, we can remove start and end times to separate dataframes
    survey_begins = all_data.loc[all_data.beg_flg == 1, ]
    all_data = all_data.loc[all_data.beg_flg != 1]

    # Filter out submission data
    survey_submits = all_data.loc[(all_data.event == 'submitted') | (all_data.event == 'User hit submit'), ]
    all_data = all_data.loc[(all_data.event != 'submitted') & (all_data.event != 'User hit submit')]

    # Filter out iOS notifications data
    survey_notifications = all_data.loc[all_data.event.isin(['notified', 'expired'])]
    all_data = all_data.loc[~all_data.event.isin(['notified', 'expired'])]

    # Filter out iOS question change times 
    survey_question_times = all_data.loc[all_data.event.isin(['present', 'unpresent'])]
    all_data = all_data.loc[~all_data.event.isin(['present', 'unpresent'])]

    # Create a rank on each instance
    all_data['instance_id_rank'] = all_data.groupby('instance_id')['timestamp'].rank(method = 'first')
    
    # Remove extra fields:
    del all_data['beg_flg']
    del all_data['event']

    return all_data, survey_begins, survey_submits, survey_notifications, survey_question_times