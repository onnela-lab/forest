import os
import logging
import pandas as pd
# from beiwetools.helpers.functions import read_json

# Explore use of logging function
logger = logging.getLogger(__name__)

# Function to stack all survey timings files for all users given a single directory location. 
def aggregate_survey_timings(path):
    '''
    TODO: add in survey data from the json file and create dummy fields for question answers

    Reads each survey file in a study folder (downloaded raw data) survey_timings files for a study, regardless of OS. 

    Args:
        path (str): 
            path to raw data
            
    Returns:
        all_data: stacked data

    '''
    # get a list of users (ignoring hidden files and registry file downloaded when using mano)
    users = [u for u in os.listdir(path) if not u.startswith('.') and u != 'registry']
    
    if len(users) == 0:
        print('No users in directory')
        return
    # for each user, list all files in survey timings
    all_data = []
    for u in users:
        st_path = os.path.join(path, u, 'survey_timings')
        if os.path.isdir(st_path):
            # get all survey timings files
            all_files = glob.glob(os.path.join(st_path, '*/*.csv'))
            #Read in all files
            survey_data = [pd.read_csv(file) for file in all_files]
            survey_data = pd.concat(survey_data, axis = 0, ignore_index = False)
            survey_data['user_id'] = u
            all_data.append(survey_data)
            
    all_data = pd.concat(all_data, axis = 0, ignore_index = False)
    
    return all_data