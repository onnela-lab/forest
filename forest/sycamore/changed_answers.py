import os
import logging
import pandas as pd
import glob
import json
import numpy as np
import datetime
import pytz
import math
import functions

### Write a function that subsets the list if it starts with nan and ends with two of the same elements
def subset_answer_choices(answer):
    '''
    If a user changes their answers multiple times, an iOS device will have redundant answers at the beginning
    and end of the list, so we remove them.
    Args:
        answer(list):
            List of changed answers
    
    Returns:
        answer(list):
            List of changed answers with redundant answers removed
    '''
    if isinstance(answer[0], float):
        answer = answer[1:]
    
    if (len(answer) > 1):
        if (answer[-1] == answer[-2]):
            answer = answer[:-1]
        
    return answer


#Function that takes aggregated data and adds list of changed answers and first and last times and answers
def agg_changed_answers(study_dir, config_path, agg, study_tz= None):
    '''
    Args:
        config_path(str):
            File path to study configuration file
        study_dir(str):
            File path to study data
        study_tz(str):
            Timezone of study. This defaults to 'America/New_York'
    
    Returns:
        agg(DataFrame):
            Dataframe with aggregated data, one line per question answered, with changed answers aggregated into a list. 
            The Final answer is in the 'last_answer' field
    '''
    
#     agg = functions.aggregate_surveys_config(study_dir, config_path, study_tz)
    
    cols = ['survey id', 'beiwe_id','question id', 'question text', 'question type', 'question index']
    
    agg['last_answer'] = agg.groupby(cols).answer.transform('last')
    # add in an answer ID and take the last of that too to join back on the time
    agg = agg.reset_index().set_index(cols)
    agg['all_answers'] = agg.groupby(cols)['answer'].apply(lambda x: list(x))
    # Subset answer lists if needed
    agg['all_answers'] = agg['all_answers'].apply(lambda x: x if isinstance(x, float) else subset_answer_choices(x))
    agg['num_answers'] = agg['all_answers'].apply(lambda x: x if isinstance(x, float) else len(list(x)))
    agg = agg.reset_index()
    
    agg['first_time'] = agg.groupby(cols)['Local time'].transform('first')
    agg['last_time'] = agg.groupby(cols)['Local time'].transform('last')
    
    # Number of changed answers and time spent answering each question
#     agg['num_answers'] = agg['all_answers'].apply(lambda x: x if isinstance(x, float) else len(x))
    agg['time_to_answer'] = agg['last_time']-agg['first_time']
    
    # Filter agg down to the line of the last question time
    agg = agg.loc[agg['Local time'] == agg['last_time']]

    
    return agg.reset_index(drop = True)


# Create a summary file that has survey, beiwe id, question id, average number of changed answers, average time spent answering question

def agg_changed_answers_summary(study_dir, config_path, agg, study_tz = None):
    '''
    Args:
        config_path(str):
            File path to study configuration file
        study_dir(str):
            File path to study data
        study_tz(str):
            Timezone of study. This defaults to 'America/New_York'
    
    Returns:
        agg(DataFrame):
            Dataframe with aggregated data, one line per question answered, with changed answers aggregated into a list. 
            The Final answer is in the 'last_answer' field
    '''
    
    detail = agg_changed_answers(study_dir, config_path, agg, study_tz)
    
    summary_cols = ['survey id', 'beiwe_id', 'question id', 'question text', 'question type']
    num_answers = detail.groupby(summary_cols)['num_answers'].count()
    avg_time = detail.groupby(summary_cols)['time_to_answer'].apply(lambda x: sum(x, datetime.timedelta())/len(x))
    avg_chgs = detail.groupby(summary_cols)['num_answers'].mean()
    
    out = pd.concat([num_answers, avg_time, avg_chgs], axis = 1).reset_index()
    
    out.columns = summary_cols + ['num_answers', 'average_time_to_answer', 'average_number_of_answers']
    
    return detail, out
    
    
    
    