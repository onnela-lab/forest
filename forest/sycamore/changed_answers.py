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


#Function that takes aggregated data and adds list of changed answers and first and last times and answers
def agg_changed_answrs(path, config_file, study_tz= None):
    
    agg = functions.aggregate_surveys_config(path, config_file, study_tz)
    
    cols = ['survey id', 'user_id','question id', 'question text', 'question type', 'question index']
    
    agg['last_answer'] = agg.groupby(cols).answer.transform('last')
    # add in an answer ID and take the last of that too to join back on the time
    agg = agg.reset_index().set_index(cols)
    agg['all_answers'] = agg.groupby(cols)['answer'].apply(list)
    agg = agg.reset_index()
    
    agg['first_time'] = agg.groupby(cols)['Local time'].transform('first')
    agg['last_time'] = agg.groupby(cols)['Local time'].transform('last')
    
    # Filter agg down to the line of the last question time
    agg = agg.loc[agg['Local time'] == agg['last_time']]
    
    return agg
    
    
    
    