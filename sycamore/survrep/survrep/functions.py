''' Functions for working with raw Beiwe survey timings files.

'''
import os
import logging
import pandas as pd
from beiwetools.helpers.functions import read_json, write_json
from .headers import raw_header


logger = logging.getLogger(__name__)


# load events & question types dictionary
this_dir = os.path.dirname(__file__)
events = read_json(os.path.join(this_dir, 'events.json'))
question_type_names = read_json(os.path.join(this_dir, 'question_types.json'))
def make_lookup():
    lookup = {'iOS':{}, 'Android':{}}
    for k in question_type_names:    
        for opsys in ['iOS', 'Android']:
            opsys_name = question_type_names[k][opsys]
            lookup[opsys][opsys_name] = k
    return(lookup)
question_types_lookup = make_lookup()


def summarize_timings(opsys, file_path,
                      n_events = [],
                      unknown_headers = [],
                      unknown_events = [],
                      unknown_question_types = [],
                      foreign_surveys = []):
    '''
    Summarize a survey timings file.

    Args:
        opsys (str): 'iOS' or 'Android'.
        file_path (str): Path to a survey timings file.       
        n_events (list): Running list of counts of events.   
        unknown_headers (list): 
            Running list of filenames with unrecognized headers.
        unknown_events (list): Running list of unrecognized events.
        unknown_question_types (list):
            Running list of unrecognized question types.
        foreign_surveys (list): 
            Running list of survey identifiers that don't match dir_name.
            
    Returns:
        None        
    '''
    # note that dir_name will be a survey identifier
    dir_name = os.path.basename(os.path.dirname(file_path))
    data = pd.read_csv(file_path)
    # filename will look like <survey_id>/<datetime>.csv
    filename = os.path.join(dir_name, os.path.basename(file_path))
    # count events
    n_events.append(len(data))
    # check header
    if not list(data.columns) == raw_header[opsys]:
        unknown_headers.append(filename)
        logger.warning('Unrecognized header: %s' % filename)
    # check events
    if opsys == 'iOS': # events are in the 'event' column
        temp_events = list(data.event)        
    if opsys == 'Android': # events are in the 'question id' column
        temp = data[pd.isnull(data['question type']) & \
                    pd.isnull(data['question text']) & \
                    pd.isnull(data['question answer options']) & \
                    pd.isnull(data['answer'])]
        temp_events = list(temp['question id'])
    for e in temp_events:
        if not e in events[opsys]:
            unknown_events.append(e)
            logger.warning('Unrecognized event: %s in %s' % (e, filename))                
    # check question types
    temp_types = [t for t in data['question type'] if not pd.isnull(t)]
    for t in temp_types:
        if not t in question_types_lookup[opsys]:
            unknown_question_types.append(t)
            logger.warning('Unrecognized question type: %s in %s' % (t, filename))
    # check if the file contains events from other surveys
    for sid in data['survey id']:
        if not sid == dir_name:
            foreign_surveys.append(sid)
            logger.warning('Unrecognized survey: %s in %s' % (sid, filename))                


def check_compatibility(opsys, file_path, config,
                        absent_survey = [],
                        absent_question = [],
                        disagree_question_type = [],
                        disagree_question_text = [],
                        disagree_answer_options = []):
    '''
    Check compatibility of a survey timings file with a study configuration.

    Args:
        opsys (str): 'iOS' or 'Android'.
        file_path (str): Path to a survey timings file.       
        config (beiwetools.configread.classes.BeiweConfig):
            Representation of a Beiwe configuration file.
        absent_survey (list):
            Running list of survey identifiers that aren't in config.
        absent_question (list):
            Running list of question identifiers that aren't in config.
        disagree_question_text (list):
            Running list of question identifiers whose text disagrees with config.
        disagree_answer_options (list):
            Running list of questions identifiers whose answers disagree with config.
            
    Returns:
        None        
    '''
    # note that dir_name will be a survey identifier
    dir_name = os.path.basename(os.path.dirname(file_path))
    data = pd.read_csv(file_path)
    # filename will look like <survey_id>/<datetime>.csv
    filename = os.path.join(dir_name, os.path.basename(file_path))
    # check compatibility of each line with configuration file 
    for i in range(len(data)):
        sid   = data['survey id'][i]
        qid   = data['question id'][i]
        qtype = data['question type'][i]
        qtext = data['question text'][i]
        qao   = data['question answer options'][i]             
        # check survey id in configuration
        if not sid in config.survey_ids['tracking']:
            absent_survey.append(sid)
        else:            
            
            # check question id in survey:
        

            # check question id
            
            # check question text
            
            # check answer options text




