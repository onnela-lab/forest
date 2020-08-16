''' Functions for working with raw Beiwe survey timings files.

'''
import os
import logging
import pandas as pd
from beiwetools.helpers.functions import read_json
from .headers import raw_header


logger = logging.getLogger(__name__)


# load events
this_dir = os.path.dirname(__file__)
events = read_json(os.path.join(this_dir, 'events.json'))
        

def summarize_timings(opsys, file_path,
                      n_events = [],
                      unknown_header = [],
                      unknown_events = [],
                      foreign_survey = []):
    '''
    Summarize a survey timings file.

    Args:
        opsys (str): 'iOS' or 'Android'.
        file_path (str): Path to a survey timings file.       
        n_events (list): Running list of counts of events.   
        unknown_header (list): 
            Running list of filenames with unrecognized headers.
        unknown_events (list): Running list of unrecognized events.
        foreign_survey (list): 
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
        unknown_header.append(filename)
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
    # check if the file contains events from other surveys
    for sid in data['survey id']:
        if not sid == dir_name:
            foreign_survey.append(sid)
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

    # check compatibility with configuration file 
    if not config is None:
         for i in range(len(data)):



             sid, qid, qt, ao = None


             
         # check survey id
         for sid in data['survey_id']:
             if not sid in config.survey_ids['tracking']:
                 pass
         # check question id
         
         # check question text

         # check answer options text


    


