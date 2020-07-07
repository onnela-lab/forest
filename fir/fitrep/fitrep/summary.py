''' Functions for working with raw Fitabase data.
'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.functions import setup_directories, setup_csv, write_to_csv
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .classes import FitabaseRegistry
from .headers import summary_header


logger = logging.getLogger(__name__)


@easy(('path_dict'))
def setup_output(proc_dir, file_types, track_time = True):
    '''
    Set up summary output.

    Args:
        proc_dir (str): 
        file_types (list): List of Fitabase file types.
        track_time (bool): If true, output to a timestamped folder.
        
    Returns:
        path_dict (dict): 
            Keys are file types, values are paths.
    '''
    if 'syncEvents' not in file_types: file_types.append('syncEvents')
    out_dir = os.path.join(proc_dir, 'fitrep', 'summary')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)
    setup_directories(out_dir)
    path_dict = {}
    for ft in file_types:
        path_dict[ft] = setup_csv(ft, out_dir, summary_header)
    logger.info('Created output directory and initialized files.')
    return(path_dict)        


@easy(('summary_dict'))
def summarize_user(fitabase_id, file_types, registry):
    summary_dict = {}
    for ft in file_types:
        pass
    
    logger.info('Summarized data for user %s.' % fitabase_id)
    pass


@easy(())
def write_user_records(fitabase_id, summary_dict, path_dict):

    logger.info('Updated records for user %s.' % fitabase_id)
    pass
    
        
        

FitrepSummary = ProcessTemplate.create('summary',
                                       [setup_output], [], 
                                       [summarize_user], 
                                       [write_user_records], [])



