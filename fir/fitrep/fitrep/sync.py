''' Functions for processing raw Fitabase syncEvents files.
'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import setup_directories, setup_csv, write_to_csv
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate




from .headers import summary_header, extra_summary_header, raw_header
from .functions import summarize_file



logger = logging.getLogger(__name__)


@easy(['path_dict', 'status'])
def setup_output(proc_dir, track_time = True):
    '''
    Set up summary output.

    Args:
        proc_dir (str): 
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
    data_dir = os.path.join(out_dir, 'data')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    # make dictionary of paths
    path_dict = {}
    for ft in file_types:
        if ft in extra_summary_header:
            header = summary_header + extra_summary_header[ft]
        else:
            header = summary_header
        path_dict[ft] = setup_csv(ft, data_dir, header)
    logger.info('Created output directory and initialized files.')
    status = 'setup_output_finished'
    return(path_dict, status)        
