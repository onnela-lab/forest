''' Functions for reformatting raw Fitabase files.
'''
import os
import logging
import numpy as np
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (setup_directories, setup_csv, 
                                          write_to_csv)
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import sync_records_header
from .functions import read_sync, process_intersync, process_offsets


logger = logging.getLogger(__name__)


@easy(['records_path', 'global_intersync_s'])
def setup_output(proc_dir, track_time):
    '''
    Set up summary output.

    Args:
        proc_dir (str): 
        track_time (bool): If True, output to a timestamped folder.
        
    Returns:
        records_path (str): Path to sync records CSV file.
        offsets_dir (str): Directory for saving offset dictionaries.
        global_intersync_s (list): Empty list to contain
    '''
    out_dir = os.path.join(proc_dir, 'fitrep', 'sync')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    records_dir = os.path.join(out_dir, 'records')
    #offsets_dir = os.path.join(out_dir, 'offsets')
    log_dir = os.path.join(out_dir, 'log')       
    setup_directories([out_dir, records_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    # initialize records file
    records_path = setup_csv('records', records_dir, sync_records_header)
    # initialize global intersync tracker
    global_intersync_s =  []   
    # success message
    logger.info('Created output directory and initialized files.')
    return(records_path, global_intersync_s)        
