''' Functions for processing raw Fitabase syncEvents files.
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


@easy(['file_path', 'followup_range', 'user_record'])
def setup_user(registry, user_id, followup_ranges):
    file_name = registry.lookup[user_id]['syncEvents'] 
    file_path= os.path.join(registry.directory, file_name)
    if user_id in followup_ranges:
        followup_range = followup_ranges[user_id]
    else:
        followup_range = None    
    user_record = [user_id]
    logger.info('Finished setup for user %s' % user_id)    
    return(file_path, followup_range, user_record)


@easy([])
def sync_user(user_id, file_path, followup_range, 
              user_record, global_intersync_s):
    sync_summary, local_dts, utc_dts = read_sync(file_path, 
                                                 followup_range)
    user_record += sync_summary
    user_record += process_intersync(user_id, utc_dts, global_intersync_s)
    user_record += process_offsets(user_id, local_dts, utc_dts)
    logger.info('Processed sync data for user %s.' % user_id)        


@easy([])
def write_user_records(user_id, records_path, user_record):
    write_to_csv(records_path, user_record)    
    logger.info('Updated sync records for user %s.' % user_id)        

    
@easy([])
def write_sync_records(global_intersync_s, records_path):
    is_s = np.diff(np.hstack(global_intersync_s))
    min_intersync_s = np.min(is_s)
    max_intersync_s = np.max(is_s)
    mean_intersync_s = np.mean(is_s)
    median_intersync_s = np.median(is_s)    
    global_summary = [min_intersync_s, max_intersync_s, 
                      mean_intersync_s, median_intersync_s]
    to_write = ['global'] + ['']*7 + global_summary + ['']*3
    write_to_csv(records_path, to_write)
    logger.info('Updated global sync records.')        


def pack_sync_kwargs(user_ids, proc_dir, registry, 
                     followup_ranges = {},
                     track_time = True, id_lookup = {}):
    '''
    Packs kwargs for fitrep.Sync.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        registry (fitrep.classes.FitabaseRegistry): 
            Registry for a folder of Fitabase files.
        followup_ranges (dict):
            Keys are identifiers from user_ids.
            Values are tuples (start, end), the local datetime strings in 
            date_time_format for the beginning and ending of followup for 
            the corresponding user id.
            If empty, all observations are included.
        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Fitabase identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Fitabase identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To process syncEvents files: Sync.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'registry': registry,
        'followup_ranges': followup_ranges,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Sync = ProcessTemplate.create(__name__,
                              [setup_output], 
                              [setup_user], 
                              [sync_user], 
                              [write_user_records], 
                              [write_sync_records])


