''' Functions for reformatting raw Fitabase files.
'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (setup_directories, setup_csv, 
                                          write_to_csv)
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import format_records_header, format_data_header
from .functions import format_data, format_sleep


logger = logging.getLogger(__name__)


@easy(['records_dir', 'data_dir'])
def setup_output(proc_dir, track_time):
    '''
    Set up summary output.

    Args:
        proc_dir (str):  Location for fitrep output.
        track_time (bool): If True, output to a timestamped folder.
        
    Returns:
        records_dir (str): Where to write format records.
        data_dir (str):  Where to write reformatted data.
    '''
    out_dir = os.path.join(proc_dir, 'fitrep', 'format')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    records_dir = os.path.join(out_dir, 'records')
    data_dir = os.path.join(out_dir, 'data')
    log_dir = os.path.join(out_dir, 'log')       
    setup_directories([out_dir, records_dir, data_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    # success message
    logger.info('Created output directories.')
    return(records_dir, data_dir)        


@easy(['user_dict', 'followup_range'])
def setup_user(registry, user_id, followup_ranges, file_types,
               records_dir, data_dir):
    # get followup dates
    if user_id in followup_ranges:
        followup_range = followup_ranges[user_id]
    else:
        followup_range = None    
    # set up input/output paths
    ud = os.path.join(data_dir, user_id)
    setup_directories(ud)
    user_dict = {}
    for ft in file_types:
        if not ft == 'syncEvents':
            fp = registry.lookup[user_id][ft]
            fr = setup_csv(user_id, records_dir, format_records_header)
            dh = format_data_header
            if ft == 'minuteSleep': dh += ['resync']
            fd = setup_csv(ft, ud, dh)
            user_dict[ft] = (fp, fr, fd)
    logger.info('Finished setup for user %s' % user_id)    
    return(user_dict, followup_range)


@easy([])
def format_user(user_dict, followup_range):
    for file_type in user_dict:
        file_path, records_path, data_path = user_dict[file_type]
        try:
            if file_type == 'minuteSleep':
                records = format_sleep(file_path, data_path, followup_range)
            else:
                records = format_data(file_path, data_path, followup_range, 
                                      file_type)
            write_to_csv(records_path, records)            
        except:
            logger.warning('Unable to format data for %s.' % file_type)    


def pack_format_kwargs(user_ids, proc_dir, file_types, registry, 
                       track_time = True, id_lookup = {}):
    '''
    Packs kwargs for fitrep.Format.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        file_types (list): 
            List of supported Fitabase file types (str) to format.
        registry (fitrep.classes.FitabaseRegistry): 
            Registry for a folder of Fitabase files.
        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Fitabase identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Fitabase identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To format data: Format.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'file_types': file_types,
        'registry': registry,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Format = ProcessTemplate.create(__name__,
                                [setup_output], [setup_user], 
                                [format_user], [], [])


