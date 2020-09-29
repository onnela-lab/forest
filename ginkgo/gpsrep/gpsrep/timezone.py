'''Extract timezone information from raw Beiwe GPS files.

'''
import os
import logging
from beiwetools.helpers.time import local_now, get_timezone, get_offset
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (setup_directories, setup_csv, 
                                          write_to_csv, write_json)
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import timezone_summary_header
from .functions import read_gps


logger = logging.getLogger(__name__)


@easy(['tz_dir', 'offset_dir', 'records_path'])
def setup_output(proc_dir, track_time):
    # set up directories
    out_dir = os.path.join(proc_dir, 'gps', 'timezone')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    tz_dir      = os.path.join(out_dir, 'timezones')
    offset_dir  = os.path.join(out_dir, 'offsets')
    records_dir = os.path.join(out_dir, 'records')
    log_dir     = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, tz_dir, offset_dir, records_dir, log_dir])    
    # set up records file
    records_path = setup_csv('records', records_dir, timezone_summary_header)
    # set up logging output
    log_to_csv(log_dir)
    logger.info('Created output directories.')
    return(tz_dir, offset_dir, records_path)        


@easy(['file_paths', 'accuracy_cutoff'])
def setup_user(user_id, project, accuracy_cutoff_dict):
    # get file paths
    file_paths = project.data[user_id].passive['gps']['files']
    # get accuracy cutoff
    accuracy_cutoff = accuracy_cutoff_dict[user_id]
    logger.info('Finished setting up for user %s.' % user_id)
    return(file_paths, accuracy_cutoff)


@easy([])
def timezone_user(user_id, file_paths, accuracy_cutoff,
                  tz_dir, offset_dir, records_path):
    tz_dict = {}
    offset_dict = {}
    last_tz = None
    last_offset = None
    for f in file_paths:
        try:
            data = read_gps(f, accuracy_cutoff = accuracy_cutoff)
            if len(data) > 0:
                index = data.accuracy.idxmin() # index with best accuracy
                best = data.loc[index]                
                tz = get_timezone(best.latitude, best.longitude)
                offset = get_offset(best.timestamp, tz)
                # what to do with the first file
                if last_tz is None:
                    last_tz = tz
                    tz_dict[int(best.timestamp)] = tz
                if last_offset is None:
                    last_offset = offset
                    offset_dict[int(best.timestamp)] = offset
                # check if dictionaries need to be updated
                if tz != last_tz:
                    tz_dict[int(best.timestamp)] = tz
                    last_tz = tz
                if offset != last_offset:
                    offset_dict[int(best.timestamp)] = offset
                    last_offset = offset
        except:
            logger.warning('Unable to get timezone info: %s' \
                           % os.path.basename(f))    
    # count timezones & offsets
    n_tz = len(set(tz_dict.values()))
    n_tz_transitions = len(tz_dict)
    n_os = len(set(offset_dict.values()))
    n_os_transitions = len(offset_dict)
    # save dictionaries
    write_json(tz_dict, user_id, tz_dir)
    write_json(offset_dict, user_id, offset_dir)
    # write counts
    write_to_csv(records_path, 
                 [user_id, 
                  n_tz, n_tz_transitions, 
                  n_os, n_os_transitions])
    logger.info('Finished extracting timezones for user %s.' % user_id)


def pack_timezone_kwargs(user_ids, proc_dir, project,
                        accuracy_cutoff_dict,
                        track_time = True, id_lookup = {}):
    '''
    Packs kwargs for gpsrep.Timezone.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        accuracy_cutoff_dict (dict):  Keys are user_ids, values are 
            accuracy cutoffs for gpsrep.functions.read_gps().
        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Fitabase identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Fitabase identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To run a summary: Timezone.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'project': project,
        'accuracy_cutoff_dict': accuracy_cutoff_dict,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Timezone = ProcessTemplate.create(__name__,
                                  [setup_output], [setup_user], 
                                  [timezone_user], [], [])


