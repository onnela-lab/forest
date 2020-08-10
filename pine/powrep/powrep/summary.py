'''Summarize raw Beiwe power state data files.

'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (read_json, setup_directories, 
                                          setup_csv, write_to_csv)
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import summary_header
from .functions import events, summarize_pow


logger = logging.getLogger(__name__)


@easy(['records_path'])
def setup_output(proc_dir, track_time):
    '''
    Set up summary output.

    Args:
        proc_dir (str): Location for powrep output.
        track_time (bool): If True, output to a timestamped folder.
        
    Returns:
        records_path (str): 
            Path to records CSV.
    '''
    out_dir = os.path.join(proc_dir, 'powrep', 'summary')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir = os.path.join(out_dir, 'records')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    # set up records CSV
    iOS_events = list(events['iOS'].keys())
    Android_events = list(events['Android'].keys())
    header = summary_header + iOS_events + Android_events
    records_path = setup_csv('records', data_dir, header)
    logger.info('Created output directory and initialized files.')
    return(records_path)        


@easy(['file_paths', 'opsys'])
def setup_user(user_id, userdata, deviceinfo, ):

    pass


@easy([])
def summarize_user(records_path):

    
    pass


def write_user_records():
    
    pass


def pack_summary_kwargs(user_ids, proc_dir, file_types, registry, 
                         track_time = True, id_lookup = {}):
    '''
    Packs kwargs for fitrep.Summary.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        file_types (list): 
            List of supported Fitabase file types (str) to summarize.
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
            To run a summary: Summary.do(**kwargs)
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


Summary = ProcessTemplate.create(__name__,
                                 [setup_output], [], 
                                 [summarize_user], 
                                 [write_user_records], [])

