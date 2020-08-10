'''Summarize raw Beiwe power state data files.

'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import setup_directories, setup_csv, write_to_csv
from beiwetools.helpers.trackers import CategoryTracker
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


@easy(['file_paths', 'opsys', 'event_tracker', 
       'n_events', 'unknown_headers', 'unknown_events'])
def setup_user(user_id, project):
    file_paths = project.data[user_id].passive['power_state']['files']
    opsys = project.lookup['os'][user_id]
    event_tracker = CategoryTracker()
    n_events = []
    unknown_headers = []
    unknown_events = []
    logger.info('Finished setting up for user %s.' % user_id)
    return(file_paths, opsys, event_tracker, 
           n_events, unknown_headers, unknown_events)


@easy(['to_write'])
def summarize_user(user_id, opsys, file_paths,
                   event_tracker, n_events, 
                   unknown_headers, unknown_events):
    for path in file_paths:
        try:
            summarize_pow(path, opsys, event_tracker = event_tracker, 
                          n_events = n_events, 
                          unknown_headers = unknown_headers, 
                          unknown_events = unknown_events)
        except:
            logger.warning('Unable to summarize: %s' % os.path.basename(path))    
    summary = [user_id, opsys, sum(n_events), len(file_paths),
               os.path.basename(file_paths[0]),
               os.path.basename(file_paths[-1]),
               len(unknown_headers), len(set(unknown_events))]
    iOS_events = list(events['iOS'].keys())
    Android_events = list(events['Android'].keys())
    event_keys = iOS_events + Android_events
    event_summary = []
    for k in event_keys:
        if k in event_tracker.stats['counts']:        
            event_summary.append(event_tracker.stats['counts'][k])
        else: event_summary.append('')
    to_write = [summary + event_summary]
    logger.info('Summarized data for user %s.' % user_id)
    return(to_write)


@easy([])
def write_user_records(user_id, to_write, records_path):
    write_to_csv(records_path, to_write)    
    logger.info('Updated records for user %s.' % user_id)


def pack_summary_kwargs(user_ids, proc_dir, project, 
                         track_time = True, id_lookup = {}):
    '''
    Packs kwargs for powrep.Summary.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
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
        'project': project,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Summary = ProcessTemplate.create(__name__,
                                 [setup_output], [setup_user], 
                                 [summarize_user], 
                                 [write_user_records], [])