''' Extract events from raw Beiwe survey timings files.

    - Note that survrep.Extract.do() only works for iPhone data.

'''
import os
import logging
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.time import local_now
from beiwetools.helpers.functions import setup_directories, join_lists
from beiwetools.helpers.process import stack_frames
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate


logger = logging.getLogger(__name__)


@easy(['data_dir'])
def setup_output(proc_dir, dir_names, track_time):
    out_dir = os.path.join(proc_dir, 'survrep', 'summary')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir = os.path.join(out_dir, 'records')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    logger.info('Created output directory and initialized files.')
    return(data_dir)        


@easy(['file_paths', 'user_dir'])
def setup_user(user_id, project, data_dir, dir_names):    
    # get operating system
    opsys = project.lookup['os'][user_id]    
    # assemble list of all survey_timings file paths        
    udata = project.data[user_id]
    key_list = [('survey_timings', sid) for sid in dir_names]    
    file_paths_dict = udata.assemble(key_list)
    list_of_paths = list(file_paths_dict.values())
    file_paths = join_lists(list_of_paths)
    # set up output directory for user
    user_dir = os.path.join(data_dir, user_id)
    setup_directories(user_dir)
    logger.info('Finished setting up parameters for user %s.' % user_id)
    return(file_paths, user_dir)


@easy([])
def extract_user(user_id, user_dir, file_paths, get_events):
    # read all timings files
    try:
        data = stack_frames(file_paths)[['timestamp', 'question id', 
                                         'survey id', 'event']]
    except:
        logger.warning('Unable to read files.')
    try:
        if get_events is None:
            get_events = list(set(data.event))
        # split up and write events
        for e in get_events:
            temp = data[data.event == e][['timestamp', 'question id', 'survey id']]
            temp.to_csv(os.path.join(user_dir, e + '.csv'), index = False)        
    except:
        logger.warning('Unable to process events.')
    logger.info('Finished summarizing data for user %s.' % user_id)


def pack_summary_kwargs(user_ids, proc_dir, dir_names, project,
                        get_events = None, track_time = True, id_lookup = {}):
    '''
    Packs kwargs for survrep.Summary.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        dir_names (list): List of directory names (str) to summarize.
            Note that directory names are survey identifiers.        
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        get_events (list): List of events to extract.  If None, All events 
            are extracted.            
        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Beiwe identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Beiwe identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To run a summary: Summary.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'dir_names': dir_names, 
        'project': project,
        'get_events': get_events,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Extract = ProcessTemplate.create(__name__,
                                 [setup_output], [setup_user], 
                                 [extract_user], 
                                 [], [])