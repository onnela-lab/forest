'''Extract events from raw Beiwe power state data files.

'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (read_json, setup_directories, 
                                          setup_csv, write_to_csv)
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import extract_header
from .functions import extract_pow


logger = logging.getLogger(__name__)


# load events
this_dir = os.path.dirname(__file__)
events = read_json(os.path.join(this_dir, 'events.json'))


# organize events into powrep variables
def organize_events():
    categories = {}
    names = {}
    for opsys in events.keys():
        categories[opsys] = {}
        names[opsys] = []
        for k in events[opsys].keys():
            cat, value = events[opsys][k]
            if not cat in categories[opsys]:    
                categories[opsys][cat] = {}
                names[opsys].append(cat)
            categories[opsys][cat][value] = k
    return(categories, names)
variables, var_names = organize_events()            


@easy(['data_dir'])
def setup_output(proc_dir, user_ids, track_time):
    # setup directories
    out_dir = os.path.join(proc_dir, 'powrep', 'extract')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir = os.path.join(out_dir, 'data')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, log_dir])
    user_dirs = []
    for u in user_ids:
        user_dirs.append(os.path.join(data_dir, u))
    setup_directories(user_dirs)
    # set up logging output
    log_to_csv(log_dir)
    logger.info('Created output directories.')
    return(data_dir)        


@easy(['file_paths', 'opsys', 'data_paths'])
def setup_user(user_id, get_variables, data_dir, project):
    # get file paths and operating system
    file_paths = project.data[user_id].passive['power_state']['files']
    opsys = project.lookup['os'][user_id]
    # initialize user data files
    user_dir = os.path.join(data_dir, user_id)    
    if opsys == 'iOS': header = extract_header + ['level']
    else: header = extract_header
    data_paths = {}
    for v in get_variables:
        data_paths[v] = setup_csv(v, user_dir, header)        
    logger.info('Finished setting up for user %s.' % user_id)
    return(file_paths, opsys, data_paths)


@easy([])
def extract_user(user_id, opsys, get_variables, 
                 file_paths, data_paths):
    for path in file_paths:
        try:
            line_dict = extract_pow(path, opsys)
        except:
            line_dict = {}
            logger.warning('Unable to extract: %s' % os.path.basename(path))    
        for k in line_dict:
            if k in get_variables:
                lines = line_dict[k]
                for line in lines:
                    write_to_csv(data_paths[k], line)
    logger.info('Finished extracting variables for user %s.' % user_id)


def pack_extract_kwargs(user_ids, proc_dir, project,
                        get_variables = [],
                        track_time = True, id_lookup = {}):
    '''
    Packs kwargs for powrep.Extract.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        get_variables (list): List of event variables to extract.
            Default is an empty list, in which case all event variables 
            will be extracted.
        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Fitabase identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Fitabase identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To run an extraction: Extract.do(**kwargs)
    '''
    if len(get_variables) == 0:
        get_variables = var_names['iOS'] + var_names['Android']
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'project': project,
        'get_variables': get_variables,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Extract = ProcessTemplate.create(__name__,
                                 [setup_output], [setup_user], 
                                 [extract_user], [], [])