''' Check configuration files against raw Beiwe survey timings files.

'''
import os
import logging
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.time import local_now
from beiwetools.helpers.functions import setup_directories, setup_csv, write_to_csv
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import compatibility_header 
from .functions import check_compatibility


logger = logging.getLogger(__name__)


@easy(['compatibility_path'])
def setup_output(proc_dir, dir_names, config_dict, track_time):
    out_dir = os.path.join(proc_dir, 'survrep', 'summary')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir = os.path.join(out_dir, 'records')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    # set up compatibility CSV
    compatibility_path = setup_csv('configuration_compatibility',
                                   data_dir, compatibility_header)
    logger.info('Created output directory and initialized files.')
    return(compatibility_path)        


@easy([])
def setup_user(user_id, project, dir_names, config):
    
    # assemble dictionary of file paths        
    
    
    # get study configuration
    
    
    pass


@easy([])
def summarize_user():
    pass


@easy([])
def write_user_records():
    pass


def pack_summary_kwargs(user_ids, proc_dir, 
                        dir_names, project, config_dict = {},
                        track_time = True, id_lookup = {}):
    '''
    Packs kwargs for survrep.Compatibility.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        dir_names (list): List of directory names (str) to summarize.
            Note that directory names are survey identifiers.        
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        config_dict (dict): Optional.
            If id_lookup = {}, keys are from user_ids.
            Otherwise, keys are id_lookup{user_ids}
            Values are beiwetools.configread.classes.BeiweConfig objects.
        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Beiwe identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Beiwe identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To run a compatibility check: Compatibility.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'dir_names': dir_names, 
        'project': project,
        'config_dict': config_dict,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Compatibility = ProcessTemplate.create(__name__,
                                       [setup_output], [setup_user], 
                                       [summarize_user], 
                                       [write_user_records], [])
