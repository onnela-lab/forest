''' Check a configuration file against raw Beiwe survey timings files.

'''
import os
import logging
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.time import local_now
from beiwetools.helpers.functions import (setup_directories, setup_csv, 
                                          write_to_csv, join_lists)
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import compatibility_header 
from .functions import check_compatibility


logger = logging.getLogger(__name__)


@easy(['compatibility_dict'])
def setup_output(proc_dir, config, track_time):
    out_dir = os.path.join(proc_dir, 'survrep', 'compatibility', 
                           config.name.replace(' ', '_'))
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir = os.path.join(out_dir, 'records')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    # set up compatibility CSV
    compatibility_dict = {}
    for sid in config.survey_ids['tracking']:
        compatibility_dict[sid] = setup_csv(sid, data_dir, 
                                            compatibility_header)
    logger.info('Created output directory and initialized files.')
    return(compatibility_dict)        


@easy(['opsys', 'file_paths'])
def setup_user(user_id, project):    
    # get operating system
    opsys = project.lookup['os'][user_id]    
    # assemble dictionary of file paths        
    udata = project.data[user_id]
    sids = udata.surveys['survey_timings']['ids'].keys()
    key_list = [('survey_timings', sid) for sid in sids]
    file_paths = udata.assemble(key_list)
    list_of_lists = [file_paths[k] for k in file_paths]
    file_paths = join_lists(list_of_lists)
    logger.info('Finished setting up parameters for user %s.' % user_id)
    return(opsys, file_paths)
 
    
@easy([])
def compatibility_user(opsys, file_paths, config):
    for file_path in file_paths:
        check_compatibility(opsys, file_path, config,
                            absent_survey = [],
                            absent_question = [],
                            disagree_question_type = [],
                            disagree_question_text = [],
                            disagree_answer_options = [])
    



@easy([])
def write_user_records():
    pass


def pack_compatibility_kwargs(user_ids, proc_dir, 
                              dir_names, project, config_dict = {},
                              track_time = True, id_lookup = {}):
    '''
    Packs kwargs for survrep.Compatibility.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        config (beiwetools.configread.classes.BeiweConfig): 
            Representation of a Beiwe study configuration.
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
        'project': project,
        'config': config,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Compatibility = ProcessTemplate.create(__name__,
                                       [setup_output], [setup_user], 
                                       [compatibility_user], 
                                       [write_user_records], [])
