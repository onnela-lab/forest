''' Summarize raw Beiwe survey timings files.

'''
import os
import logging
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.time import local_now
from beiwetools.helpers.functions import setup_directories, setup_csv, write_to_csv
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import summary_header 
from .functions import summarize_timings


logger = logging.getLogger(__name__)


@easy(['records_dict'])
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
    # set up records CSVs
    records_dict = {}
    for sid in dir_names:    
        records_dict[sid] = setup_csv(sid, data_dir, summary_header)
    logger.info('Created output directory and initialized files.')
    return(records_dict)        


@easy(['opsys', 'file_paths_dict'])
def setup_user(user_id, project, dir_names):    
    # get operating system
    opsys = project.lookup['os'][user_id]    
    # assemble dictionary of file paths        
    udata = project.data[user_id]
    key_list = [('survey_timings', sid) for sid in dir_names]
    file_paths_dict = udata.assemble(key_list)
    logger.info('Set up parameters for user %s.' % user_id)
    return(opsys, file_paths_dict)


@easy(['summary'])
def summarize_user(user_id, opsys, dir_names, file_paths_dict):
    summary = {}
    for sid in dir_names:
        n_events = []
        unknown_header = []
        unknown_events = []
        foreign_survey = []
        file_paths = file_paths_dict[('survey_timings', sid)]
        for file_path in file_paths:
            try:
                summarize_timings(opsys, file_path, n_events, 
                                  unknown_header, unknown_events, foreign_survey)            
            except:
                logger.warning('Unable to summarize file %s for user %s.' % (file_path, user_id))
        try: first = os.path.basename(file_paths[0])
        except: first = None
        try: last = os.path.basename(file_paths[-1])
        except: last = None
        summary[sid] = [user_id, opsys,
                        first,
                        last,
                        len(file_paths),            
                        sum(n_events), 
                        len(set(unknown_header)),
                        len(set(unknown_events)), 
                        len(set(foreign_survey))]
    logger.info('Finished summarizing data for user %s.' % user_id)
    return(summary)


@easy([])
def write_user_records(user_id, summary, records_dict):
    for sid in summary:
        to_write = summary[sid]
        write_to_csv(records_dict[sid], to_write)
    logger.info('Finished writing records for user %s.' % user_id)                


def pack_summary_kwargs(user_ids, proc_dir, dir_names, project,
                        track_time = True, id_lookup = {}):
    '''
    Packs kwargs for survrep.Summary.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        dir_names (list): List of directory names (str) to summarize.
            Note that directory names are survey identifiers.        
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
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
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Summary = ProcessTemplate.create(__name__,
                                 [setup_output], [setup_user], 
                                 [summarize_user], 
                                 [write_user_records], [])
