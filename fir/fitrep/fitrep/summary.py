''' Functions for summarizing a directory of raw Fitabase data.
'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import setup_directories, setup_csv, write_to_csv
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import summary_header, extra_summary_header, raw_header
from .functions import summarize_file


logger = logging.getLogger(__name__)


@easy(['path_dict', 'status'])
def setup_output(proc_dir, file_types, track_time = True):
    '''
    Set up summary output.

    Args:
        proc_dir (str): 
        file_types (list): List of Fitabase file types.
        track_time (bool): If true, output to a timestamped folder.
        
    Returns:
        path_dict (dict): 
            Keys are file types, values are paths.
    '''
    if 'syncEvents' not in file_types: file_types.append('syncEvents')
    out_dir = os.path.join(proc_dir, 'fitrep', 'summary')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir = os.path.join(out_dir, 'data')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    # make dictionary of paths
    path_dict = {}
    for ft in file_types:
        if ft in extra_summary_header:
            header = summary_header + extra_summary_header[ft]
        else:
            header = summary_header
        path_dict[ft] = setup_csv(ft, data_dir, header)
    logger.info('Created output directory and initialized files.')
    status = 'setup_output_finished'
    return(path_dict, status)        


@easy(['summary_dict', 'status'])
def summarize_user(user_id, file_types, registry):
    summary_dict = {}
    for ft in file_types:
        if not ft in raw_header:
            logger.warning('The fitrep package does not handle %s files.' % ft)
        else:
            fn = registry.lookup[user_id][ft] 
            fp = os.path.join(registry.directory, fn)
            summary_dict[ft] = [user_id] + summarize_file(fp, ft)
    logger.info('Summarized data for user %s.' % user_id)
    status = '%s_summary_finished' % user_id
    return(summary_dict, status)


@easy(['status'])
def write_user_records(user_id, summary_dict, path_dict):
    for k in summary_dict:
        write_to_csv(path_dict[k], summary_dict[k])        
    logger.info('Updated records for user %s.' % user_id)
    status = '%s_records_finished' % user_id
    return(status)

    
# For FitrepSummary.do, the process_kwargs dictionary must include these keys:
#   'proc_dir' (str): 
#       Path to folder where processed data can be written.
#   'file_types' (list): 
#       List of supported Fitabase file types (str) to summarize.
#   'registry' (fitrep.classes.FitabaseRegistry): 
#       Registry for a folder of Fitabase files.
FitrepSummary = ProcessTemplate.create(__name__,
                                       [setup_output], [], 
                                       [summarize_user], 
                                       [write_user_records], [])


