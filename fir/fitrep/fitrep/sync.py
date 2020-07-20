''' Functions for processing raw Fitabase syncEvents files.
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


@easy(['records_path', 'offsets_dir'])
def setup_output(proc_dir, track_time):
    '''
    Set up summary output.

    Args:
        proc_dir (str): 
        track_time (bool): If True, output to a timestamped folder.
        
    Returns:
        path_dict (dict): 
            Keys are file types, values are paths.
    '''
    out_dir = os.path.join(proc_dir, 'fitrep', 'sync')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    records_dir = os.path.join(out_dir, 'records')
    offsets_dir = os.path.join(out_dir, 'offsets')
    log_dir = os.path.join(out_dir, 'log')       
    setup_directories([out_dir, records_dir, offsets_dir, log_dir])
    # set up logging output
    log_to_csv(log_dir)
    # initialize records file
    records_path = setup_csv(ft, records_dir, header)
    # success message
    logger.info('Created output directory and initialized files.')
    return(records_path, offsets_dir)        


@easy([])
def setup_user()






def setup_kwargs(user_ids, proc_dir, registry, 


                 track_time = True, id_lookup = {}):
    '''
    Packs kwargs for fitrep.Sync.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
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
            To run a summary: FitrepSummary.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'registry': registry,
        
        
        
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Sync = ProcessTemplate.create(__name__,
                              [setup_output], [], 
                              [summarize_user], 
                              [write_user_records], [])


