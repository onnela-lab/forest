'''Summarize variables from raw Beiwe GPS files.

'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import setup_directories, setup_csv, write_to_csv
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.trackers import (Histogram, SamplingSummary, 
                                         RangeTracker)
from beiwetools.helpers.templates import ProcessTemplate
from .headers import variable_ranges
from .functions import read_gps


logger = logging.getLogger(__name__)


@easy(['out_file'])
def setup_output(proc_dir, track_time):
    # setup directories
    out_dir = os.path.join(proc_dir, 'gps', 'range')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir = os.path.join(out_dir, 'data')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, log_dir])
    # setup output file
    header = ['user_id'] + variable_ranges
    out_file = setup_csv('records', data_dir, header)    
    # set up logging output
    log_to_csv(log_dir)
    logger.info('Created output directories.')
    return(out_file)        


@easy(['file_paths', 'trackers'])
def setup_user(user_id, project):
    # get file paths
    file_paths = project.data[user_id].passive['gps']['files']
    # setup trackers
    trackers = {}
    for v in ['latitude', 'longitude', 'altitude', 'accuracy']:
        trackers[v] = RangeTracker()
    return(file_paths, trackers)


@easy([])
def range_user(user_id, file_paths, trackers):
    for f in file_paths:
        try:
            data = read_gps(f)
            for v in trackers:
                trackers[v].update(data[v])
        except:
            logger.warning('Unable to get variable ranges: %s' \
                           % os.path.basename(f))    
    logger.info('Finished getting variable ranges for user %s.' % user_id)


@easy([])
def write_user(user_id, trackers, out_file):
    



def pack_range_kwargs(user_ids, proc_dir, project,
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
                                 [range_user], [], [])



def range_gps(filepath, 
              sampling_summary = SamplingSummary(hour_ms),
              latitude_range = RangeTracker(),
              longitude_range = RangeTracker(), 
              altitude_range = RangeTracker(),                  
              accuracy_range = RangeTracker()):
    '''
    Get variable ranges from one file of raw GPS data.

    Args:
        filepath (str): Path to the file.
        samplingsum ():
        latrange ():
        longrange (): 
        altrange (): 
        accyrange ():
            
    Returns:
        None
    '''
    try:
        # read data
        data = read_gps(filepath)
        # update trackers
        sampling_summary.update(data.timestamp)
        latitude_range.update(data.latitude)
        longitude_range.update(data.longitude)
        altitude_range.update(data.altitude)
        accuracy_range.update(data.accuracy)    
    except:
        basename = os.path.basename(filepath)
        logger.warning('Unable to get variable ranges: %s' % basename)

    