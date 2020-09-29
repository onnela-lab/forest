'''Extract displacement information from raw Beiwe GPS files.

'''
import os
import logging
import numpy as np
from beiwetools.helpers.time import (local_now, to_timestamp, 
                                     filename_time_format, hour_ms, min_ms)
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import setup_directories
from beiwetools.helpers.process import get_windows
from beiwetools.helpers.classes import WriteQueue
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import displacement_header
from .functions import read_gps, gps_project, get_displacement


logger = logging.getLogger(__name__)


@easy(['hours_dir', 'windows_dir'])
def setup_output(proc_dir, track_time):
    # set up directories
    out_dir = os.path.join(proc_dir, 'gps', 'displacement')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    hours_dir   = os.path.join(out_dir, 'hours')
    windows_dir = os.path.join(out_dir, 'windows')
    log_dir     = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, hours_dir, windows_dir, log_dir])    
    # set up logging output
    log_to_csv(log_dir)
    logger.info('Created output directories.')
    return(hours_dir, windows_dir)        


@easy(['file_paths', 'accuracy_cutoff', 'hours_writer', 'windows_writer'])
def setup_user(user_id, project, accuracy_cutoff_dict,
               hours_dir, windows_dir):
    # get file paths
    file_paths = project.data[user_id].passive['gps']['files']
    # get accuracy cutoff
    accuracy_cutoff = accuracy_cutoff_dict[user_id]
    # set up writers
    hours_writer = WriteQueue(user_id, hours_dir, displacement_header,
                              lines_per_csv = 1000)
    windows_writer = WriteQueue(user_id, windows_dir, displacement_header,
                                lines_per_csv = 1000)    
    logger.info('Finished setting up for user %s.' % user_id)
    return(file_paths, accuracy_cutoff, hours_writer, windows_writer)


@easy([])
def displacement_user(user_id, file_paths, accuracy_cutoff, 
                      hours_writer, windows_writer, project_with,
                      window_length_ms):
    for f in file_paths:
        try:
            data = read_gps(f, accuracy_cutoff = accuracy_cutoff)
            if len(data) > 1: # need at least two observations
                mean_lat  = np.mean(data.latitude)
                mean_long = np.mean(data.longitude)
                h = project_with/2
                location_ranges = dict(zip(['latitude', 'longitude', 'altitude'],
                                           [[mean_lat-h, mean_lat+h],
                                           [mean_long-h, mean_long+h],
                                           [np.min(data.altitude),
                                            np.max(data.altitude)]]))
                tests = [np.max(data.latitude)<=location_ranges['latitude'][1],
                         np.min(data.latitude)>=location_ranges['latitude'][0],
                         np.max(data.longitude)<=location_ranges['longitude'][1],    
                         np.min(data.longitude)>=location_ranges['longitude'][0]]
                if all(tests):
                    new = gps_project(data, location_ranges, action = 'new')
                    t = to_timestamp(os.path.basename(f).split('.')[0],
                                     filename_time_format)
                    d = get_displacement(new.loc[0], new.iloc[-1])
                    hours_writer.write([t] + d)
                    windows = get_windows(new, t, t + hour_ms, 
                                          window_length_ms)
                    for w in windows:
                        if not windows[w] is None:
                            first, last = windows[w][0], windows[w][-1]
                            d = get_displacement(new.loc[first], 
                                                 new.loc[last])
                            windows_writer.write([w] + d)
        except:
            logger.warning('Unable to get displacements: %s' \
                           % os.path.basename(f))    
    logger.info('Finished extracting displacements for user %s.' % user_id)


def pack_displacement_kwargs(user_ids, proc_dir, project,
                             accuracy_cutoff_dict,
                             project_with = 2,
                             window_length_ms = min_ms,
                             track_time = True, id_lookup = {}):
    '''
    Packs kwargs for Displacement.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        accuracy_cutoff_dict (dict):  Keys are user_ids, values are 
            accuracy cutoffs for gpsrep.functions.read_gps().
        project_with (int or float): The size (degrees) of the sides of the
            box used to project GPS data.
        window_length_ms (int): Size of uniform windows to calculate
            displacements over, e.g. 60*1000 for 1-minute windows.
        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Fitabase identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Fitabase identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To extract displacements: Displacement.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'project': project,
        'accuracy_cutoff_dict': accuracy_cutoff_dict,
        'project_with': project_with,
        'window_length_ms': window_length_ms,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Displacement = ProcessTemplate.create(__name__,
                                      [setup_output], [setup_user], 
                                      [displacement_user], [], [])


