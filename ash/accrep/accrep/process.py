'''Get window-by-window summaries of raw Beiwe accelerometer data.

    - Do this AFTER accrep.calibration.Calibration.do().

'''
'''Process data from raw Beiwe GPS files.

'''
import os
import logging
import numpy as np
from beiwetools.helpers.time import (local_now, to_timestamp, 
                                     filename_time_format, hour_ms, min_ms)
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (setup_directories, setup_csv, 
                                          write_to_csv)
from beiwetools.helpers.process import get_windows
from beiwetools.helpers.classes import WriteQueue
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .headers import process_header
from .functions import read_acc


logger = logging.getLogger(__name__)


@easy(['data_dir'])
def setup_output(proc_dir, track_time):
    # set up directories
    out_dir = os.path.join(proc_dir, 'accrep', 'process')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    windows_dir = os.path.join(out_dir, 'windows')
    log_dir     = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, windows_dir, log_dir])    
    # set up logging output
    log_to_csv(log_dir)
    logger.info('Created output directories.')
    return(windows_dir)        


@easy(['file_paths', 'cal_dict'])
def setup_user(user_id, project, calibration_dict, windows_dir,
               get_summaries, get_metrics):
    # get os
    device_os = project.lookup['os'][user_id]   
    # get file paths
    file_paths = project.data[user_id].passive['accelerometer']['files']
    # get calibration dictionary
    cal_dict = calibration_dict[user_id]
    # set up user file
    header = ['timestamp'] + \
             [s[1]+'_'+s[0] for s in get_summaries] + \
             get_metrics
    user_file = setup_csv(user_id, windows_dir, header)
    # set up writers
    #windows_writer = WriteQueue(user_id, windows_dir, process_header,
    #                            lines_per_csv = 1000)    
    logger.info('Finished setting up for user %s.' % user_id)
    return(device_os, file_paths, cal_dict, user_file)


@easy([])
def process_user(user_id, device_os, file_paths, cal_dict, 
                 user_file, window_length_ms):
    for f in file_paths:
        refpath = None
        refsum  = None
        try:
            data = read_gps(f, device_os)
            t = to_timestamp(os.path.basename(f).split('.')[0],
                             filename_time_format)
            #last_refpath = [k for k in cal_dict if t - int(k) > 0][-1]
            #if refpath != last_refpath:
            #    refpath = last_refpath
            #    refsum  = get_ref(refpath, do_transformations,
            #                      get_summaries, get_metrics)
            windows = get_windows(new, t, t + hour_ms, window_length_ms)
            for w in windows:
                if not windows[w] is None:
                    first, last = windows[w][0], windows[w][-1]
                    
                    
                    f = 
                    
                    
                    d = get_displacement(new.loc[first], 
                                         new.loc[last])
                    windows_writer.write([w] + d)
        except:
            logger.warning('Unable to process: %s' \
                           % os.path.basename(f))    
    logger.info('Finished processing user %s.' % user_id)





            # update trackers
            for v in trackers:
                trackers[v].update(data[v])
            # do transformations
            for t in do_transformations:
                tt = transformations[t]
                tt.apply(data)
            # get summaries
            for s in get_summaries:
                function_name, variable = s                        
                function = summary_statistics[function_name]
                to_write.append(function(data[variable]))
            # get metrics
            for m in get_metrics:
                function = metrics[m]
                to_write.append(function(data))
            # write results
            write_to_csv(user_file, to_write)
        except:
            logger.warning('Unable to process file: %s' \
                           % os.path.basename(f))       
    # save intersample times
    trackers['timestamp'].save(sampling_dir, user_id)
    # write summary
    to_write = [user_id, len(file_paths), n_observations,
                trackers['timestamp'].frequency(),
                trackers['x'].stats['min'], trackers['x'].stats['max'],
                trackers['y'].stats['min'], trackers['y'].stats['max'],
                trackers['z'].stats['min'], trackers['z'].stats['max']]
    write_to_csv(records_file, to_write)
    logger.info('Finished processing accelerometer data for user %s.' % user_id)
    return()









all_transformations = ['vecmag', 'enmo', 'ampdev', 'sumamp', 'ppz', 'zmo']
all_summaries = [('mean', 'ppz'),   ('mean', 'zmo'), 
                 ('std', 'vecmag'), ('range', 'vecmag'), 
                 ('std', 'enmo'),   ('range', 'enmo'), 
                 ('std', 'ampdev'), ('range', 'ampdev'),
                 ('std', 'x'), ('range', 'x'), 
                 ('std', 'y'), ('range', 'y'),
                 ('std', 'z'), ('range', 'z'),
                 ('std', 'sumamp'), ('range', 'sumamp')]
all_metrics = ['ai', 'ai0', 'ari', 'ari0']


def pack_process_kwargs(user_ids, proc_dir, project,

                        window_length_ms = min_ms,
                        minimum_s, minimum_obs,

                        
                        do_transformations = all_transformations,
                        get_summaries = all_summaries,
                        get_metrics = all_metrics,



                        track_time = True, id_lookup = {}):
    '''
    Packs kwargs for Process.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        window_length_ms (int): Size of uniform windows to calculate
            displacements over, e.g. 60*1000 for 1-minute windows.

        minimum_s (int):
        
        minimum_obs (int):

        calibration_dict (dict):

        do_transformations (list):

        get_summaries (list):
            (<variable>, <summary statistic>)

        get_metrics (list):


        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Fitabase identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Fitabase identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To process: Process.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'project': project,
        'window_length_ms': window_length_ms,        
        'minimum_s': minimum_s,
        'minimum_obs': minimum_obs,
        'calibration_dict': calibration_dict,
        'do_transformations': do_transformations,
        'get_summaries': get_summaries,
        'get_metrics': get_metrics,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Process = ProcessTemplate.create(__name__,
                                 [setup_output], [setup_user], 
                                 [process_user], [], [])






