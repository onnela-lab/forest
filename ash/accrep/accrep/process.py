'''Get window-by-window summaries of raw Beiwe accelerometer data.

    - Do this AFTER accrep.calibration.Calibration.do().

'''
import os
import logging
import numpy as np
from scipy.stats import percentileofscore
from beiwetools.helpers.time import (local_now, to_timestamp, 
                                     filename_time_format, hour_ms)
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (setup_directories, read_json, 
                                          summary_statistics)
from beiwetools.helpers.process import get_windows
from beiwetools.helpers.classes import WriteQueue
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.templates import ProcessTemplate
from .functions import read_acc, transformations, metrics


logger = logging.getLogger(__name__)


def most_recent(timestamp, dictionary):
    try:
        keys = [int(k) for k in dictionary.keys()]
        before = [k for k in keys if k <= timestamp]
        return(before[-1])
    except:
        logger.warning('Unable to find most recent reference file for %s.'
                       % timestamp)


def update_reference(calpath, device_os, minimum_obs, do_transformations,
                     get_summaries, get_metrics, n_resamples):
    refdata = read_acc(calpath, device_os)    
    refsum = {}
    # do transformations
    for t in do_transformations:
        tt = transformations[t]
        tt.apply(refdata)
    # estimate reference distributions for summaries
    for s in get_summaries:
        try:
            function_name, variable = s                        
            function = summary_statistics[function_name]
            refsum[s] = np.full(n_resamples, np.nan)
            for i in range(n_resamples):
                sample = refdata.sample(n = minimum_obs, replace = True)
                refsum[s][i] = function(sample[variable])
        except:
            logger.warning('Unable to process reference data for %s, %s.' %s)
    # get reference values for metrics
    for m in get_metrics:
        try:
            if m in ['ai', 'ai0']:
                refsum[m] = np.sum([np.var(refdata[a], ddof=1) 
                                    for a in ['x', 'y', 'z']])
            if m in ['ari', 'ari0']:
                refsum[m] = np.sum([summary_statistics['iqr'](refdata[a]) 
                                    for a in ['x', 'y', 'z']])    
        except:
            logger.warning('Unable to process reference data for %s.' %m)
    # estimate reference distributions for metrics
    for m in get_metrics:
        try:
            function = metrics[m]
            refsum[m+'_'+'samples'] = np.full(n_resamples, np.nan)
            for i in range(n_resamples):
                sample = refdata.sample(n = minimum_obs, replace = True)
                refsum[m+'_'+'samples'][i] = function(sample, refsum[m])
        except:
            logger.warning('Unable to process reference data for %s.' %m)
    return(refsum)


@easy(['windows_dir'])
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


@easy(['device_os', 'file_paths', 'calibration_dict', 'writer'])
def setup_user(user_id, project, calibration_dir, calibration_metric,
               windows_dir, get_summaries, get_metrics, n_resamples):
    # get os
    device_os = project.lookup['os'][user_id]   
    # get file paths
    file_paths = project.data[user_id].passive['accelerometer']['files']
    # get calibration dictionary
    calibration_dict = read_json(os.path.join(calibration_dir, 
                                 user_id + '.json'))[calibration_metric]
    # set up writer
    header = ['timestamp'] + \
             [s[1]+'_'+s[0] for s in get_summaries] + \
             get_metrics + \
             [s[1]+'_'+s[0]+'_'+'percentile' for s in get_summaries] + \
             [m+'_'+'percentile' for m in get_metrics]
    writer = WriteQueue(user_id, windows_dir, header, lines_per_csv = 10000)    
    logger.info('Finished setting up for user %s.' % user_id)
    return(device_os, file_paths, calibration_dict, writer)


@easy([])
def process_user(user_id, device_os, file_paths, calibration_dict, 
                 writer, window_length_ms, minimum_s, minimum_obs,
                 do_transformations, get_summaries, get_metrics,
                 n_resamples):
    refkey = None
    refsum  = None
    for f in file_paths: 
        try:
            data = read_acc(f, device_os)
            # check if calibration needs to be updated
            t = to_timestamp(os.path.basename(f).split('.')[0],
                             filename_time_format)
            this_refkey = most_recent(t, calibration_dict)
            if this_refkey != refkey:
                refkey = this_refkey
                refsum = update_reference(calibration_dict[str(refkey)],
                                          device_os, minimum_obs, 
                                          do_transformations,
                                          get_summaries, get_metrics, 
                                          n_resamples)
                logger.info('Updated reference data.')
            # split file into windows and get summaries
            windows = get_windows(data, t, t+hour_ms, window_length_ms)
            for w in windows:
                if not windows[w] is None:
                    if len(windows[w]) >= minimum_obs:                        
                        temp = data.iloc[windows[w]]
                        t0 = temp.timestamp.iloc[0]
                        t1 = temp.timestamp.iloc[-1]
                        if (t1-t0)/1000 >= minimum_s:
                            sample = temp.sample(n = minimum_obs, 
                                                 replace = True)
                            to_write = [w]
                            percentiles = []
                            # do transformations
                            for t in do_transformations:
                                tt = transformations[t]
                                tt.apply(sample)
                            # get summaries
                            for s in get_summaries:
                                function_name, variable = s                        
                                function = summary_statistics[function_name]
                                value = function(sample[variable])
                                to_write.append(value)
                                percentiles.append(percentileofscore(refsum[s], 
                                                                     value))
                            # get metrics
                            for m in get_metrics:
                                function = metrics[m]
                                value = function(sample, refsum[m])
                                to_write.append(value)
                                percentiles.append(percentileofscore(
                                        refsum[m+'_'+'samples'], value))
                            writer.write(to_write + percentiles)
        except:
            logger.warning('Unable to process: %s' \
                           % os.path.basename(f))    
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
                        window_length_ms,
                        minimum_s, minimum_obs,
                        n_resamples,
                        calibration_dir, 
                        calibration_metric = 'total_variance',                        
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

        n_resamples (int):            

        calibration_dir (str):

        calibration_metric (str):

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
        'n_resamples': n_resamples,
        'calibration_dir': calibration_dir,
        'calibration_metric': calibration_metric,
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
