''' Functions for working with raw Fitabase data.
'''
import os
import datetime
import logging
from collections import OrderedDict
import numpy as np
import pandas as pd
from beiwetools.helpers.time import (UTC, day_s, hour_s, 
                                     date_only, data_time_format, 
                                     reformat_datetime)
from beiwetools.helpers.process import to_1Darray
from beiwetools.helpers.decorators import easy
from .headers import datetime_header


logger = logging.getLogger(__name__)


# Fitabase time formats
fitabase_filename_date_format = '%Y%m%d'
fitabase_date_format = '%m/%d/%Y'    # without zero padding for %m, %d
fitabase_time_format = '%I:%M:%S %p' # without zero padding for %I
fitabase_datetime_format = fitabase_date_format + ' ' + fitabase_time_format


def fbdt_to_dt(fitabase_datetime):
    '''
    Convert a Fitabase datetime string to a datetime object.

    Args:
        fitabase_datetime (str): A datetime string in fitabase_datetime_format,
            without leading zeros.
        
    Returns:
        dt (datetime.datetime): The corresponding unaware datetime object.
    '''
    dt = datetime.datetime.strptime(fitabase_datetime, fitabase_datetime_format)
    return(dt)


@easy(['fitabase_id', 'file_type', 'local_start', 'local_end'])
def parse_filename(filename):
    '''
    Get info from a fitabase file name.

    Args:
        filename (str): Name of fitabase data file.

    Returns:
        fitabase_id (str): fitabase user ID.
        filetype (str): 
        local_start, local_end (str):
    '''
    # get rid of extension
    n = filename.split('.')[0]
    # split at underscores
    s = n.split('_')
    # get identifier:
    fitabase_id = s.pop(0)
    # get dates:
    end = s.pop()
    start = s.pop()
    # whatever's left is the variable name:
    file_type = '_'.join(s)
    local_start = reformat_datetime(start, fitabase_filename_date_format, date_only) + ' 00:00:00'
    local_end = reformat_datetime(end, fitabase_filename_date_format, date_only) + ' 23:59:59'
    return(fitabase_id, file_type, local_start, local_end)


def summarize_file(file_path, file_type):
    '''
    Get some info from a raw file of minute-by-minute Fitabase data.

    Args:
        file_path (str): Path to raw Fitabase data file.
        file_type (str): A key from headers.raw_header.
        
    Returns:
        summary (list): List of summary items.
    '''    
    file_name = os.path.basename(file_path)
    data = pd.read_csv(file_path)    
    datetimes = list(data[datetime_header[file_type]])
    if len(datetimes) > 0:
        first_observation = datetimes[0]
        last_observation = datetimes[-1]
        n_observations = len(datetimes)
    else: 
        first_observation, last_observation, n_observations = None, None, 0
    try:
        followup_s = (fbdt_to_dt(last_observation) - fbdt_to_dt(first_observation)).total_seconds()
        followup_days = followup_s / day_s
    except:
        followup_days = None
        logger.warning('Unable to calculate followup duration: %s' % file_name)
    summary = [first_observation, last_observation, 
               n_observations, followup_days]
    if file_type == 'minuteSleep':
        # get number of offsets        
        if len(datetimes) > 0:
            n_offsets = len([d for d in datetimes if ':30 ' in d])
            p_offsets = n_offsets / n_observations
        else: 
            n_offsets, p_offsets = None, None
        summary += [n_offsets, p_offsets]    
    if file_type == 'syncEvents':
        # get service provider
        p = list(set(data['Provider'])) # should be one unique provider
        if len(p) == 1:
            provider = p[0]
        else:
            provider = 'Unknown'
            logger.warning('Unknown service provider: %s' % file_name)
        # get device
        d = list(set(data['DeviceName'])) # should be one unique device
        if 'MobileTrack' in d:
            logger.warning('Contains smartphone app data: %s' % file_name)            
        if len(d) == 1:
            device = d[0]
        else:
            device = 'Unknown'
            logger.warning('Unknown device: %s' % file_name)
        summary += [provider, device]                                                          
    return(summary)


def read_sync(file_path):
    '''
    Load a syncEvents file and get some information.
    
    Args:
        file_path (str): Path to a syncEvents file.

    Returns:
        sync_summary (list): List of summary items (str).
        local_dts (list): 
            List of local sync times as datetime.datetime objects.
        utc_dts (list): 
            List of UTC sync times as datetime.datetime objects.
    '''
    data = pd.read_csv(file_path)    
    file_name = os.path.basename(file_path)
    # get provider
    p = list(set(data.Provider)) # should be one unique provider 
    if len(p) == 0:
        provider = 'missing'
        logger.warning('Missing service provider: %s' % file_name)
    elif len(p) == 1:
        provider = p[0]
    else: 
        provider = 'multiple'
        logger.warning('Multiple service providers logged: %s' % file_name)
    # get device
    d = list(set(data.DeviceName)) # should be one unique device
    if len(d) == 0:
        device = 'missing'
        logger.warning('Missing device log: %s' % file_name)
    elif len(d) == 1:
        device = d[0]
    else:
        device = 'multiple'
        logger.warning('Multiple devices logged: %s' % file_name)
    # count syncs
    n_syncs = len(data)
    n_app_syncs = len(data[data.DeviceName == 'MobileTrack'])
    # process datetimes
    local = list(data.DateTime)
    local_dts = [fbdt_to_dt(fbdt) for fbdt in local]
    utc = list(data.SyncDateUTC)
    utc_dts = [fbdt_to_dt(fbdt) for fbdt in utc]    
    # return summary & datetime lists
    sync_summary = [provider, device, n_syncs, n_app_syncs]
    return(sync_summary, local_dts, utc_dts)


def process_intersync(utc_dts, intersync_tracker_s):
    first_sync = utc_dts[0].strftime(data_time_format)
    last_sync = utc_dts[-1].strftime(data_time_format)
    # convert to Unix timestamps
    utc_ts = to_1Darray([UTC.localize(dt).timestamp() for dt in utc_dts])
    # process intersync times
    is_s = np.diff(utc_ts)
    min_intersync_s = np.min(is_s)
    max_intersync_s = np.max(is_s)
    mean_intersync_s = np.mean(is_s)
    median_intersync_s = np.median(is_s)
    # update global tracker
    intersync_tracker_s.update(is_s)
    # return summary
    intersync_summary = [first_sync, last_sync, 
                         min_intersync_s, max_intersync_s, 
                         mean_intersync_s, median_intersync_s]
    return(intersync_summary)
    

def get_offset(local_dt, utc_dt):
    s = (local_dt-utc_dt).total_seconds()
    offset = round(s/hour_s)
    return(offset)


def process_offsets(local_dts, utc_dts):
    offsets = [get_offset(local_dts[i], utc_dts[i]) for i in range(len(local_dts))]
    offset_dict = OrderedDict()
    last_offset = None
    for i in range(len(offsets)):
        if last_offset is None or last_offset != offsets[i]:
            offset_dict[local_dts[i]] = offsets[i]            
        last_offset = offsets[i]
    n_transitions = len(offset_dict) - 1
    n_offsets = len(set(offset_dict.values()))
    # return summary
    offset_summary = [n_transitions, n_offsets]
    return(offset_summary, offset_dict)


def process_data(file_path, offset_dict):
    pass


def process_sleep(file_path, offset_dict):
    pass    
    
    
    
    
    
