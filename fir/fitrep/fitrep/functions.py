''' Functions for working with raw Fitabase data.
'''
import os
import datetime
import logging
from collections import OrderedDict
import numpy as np
import pandas as pd
from beiwetools.helpers.time import (UTC, day_s, hour_s, 
                                     date_only, date_time_format, 
                                     datatime_to_dt, reformat_datetime)
from beiwetools.helpers.functions import write_to_csv
from beiwetools.helpers.process import to_1Darray
from .headers import datetime_header, content_header


logger = logging.getLogger(__name__)


# Fitabase time formats
fitabase_filename_date_format = '%Y%m%d'
fitabase_date_format = '%m/%d/%Y'    # without zero padding for %m, %d
fitabase_time_format = '%I:%M:%S %p' # without zero padding for %I
fitabase_datetime_format = fitabase_date_format + ' ' + fitabase_time_format


def fbdt_to_dt(fitabase_datetime, timezone = None):
    '''
    Convert a Fitabase datetime string to a datetime object.

    Args:
        fitabase_datetime (str): A datetime string in fitabase_datetime_format,
            without leading zeros.
        timezone (timezone from pytz.tzfile or Nonetype):
            Optionally, return a timezone-aware datetime object.
            
    Returns:
        dt (datetime.datetime): 
            The corresponding datetime object.
    '''
    dt = datetime.datetime.strptime(fitabase_datetime, fitabase_datetime_format)
    if not timezone is None:
        dt = timezone.localize(dt)
    return(dt)


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


def read_sync(file_path, followup_range):
    '''
    Load a syncEvents file and get some information.
    
    Args:
        file_path (str): Path to a syncEvents file.
        followup_range (tuple or Nonetype): A pair of local datetime strings in 
            date_time_format for the beginning and ending of followup.
            If None, no observations are dropped.

    Returns:
        sync_summary (list): List of summary items (str).
        local_dts (list): 
            List of local sync times as datetime.datetime objects.
        utc_dts (list): 
            List of UTC sync times as datetime.datetime objects.
    '''
    data = pd.read_csv(file_path) 
    file_name = os.path.basename(file_path)    
    # drop duplicate sync times
    # may occur rarely when timezone changes, look in SyncDateUTC
    data.drop_duplicates(subset = 'SyncDateUTC',
                         inplace = True)
    # drop observations outside of followup range
    # replace this with smart_read()
    if not followup_range is None:
        t0 = datatime_to_dt(followup_range[0])
        t1 = datatime_to_dt(followup_range[1])
        dts = [fbdt_to_dt(fbdt, UTC) for fbdt in data.SyncDateUTC]
        i_to_keep = [i for i in range(len(dts)) if dts[i] >= t0 and dts[i] < t1]
        data = data.iloc[i_to_keep]
        data.set_index(np.arange(len(data)), inplace = True)
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
    dd = [dn for dn in d if type(dn) is str] # drop nan
    if d != dd:
        d = dd
        logger.warning('Incomplete device log: %s' % file_name)    
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
    try:
        n_app_syncs = len(data[data.DeviceName == 'MobileTrack'])
    except:
        n_app_syncs = ''
        logger.warning('Unable to subset dataframe: %s' % file_name)        
    # process datetimes
    local = list(data.DateTime)
    local_dts = [fbdt_to_dt(fbdt) for fbdt in local]
    utc = list(data.SyncDateUTC)
    utc_dts = [fbdt_to_dt(fbdt) for fbdt in utc]    
    # return summary & datetime lists
    sync_summary = [provider, device, n_syncs, n_app_syncs]
    return(sync_summary, local_dts, utc_dts)


def process_intersync(user_id, utc_dts, global_intersync_s):
    if len(utc_dts) > 0:
        first_sync = utc_dts[0].strftime(date_time_format)
        last_sync = utc_dts[-1].strftime(date_time_format)
        # convert to Unix timestamps
        utc_ts = to_1Darray([UTC.localize(dt).timestamp() for dt in utc_dts])
        # get intersync times
        is_s = np.diff(utc_ts)
    else:
        first_sync, last_sync = '', ''
        is_s = []
        logger.warning('No sync events for user %s.' % user_id )
    if len(is_s) > 0:    
        # update global tracker
        global_intersync_s.append(is_s)
        # get stats
        min_intersync_s = np.min(is_s)
        max_intersync_s = np.max(is_s)
        mean_intersync_s = np.mean(is_s)
        median_intersync_s = np.median(is_s)
        is_stats = [min_intersync_s, max_intersync_s, 
                    mean_intersync_s, median_intersync_s]
    else: is_stats = ['']*4
    # return summary
    intersync_summary = [first_sync, last_sync] + is_stats
    return(intersync_summary)
    

def get_offset(local_dt, utc_dt):
    '''
    Get UTC offset in hours.
    '''
    s = (local_dt-utc_dt).total_seconds()
    offset = round(s/hour_s)
    return(offset)


def process_offsets(user_id, local_dts, utc_dts):
    if len(utc_dts) > 0:
        offsets = [get_offset(local_dts[i], utc_dts[i]) for i in range(len(local_dts))]
        offset_dict = OrderedDict()
        offset_dict[local_dts[0]] = offsets[0]
        last_offset = offsets[0]
        for i in range(len(offsets)):
            if last_offset != offsets[i]:
                offset_dict[local_dts[i]] = offsets[i]            
            last_offset = offsets[i]
        n_transitions = len(offset_dict) - 1
        n_offsets = len(set(offset_dict.values()))
        # return summary
        offset_summary = [n_transitions, n_offsets]
    else:
        offset_summary = ['', '']
        logger.warning('No intersync times for user %s.' % user_id )
    # don't return the offset dictionary
    return(offset_summary)


def smart_read(file_path, followup_range):    
    '''
    Load a raw Fitabase file and drop observations that are outside of 
    the followup period.

    Args:
        file_path (str): Path to raw Fitabase data file.
        followup_range (tuple or Nonetype): A pair of local datetime strings in 
            date_time_format for the beginning and ending of followup.
            If None, no observations are dropped.

    Returns:
        data (pd.DataFrame): The contents of the raw Fitabase file, excluding
            observations that are out of range.
    '''
    data = pd.read_csv(file_path)
    try:
        if not followup_range is None:
            t0 = datatime_to_dt(followup_range[0])
            t1 = datatime_to_dt(followup_range[1])
            dts = [fbdt_to_dt(fbdt, UTC) for fbdt in data.SyncDateUTC]
            i_to_keep = [i for i in range(len(dts)) if dts[i] >= t0 and dts[i] < t1]
            data = data.iloc[i_to_keep]
            data.set_index(np.arange(len(data)), inplace = True)
    except:
        logger.warning('Unable to apply followup range: %s' % file_path)
    return(data)


def format_data(file_path, data_path, followup_range, file_type):
    data = smart_read(file_path, followup_range)
    time_name = datetime_header[file_type]
    var_name = content_header[file_type]    
    local_dts = [fbdt_to_dt(fbdt) for fbdt in data[time_name]]
    var = list(data[var_name])
    for i in range(len(local_dts)):
        line = [local_dts[i].strftime(date_time_format), var[i]]
        write_to_csv(file_path, line)
    followup_s = (local_dts[-1] - local_dts[0]).total_seconds()
    followup_days = followup_s / day_s
    first_observation = local_dts[0].strftime(date_time_format)
    last_observation = local_dts[-1].strftime(date_time_format)
    records = [file_type, first_observation, last_observation,
               len(data), followup_days]
    return(records)


def format_sleep(file_path, data_path, followup_range):
    data = smart_read(file_path, followup_range)
    async_dts = [fbdt_to_dt(fbdt) for fbdt in data['date']]
    async_var = list(data['value'])
    # resync
    already_synced = []
    first  = OrderedDict({}) #  first 30 seconds of every minute
    second = OrderedDict({}) # second 30 seconds of every minute
    for i in range(len(async_dts)):
        if async_dts[i].second == 0:
            t = async_dts[i].strftime(date_time_format)
            first[t]  = async_var[i]
            second[t] = async_var[i]
            already_synced.append(t)
        if async_dts[i].second == 30:
            delta = datetime.timedelta(seconds = 30)
            before = (async_dts[i] - delta).strftime(date_time_format)
            after  = (async_dts[i] + delta).strftime(date_time_format)
            first[after]   = async_var[i]
            second[before] = async_var[i]
    first_observation = None
    last_observation = None
    n_observations = 0
    for t in first:
        # only write minutes in which first & second agree        
        if first[t] == second[t]:
            line = [t, first[t], t in already_synced]
            write_to_csv(file_path, line)            
            if first_observation is None: first_observation = t
            last_observation = t
            n_observations += 1
    first_dt = datetime.datetime.strptime(first_observation, date_time_format)
    last_dt =  datetime.datetime.strptime(last_observation, date_time_format)
    followup_s = (last_dt - first_dt).total_seconds()
    followup_days = followup_s / day_s
    records = ['minuteSleep', first_observation, last_observation,
               followup_days]
    return(records)