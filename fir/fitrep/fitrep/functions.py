''' Functions for working with raw Fitabase data.
'''
import os
import logging
import pandas as pd
from beiwetools.helpers.time import date_only, date_time_format, reformat_datetime
from beiwetools.helpers.decorators import easy
from .headers import datetime_header


logger = logging.getLogger(__name__)


# Fitabase time formats
fitabase_filename_date_format = '%Y%m%d'
fitabase_date_format = ''
fitabase_time_format = ''
fitabase_datetime_format = ''


@easy(('fitabase_id', 'file_type', 'local_start', 'local_end'))
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
    data = pd.read_csv(file_path)    
    datetimes = list(data[datetime_header[file_type]])
    if len(datetimes) > 0:
        first_observation = datetimes[0]
        last_observation = datetimes[-1]
        n_observations = len(datetimes)
    else: 
        first_observation, last_observation, n_observations = None, None, 0
    summary = [first_observation, last_observation, n_observations]
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
            file_name = os.path.basename(file_path)
            logger.warning('Unknown service provider: %s' % file_name)
        # get device
        d = list(set(data['DeviceName'])) # should be one unique device
        if 'MobileTrack' in d:
            logger.warning('Contains smartphone app data: %s' % file_name)            
        if len(d) == 1:
            device = d[0]
        else:
            device = 'Unknown'
            file_name = os.path.basename(file_path)
            logger.warning('Unknown device: %s' % file_name)
        summary += [provider, device]                                                          
    return(summary)


def to_Beiwe_datetime(fitabase_datetime, utc_offset = 0):
    '''
    Convert a local fitabase datetime to a Beiwe UTC datetime.

    Args:
        
    Returns:

    
    '''
    
    
    pass



