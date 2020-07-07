''' Functions for working with raw Fitabase data.
'''
import os
import logging
import pandas as pd
from collections import OrderedDict
from beiwetools.helpers.time import date_only, date_time_format, reformat_datetime
from beiwetools.helpers.decorators import easy


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


def summarize_minute_file(file_path, check_offset = True):
    '''
    Get some info from a raw file of minute-by-minute Fitabase data.

    Args:
        file_path (str): Path to raw Fitabase data file.
        check_offset (bool): 
            If true, count how many observations aren't synced to UTC minutes.
        
    Returns:
        summary (list): List of summary items.
    '''
    data = pd.read_csv(file_path)



    
    return(summary)


def summarize_sync_file(file_path):
    '''
    
    '''
    pass    
    
