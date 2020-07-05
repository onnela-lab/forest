''' Functions for working with raw Fitabase data.
'''
import os
import logging
import pandas as pd
from collections import OrderedDict
from beiwetools.helpers.time import date_only, date_time_format, reformat_datetime
from beiwetools.helpers.decorators import easy
#from beiwetools.helpers.templates import template


logger = logging.getLogger(__name__)


# Fitabase file headers
from headers import raw_


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


@easy(('summary', 'file_types'))
def summarize_directory(fitabase_dir):
    '''
    Get available identifiers and filetypes from a directory of raw Fitabase files.

    Args:
        fitabase_dir (str):  Directory containing raw Fitabase data.

    Returns:
        summary(OrderedDict):
            Keys are sorted fitabase identifiers.
            Values are OrderedDicts.
            summary[fitabase_id]['file_type'] is a list of available file types.
            summary[fitabase_id]['file_path'] is a corresponding list of paths.            
        file_types(list):
            Sorted list of all file types found among all fitabase identifiers.
    '''
    file_names = sorted(os.listdir(fitabase_dir))
    summary = OrderedDict()
    file_types = []
    start = []
    end = []
    for f in file_names:
        p = parse_filename.to_dict(f)
        fid = p['fitabase_id']
        ft  = p['file_type']
        fp  = os.path.join(fitabase_dir, f)
        file_types.append(ft)
        start.append(p['local_start'])
        end.  append(p['local_end'])
        if not fid in summary:
            summary[fid] = OrderedDict({
                                        'file_types': [ft], 
                                        'file_paths': [fp]
                                        })
        else:
            summary[fid]['file_types'].append(ft)
            summary[fid]['file_paths'].append(fp)
    if len(set(start)) > 1:  logger.warning('Multiple start dates.')
    if len(set(end))   > 1:  logger.warning('Multiple end dates.')
    file_types = sorted(list(set(file_types)))
    return(summary, file_types)


