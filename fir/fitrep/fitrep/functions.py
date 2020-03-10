''' Tools for working with raw Fitabase data.
'''
import os
import logging
import pandas as pd
from collections import OrderedDict
from beiwetools.helpers.time import date_only, date_time_format, reformat_datetime
from beiwetools.helpers.functions import coerce_to_dict
from beiwetools.helpers.decorators import easy
#from beiwetools.helpers.templates import template


logger = logging.getLogger(__name__)


# Fitabase time formats
fitabase_filename_time_format = '%Y%m%d'
fitabase_data_time_format = ''


@easy(('fitabase_id', 'filetype', 'local_start', 'local_end'))
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
    filetype = '_'.join(s)
    local_start = reformat_datetime(start, fitabase_filename_time_format, date_only) + ' 00:00:00'
    local_end = reformat_datetime(end, fitabase_filename_time_format, date_only) + ' 23:59:59'
    return(fitabase_id, filetype, local_start, local_end)




class FitabaseDirectory():
    '''
    Represent available data in a directory of fitabase data sets.
    
    '''
    pass