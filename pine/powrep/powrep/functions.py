'''Functions for working with raw Beiwe power state data

'''
import os
import logging
import numpy as np
import pandas as pd
from collections import OrderedDict
from beiwetools.helpers.functions import read_json
from beiwetools.helpers.process import clean_dataframe, stack_frames
from beiwetools.helpers.trackers import CategoryTracker
from .headers import raw_header, keep_header


logger = logging.getLogger(__name__)


# load events
this_dir = os.path.dirname(__file__)
events = read_json(os.path.join(this_dir, 'events.json'))


def summarize_pow(path, opsys):
    '''
    Summarize a raw Beiwe power state file.

    Args:
        path (str): Path to a raw Beiwe power state file.
        opsys (str): Either 'iOS' or 'Android'.
        
    Returns:
        
        
    '''
    data = pd.read_csv(path)
    opsys_events = events[opsys]    
        
    
    pass


def read_pow(path, opsys, keep = raw_header,
             clean_args = (True, True, True)):
    '''
    Open a raw Beiwe power state file.
    
    Args:
        path (str): Path to the file.
        opsys (str): 'Android' or 'iOS'
        keep (list): Which columns to keep.
        clean_args (tuple): Args for clean_dataframe().
        
    Returns:
        df (DataFrame): Pandas dataframe of power state data.
    '''
    if isinstance(keep, dict): opsys_keep = keep[opsys]
    df = pd.read_csv(path, usecols = opsys_keep)
    clean_dataframe(df, *clean_args)
    return(df)







def proc_pow(paths, opsys):
    '''

    Args:        
        paths (list): List of paths to raw Beiwe power state files.
        opsys (str): 'Android' or 'iOS'
        keep (list): Which columns to keep.
        clean_args (tuple): Args for final clean_dataframe() before return.
        
    Returns:
        df (DataFrame): Pandas dataframe of power state data.    
    '''

    # Read and stack the files.

        
    #
    pass


def split_events():
    '''
    Get a 1D-array for a category of event.
    '''

    pass