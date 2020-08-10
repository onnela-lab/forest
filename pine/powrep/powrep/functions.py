'''Functions for working with raw Beiwe power state data

'''
import os
import logging
import pandas as pd
from beiwetools.helpers.functions import read_json
from beiwetools.helpers.process import clean_dataframe, stack_frames
from beiwetools.helpers.trackers import CategoryTracker
from .headers import raw_header, keep_header


logger = logging.getLogger(__name__)


# load events
this_dir = os.path.dirname(__file__)
events = read_json(os.path.join(this_dir, 'events.json'))


def summarize_pow(path, opsys, event_tracker = CategoryTracker(), 
                  n_events = 0, unknown_headers = 0, unknown_events = []):
    '''
    Summarize a raw Beiwe power state file.

    Args:
        path (str): Path to a raw Beiwe power state file.
        opsys (str): Either 'iOS' or 'Android'.
        event_tracker (beiwetools.helpers.trackers.CategoryTracker):
            Online tracker for a categorical variable.
        n_events (int): Running count of events.
        unknown_headers (int): Running count of unrecognized headers.
        unknown_events (list): Running list of unrecognized events.
            
    Returns:
        None                
    '''
    filename = os.path.basename(path)
    data = pd.read_csv(path)
    # check header
    opsys_header = raw_header[opsys]
    if not list(data.columns) == opsys_header:
        unknown_headers += 1
        logger.warning('Header is not recognized: %s' % filename)
    # check events
    opsys_events = list(events[opsys].keys())
    temp_events = list(data.event)
    for e in temp_events:
        if not e in opsys_events:
            unknown_events.append(e)
            logger.warning('Unrecognized event \"%s\" in %s' % (e, filename))
    # update running stats
    event_tracker.update(temp_events)
    n_events += len(temp_events)


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







def extract_pow(paths, opsys):
    '''

    Args:        
        paths (list): List of paths to raw Beiwe power state files.
        opsys (str): 'Android' or 'iOS'
        
    Returns:

    '''

    # Read and stack the files.

        
    #
    pass


