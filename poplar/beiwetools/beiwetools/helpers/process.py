'''Functions for processing Beiwe data.

'''
import os
import logging
import numpy as np
from pandas import Series, DataFrame, read_csv, concat

from collections import OrderedDict

from .time import filename_time_format, to_timestamp


logger = logging.getLogger(__name__)


def to_1Darray(x, name = None):
    '''
    Convert list or pandas series to a 1D numpy array.
    If x is a pandas dataframe, converts x[name] to a numpy array.
    '''
    if   isinstance(x, DataFrame): 
        try:
            return(x[name].to_numpy())
        except:
            logger.warning('Must provide a column name in the dataframe.')
    elif isinstance(x, Series):     return(x.to_numpy())
    elif isinstance(x, list):       return(np.array(x))
    elif isinstance(x, np.ndarray): return(x)
    else: 
        logger.warning('Unable to convert this type to numpy array.')


def directory_range(directory):
	'''
	Finds the first and last date/time in a directory that contains Beiwe data.  Searches all subdirectories.

	Args:
		directory (str): Path to a directory that contains files named using the Beiwe convention (%Y-%m-%d %H_%M_%S.csv).

	Returns:
		first (str): Earliest date/time (%Y-%m-%d %H_%M_%S) found among filenames in the directory..
		last (str): Last date/time (%Y-%m-%d %H_%M_%S) found among filenames in the directory.
	'''
	start = True
	for path, dirs, filenames in os.walk(directory):
		for f in filenames:
			dt = f.split('.')[0]
			if start:
				first = dt
				last = dt
				start = False
			else:
				if dt < first: first = dt
				if dt > last: last = dt
	return(first, last)


def clean_dataframe(df, 
                    drop_duplicates = True, 
                    sort = True, 
                    update_index = True):
    '''
    Clean up a pandas dataframe.
    
    Args:
        df (DataFrame): A pandas dataframe.        
        drop_duplicates (bool):  Drop extra copies of rows, if any exist.
        sort (bool):  Sort by timestamp.  
            If True, df must have a timestamp column.
        update_index (bool):  Set index to range(len(df)).
            
    Returns:
        None
    '''
    if drop_duplicates:
        df.drop_duplicates(inplace = True)
    if sort:
        try:
            df.sort_values(by = ['timestamp'], inplace = True)
        except:
            logger.exception('Unable to sort by timestamp.')
    if update_index:
        df.set_index(np.arange(len(df)), inplace = True)


def stack_frames(paths, clean_args = (True, True, True), 
                 reader = read_csv, **kwargs):
    '''
    Read multiple CSVs into DataFrames and stack them into a single DataFrame.

    Args:
        paths (list): List of paths (strings) to CSVs.
            It's assumed that the paths correspond to chronologically 
            ordered files.  If not, be sure to set clean_args[1] = True 
            to return sorted observations.
        clean_args (tuple): Args for final clean_dataframe() before return.
        reader (func): Function for reading CSVs into DataFrames.
            Input should include a filepath.
            Output should be a pandas DataFrame.
        kwargs: Keyword arguments for reader.   These might include lists of 
            columns to keep and how to clean each DataFrame before stacking.  
        
    Returns:
        stack (DataFrame):
    '''    
    df_list = []
    for p in paths:
        df_list.append(reader(p, **kwargs))
    df = concat(df_list, ignore_index = True)
    if True in clean_args:
        clean_dataframe(df, *clean_args)               
    return(df)


def summarize_filepath(filepath = None, ndigits = 3, basename_only = False):
    '''
    Get some basic information from a single raw Beiwe data file.
    
    Args:
        filepath (str): Path to a raw Beiwe data file.
            If None, then just returns a list of summary names.
        ndigits (int):  For rounding disk_MB.
    
    Returns:
        timestamp (int): Millisecond timestamp.  The start of the hour covered by the file.
        filename (str): The basename of the file.
        disk_MB (float): Size of the file on disk in Megabytes.
    '''
    returns = ['timestamp', 'file', 'disk_MB']
    if filepath is None:
        return(returns)
    else:
        basename = os.path.basename(filepath)
        if basename_only: file = basename
        else: file = filepath
        timestamp = to_timestamp(basename.split('.')[0], filename_time_format)
        return(timestamp, file, round(os.path.getsize(filepath)/(1000**2), ndigits))
        

def summarize_timestamps(t = None, ndigits = 3, unit = 'seconds'):
    '''
    Get some basic information from a list, Series or 1-D array of timestamps.
    
    Args:
        t (list, Series or 1-D array): List of millisecond timestamps.
            If None, then just returns a list of summary names.
        ndigits (int):  For rounding duration.
        unit (str): 'seconds' or 'ms'.
    
    Returns:
        first_observation, last_observation (int):  
            Millisecond timestamps of the first and last observations.
        n_observations (int): Number of rows in the csv.
        duration_minutes (float): Elapsed time (minutes) between first and last observation.
    '''
    returns = ['first_observation', 'last_observation',
               'n_observations', 'duration_%s' % unit]
    if t is None:
        return(returns)
    else:
        t = np.array(t) # avoid future warning with np.ptp() and pandas series objects
        if len(t) > 0:
            if unit == 'seconds': d = 1000
            elif unit == 'ms': d = 1
            
            return(np.min(t), np.max(t), 
                   len(t), round(np.ptp(t) / d, ndigits))
        else:
            return(None, None, 0, None)


def get_windows(df, start, stop, window_length_ms):
    '''
    Generate evenly spaced windows over a time period (e.g. one hour).
    For each window, figure out which rows of a data frame were observed during the window.

    Args:
        
    Returns:
        
    '''    
    if (stop - start) % window_length_ms != 0:
        logger.warning('The window length doesn\'t evenly divide the interval.')
    else:
        windows = OrderedDict.fromkeys(np.arange(start, stop, window_length_ms))
        for i in range(len(df)):
            key = df.timestamp[i] - (df.timestamp[i] % window_length_ms)
            if windows[key] is None:
                windows[key] = [i]
            else:
                windows[key].append(i)
        return(windows)
