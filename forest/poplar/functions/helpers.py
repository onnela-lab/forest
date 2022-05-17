"""Tools for common data processing tasks.

"""
import os
from logging import getLogger
from collections import OrderedDict
import numpy as np


logger = getLogger(__name__)


def clean_dataframe(df, drop_duplicates = True, sort = True, 
                    update_index = True):
    """
    Clean up a pandas dataframe that contains Beiwe data.
    
    Args:
        df (pandas.DataFrame): A pandas dataframe.        
        drop_duplicates (bool):  Drop extra copies of rows, if any exist.
        sort (bool):  Sort by timestamp.  If True, df must have a 'timestamp' 
            column.
        update_index (bool):  Set index to range(len(df)).
            
    Returns:
        None
    """
    if drop_duplicates:
        df.drop_duplicates(inplace = True)
    if sort:
        try:
            df.sort_values(by = ['timestamp'], inplace = True)
        except:
            logger.exception('Unable to sort by timestamp.')
    if update_index:
        df.set_index(np.arange(len(df)), inplace = True)


def get_windows(df, start, end, window_length_ms):
    """
    Generate evenly spaced windows over a time period.  For each window, 
    figure out which rows of a data frame were observed during the window.
    Usually df contains an hour of raw sensor data, and start/end are the 
    beginning/ending timestamps for that hour.

    Args:
        df (pandas.DataFrame): A pandas dataframe with a 'timestamp' column.
        start (int):  Millisecond timestamp for the beginning of the time 
            period.  Must be before the first timestamp in df.
        end (int):  Millisecond timestamp for the end of the time period.
            Must be after the last timestamp in df.
        window_length_ms (int):  Length of window, e.g. use 60*1000 for 
            1-minute windows.
        
    Returns:
        windows (OrderedDict):  Keys are timestamps corresponding to the 
            beginning of each window.  Values are lists of row indices in df
            that fall within the corresponding window.
    """
    if (end - start) % window_length_ms != 0:
        logger.warning('The window length doesn\'t \
                       evenly divide the interval.')
    else:
        try:
            windows = OrderedDict.fromkeys(np.arange(start, end,
                                                     window_length_ms))
            for i in range(len(df)):
                key = df.timestamp[i] - (df.timestamp[i] % window_length_ms)
                if windows[key] is None:
                    windows[key] = [i]
                else:
                    windows[key].append(i)
            return windows
        except:
            logger.warning('Unable to identify windows.')
            

def directory_size(dirpath, ndigits = 1):
    """
    Get the total size in megabytes of all files in a directory (including 
    subdirectories).
    
    Args:
        dirpath (str):  Path to a directory.
        ndigits (int): Precision.         

    Returns:
        size (float): Size in megabytes.
        file_count (int): Number of files in the directory.
    """
    size = 0
    file_count = 0
    for path, dirs, filenames in os.walk(dirpath):
        for f in filenames:
            filepath = os.path.join(path, f)
            size += os.path.getsize(filepath)
            file_count += 1
    size = round(size / (2**20), ndigits)    
    return size, file_count


def sort_by(list_to_sort, list_to_sort_by):
    """
    Sort a list according to items in a second list.  For example, sort 
    person identifiers according to study entry date.

    Args:
        list_to_sort (list): A list.
        list_to_sort_by (list): Usually a list of strings or numbers.

    Returns:
        sorted_list (list): Items of list_to_sort, arranged according 
            to values in list_to_sort_by.

    Example:
        >>> sort_by([1, 2, 3, 4], ['b', 'c', 'a', 'd'])
        [3, 1, 2, 4]        
    """
    if len(list_to_sort) != len(list_to_sort_by):
        logger.warning('Lists are not the same length.')
    else:
        try:
            sorted_list = [i for _, i in \
                           sorted(zip(list_to_sort_by, list_to_sort))]
            return sorted_list
        except:
            logger.warning('Unable to sort list.')               
        

def join_lists(list_of_lists):
    """
    Join all lists in a list of lists.    

    Args:
        list_of_lists (list): Each item is also a list.

    Returns:
        joined_list (list)
    """
    if not all([type(i) is list for i in list_of_lists]):
        logger.warning('This is not a list of lists.')
    else:
        joined_list = [i for sublist in list_of_lists for i in sublist]
        return joined_list


# Dictionary of some summary statistics:
def sample_range(a): return np.max(a) - np.min(a)
def iqr(a):          return np.percentile(a, 75) - np.percentile(a, 25)
def sample_std(a):   return np.std(a, ddof = 1)
def sample_var(a):   return np.var(a, ddof = 1)
STATS = {'mean':   np.mean,
         'median': np.median,
         'range':  sample_range,
         'iqr':    iqr,
         'std':    sample_std,
         'var':    sample_var
         }


def get_ids(study_folder: str) -> list:
    """Get subfolders in directory, excluding registry and hidden folders

    Args:
        study_folder(str): Filepath to the folder containing desired
            subdirectories

    Returns:
        List of subdirectories of the study_folder.
    """
    list_of_dirs = []
    for subdir in os.listdir(study_folder):
        is_folder = os.path.isdir(os.path.join(study_folder, subdir))
        if (not subdir.startswith(".")) and subdir != "registry" and is_folder:
            list_of_dirs.append(subdir)
    return list_of_dirs