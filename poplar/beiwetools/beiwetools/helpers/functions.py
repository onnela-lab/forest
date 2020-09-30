'''Miscellaneous functions for working with Beiwe data and study configurations.

'''
import os
import json
import logging
import numpy as np
from collections import OrderedDict


logger = logging.getLogger(__name__)


# Dictionary of some summary statistics:
def range_(a):     return(np.max(a) - np.min(a))
def iqr(a):        return(np.percentile(a, 75) - np.percentile(a, 25))
def sample_std(a): return(np.std(a, ddof = 1))
def sample_var(a): return(np.var(a, ddof = 1))
summary_statistics = OrderedDict([('mean',   np.mean),
                                  ('median', np.median),
                                  ('range',  range_),
                                  ('iqr',    iqr),
                                  ('std',    sample_std),
                                  ('var',    sample_var)
                                  ])


def write_string(to_write, name, directory, title = '', mode = 'w'):
    '''
    Writes a string to a human-readable text file.

    Args:
        to_write (str):  String to write.
		name (str):  Name for the file to create.
		directory (str):  Path to location for the text file.
        title (str):  Header or title for the text file.
		mode (str):  Mode for open().  Use 'w' to overwrite, 'a' to append.

    Returns:
		None
    '''
    path = os.path.join(directory, name + '.txt')
    with open(path, mode) as f:
        if len(title) > 0:
            f.write(title + '\n')
        f.write(to_write + '\n')


def write_json(dictionary, name, directory, indent = 4):
	'''
	Writes a dictionary to a json file.

	Args:
		dictionary (dict or OrderedDict):  Dictionary to write.
		name (str):  Name for the file to create.
		directory (str):  Path to location for the json file.
		indent (int):  Indentation for pretty printing.

	Returns:
		None
	'''
	path = os.path.join(directory, name + '.json')
	with open(path, 'w') as f:
		json.dump(dictionary, f, indent = indent)


def read_json(path, ordered = True):
	'''
	Read a json file into a dictionary.

	Args:
		path (str):  Path to json file.
		ordered (bool):  Returns an ordered dictionary if True.

	Returns:
		dictionary (dict or OrderedDict)
	'''
	with open(path, 'r') as f:        
		if ordered:
			dictionary = json.load(f, object_pairs_hook = OrderedDict)
		else:
			dictionary = json.load(f)
	return(dictionary)


def setup_directories(dir_list):
    '''
    Checks if directories exist; creates them if not.

    Args:
        dir_list (list): List of directory paths (str) to create.  
            Can also be a single path (str).

    Returns:
        None
    '''
    if type(dir_list) is str:
        dir_list = [dir_list]
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
        else:
            logging.warning('Directory already exists.')


def setup_csv(name, directory, header):
    '''
    Creates a csv file with the given column labels.
    Overwrites a file with the same name.

    Args:
        name (str):  Name of csv file to create.
        directory (str):  Path to location for csv file.
        header (list):  List of column headers (str).
	
    Returns:
        path (str): Path to the new csv file.
    '''
    path = os.path.join(directory, name + '.csv')
    if os.path.exists(path):
        logging.warning('Overwriting existing file with that name.')
    f = open(path, 'w')
    f.write(','.join(header) + '\n')
    f.close()
    return(path)	


def write_to_csv(path, line):
    '''
    Writes line to a csv file.

    Args:
        path (str):  Path to a text file.
        line (list):  
            Line of items to add to the csv.  
            Line items are converted to strings, joined with ',' and terminated with '\n'.
	Returns:
		None
    '''
    # Replace missing values with ''
    missing = (None, np.nan)
    line = ['' if i in missing else i for i in line]
    # Make sure everything is a string
    line = [str(i) for i in line]
    # Write to file
    f = open(path, "a")
    f.write(','.join(line) + "\n")
    f.close()


def directory_size(directory, ndigits = 1):
	'''
	Get the total size in megabytes of all files in a directory (including subdirectories).
	
	Args:
		directory (str):  Path to a directory.
		ndigits (int): Precision. 		
	Returns:
		size (float): Size in megabytes.
		file_count (int): Number of files in the directory.
	'''
	size = 0
	file_count = 0
	for path, dirs, filenames in os.walk(directory):
		for f in filenames:
			file_path = os.path.join(path, f)
			size += os.path.getsize(file_path)
			file_count += 1
	size = round(size / (2**20), ndigits)    
	return(size, file_count)


def sort_by(list_to_sort, list_to_sort_by):
    '''
    Sort a list according to items in a second list.

    Args:
        list_to_sort (list)
        list_to_sort_by (list): Usually a list of strings or numbers.

    Returns:
        sorted_list (list): Items of list_to_sort, arranged according to values in list_to_sort_by.

    Example:
        >>> sort_by([1, 2, 3, 4], ['b', 'c', 'a', 'd'])
        [3, 1, 2, 4]        
    '''
    if len(list_to_sort) != len(list_to_sort_by):
        logging.warning('Lists are not the same length.')
    try:
        sorted_list = [i for _, i in sorted(zip(list_to_sort_by, list_to_sort))]
        return(sorted_list)
    except:
        pass               
        

def join_lists(list_of_lists):
    '''
    Join all lists in a list of lists.    

    Args:
        list_of_lists (list)

    Returns:
        joined_list (list)
    '''
    if not all([type(i) is list for i in list_of_lists]):
        logging.warning('This is not a list of lists.')
    joined_list = [i for sublist in list_of_lists for i in sublist]
    return(joined_list)    


def check_same(x, y, to_check = 'all'):
    '''
    For use in defining equality for classes.
    Checks if x, y are of the same class.
    Checks equality of keys in x.__dict__ and y.__dict__.

    Args:    
        x, y: Two objects with __dict__ attributes.
        to_check (list or str): List of keys to check.
            If 'all' then all keys are checked.

    Returns:
        same (bool):
            True if x.__dict__ and y.__dict__ agree on all keys in to_check.
    '''    
    same = False
    results = []
    if type(x) is type(y):  
        try:
            if to_check == 'all':
                to_check = list(x.__dict__.keys()) + list(y.__dict__.keys())
            for k in to_check:
                results.append(x.__dict__[k] == y.__dict__[k])            
        except:
            logger.warning('Equality check is not implemented for type %s.' % str(type(x)))
            return(False)
    if len(results) > 0:
        same = all(results)
    return(same)
    

def coerce_to_dict(to_coerce, to_keys):
    '''
    Return a dictionary with the desired keys.
    Values are lists.

    Args:
        to_coerce (dict, str, list, or Nonetype):
            If dict, keys should be a subset of to_keys.
                For keys in to_coerce.keys: 
                    to_dict agrees with to_coerce (str and None are convereted to lists).
                For keys not in to_coerce.keys: 
                    Values of to_dict are [].
            If str, values of to_dict are all [to_coerce].
            If list, values of to_dict are all to_coerce.            
            If None, values of to_dict are all [].
        to_keys (list): List of keys.

    Returns:
        to_dict (OrderedDict):
            Keys are keys, values are as described above.
    '''
    if isinstance(to_coerce, (dict, OrderedDict)):
        t = OrderedDict()
        for k in to_coerce:
            if isinstance(to_coerce[k], str): t[k] = [to_coerce[k]]
            if to_coerce[k] is None: t[k] = []
        to_dict = OrderedDict(zip(to_keys, [[] for k in to_keys]))
        to_dict.update(t)
    else:
        if isinstance(to_coerce, str): t = [to_coerce]        
        elif isinstance(to_coerce, list): t = list(to_coerce)
        elif to_coerce is None: t = []
        to_dict = OrderedDict(zip(to_keys, [t]*len(to_keys)))
    return(to_dict)