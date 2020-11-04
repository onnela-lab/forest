'''Functions for input/output tasks.

'''
import os
import json
from collections import OrderedDict
from logging import getLogger


logger = getLogger(__name__)


def setup_directories(dirpath_list):
    '''
    Checks if directories exist; creates them if not.
    Creates intermediate directories if necessary.

    Args:
        dirpath_list (str or list): List of directory paths (str) to create.  
            Can also be a single path (str).

    Returns:
        None
    '''
    if type(dirpath_list) is str:
        dirpath_list = [dirpath_list]
    for d in dirpath_list:
        if not os.path.exists(d):
            os.makedirs(d)
        else:
            logger.warning('Directory already exists.')


def write_json(dictionary, name, dirpath, indent = 4):
    '''
    Writes a dictionary to a JSON file.

    Args:
        dictionary (dict or OrderedDict):  Dictionary to write.
        name (str):  Name for the file to create, without extension.
        dirpath (str):  Path to location for the JSON file.
        indent (int):  Indentation for pretty printing.

    Returns:
        filepath (str): Path to the JSON file.
    '''
    try:
        filepath = os.path.join(dirpath, name + '.json')
        with open(filepath, 'w') as f:
            json.dump(dictionary, f, indent = indent)
        return(filepath)        
    except:
        logger.warning('Unable to write JSON file.')


def read_json(filepath, ordered = False):
    '''
    Read a JSON file into a dictionary.

    Args:
        filepath (str):  Path to JSON file.
        ordered (bool):  Returns an ordered dictionary if True.

    Returns:
        dictionary (dict or OrderedDict): The deserialized contents of 
            the JSON file.
    '''
    try:
        with open(filepath, 'r') as f:        
            if ordered:
                dictionary = json.load(f, object_pairs_hook = OrderedDict)
            else:
                dictionary = json.load(f)
        return(dictionary)
    except:
        logger.warning('Unable to read JSON file.')


def setup_csv(name, dirpath, header):
    '''
    Creates a csv file with the given column labels.
    Overwrites a file with the same name.

    Args:
        name (str):  Name of csv file to create, without extension.
        dirpath (str):  Path to location for csv file.
        header (list):  List of column headers (str).
    
    Returns:
        filepath (str): Path to the new csv file.
    '''
    filepath = os.path.join(dirpath, name + '.csv')
    if os.path.exists(filepath):
        logger.warning('Overwriting existing file with that name.')
    f = open(filepath, 'w')
    f.write(','.join(header) + '\n')
    f.close()
    return(filepath)    


def write_to_csv(filepath, line, missing_strings = ['nan']):
    '''
    Writes line to a csv file.

    Args:
        filepath (str): Path to a text file.
        line (list): Line of items to add to the csv.  Line items are 
            converted to strings, joined with ',' and terminated with '\n'.
        missing_strings (list): List of strings to replace with ''.  Note that 
            'nan' covers both float('NaN') and np.nan.
    
    Returns:
        None
    '''
    try:
        # Replace None with ''
        line = ['' if i is None else i for i in line]
        # Make sure everything is a string
        line = [str(i) for i in line]
        # Replace missing values with ''
        line = ['' if i in missing_strings else i for i in line]
        # Write to file
        f = open(filepath, "a")
        f.write(','.join(line) + "\n")
        f.close()
    except:
        logger.warning('Unable to append line to CSV.')