"""
Functions for input/output tasks.
Original Authors: Georgios Efstathiadis, Josh Barback
"""

import json
import os
from collections import OrderedDict
from logging import getLogger


logger = getLogger(__name__)


def setup_directories(dirpath_list: str | list[str]) -> None:
    """
    Checks if directories exist; creates them if not. Creates intermediate directories if necessary.

    Args:
        dirpath_list (str or list): List of directory paths (str) to create.
            Can also be a single path (str).

    Returns:
        None
    """
    if isinstance(dirpath_list, str):
        dirpath_list = [dirpath_list]
    for directory in dirpath_list:
        os.makedirs(directory, exist_ok=True)


def write_json(
    dictionary: dict | OrderedDict, name: str, dirpath: str, indent: int = 4
) -> str | None:
    """ Writes a dictionary to a JSON file.

    Args:
        dictionary (dict or OrderedDict):  Dictionary to write.
        
        name (str):  Name for the file to create, without extension.
        
        dirpath (str):  Path to location for the JSON file.
        
        indent (int):  Indentation for pretty printing.

    Returns:
        filepath (str): Path to the JSON file.
    """
    try:
        filepath = os.path.join(dirpath, name + ".json")
        with open(filepath, "w") as file:
            json.dump(dictionary, file, indent=indent)
        return filepath
    except Exception:
        logger.warning("Unable to write JSON file.")
        return None


def read_json(filepath: str) -> dict | None:
    """ Read a JSON file into a dictionary.

    Args:
        filepath (str):  Path to JSON file.

    Returns:
        dictionary (dict): The deserialized contents of the JSON file.
    """
    try:
        with open(filepath) as file:
            return json.load(file)
    except Exception:
        logger.warning("Unable to read JSON file.")
        return None


def setup_csv(
    name: str, dirpath: str, header: list[str]
) -> str:
    """
    Creates a csv file with the given column labels. Overwrites a file with the same name.

    Args:
        name (str):  Name of csv file to create, without extension.
        dirpath (str):  Path to location for csv file.
        header (list):  List of column headers (str).

    Returns:
        filepath (str): Path to the new csv file.
    """
    filepath = os.path.join(dirpath, name + ".csv")
    if os.path.exists(filepath):
        logger.warning("Overwriting existing file with that name.")
    with open(filepath, "w") as file:
        file.write(",".join(header) + "\n")
    return filepath


def write_to_csv(filepath: str, line: list[str], missing_strings: list[str] = ["nan"]) -> None:
    """ Writes line to a csv file.

    Args:
        filepath (str): Path to a text file.
        
        line (list): Line of items to add to the csv.  Line items are
            converted to strings, joined with ',' and terminated with '\n'.
        
        missing_strings (list): List of strings to replace with ''.  Note that
            'nan' covers both float('NaN') and np.nan.

    Returns:
        None
    """
    try:
        line = ["" if i is None else i for i in line]             # Replace None with ''
        line = [str(i) for i in line]                             # Make sure everything is a string
        line = ["" if i in missing_strings else i for i in line]  # Replace missing values with ''
        
        with open(filepath, "a") as file:
            file.write(",".join(line) + "\n")

    except Exception:
        logger.warning("Unable to append line to CSV.")
