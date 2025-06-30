"""Functions for input/output tasks."""
import os
import json
from collections import OrderedDict
from logging import getLogger

from typing import List, Union


logger = getLogger(__name__)


def setup_directories(dirpath_list: Union[str, List[str]]) -> None:
    """
    Checks if directories exist; creates them if not.
    Creates intermediate directories if necessary.

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
    dictionary: Union[dict, OrderedDict], name: str,
    dirpath: str, indent: int = 4
) -> Union[str, None]:
    """Writes a dictionary to a JSON file.

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


def read_json(
    filepath: str, ordered: bool = False
) -> Union[dict, OrderedDict, None]:
    """
    Read a JSON file into a dictionary.

    Args:
        filepath (str):  Path to JSON file.
        ordered (bool):  Returns an ordered dictionary if True.

    Returns:
        dictionary (dict or OrderedDict): The deserialized contents of
            the JSON file.
    """
    try:
        with open(filepath, "r") as file:
            if ordered:
                dictionary = json.load(file, object_pairs_hook=OrderedDict)
            else:
                dictionary = json.load(file)
        return dictionary
    except Exception:
        logger.warning("Unable to read JSON file.")
        return None


def setup_csv(
    name: str, dirpath: str, header: List[str]
) -> str:
    """
    Creates a csv file with the given column labels.
    Overwrites a file with the same name.

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


def write_to_csv(
    filepath: str, line: List[str], missing_strings: List[str] = ["nan"]
) -> None:
    """
    Writes line to a csv file.

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
        # Replace None with ''
        line = ["" if i is None else i for i in line]
        # Make sure everything is a string
        line = [str(i) for i in line]
        # Replace missing values with ''
        line = ["" if i in missing_strings else i for i in line]
        # Write to file
        with open(filepath, "a") as file:
            file.write(",".join(line) + "\n")
    except Exception:
        logger.warning("Unable to append line to CSV.")
