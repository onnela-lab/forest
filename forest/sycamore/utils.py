"""Utility functions in Sycamore.
This file exists to avoid a circular imports from common.py.
In the future, it would be good to refactor code and put more things in here,
but for now, I'll just include the three necessary functions so this pull
request isn't massive"""

import datetime
import json
import re

import pandas as pd


def read_json(study_dir: str) -> dict:
    """Read a json file into a dictionary

    Args:
        study_dir (str):  filepath to json file.
    Returns:
        A dict representation of the json file
    """
    with open(study_dir, "r") as f:
        dictionary = json.load(f)
    return dictionary


def get_month_from_today():
    """Get the date 31 days from today, in YYYY-MM-DD format"""
    return (datetime.datetime.today() +
            datetime.timedelta(31)).strftime("%Y-%m-%d")


def filename_to_timestamp(filename: str, tz_str: str = "UTC"
                          ) -> pd.Timestamp:
    """Extract a datetime from a filepath.

    Args:
        filename(str):
            a string, with a csv file at the end formatted like
            "YYYY-MM-DD HH_MM_SS+00_00"
        tz_str(str):
            Output Timezone

    Returns:
        The Timestamp corresponding to the date in the csv filename, in the
        time zone corresponding to tz_str
    """
    str_to_convert = re.sub("_", ":", filename)[0:25]
    time_written = pd.to_datetime(str_to_convert
                                  ).tz_convert(tz_str).tz_localize(None)
    return time_written
