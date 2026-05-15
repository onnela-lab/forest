"""Common functions for the forest package"""

import logging
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from dateutil.tz import gettz, UTC

logger = logging.getLogger(__name__)


def datetime2stamp(time_list: Sequence[int], tz_str: str) -> int:
    """ Convert a componentized datetime (year, month, day, hour, min, sec, us) to Unix time
    
    Args:
        time_list: list or tuple,
            a list of integers [year, month, day, hour (0-23), min, sec],
        tz_str: str,
            timezone where the study is conducted
            
            To check that a timezone string is valid use:
            # from dateutil.tz import gettz
            # assert gettz(tz_str) is not None
    
    Returns:
        Unix time, which is what Beiwe uses
    """
    t = list(time_list)
    loc_dt = datetime(t[0], t[1], t[2], t[3], t[4], t[5], t[6], tzinfo=gettz(tz_str))
    # this function used to mess around with the calendar library and returned an int
    return int(loc_dt.astimezone(UTC).timestamp())


def stamp2datetime(stamp: float | int, tz_str: str) -> list:
    """Convert a Unix time to datetime
    
    Args:
        stamp: int or float,
            Unix time, the timestamp in Beiwe
        tz_str: str,
            timezone where the study is conducted
            
            To check that a timezone string is valid use:
            # from dateutil.tz import gettz
            # assert gettz(tz_str) is not None
    
    Returns:
        a list of integers [year, month, day, hour (0-23), min, sec] in the specified tz
    """
    utc_dt = datetime.fromtimestamp(stamp, UTC)
    loc_dt = utc_dt.astimezone(gettz(tz_str))
    return [loc_dt.year, loc_dt.month, loc_dt.day, loc_dt.hour, loc_dt.minute, loc_dt.second]


def filename2stamp(filename: str) -> int:
    """Convert a filename to Unix time - Beiwe filenames are of the form "YYYY-MM-DD HH_MM_SS....."
    
    Args:
        filename: str,
            the filename of communication log
    
    Returns:
        UNIX time (int)
    """
    [d_str, h_str] = filename.split(" ")
    [year, month, day] = np.array(d_str.split("-"), dtype=int)
    hour = int(h_str.split("_")[0])
    
    return datetime2stamp((year, month, day, hour, 0, 0), "UTC")


def get_files_timestamps(folder_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Get List of Files and Timestamps in a folder
    
    Args:
        folder_path: str,
            The directory containing files
    
    Returns:
        filenames: An np.array containing all csv files in the directory
        
        filestamps: An np.array containing all timestamps of csv files in
            directory, in the same order as those in filenames
    """
    # get list of all files in path
    filenames = np.sort(np.array([f for f in os.listdir(folder_path) if not f.startswith(".")]))
    
    # create a list to convert all filenames to UNIX time
    filestamps = np.array([filename2stamp(filename) for filename in filenames])
    
    return filenames, filestamps


def read_data(
    beiwe_id: str,
    study_folder: str,
    datastream: str,
    tz_str: str,
    time_start: list[int] | None,
    time_end: list[int] | None,
) -> tuple[Any, float, float]:
    """Read data from a user's datastream folder

    Args:
        beiwe_id: str,
            beiwe ID; study_folder: the path of the folder which contains all the users' data
        study_folder: str,
            the path of the folder which contains all the users' data
        datastream: str,
            'gps','accelerometer','texts' or 'calls'
        tz_str: str,
            where the study is/was conducted
        time_start, time_end: list of integers or None,
            starting time and ending time of the window of interest
            
            time should be a list of integers with format [year, month, day, hour, minute, second]
            
            if time_start is None and time_end is None: then it reads all the available files
            
            if time_start is None and time_end is given, then it reads all the files before the
            given time
            
            if time_start is given and time_end is None, then it reads all the files after the given
            time
            
            if identifiers files are present and the earliest identifiers registration timestamp
            occurred after the provided time_start (or if time_start is None) then that identifier
            timestamp will be used instead.

    Returns:
        a panda dataframe of the datastream (not for accelerometer data!) and corresponding
        starting/ending timestamp (UTC), you can convert it to numpy array as needed
        
        For accelerometer data, instead of a panda dataframe, it returns a list of filenames. The
        reason is the volume of accelerometer data is too large, we need to process it on the fly:
        read one csv file, process one, not wait until all the csv's are imported (that may be too
        large in memory!)
    """
    res = pd.DataFrame()
    
    stamp_start: float = 1e12
    stamp_end: float = 0.
    
    folder_path = os.path.join(study_folder, beiwe_id, datastream)
    files_in_range: list[str] = []
    # if text folder exists, call folder must exists
    if not os.path.exists(os.path.join(study_folder, beiwe_id)):
        logger.warning("User %s does not exist, please check the ID again.", beiwe_id)
    elif not os.path.exists(folder_path):
        logger.warning("User %s: %s data are not collected.", beiwe_id, datastream)
    else:
        filenames, filestamps = get_files_timestamps(folder_path)
        
        # find the timestamp in the identifier (when the user was enrolled)
        if os.path.exists(os.path.join(study_folder, beiwe_id, "identifiers")):
            identifier_files, _ = get_files_timestamps(
                os.path.join(study_folder, beiwe_id, "identifiers")
            )
            identifiers = pd.read_csv(
                os.path.join(study_folder, beiwe_id, "identifiers", identifier_files[0]),
                sep=",",
            )
            # now determine the starting and ending time according to the
            # Docstring
            if identifiers.index[0] > 10**10:
                # sometimes the identifier has mismatched colnames and columns
                stamp_start1 = identifiers.index[0] / 1000
            else:
                stamp_start1 = identifiers["timestamp"][0] / 1000
        else:
            stamp_start1 = sorted(filestamps)[0]
        # now determine the starting and ending time according to the Docstring
        if time_start is None:
            stamp_start = stamp_start1
        else:
            stamp_start2 = datetime2stamp(time_start, tz_str)
            # only allow data after the participant registered (this condition may be violated under
            # test conditions of the beiwe backend.)
            stamp_start = max(stamp_start1, stamp_start2)
        # Last hour: look at all the subject's directories (except survey) and find the latest date
        # for each directory
        directories = [
            directory for directory in os.listdir(os.path.join(study_folder, beiwe_id))
            if os.path.isdir(os.path.join(study_folder, beiwe_id, directory))
        ]
        
        directories = list(
            set(directories) - {"survey_answers", "survey_timings", "audio_recordings"}
        )
        
        all_timestamps: list = []
        for i in directories:
            _, directory_filestamps = get_files_timestamps(os.path.join(study_folder, beiwe_id, i))
            all_timestamps += directory_filestamps.tolist()
        
        ordered_timestamps = sorted(all_timestamps)
        stamp_end1 = ordered_timestamps[-1]
        
        if time_end is None:
            stamp_end = stamp_end1
        else:
            stamp_end2 = datetime2stamp(time_end, tz_str)
            stamp_end = min(stamp_end1, stamp_end2)
        
        # extract the filenames in range
        files_in_range = list(filenames[(filestamps >= stamp_start) * (filestamps < stamp_end)])
        if len(files_in_range) == 0:
            logger.warning("User %s: There are no %s data in range.", beiwe_id, datastream)
        else:
            if datastream != "accelerometer":
                # read in the data one by one file and stack them
                for data_file in files_in_range:
                    dest_path = os.path.join(folder_path, data_file)
                    hour_data = pd.read_csv(dest_path)
                    if res.shape[0] == 0:
                        res = hour_data
                    else:
                        res = pd.concat([res, hour_data], ignore_index=True)
    
    if datastream == "accelerometer":
        return files_in_range, stamp_start, stamp_end
    
    return res, stamp_start, stamp_end


def write_all_summaries(
    beiwe_id: str,
    stats_pdframe: pd.DataFrame,
    output_path: str,
    columns: list[str] | None = None,
):
    """Write out all the summary stats for a user
    
    Args:
        beiwe_id: str,
            beiwe ID
        stats_pdframe: pd.DataFrame,
            the summary stats for a user
        output_path: str,
            the path to write out the summary stats
    """
    os.makedirs(output_path, exist_ok=True)
    stats_pdframe.to_csv(os.path.join(output_path, f"{beiwe_id}.csv"), index=False, columns=columns)
