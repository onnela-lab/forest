import calendar
from datetime import datetime
import os
from pytz import timezone
import sys
from typing import Optional, Any, List, Union

import numpy as np
import pandas as pd


def datetime2stamp(time_list: Union[list, tuple], tz_str: str) -> int:
    """
    Docstring
    Args: time_list: a list of integers [year, month, day, hour (0-23), min,
                sec],
          tz_str: timezone (str), where the study is conducted
    please use
    # from pytz import all_timezones
    # all_timezones
    to check all timezones
    Return: Unix time, which is what Beiwe uses
    """
    loc_tz = timezone(tz_str)
    loc_dt = loc_tz.localize(
        datetime(time_list[0], time_list[1], time_list[2], time_list[3],
                 time_list[4], time_list[5])
    )
    utc = timezone("UTC")
    utc_dt = loc_dt.astimezone(utc)
    timestamp = calendar.timegm(utc_dt.timetuple())
    return timestamp


def stamp2datetime(stamp: Union[float, int], tz_str: str) -> list:
    """
    Docstring
    Args: stamp: Unix time, integer, the timestamp in Beiwe
          tz_str: timezone (str), where the study is conducted
    please use
    # from pytz import all_timezones
    # all_timezones
    to check all timezones
    Return: a list of integers [year, month, day, hour (0-23), min, sec] in the
        specified tz
    """
    loc_tz = timezone(tz_str)
    utc = timezone("UTC")
    utc_dt = utc.localize(datetime.utcfromtimestamp(stamp))
    loc_dt = utc_dt.astimezone(loc_tz)
    return [loc_dt.year, loc_dt.month, loc_dt.day, loc_dt.hour, loc_dt.minute,
            loc_dt.second]


def filename2stamp(filename: str) -> int:
    """
    Docstring
    Args: filename (str), the filename of communication log
    Return: UNIX time (int)
    """
    [d_str, h_str] = filename.split(' ')
    [y, m, d] = np.array(d_str.split('-'), dtype=int)
    h = int(h_str.split('_')[0])
    stamp = datetime2stamp((y, m, d, h, 0, 0), 'UTC')
    return stamp


def read_data(beiwe_id: str, study_folder: str, datastream: str, tz_str: str,
              time_start: Optional[List[Any]], time_end: Optional[List[Any]]
              ) -> pd.DataFrame:
    """
    Docstring
    Args: beiwe_id: beiwe ID; study_folder: the path of the folder which
              contains all the users
          datastream: 'gps','accelerometer','texts' or 'calls'
          tz_str: where the study is/was conducted
          starting time and ending time of the window of interest
          time should be a list of integers with format [year, month, day,
          hour, minute, second]
          if time_start is None and time_end is None: then it reads all the
          available files
          if time_start is None and time_end is given, then it reads all the
          files before the given time
          if time_start is given and time_end is None, then it reads all the
          files after the given time
          if identifiers files are present and the earliest identifiers
          registration timestamp occurred
            after the provided time_start (or if time_start is None) then that
            identifier timestamp
            will be used instead.
    return: a panda dataframe of the datastream (not for accelerometer data!)
    and corresponding starting/ending timestamp (UTC),
            you can convert it to numpy array as needed
            For accelerometer data, instead of a panda dataframe, it returns a
            list of filenames
            The reason is the volume of accelerometer data is too large, we
            need to process it on the fly:
            read one csv file, process one, not wait until all the csv's are
            imported (that may be too large in memory!)
    """
    df = pd.DataFrame()
    stamp_start = 1e12
    stamp_end: int = 0
    folder_path = os.path.join(study_folder, beiwe_id, datastream)
    files_in_range = []
    # if text folder exists, call folder must exists
    if not os.path.exists(os.path.join(study_folder, beiwe_id)):
        print('User ' + str(beiwe_id) +
              ' does not exist, please check the ID again.')
    elif not os.path.exists(folder_path):
        print('User ' + str(beiwe_id) + ' : ' + str(
            datastream) + ' data are not collected.')
    else:
        filenames = np.sort(np.array(os.listdir(folder_path)))
        filenames = np.array(
            [file for file in filenames if not file.startswith(".")]
        )
        # create a list to convert all filenames to UNIX time
        filestamps = np.array(
            [filename2stamp(filename) for filename in filenames])
        # find the timestamp in the identifier (when the user was enrolled)
        if os.path.exists(os.path.join(study_folder, beiwe_id, "identifiers")):
            identifier_files = os.listdir(
                os.path.join(study_folder, beiwe_id, "identifiers"))
            identifiers = pd.read_csv(
                os.path.join(study_folder, beiwe_id, "identifiers",
                             identifier_files[0]), sep=","
            )
            # now determine the starting and ending time according to the
            # Docstring
            if identifiers.index[0] > 10 ** 10:
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
            # only allow data after the participant registered (this condition
            # may be violated under test conditions of the beiwe backend.)
            stamp_start = max(stamp_start1, stamp_start2)
        # Last hour: look at all the subject's directories (except survey) and
        # find the latest date for each directory
        directories = os.listdir(os.path.join(study_folder, beiwe_id))
        directories = list(set(directories) - {
            "survey_answers", "survey_timings", "audio_recordings"})
        all_timestamps = []
        for i in directories:
            files = os.listdir(os.path.join(study_folder, beiwe_id, i))
            all_timestamps += [filename2stamp(filename) for filename in files]
        ordered_timestamps = sorted(
            [timestamp for timestamp in all_timestamps if
             timestamp is not None])
        stamp_end1 = ordered_timestamps[-1]
        if time_end is None:
            stamp_end = stamp_end1
        else:
            stamp_end2 = datetime2stamp(time_end, tz_str)
            stamp_end = min(stamp_end1, stamp_end2)

        # extract the filenames in range
        files_in_range = filenames[
            (filestamps >= stamp_start) * (filestamps < stamp_end)
        ]
        if len(files_in_range) == 0:
            sys.stdout.write('User ' + str(beiwe_id) + ' : There are no ' +
                             str(datastream) + ' data in range.' + '\n')
        else:
            if datastream != 'accelerometer':
                # read in the data one by one file and stack them
                for data_file in files_in_range:
                    dest_path = os.path.join(folder_path, data_file)
                    hour_data = pd.read_csv(dest_path)
                    if df.shape[0] == 0:
                        df = hour_data
                    else:
                        df = pd.concat([df, hour_data], ignore_index=True)

    if datastream == "accelerometer":
        return files_in_range, stamp_start, stamp_end
    else:
        return df, stamp_start, stamp_end


def write_all_summaries(beiwe_id: str, stats_pdframe: pd.DataFrame,
                        output_folder: str):
    """
    Docstring
    Args: beiwe_id: str, stats_pdframe is pd dataframe (summary stats)
          output_path should be the folder path where you want to save the
          output
    Return: write out as csv files named by user ID
    """
    os.makedirs(output_folder, exist_ok=True)
    stats_pdframe.to_csv(os.path.join(output_folder, str(beiwe_id) + ".csv"),
                         index=False)
