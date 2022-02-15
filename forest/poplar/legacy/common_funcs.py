import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone
import calendar

def datetime2stamp(time_list,tz_str):
    """
    Docstring
    Args: time_list: a list of integers [year, month, day, hour (0-23), min, sec],
          tz_str: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: Unix time, which is what Beiwe uses
    """
    loc_tz =  timezone(tz_str)
    loc_dt = loc_tz.localize(datetime(time_list[0], time_list[1], time_list[2], time_list[3], time_list[4], time_list[5]))
    utc = timezone("UTC")
    utc_dt = loc_dt.astimezone(utc)
    timestamp = calendar.timegm(utc_dt.timetuple())
    return timestamp

def stamp2datetime(stamp,tz_str):
    """
    Docstring
    Args: stamp: Unix time, integer, the timestamp in Beiwe
          tz_str: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: a list of integers [year, month, day, hour (0-23), min, sec] in the specified tz
    """
    tz = timezone(tz_str)
    localized_dt = tz.localize(datetime.utcfromtimestamp(stamp))
    return [localized_dt.year, localized_dt.month, localized_dt.day,
            localized_dt.hour, localized_dt.minute, localized_dt.second]

def filename2stamp(filename):
    """
    Docstring
    Args: filename (str), the filename of communication log
    Return: UNIX time (int)
    """
    [d_str,h_str] = filename.split(' ')
    [y,m,d] = np.array(d_str.split('-'),dtype=int)
    h = int(h_str.split('_')[0])
    stamp = datetime2stamp((y,m,d,h,0,0),'UTC')
    return stamp

def read_data(ID:str, study_folder: str, datastream:str, tz_str: str, time_start, time_end):
    """
    Docstring
    Args: ID: beiwe ID; study_folder: the path of the folder which contains all the users
          datastream: 'gps','accelerometer','texts' or 'calls'
          tz_str: where the study is/was conducted
          starting time and ending time of the window of interest
          time should be a list of integers with format [year, month, day, hour, minute, second]
          if time_start is None and time_end is None: then it reads all the available files
          if time_start is None and time_end is given, then it reads all the files before the given time
          if time_start is given and time_end is None, then it reads all the files after the given time
          if identifiers files are present and the earliest identifiers registration timestamp occurred
            after the provided time_start (or if time_start is None) then that identifier timestamp
            will be used instead.
    return: a panda dataframe of the datastream (not for accelerometer data!) and corresponding starting/ending timestamp (UTC),
            you can convert it to numpy array as needed
            For accelerometer data, instead of a panda dataframe, it returns a list of filenames
            The reason is the volume of accelerometer data is too large, we need to process it on the fly:
            read one csv file, process one, not wait until all the csv's are imported (that may be too large in memory!)
    """
    df = pd.DataFrame()
    stamp_start = 1e12
    stamp_end = 0
    folder_path = study_folder + "/" + ID +  "/" + str(datastream)
    ## if text folder exists, call folder must exists
    if not os.path.exists(study_folder + "/" + ID):
        print('User '+ str(ID) + ' does not exist, please check the ID again.')
    elif not os.path.exists(folder_path):
        print('User '+ str(ID) + ' : ' + str(datastream) + ' data are not collected.')
    else:
        filenames = np.sort(np.array(os.listdir(folder_path)))
        ## create a list to convert all filenames to UNIX time
        filestamps = np.array([filename2stamp(filename) for filename in filenames])
        ## find the timestamp in the identifier (when the user was enrolled)
        if os.path.exists(study_folder + "/" + ID + "/identifiers"):
            identifier_Files = os.listdir(study_folder + "/" + ID + "/identifiers")
            identifiers = pd.read_csv(study_folder + "/" + ID + "/identifiers/"+ identifier_Files[0], sep = ",")
            ## now determine the starting and ending time according to the Docstring
            if identifiers.index[0]>10**10:  ## sometimes the identifier has mismatched colnames and columns
                stamp_start1 = identifiers.index[0]/1000
            else:
                stamp_start1 = identifiers["timestamp"][0]/1000
        else:
            stamp_start1 = sorted(filestamps)[0]
        ## now determine the starting and ending time according to the Docstring
        if time_start == None:
            stamp_start = stamp_start1
        else:
            stamp_start2 = datetime2stamp(time_start,tz_str)
            # only allow data after the participant registered (this condition may be violated under
            # test conditions of the beiwe backend.)
            stamp_start = max(stamp_start1,stamp_start2)
        ##Last hour: look at all the subject's directories (except survey) and find the latest date for each directory
        directories = os.listdir(study_folder + "/" + ID)
        directories = list(set(directories)-set(["survey_answers","survey_timings","audio_recordings"]))
        all_timestamps = []
        for i in directories:
            files = os.listdir(study_folder + "/" + ID + "/" + i)
            all_timestamps += [filename2stamp(filename) for filename in files]
        ordered_timestamps = sorted([timestamp for timestamp in all_timestamps if timestamp is not None])
        stamp_end1 = ordered_timestamps[-1]
        if time_end == None:
            stamp_end = stamp_end1
        else:
            stamp_end2 = datetime2stamp(time_end,tz_str)
            stamp_end = min(stamp_end1,stamp_end2)

        ## extract the filenames in range
        files_in_range = filenames[(filestamps>=stamp_start)*(filestamps<stamp_end)]
        if len(files_in_range) == 0:
            sys.stdout.write('User '+ str(ID) + ' : There are no ' + str(datastream) + ' data in range.'+ '\n')
        else:
            if datastream!='accelerometer':
                ## read in the data one by one file and stack them
                for data_file in files_in_range:
                    dest_path = folder_path + "/" + data_file
                    hour_data = pd.read_csv(dest_path)
                    if df.shape[0]==0:
                        df = hour_data
                    else:
                        df = df.append(hour_data,ignore_index=True)
    
    if datastream == "accelerometer":
        return files_in_range, stamp_start, stamp_end
    else:
        return df, stamp_start, stamp_end

def write_all_summaries(ID, stats_pdframe, output_folder):
    """
    Docstring
    Args: ID: str, stats_pdframe is pd dataframe (summary stats)
          output_path should be the folder path where you want to save the output
    Return: write out as csv files named by user ID
    """
    if os.path.exists(output_folder)==False:
        os.mkdir(output_folder)
    stats_pdframe.to_csv(output_folder + "/" + str(ID) +  ".csv",index=False)
