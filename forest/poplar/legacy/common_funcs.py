import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone
import calendar
import logging


logger = logging.getLogger(__name__)


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
    try:
        loc_tz =  timezone(tz_str)
        loc_dt = loc_tz.localize(datetime(time_list[0], time_list[1], time_list[2], time_list[3], time_list[4], time_list[5]))
        utc = timezone("UTC")
        utc_dt = loc_dt.astimezone(utc)
        timestamp = calendar.timegm(utc_dt.timetuple())
        return timestamp
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))


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
    try:
        loc_tz =  timezone(tz_str)
        utc = timezone("UTC")
        utc_dt = utc.localize(datetime.utcfromtimestamp(stamp))
        loc_dt = utc_dt.astimezone(loc_tz)
        return [loc_dt.year, loc_dt.month,loc_dt.day,loc_dt.hour,loc_dt.minute,loc_dt.second]
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))

def filename2stamp(filename):
    """
    Docstring
    Args: filename (str), the filename of communication log
    Return: UNIX time (int)
    """
    try:
        [d_str,h_str] = filename.split(' ')
        [y,m,d] = np.array(d_str.split('-'),dtype=int)
        h = int(h_str.split('_')[0])
        stamp = datetime2stamp((y,m,d,h,0,0),'UTC')
        return stamp
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))

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
    return: a panda dataframe of the datastream (not for accelerometer data!) and corresponding starting/ending timestamp (UTC),
            you can convert it to numpy array as needed
            For accelerometer data, intsead of a panda dataframe, it returns a list of filenames
            The reason is the volumn of accelerometer data is too large, we need to process it on the fly:
            read one csv file, process one, not wait until all the csv's are imported (that may be too large in memory!)
    """
    df = pd.DataFrame()
    stamp_start = 0 ; stamp_end = 0
    folder_path = study_folder + "/" + ID +  "/" + str(datastream)
    ## if text folder exists, call folder must exists
    try:
        if not os.path.exists(study_folder + "/" + ID):
            logger.warning('User '+ str(ID) + ' does not exist, please check the ID again.')
        elif not os.path.exists(folder_path):
            logger.warning('User '+ str(ID) + ' : ' + str(datastream) + ' data are not collected.')
        else:
            filenames = np.array(os.listdir(folder_path))
            ## create a list to convert all filenames to UNIX time
            filestamps = np.array([filename2stamp(filename) for filename in filenames])
            ## find the timestamp in the identifier (when the user was enrolled)
            identifier_Files = os.listdir(study_folder + "/" + ID + "/identifiers")
            identifiers = pd.read_csv(study_folder + "/" + ID + "/identifiers/"+ identifier_Files[0], sep = ",")
            ## now determine the starting and ending time according to the Docstring
            stamp_start1= identifiers["timestamp"][0]/1000
            if time_start == None:
                stamp_start = stamp_start1
            else:
                stamp_start2 = datetime2stamp(time_start,tz_str)
                stamp_start = max(stamp_start1,stamp_start2)
            ##Last hour: look at all the subject's directories (except survey) and find the latest date for each directory
            directories = os.listdir(study_folder + "/" + ID)
            directories = list(set(directories)-set(["survey_answers","survey_timings"]))
            lastDate = []
            for i in directories:
                files = os.listdir(study_folder + "/" + ID + "/" + i)
                lastDate.append(files[-1])
            stamp_end_vec = [filename2stamp(j) for j in lastDate]
            stamp_end1 = max(stamp_end_vec)
            if time_end == None:
                stamp_end = stamp_end1
            else:
                stamp_end2 =  datetime2stamp(time_end,tz_str)
                stamp_end = min(stamp_end1,stamp_end2)
            ## extract the filenames in range
            files_in_range = filenames[(filestamps>=stamp_start)*(filestamps<stamp_end)]
            if len(files_in_range) == 0:
                logger.warning('User '+ str(ID) + ' : There are no ' + str(datastream) + ' data in range.')
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
                    sys.stdout.write("Data imported ..." + '\n')
                    return df, stamp_start, stamp_end
                else:
                    return files_in_range, stamp_start, stamp_end
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))