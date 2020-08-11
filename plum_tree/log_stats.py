import os
import sys
import pandas as pd
import numpy as np
from dateutil import tz
from datetime import datetime
import pytz
from pytz import timezone
import calendar
import logging
logger = logging.getLogger(__name__)



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
        logger.warning('Overwriting existing file with that name.')
    f = open(path, 'w')
    f.write(','.join(header) + '\n')
    f.close()
    return(path)

# Dictionary of log record attributes:
# (For details:  https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes)
log_attributes = {
'asctime,msecs': '%(asctime)s', # Human-readable time with milliseconds.
'created': '%(created)f',       # Unix timestamp (seconds since epoch).
'filename': '%(filename)s',     # Filename portion of pathname.
'funcName': '%(funcName)s',     # Originating function.
'levelname': '%(levelname)s', # Message level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
'levelno': '%(levelno)s',     # Numeric message level.
'lineno': '%(lineno)d',       # Source line number, if available.
'message': '%(message)s',     # Logged message.
'module': '%(module)s',       # Module name.
'msecs': '%(msecs)d',         # Millisecond portion of timestamp.
'name': '%(name)s',              # Name of originating logger.
'pathname': '%(pathname)s',      # Path to originating file.
'process': '%(process)d',        # Process id, if available.
'processName': '%(processName)s',# Process name, if available.
'relativeCreated': '%(relativeCreated)d', # Milliseconds since logging was loaded.
'thread': '%(thread)d',          # Thread id, if available.
'threadName': '%(threadName)s',  # Thread name, if available.
 }

def attributes_to_csv(attribute_list):
    '''
    Given a list of attributes (keys of log_attributes), returns a logging
    format with header for writing records to CSV.
    '''
    try:
        log_format = type('logging_format', (), {})()
        attributes = [log_attributes[a] for a in attribute_list]
        log_format.attributes = ','.join(attributes)
        log_format.header = []
        for a in attribute_list:
            if ',' in a: log_format.header += a.split(',') # hack for asctime
            else: log_format.header.append(a)
    except:
        logger.warning('Unable to assemble logging format.')
    return(log_format)

# Simple format for logging messages:
basic_format = attributes_to_csv([
    'created',
    'asctime,msecs',
    'levelname',
    'module',
    'funcName',
    'message',
    ])

# More comprehensive format for logging messages, including traceback info:
traceback_format = attributes_to_csv([
    'created',
    'asctime,msecs',
    'levelname',
    'module',
    'funcName',
    'message',
    'lineno',
    'pathname'
    ])

def log_to_csv(log_dir, level = logging.DEBUG,
               log_name = 'log',
               log_format = traceback_format.attributes,
               log_header = traceback_format.header):
    '''
    Configure the logging system to write messages to a csv.
    Overwrites any existing logging handlers and configurations.

    Args:
        log_dir (str): Path to a directory where log messages should be written.
        level (int):  An integer between 0 and 50.
            Set level = logging.DEBUG to log all messages.
        log_name (str): Name for the log file.
        log_format (str): The format argument for logging.basicConfig.
            For available attributes and formatting instructions, see:
            https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes)
        log_header (list): Header for the csv.

    Returns:
        None
    '''
    try:
        # initialize csv
        filepath = setup_csv(name = log_name, directory = log_dir, header = log_header)
        # configure logging output
        logging.basicConfig(format = log_format,
                            filename = filepath,
                            level = level, force = True)
        # success message
        logger.info('Writing log messages to %s.csv...' % log_name)
    except:
        logger.warning('Unable to write logging messages.')

def datetime2stamp(time_tuple,tz_str):
    """
    Docstring
    Args: time_tupe: a tuple of integers (year, month, day, hour (0-23), min, sec),
          tz_str: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: Unix time, which is what Beiwe uses
    """
    try:
        loc_tz =  timezone(tz_str)
        loc_dt = loc_tz.localize(datetime(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3], time_tuple[4], time_tuple[5]))
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
    Return: a tuple of integers (year, month, day, hour (0-23), min) in specified tz
    """
    try:
        loc_tz =  timezone(tz_str)
        utc = timezone("UTC")
        utc_dt = utc.localize(datetime.utcfromtimestamp(stamp))
        loc_dt = utc_dt.astimezone(loc_tz)
        return (loc_dt.year, loc_dt.month,loc_dt.day,loc_dt.hour,loc_dt.minute,loc_dt.second)
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

def read_comm_logs(ID:str, study_folder: str, tz_str: str, time_start = None, time_end = None):
    """
    Docstring
    Args: a beiwe IDs (str), starting time and ending time of the window of interest
          time should be a tuple of integers with format (year, month, day, hour, minute, second)
          tz_str(timezone) should be where the study is/was conducted (str)
          if time_start is None and time_end is None: then it reads all the available files
          if time_start is None and time_end is given, then it reads all the files before the given time
          if time_start is given and time_end is None, then it reads all the files after the given time
    return: two pandas dataframes of records (one for calls and one for texts)
            and corresponding starting/ending timestamp (UTC)
    """
    df_text = pd.DataFrame()
    df_call = pd.DataFrame()
    stamp_start = 0 ; stamp_end = 0
    folder_path_text = study_folder + "/" + ID +  "/texts"
    folder_path_call = study_folder + "/" + ID +  "/calls"
    ## if text folder exists, call folder must exists
    try:
        if os.path.exists(folder_path_text):
            text_files = np.array(os.listdir(folder_path_text))
            call_files = np.array(os.listdir(folder_path_call))
            ## create a list to convert all filenames to UNIX time
            text_filestamp = np.array([filename2stamp(filename) for filename in text_files])
            call_filestamp = np.array([filename2stamp(filename) for filename in call_files])

            ## find the timestamp in the identifier (when the user was enrolled)
            identifier_Files = os.listdir(study_folder + "/" + ID + "/identifiers")
            identifiers = pd.read_csv(study_folder + "/" + ID + "/identifiers/"+ identifier_Files[0], sep = ",")
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

            text_files_in_range = text_files[(text_filestamp>=stamp_start)*(text_filestamp<stamp_end)]
            call_files_in_range = call_files[(call_filestamp>=stamp_start)*(call_filestamp<stamp_end)]

            if len(text_files_in_range)==0:
                df_text = pd.DataFrame()
            else:
                for data_file in text_files_in_range:
                    dest_path = folder_path_text + "/" + data_file
                    hour_text = pd.read_csv(dest_path)
                    if df_text.shape[0]==0:
                        df_text = hour_text
                    else:
                        df_text = df_text.append(hour_text,ignore_index=True)

            if len(call_files_in_range)==0:
                df_call = pd.DataFrame()
            else:
                for data_file in call_files_in_range:
                    dest_path = folder_path_call + "/" + data_file
                    hour_call = pd.read_csv(dest_path)
                    if df_call.shape[0]==0:
                        df_call = hour_call
                    else:
                        df_call = df_call.append(hour_call,ignore_index=True)

        else:
            if os.path.exists(study_folder + "/" + ID):
                logger.info('User '+ str(ID) + ' does not have call/text data (not collected).')
            else:
                logger.info('User '+ str(ID) + ' does not exist, please check the ID again.')
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))
    return df_text, df_call, stamp_start, stamp_end

def comm_logs_summaries(ID:str, df_text, df_call, stamp_start, stamp_end, tz_str, option):
    """
    Docstring
    Args: The outputs from read_comm_logs(). Option is 'daily' or 'hourly', determining the resolution of the summary stats
          tz_str: timezone where the study was/is conducted
    Return: pandas dataframe of summary stats
    """
    try:
        summary_stats = []
        (start_year, start_month, start_day, start_hour, start_min, start_sec) = stamp2datetime(stamp_start,tz_str)
        (end_year, end_month, end_day, end_hour, end_min, end_sec) = stamp2datetime(stamp_end,tz_str)
        if option == 'hourly':
            table_start = datetime2stamp((start_year, start_month, start_day, start_hour,0,0),tz_str)
            table_end = datetime2stamp((end_year, end_month, end_day, end_hour,0,0),tz_str)
            step_size = 3600
        if option == 'daily':
            table_start = datetime2stamp((start_year, start_month, start_day, 0,0,0),tz_str)
            table_end = datetime2stamp((end_year, end_month, end_day,0,0,0),tz_str)
            step_size = 3600*24
        for stamp in np.arange(table_start,table_end+1,step=step_size):
            (year, month, day, hour, minute, second) = stamp2datetime(stamp,tz_str)
            if df_text.shape[0] > 0:
                temp_text = df_text[(df_text["timestamp"]/1000>=stamp)&(df_text["timestamp"]/1000<stamp+step_size)]
                m_len = np.array(temp_text['message length'])
                for k in range(len(m_len)):
                    if m_len[k]=="MMS":
                        m_len[k]=0
                    if isinstance(m_len[k], str)==False:
                        if np.isnan(m_len[k]):
                            m_len[k]=0
                m_len = m_len.astype(int)
                index_s = np.array(temp_text['sent vs received'])=="sent SMS"
                index_r = np.array(temp_text['sent vs received'])=="received SMS"
                index_mms_s = np.array(temp_text['sent vs received'])=="sent MMS"
                index_mms_r = np.array(temp_text['sent vs received'])=="received MMS"
                num_s = sum(index_s.astype(int))
                num_r = sum(index_r.astype(int))
                num_mms_s = sum(index_mms_s.astype(int))
                num_mms_r = sum(index_mms_r.astype(int))
                num_s_tel = len(np.unique(np.array(temp_text['hashed phone number'])[index_s]))
                num_r_tel = len(np.unique(np.array(temp_text['hashed phone number'])[index_r]))
                total_char_s = sum(m_len[index_s])
                total_char_r = sum(m_len[index_r])

            if df_call.shape[0] > 0:
                temp_call = df_call[(df_call["timestamp"]/1000>=stamp)&(df_call["timestamp"]/1000<stamp+step_size)]
                dur_in_sec = np.array(temp_call['duration in seconds'])
                dur_in_sec[np.isnan(dur_in_sec)==True] = 0
                dur_in_min = dur_in_sec/60
                index_in_call = np.array(temp_call['call type'])=="Incoming Call"
                index_out_call = np.array(temp_call['call type'])=="Outgoing Call"
                index_mis_call = np.array(temp_call['call type'])=="Missed Call"
                num_in_call = sum(index_in_call)
                num_out_call = sum(index_out_call)
                num_mis_call = sum(index_mis_call)
                num_uniq_in_call = len(np.unique(np.array(temp_call['hashed phone number'])[index_in_call]))
                num_uniq_out_call = len(np.unique(np.array(temp_call['hashed phone number'])[index_out_call]))
                num_uniq_mis_call = len(np.unique(np.array(temp_call['hashed phone number'])[index_mis_call]))
                total_time_in_call = sum(dur_in_min[index_in_call])
                total_time_out_call = sum(dur_in_min[index_out_call])
            newline = [year, month, day, hour, num_in_call, num_out_call, num_mis_call, num_uniq_in_call, num_uniq_out_call,
                  num_uniq_mis_call, total_time_in_call, total_time_out_call, num_s, num_r, num_mms_s, num_mms_r, num_s_tel,
                  num_r_tel, total_char_s, total_char_r]
            summary_stats.append(newline)
        summary_stats = np.array(summary_stats)
        if option == 'daily':
            summary_stats = np.delete(summary_stats, 3, 1)
            stats_pdframe = pd.DataFrame(summary_stats, columns=['year', 'month', 'day','num_in_call', 'num_out_call', 'num_mis_call',
                    'num_in_caller', 'num_out_caller','num_mis_caller', 'total_mins_in_call', 'total_mins_out_call',
                    'num_s', 'num_r', 'num_mms_s', 'num_mms_r', 'num_s_tel','num_r_tel', 'total_char_s', 'total_char_r'])
        if option == 'hourly':
            stats_pdframe = pd.DataFrame(summary_stats, columns=['year', 'month', 'day','hour','num_in_call', 'num_out_call',
                    'num_mis_call','num_in_caller', 'num_out_caller','num_mis_caller', 'total_mins_in_call', 'total_mins_out_call',
                    'num_s', 'num_r', 'num_mms_s', 'num_mms_r', 'num_s_tel','num_r_tel', 'total_char_s', 'total_char_r'])
        return stats_pdframe
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))

def write_all_summaries(ID, stamp_start, stamp_end, stats_pdframe, output_folder):
    """
    Docstring
    Args: ID: str, stamp_start, stamp_end are int, stats_pdframe is pd dataframe
          output_path should be the folder path where you want to save the output
    Return: write out as csv files named by user ID and timestamps
    """
    try:
        if os.path.exists(output_folder)==False:
            os.mkdir(output_folder)
        stats_pdframe.to_csv(output_folder + "/" + str(ID) + "_" + str(int(stamp_start)) + "_"+str(int(stamp_end)) + ".csv",index=False)
        print("User " + str(ID) + ' : Done.')
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))


# Main function/wrapper should take standard arguments with Beiwe names:
def log_stats_main(study_folder: str, output_folder:str, tz_str: str,  option: str, time_start = None, time_end = None, beiwe_id = None):
    log_to_csv(output_folder)
    logger.info("Begin")
    ## beiwe_id should be a list of str
    if beiwe_id == None:
        beiwe_id = os.listdir(study_folder)
    for ID in beiwe_id:
        try:
            text_data, call_data, stamp_start, stamp_end = read_comm_logs(ID, study_folder, tz_str, time_start, time_end)
            stats_pdframe = comm_logs_summaries(ID, text_data, call_data, stamp_start, stamp_end, tz_str, option)
            write_all_summaries(ID, stamp_start, stamp_end, stats_pdframe, output_folder)
        except:
            if text_data.shape[0]>0 or call_data.shape[0]>0:
                logger.debug("There is a problem unrelated to data for user %s." % str(ID))
    logger.info("End")
    temp = pd.read_csv(output_folder + "/log.csv")
    if temp.shape[0]==3:
      print("Finished without any warning messages.")
    else:
      print("Finished. Please check log.csv for warning messages.")

## test the code
study_folder = 'F:/DATA/hope'
output_folder = 'C:/Users/glius/Downloads/hope_log'
tz_str = 'America/New_York'
option = 'hourly'
log_stats_main(study_folder,output_folder,tz_str,option)
