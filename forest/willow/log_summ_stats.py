# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:57:46 2020


[What needs to be modified by the user]

[Reference to paper / reference to Beiwe blog]

This is version 1.6 (last modified 2020-07-01: unification with forest)
[Details on last modifications (automatically by Github @ comment section)]
[Onnela Lab signature]

"""

import os
import pandas as pd
import numpy as np
from dateutil import tz
from datetime import datetime
import pytz
from pytz import timezone
import calendar

def datetime2stamp(time_tuple, tz):
    """
    Docstring
    Args: time_tupe: a tuple of integers (year, month, day, hour (0-23), min),
          tz: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: Unix time, which is what Beiwe uses
    """
    loc_tz =  timezone(tz)
    loc_dt = loc_tz.localize(datetime(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3], time_tuple[4], time_tuple[5]))
    utc = timezone("UTC")
    utc_dt = loc_dt.astimezone(utc)
    timestamp = calendar.timegm(utc_dt.timetuple())
    return timestamp

def stamp2datetime(stamp,tz):
    """
    Docstring
    Args: stamp: Unix time, integer, the timestamp in Beiwe
          tz: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: a tuple of integers (year, month, day, hour (0-23), min) in specified tz
    """
    loc_tz =  timezone(tz)
    utc = timezone("UTC")
    utc_dt = utc.localize(datetime.utcfromtimestamp(stamp))
    loc_dt = utc_dt.astimezone(loc_tz)
    return (loc_dt.year, loc_dt.month,loc_dt.day,loc_dt.hour,loc_dt.minute,loc_dt.second)

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

def read_comm_logs(study_folder: str, tz: str, ID:str, time_start = None, time_end = None):
    """
    Docstring
    Args: a beiwe IDs (str), starting time and ending time of the window of interest
          time should be a tuple of integers with format (year, month, day, hour, minute, second)
          tz(timezone) should be where the study is/was conducted (str)
          if time_start is None and time_end is None: then it reads all the available files
          if time_start is None and time_end is given, then it reads all the files before the given time
          if time_start is given and time_end is None, then it reads all the files after the given time
    return: two pandas dataframes of records (one for calls and one for texts)
            and corresponding starting/ending timestamp (UTC)
    """
    df_text = pd.DataFrame()
    df_call = pd.DataFrame()
    folder_path_text = study_folder + "/" + ID +  "/texts"
    folder_path_call = study_folder + "/" + ID +  "/calls"
    ## if text folder exists, call folder must exists
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
            stamp_start2 = datetime2stamp(time_start,tz)
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
            stamp_end2 =  datetime2stamp(time_end,tz)
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
            print('User '+ str(ID) + ' does not have call/text data (not collected).')
        else:
            print('User '+ str(ID) + ' does not exist, please check the ID again.')
    return df_text, df_call, stamp_start, stamp_end

def comm_logs_summaries(df_text, df_call, stamp_start, stamp_end, tz, option):
    """
    Docstring
    Args: The outputs from read_comm_logs(). Option is 'daily' or 'hourly', determining the resolution of the summary stats
          tz: timezone where the study was/is conducted
    Return: pandas dataframe of summary stats
    """
    summary_stats = []
    (start_day, start_hour, start_min, start_sec) = stamp2datetime(stamp_start,tz)
    (end_year, end_month, end_day, end_hour, end_min, end_sec) = stamp2datetime(stamp_end,tz)
    if option == 'hourly':
        table_start = datetime2stamp((start_year, start_month, start_day, start_hour,0,0),tz)
        table_end = datetime2stamp((end_year, end_month, end_day, end_hour,0,0),tz)
        step_size = 3600
    if option == 'daily':
        table_start = datetime2stamp((start_year, start_month, start_day, 0,0,0),tz)
        table_end = datetime2stamp((end_year, end_month, end_day,0,0,0),tz)
        step_size = 3600*24
    for stamp in np.arange(table_start,table_end+1,step=step_size):
        (year, month, day, hour, minute, second) = stamp2datetime(stamp,tz)
        if df_text.shape[0] == 0:
            num_s = 0
            num_r = 0
            num_mms_s = 0
            num_mms_r = 0
            num_s_tel = 0
            num_r_tel = 0
            total_char_s = 0
            total_char_r = 0
        else:
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

        if df_call.shape[0] == 0:
            num_in_call = 0
            num_out_call = 0
            num_mis_call = 0
            num_uniq_in_call = 0
            num_uniq_out_call = 0
            num_uniq_mis_call = 0
            total_time_in_call = 0
            total_time_out_call = 0
        else:
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

def write_all_summaries(ID, stamp_start, stamp_end, stats_pdframe, output_folder):
    """
    Docstring
    Args: ID: str, stamp_start, stamp_end are int, stats_pdframe is pd dataframe
          output_path should be the folder path where you want to save the output
    Return: write out as csv files named by user ID and timestamps
    """
    if os.path.exists(output_folder)==False:
        os.mkdir(output_folder)
    df.to_csv(output_folder + "/" + str(ID) + "_" + str(stamp_start) + "_"+str(stamp_end) + ".csv",index=False)
    print("User" + str(ID) + ' : CSV file of summary statistics is generated.')

# Main function/wrapper should take standard arguments with Beiwe names:
def log_stats_main(study_folder: str, output_folder:str, tz: str,  option: str, time_start: str = None, time_end: str = None, beiwe_id = None):
    ## beiwe_id should be a list of str
    if beiwe_id == None:
        beiwe_id = os.listdir(study_folder)
    for ID in beiwe_id:
        try:
            text_data, call_data, stamp_start, stamp_end = read_comm_logs(study_folder, tz, ID, time_start, time_end)
            stats_pdframe = comm_logs_summaries(text_data, call_data, stamp_start, stamp_end, tz, option)
            write_all_summaries(ID, stamp_start, stamp_end, stats_pdframe, output_folder)
        except:
            print("something")
        ## ??
