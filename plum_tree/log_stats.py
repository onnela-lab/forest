import os
import pandas as pd
import numpy as np
from dateutil import tz
from datetime import datetime, timedelta

def comm_logs(data_filepath,output_path,option):
  if option == "both":
    os.mkdir(output_path+"/hourly")
    os.mkdir(output_path+"/daily")
  patient_names = os.listdir(data_filepath)
  for patient_name in patient_names:
    print(patient_name)
    data_folder_calls = data_filepath + "/" + patient_name +  "/calls"
    data_folder_texts = data_filepath + "/" + patient_name +  "/texts"
    ## deal with calls
    call_stats = pd.DataFrame([])
    if os.path.exists(data_folder_calls):
      data_files_calls = os.listdir(data_folder_calls)
      for data_file in data_files_calls:
        dest_path = data_folder_calls + "/" + data_file
        hour_calls = pd.read_csv(dest_path)
        current_t = datetime.fromtimestamp(hour_calls["timestamp"][0]/1000)
        final_t = current_t - timedelta(minutes=current_t.minute,
                    seconds=current_t.second, microseconds=current_t.microsecond)
        final_t = datetime.timestamp(final_t)
        hour_calls['duration in seconds'] = hour_calls['duration in seconds'].fillna(0.0).astype(int)
        hour_calls['duration in minutes'] = hour_calls['duration in seconds']/60
        index_in = np.array(hour_calls['call type'])=="Incoming Call"
        index_out = np.array(hour_calls['call type'])=="Outgoing Call"
        index_mis = np.array(hour_calls['call type'])=="Missed Call"
        num_in = sum(index_in)
        num_out = sum(index_out)
        num_mis = sum(index_mis)
        num_uniq_in_tel = len(np.unique(np.array(hour_calls['hashed phone number'])[index_in]))
        num_uniq_out_tel = len(np.unique(np.array(hour_calls['hashed phone number'])[index_out]))
        num_uniq_mis_tel = len(np.unique(np.array(hour_calls['hashed phone number'])[index_mis]))
        total_time_in = sum(np.array(hour_calls['duration in minutes'])[index_in])
        total_time_out = sum(np.array(hour_calls['duration in minutes'])[index_out])
        call_stats = call_stats.append({'ID':patient_name,'time':final_t,'num_in':num_in,'num_out':num_out,
                                       'num_mis':num_mis,'uniq_in':num_uniq_in_tel,'uniq_out':num_uniq_out_tel,
                                       'uniq_mis':num_uniq_mis_tel,'tt_in':total_time_in,'tt_out':total_time_out},
                                       ignore_index=True)
    call_stats.fillna(0)

    ## deal with texts
    text_stats = pd.DataFrame([])
    if os.path.exists(data_folder_texts):
      data_files_texts = os.listdir(data_folder_texts)
      for data_file in data_files_texts:
        dest_path = data_folder_texts + "/" + data_file
        hour_texts = pd.read_csv(dest_path)
        current_t = datetime.fromtimestamp(hour_texts["timestamp"][0]/1000)
        final_t = current_t - timedelta(minutes=current_t.minute,
                    seconds=current_t.second, microseconds=current_t.microsecond)
        final_t = datetime.timestamp(final_t)
        m_len = np.array(hour_texts['message length'])
        for k in range(len(m_len)):
          if m_len[k]=="MMS":
            m_len[k]=0
          if isinstance(m_len[k], str)==False:
            if np.isnan(m_len[k]):
               m_len[k]=0
        m_len = m_len.astype(int)
        index_s = np.array(hour_texts['sent vs received'])=="sent SMS"
        index_r = np.array(hour_texts['sent vs received'])=="received SMS"
        index_mms_s = np.array(hour_texts['sent vs received'])=="sent MMS"
        index_mms_r = np.array(hour_texts['sent vs received'])=="received MMS"
        num_s = sum(index_s.astype(int))
        num_r = sum(index_r.astype(int))
        num_mms_s = sum(index_mms_s.astype(int))
        num_mms_r = sum(index_mms_r.astype(int))
        num_s_tel = len(np.unique(np.array(hour_texts['hashed phone number'])[index_s]))
        num_r_tel = len(np.unique(np.array(hour_texts['hashed phone number'])[index_r]))
        total_words_s = sum(m_len[index_s])
        total_words_r = sum(m_len[index_r])
        text_stats = text_stats.append({'ID':patient_name,'time':final_t,'num_s':num_s,'num_r':num_r,
                                       'uniq_s':num_s_tel,'uniq_r':num_r_tel,'tw_s':total_words_s,
                                       'tw_r':total_words_r,'num_mms_s':num_mms_s,'num_mms_r':num_mms_r},
                                       ignore_index=True)
    text_stats = text_stats.fillna(0)

    ## create a full table, fill the days without records as 0
    ## use identifier to find the first hour
    if os.path.exists(data_folder_calls) and os.path.exists(data_folder_texts):
      identifier_Files = os.listdir(data_filepath + "/" + patient_name +  "/identifiers")
      identifiers = pd.read_csv(data_filepath + "/" + patient_name +  "/identifiers/"+ identifier_Files[0], sep = ",")
      start_t = datetime.fromtimestamp(identifiers["timestamp"][0]/1000)
      start_t = start_t - timedelta(minutes=start_t.minute,seconds=start_t.second, microseconds=start_t.microsecond)+timedelta(hours=1)

      ##Last hour: look at all the subject's directories (except survey) and find the latest datefor each directory
      directories = os.listdir(data_filepath + "/" + patient_name)
      directories = list(set(directories)-set(["survey_answers","survey_timings"]))
      lastDate = []
      for i in directories:
        files = os.listdir(data_filepath + "/" + patient_name + "/" + i)
        lastDate.append(files[-1])
      from_zone = tz.gettz('UTC')
      to_zone = tz.gettz('America/New_York')
      UTC = [datetime.strptime(i, '%Y-%m-%d %H_%M_%S.csv') for i in lastDate]
      EST = [i.replace(tzinfo=from_zone).astimezone(to_zone) for i in UTC]
      end_t = max(EST).replace(tzinfo=None)
      hour_diff = int((end_t - start_t).total_seconds()/(60*60))+1
      call_t_array = np.array(call_stats["time"])
      text_t_array = np.array(text_stats["time"])
      full_logs = np.zeros([1,20]).astype(float)  ##  number of columns in the final output
      for j in range(hour_diff):
        current_t = datetime.timestamp(start_t) + 3600*j
        if sum(call_t_array==current_t)==1:
          call_log = (np.array(call_stats.loc[call_t_array==current_t])[0])[np.array([1,2,3,5,6,7,8,9])].astype(float)
        else:
          call_log = np.zeros(8).astype(float)
        if sum(text_t_array==current_t)==1:
          text_log = (np.array(text_stats.loc[text_t_array==current_t])[0])[np.array([1,2,3,4,6,7,8,9])].astype(float)
        else:
          text_log = np.zeros(8).astype(float)
        t = datetime.fromtimestamp(current_t)
        hour_log = np.concatenate((np.array([t.year,t.month,t.day,t.hour]),call_log,text_log))
        full_logs = np.vstack((full_logs,hour_log))
      full_logs = pd.DataFrame(np.delete(full_logs,0,0))
      full_logs.columns = ["year","month","day","hour","num_in","num_mis","num_out","time_in","time_out","uniq_in",
                     "uniq_mis","uniq_out","mms_r","mms_s","num_r","num_s","char_r","char_s","uniq_r","uniq_s"]
      if option == "hourly":
        full_logs.to_csv(output_path + "/" + patient_name + "_hourly_log.csv",index=False)
      if option == "daily":
        day_logs = full_logs.groupby(["year","month","day"]).sum().reset_index()
        day_logs = day_logs.drop(columns = ["hour"])
        day_logs.to_csv(output_path + "/" + patient_name + "_daily_log.csv",index=False)
      if option == "both":
        full_logs.to_csv(output_path + "/hourly/" + patient_name + "_hourly_log.csv",index=False)
        day_logs = full_logs.groupby(["year","month","day"]).sum().reset_index()
        day_logs = day_logs.drop(columns = ["hour"])
        day_logs.to_csv(output_path + "/daily/" + patient_name + "_daily_log.csv",index=False)


## example:
option = "both"
data_filepath = "F:/DATA/hope"
output_path = "C:/Users/glius/Downloads/hope_log"
comm_logs(data_filepath,output_path,option)
