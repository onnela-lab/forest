import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from dateutil import tz
from scipy import fftpack, stats
from itertools import chain
from datetime import datetime,timedelta
import pytz
from pytz import timezone
import calendar

def stamp2datetime(stamp,tz_str):
    """
    Docstring
    Args: stamp: Unix time, integer, the timestamp in Beiwe
          tz_str: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: a list of integers (year, month, day, hour (0-23), min) in specified tz
    """
    loc_tz =  timezone(tz_str)
    utc = timezone("UTC")
    utc_dt = utc.localize(datetime.utcfromtimestamp(stamp))
    loc_dt = utc_dt.astimezone(loc_tz)
    return [loc_dt.year, loc_dt.month,loc_dt.day,loc_dt.hour,loc_dt.minute,loc_dt.second]

def datetime2stamp(time_list,tz_str):
    """
    Docstring
    Args: time_list: a list of integers (year, month, day, hour (0-23), min, sec),
          tz_str: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: Unix time, which is what Beiwe uses
    """
    loc_tz =  timezone(tz_str)
    loc_dt = loc_tz.localize(datetime(int(time_list[0]), int(time_list[1]), int(time_list[2]),
                           int(time_list[3]), int(time_list[4]), int(time_list[5])))
    utc = timezone("UTC")
    utc_dt = loc_dt.astimezone(utc)
    timestamp = calendar.timegm(utc_dt.timetuple())
    return timestamp

def smooth_data(data,hz,tz_str):
  """
  Args: data: pd dataframe of the raw acc data
        hz: the target sampling frequency
        tz_str: timezone (str), where the study is conducted
  Return: a list of integers (year, month, day, hour (0-23), min) in specified tz
  """
  stamp = data['timestamp'][0]/1000
  time_list = stamp2datetime(stamp,tz_str)
  time_list[4] = 0 ; time_list[5]=0
  t = np.array(data["timestamp"])
  x = np.array(data["x"])
  y = np.array(data["y"])
  z = np.array(data["z"])
  mag = np.sqrt(x**2+y**2+z**2)
  t_diff = t[1:]-t[:-1]
  t_active = sum(t_diff[t_diff<5*1000])
  t_active = t_active/1000/60  ## in minute
  a = np.floor((t - min(t))/(1/hz*1000))  ## bin
  b = []
  for i in np.unique(a):
    index = a==i
    b.append(np.mean(mag[index]))
  b = np.array(b)
  return time_list,t_active,np.unique(a)*(1/hz*1000),b

def step_est(t,mag,t_active,q,c):
  if np.mean(mag)>8:
    g = 9.8
  else:
    g = 1
  h = max(np.percentile(mag,q),c*g)
  step = 0
  current = -350
  for j in range(len(t)):
    if(mag[j]>=h and t[j]>=current+350):
      step = step + 1
      current = t[j]
  final_step = int(step/t_active*60)
  return final_step

def acc_stats(mag,hz):
  if np.mean(mag)>8:
    mag = mag/9.8
  m_mag = np.mean(mag)
  sd_mag = np.std(mag)
  cur_len = np.mean(abs(mag[1:]-mag[:-1]))
  X = fftpack.fft(mag)
  amplitude_spectrum = np.abs(X)/hz
  eg = sum(amplitude_spectrum**2)*hz/len(mag)**2
  entropy = stats.entropy(mag)
  return [m_mag,sd_mag,cur_len,eg,entropy]

def GetAccStats(time_list,t_active,t,mag,hz,q,c):
  steps = step_est(t,mag,t_active,q,c)
  others = acc_stats(mag,hz)
  result = [time_list[0],time_list[1],time_list[2],time_list[3],t_active,steps,
            others[0],others[1],others[2],others[3],others[4]]
  return np.array(result)

def patient_stats(path,hz,q,c,tz_str):
  files = os.listdir(path)
  result = []
  for i in range(len(files)):
    dest_path = path + "/" + files[i]
    data = pd.read_csv(dest_path)
    t_list,t_active,t,mag = smooth_data(data,hz,tz_str)
    if t_active>1:
      result.append(GetAccStats(t_list,t_active,t,mag,hz,q,c))
  result = np.array(result)
  return result

def hour_range(h1,h2):
  """
  Args: h1: the hour in the center
        h2: the window size on two sides
        for example: (4,2)--> 2,3,4,5,6,   (23,1)-->22,23,0
  Return: a numpy array of hours
  """
  if h1 + h2 > 23:
    out = np.arange(0,h1+h2-24+1)
    out = np.append(np.arange(h1-h2,24),out)
  elif h1 - h2 < 0:
    out = np.arange(h1-h2+24,24)
    out = np.append(out,np.arange(0,h1+h2+1))
  else:
    out = np.arange(h1-h2,h1+h2+1)
  return out

def check_exist(a1,a2):
  """
  check if each element in a1 is in a2, both a1 and a2 should be numpy array
  Return: a bool vector of len(a1)
  """
  b = np.zeros(len(a1))
  for i in range(len(a1)):
    if sum(a1[i]==a2)>0:
      b[i] = 1
  return np.array(b,dtype=bool)

def fill_missing(p_stats,tz_str):
  full_stats = []
  start_t = datetime2stamp([p_stats[0,0],p_stats[0,1],p_stats[0,2],p_stats[0,3],0,0],tz_str)
  end_t = datetime2stamp([p_stats[-1,0],p_stats[-1,1],p_stats[-1,2],p_stats[-1,3],0,0],tz_str)
  k = int((end_t - start_t)/3600 + 1)
  j = 0
  for i in range(k):
    current_t = start_t + 3600*i
    if current_t == datetime2stamp([p_stats[j,0],p_stats[j,1],p_stats[j,2],p_stats[j,3],0,0],tz_str):
      full_stats.append(p_stats[j,:])
      j = j + 1
    else:
      time_list = stamp2datetime(current_t,tz_str)
      if sum(time_list[3]==p_stats[:,3])>15:
        candidates = p_stats[p_stats[:,3]==time_list[3],:]
      else:
        index = check_exist(p_stats[:,3],hour_range(time_list[3],1))
        if sum(index)<15:
          index = np.arange(p_stats.shape[0])
        candidates = p_stats[index,:]
      r = np.random.randint(candidates.shape[1])
      temp = [[time_list[0],time_list[1],time_list[2],time_list[3],0],candidates[r,np.arange(5,11)].tolist()]
      newline = np.array(list(itertools.chain(*temp)))
      full_stats.append(newline)
  return np.array(full_stats)

def hour2day(full_stats):
  daily_stats = []
  if full_stats.shape[0]>24:
    hours = full_stats[:,3]
    start_index = np.where(hours==0)[0]
    end_index = np.where(hours==23)[0]
    end_index = end_index[end_index>start_index[0]]
    for i in range(len(end_index)):
      index = np.arange(start_index[i],end_index[i]+1)
      temp = full_stats[index,:]
      newline = np.concatenate((temp[0,np.arange(3)],np.sum(temp[:,np.arange(4,11)],axis=0)))
      daily_stats.append(newline)
  return np.array(daily_stats)

def summarize_acc(input_path,output_path,option,tz_str,hz=10,q=75,c=1.05):
  user_list = os.listdir(input_path)
  if option == "both" and os.path.exists(output_path+"/hourly")==False:
    os.mkdir(output_path+"/hourly")
    os.mkdir(output_path+"/daily")
  for i in range(len(user_list)):
    sys.stdout.write( "Processing data from "+ user_list[i]  + '\n')
    acc_path = input_path + "/" + user_list[i] +"/accelerometer"
    if os.path.isdir(acc_path):
      p_stats = patient_stats(acc_path,hz,q,c,tz_str)
      full_stats = fill_missing(p_stats,tz_str)
      if option == "hourly":
        full_stats = pd.DataFrame(full_stats)
        full_stats.columns = ["year","month","day","hour","active_min","steps","mean_mag","sd_mag",
                              "cur_len","energy","entropy"]
        dest_path = output_path + "/" + user_list[i] + "_hourly_acc.csv"
        full_stats.to_csv(dest_path,index=False)
      if option == "daily":
        daily_stats = hour2day(full_stats)
        if daily_stats.shape[0]>0:
          daily_stats = pd.DataFrame(daily_stats)
          daily_stats.columns = ["year","month","day","active_min","steps","mean_mag","sd_mag",
                              "cur_len","energy","entropy"]
          dest_path = output_path + "/" + user_list[i] + "_daily_acc.csv"
          daily_stats.to_csv(dest_path,index=False)
        else:
          print('The duration of data collection is less than a day.')
      if option == "both":
        daily_stats = hour2day(full_stats)
        if daily_stats.shape[0]>0:
          daily_stats = pd.DataFrame(daily_stats)
          daily_stats.columns = ["year","month","day","active_min","steps","mean_mag","sd_mag",
                              "cur_len","energy","entropy"]
          dest_path = output_path+"/daily/" + user_list[i] + "_daily_acc.csv"
          daily_stats.to_csv(dest_path,index=False)
        else:
          print('The duration of data collection is less than a day.')
        full_stats = pd.DataFrame(full_stats)
        full_stats.columns = ["year","month","day","hour","active_min","steps","mean_mag","sd_mag",
                              "cur_len","energy","entropy"]
        dest_path = output_path+"/hourly/" + user_list[i] + "_hourly_acc.csv"
        full_stats.to_csv(dest_path,index=False)
    sys.stdout.write( "Done" + '\n')

input_path = "F:/DATA/hope"
output_path = "C:/Users/glius/Downloads/hope_acc"
option = "both"
tz_str = "America/New_York"
summarize_acc(input_path,output_path,option,tz_str,hz=10,q=85,c=1.1)
