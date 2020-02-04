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

def smooth_data(data,hz):
  stamp0 = datetime.fromtimestamp(data['timestamp'][0]/1000)
  stamp1 = [stamp0.year,stamp0.month,stamp0.day,stamp0.hour]
  stamp0 = np.floor(data['timestamp'][0]/1000/60/60)*60*60
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
  return stamp0,stamp1,t_active,np.unique(a)*(1/hz*1000),b

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

def GetAccStats(stamp0,stamp1,t_active,t,mag,hz,q,c):
  steps = step_est(t,mag,t_active,q,c)
  others = others = acc_stats(mag,hz)
  result = [stamp0,stamp1[0],stamp1[1],stamp1[2],stamp1[3],t_active,steps,
            others[0],others[1],others[2],others[3],others[4]]
  return np.array(result)

def patient_stats(path,hz,q,c):
  files = os.listdir(path)
  result = []
  for i in range(len(files)):
    dest_path = path + "/" + files[i]
    data = pd.read_csv(dest_path)
    stamp0,stamp1,t_active,t,mag = smooth_data(data,hz)
    if t_active>1:
      result.append(GetAccStats(stamp0,stamp1,t_active,t,mag,hz,q,c))
  result = np.array(result)
  return result

def hour_range(h1,h2):
  if h1 + h2 > 23:
    out = np.arange(0,h1+h2-24+1)
    out = np.append(out,np.arange(h1-h2,24))
  elif h1 - h2 < 0:
    out = np.arange(h1-h2+24,24)
    out = np.append(out,np.arange(0,h1+h2+1))
  else:
    out = np.arange(h1-h2,h1+h2+1)
  return out

def check_exist(a1,a2):
  b = np.zeros(len(a1))
  for i in range(len(a1)):
    if sum(a1[i]==a2)>0:
      b[i] = 1
  return np.array(b,dtype=bool)

def fill_missing(p_stats):
  full_stats = []
  start_t = p_stats[0,0]
  end_t = p_stats[-1,0]
  k = int((end_t - start_t)/3600 + 1)
  j = 0
  for i in range(k):
    current_t = start_t + 3600*i
    if current_t == p_stats[j,0]:
      full_stats.append(p_stats[j,:])
      j = j + 1
    else:
      t = datetime.fromtimestamp(current_t)
      if sum(t.hour==p_stats[:,4])>15:
        candidates = p_stats[p_stats[:,4]==t.hour,:]
      else:
        index = check_exist(p_stats[:,4],hour_range(t.hour,2))
        if sum(index)<15:
          index = np.arange(p_stats.shape[0])
        candidates = p_stats[index,:]
      r = np.random.randint(candidates.shape[1])
      temp = [[current_t,t.year,t.month,t.day,t.hour,0],candidates[r,np.arange(6,12)].tolist()]
      newline = np.array(list(itertools.chain(*temp)))
      full_stats.append(newline)
  return np.array(full_stats)

def hour2day(full_stats):
  daily_stats = []
  t = full_stats[:,0]
  days = t/(60*60*24)
  start_day = np.ceil(days[0])
  end_day = np.floor(days[-1])
  for i in np.arange(start_day,end_day+1):
    temp = full_stats[(days>=i)*(days<i+1)]
    newline = np.append(temp[0,np.arange(1,4)],np.sum(temp[:,np.arange(5,12)],axis=0))
    daily_stats.append(newline)
  return np.array(daily_stats)

def summarize_acc(input_path,output_path,option,hz=10,q=75,c=1.05):
  user_list = os.listdir(input_path)
  if option == "both":
    os.mkdir(output_path+"/hourly")
    os.mkdir(output_path+"/daily")
  for i in range(len(user_list)):
    sys.stdout.write( "Processing data from "+ user_list[i]  + '\n')
    acc_path = input_path + "/" + user_list[i] +"/accelerometer"
    if os.path.isdir(acc_path):
      p_stats = patient_stats(acc_path,hz,q,c)
      full_stats = fill_missing(p_stats)
      if option == "hourly":
        full_stats = full_stats[:,1:]
        full_stats = pd.DataFrame(full_stats)
        full_stats.columns = ["year","month","day","hour","active_min","steps","mean_mag","sd_mag",
                              "cur_len","energy","entropy"]
        dest_path = output_path + "/" + user_list[i] + "_hourly_acc.csv"
        full_stats.to_csv(dest_path,index=False)
      if option == "daily":
        daily_stats = hour2day(full_stats)
        daily_stats = pd.DataFrame(daily_stats)
        daily_stats.columns = ["year","month","day","active_min","steps","mean_mag","sd_mag",
                              "cur_len","energy","entropy"]
        dest_path = output_path + "/" + user_list[i] + "_daily_acc.csv"
        daily_stats.to_csv(dest_path,index=False)
      if option == "both":
        daily_stats = hour2day(full_stats)
        daily_stats = pd.DataFrame(daily_stats)
        daily_stats.columns = ["year","month","day","active_min","steps","mean_mag","sd_mag",
                              "cur_len","energy","entropy"]
        dest_path = output_path+"/daily/" + user_list[i] + "_daily_acc.csv"
        daily_stats.to_csv(dest_path,index=False)
        full_stats = full_stats[:,1:]
        full_stats = pd.DataFrame(full_stats)
        full_stats.columns = ["year","month","day","hour","active_min","steps","mean_mag","sd_mag",
                              "cur_len","energy","entropy"]
        dest_path = output_path+"/hourly/" + user_list[i] + "_hourly_acc.csv"
        full_stats.to_csv(dest_path,index=False)
    sys.stdout.write( "Done" + '\n')

input_path = "F:/DATA/hope"
output_path = "C:/Users/glius/Downloads/hope_acc"
option = "both"
summarize_acc(input_path,output_path,option,hz=10,q=85,c=1.1)
