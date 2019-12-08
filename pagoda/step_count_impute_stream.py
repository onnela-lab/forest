import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime,timedelta

with open('empirical_steps.pickle', 'rb') as handle:
  empirical_steps = pickle.load(handle)

## 1. smooth to a certain Hz
def smooth_data(data,hz):
  t = datetime.fromtimestamp(data['timestamp'][0]/1000)
  hr = t.hour
  t0 = datetime.timestamp(t-timedelta(minutes=t.minute,seconds=t.second,microseconds=t.microsecond))*1000
  t_seq = np.array(data['timestamp'])
  ## first locate those active bouts (let min interval be s seconds)
  num = np.floor((t_seq - t0)/(1/hz*1000))
  j = 0; i = 1
  new_data = []
  while i<(len(t_seq)-1):
    if num[i]==num[j]:
      i = i + 1
      if i == len(t_seq):
        index = np.arange(j,i)
        mean_x = np.mean(data['x'][index])
        mean_y = np.mean(data['y'][index])
        mean_z = np.mean(data['z'][index])
        new_data.append([t0+1/hz*1000*num[i],mean_x,mean_y,mean_z])
    else:
      index = np.arange(j,i)
      mean_x = np.mean(data['x'][index])
      mean_y = np.mean(data['y'][index])
      mean_z = np.mean(data['z'][index])
      new_data.append([t0+1/hz*1000*num[i],mean_x,mean_y,mean_z])
      j = i
      i = i+1
  w = len(new_data)
  new_data = np.array(new_data).reshape((w,4))
  mag = np.sqrt(new_data[:,1]**2+new_data[:,2]**2+new_data[:,3]**2)
  stamp = new_data[:,0]
  return w,hr,t0,stamp,mag

## 2. minute-wise step estimation funciton
def step_est_min(stamp,mag,t0,hz,q,c):
  if np.mean(mag)>8:
    g = 9.8
  else:
    g = 1
  h = max(np.percentile(mag,q),c*g)
  output = []
  for i in range(60):
    index = (stamp>=t0+i*60*1000)*(stamp<t0+(i+1)*60*1000)
    sub_mag = mag[index]
    sub_stamp = stamp[index]
    if len(sub_mag)<=1:
      output.append([i,0,0,0])
    else:
      step = 0
      current = min(sub_stamp)-350
      for j in np.arange(1,len(sub_mag)):
        if(sub_mag[j]>=h and sub_stamp[j]>=current+350):
          step = step + 1
          current = sub_stamp[j]
      on_time = len(sub_mag)/hz
      output.append([i,on_time,step,np.floor(step*60/on_time)])
  output = pd.DataFrame(np.array(output))
  output.columns = ["min","active_s","step_obs","step_infer"]
  return output

## 3. check if there exists any walk within an interval
def nearby_walk(data,k,h):
  walk = (np.array(data['step_infer'])>h)*1
  record = (np.array(data['active_s'])>0)*1
  nearby = walk
  active = record
  n = data.shape[0]
  for i in np.arange(1,k+1):
    nearby = nearby + np.concatenate((walk[np.arange(i,n)],np.zeros(i))) + np.concatenate((np.zeros(i), walk[np.arange(0,n-i)]))
    active = active + np.concatenate((record[np.arange(i,n)],np.zeros(i))) + np.concatenate((np.zeros(i), record[np.arange(0,n-i)]))
  final = (np.array(active)>=1)*1
  final[(active>=1)*(nearby==0)] = 1
  final[(active>=1)*(nearby>=1)] = 2
  return final

## 4. impute the steps based on the output of nearby_walk
def imp_steps(output,final,hr):
  steps = np.zeros(output.shape[0])
  for i in range(output.shape[0]):
    if np.array(output['active_s'])[i]>5:
      steps[i] = np.array(output['step_infer'])[i]
    elif final[i] == 0:
      r = np.random.choice(range(len(empirical_steps['no_records'][hr])),1)
      steps[i] = empirical_steps['no_records'][hr][r]
    elif final[i] == 1:
      r = np.random.choice(range(len(empirical_steps['non_walk'][hr])),1)
      steps[i] = empirical_steps['non_walk'][hr][r]
    else:
      r = np.random.choice(range(len(empirical_steps['walk'][hr])),1)
      steps[i] = empirical_steps['walk'][hr][r]
  return(sum(steps))


## 5. read in all the data from the folder one by one, and fill in the hours without records
def hourly_step_count(data_path,output_path,hz,q,c,k,h):
  for i in os.listdir(data_path):
    result0 = []
    t_vec = []
    patient_path = data_path+"/"+i+"/accelerometer"
    if os.path.exists(patient_path):
      sys.stdout.write("Read and preprocess the data from user "+i+"..." + '\n')
      for j in os.listdir(patient_path):
        path = patient_path + "/" + j
        sys.stdout.write(j+"...")
        data = pd.read_csv(path)
        w,hr,t0,stamp,mag = smooth_data(data,hz)
        if w>10:
          sys.stdout.write("  Estimating Steps..."+'\n')
          output = step_est_min(stamp,mag,t0,hz,q,c)
          final = nearby_walk(output,k,h)
          step = imp_steps(output,final,hr)
          t1 = datetime.fromtimestamp(t0/1000)
          t_vec.append(t0)
          result0.append([t1.year,t1.month,t1.day,t1.hour,step])
        else:
          sys.stdout.write("  Data quality is too low..."+'\n')
      t_vec = np.array(t_vec)
      nrow = int((t_vec[-1]-t_vec[0])/(1000*60*60)+1)
      result1 = []
      m = 0
      sys.stdout.write("Impute the steps during inactive hours..."+'\n')
      for b in range(nrow):
        current_t = t_vec[0] + 1000*60*60*b
        if current_t == t_vec[m]:
          result1.append(result0[m])
          m = m + 1
        else:
          stamp = datetime.fromtimestamp(current_t/1000)
          r = np.random.choice(range(len(empirical_steps['no_records'][stamp.hour])),60)
          step = sum(empirical_steps['no_records'][stamp.hour][r])
          result1.append([stamp.year,stamp.month,stamp.day,stamp.hour,step])
      result1 = pd.DataFrame(np.array(result1))
      result1.columns = ["year","month","day","hour","steps"]
      result1.to_csv(output_path + "/" + i + ".csv", index=False)
      sys.stdout.write("Done"+'\n')


## recommended parameters
hz = 10; q=75; c=1.05; k=60; h=60
## usage
data_path = "C:/Users/glius/Desktop/trial"
output_path = "C:/Users/glius/Desktop/output"
hourly_step_count(data_path,output_path,hz,q,c,k,h)
