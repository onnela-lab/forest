import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from dateutil import tz
import matplotlib.pyplot as plt
data_filepath = "/home/gl176/data"
with open(data_filepath + '/speed1_all.pickle', 'rb') as handle:
    speed1_all = pickle.load(handle)
with open(data_filepath + '/speed2_all.pickle', 'rb') as handle:
    speed2_all = pickle.load(handle)

par1 = 0.95
par2 = 0.95
par3 = 0.05
par4 = 0.30
par5 = 0.5
gap = 15
K = 3

def data_stream(patient_id,j):
  data_folder = data_filepath + "/beiwe_accelerometer/" + patient_id +  "/accelerometer"
  if os.path.exists(data_folder):
    data_files = os.listdir(data_folder)
    dest_path = data_folder + "/" + data_files[j]
    hour_data = pd.read_csv(dest_path)
    hour_data["patient"] = patient_id
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/New_York')
    UTC = [datetime.strptime(i, '%Y-%m-%dT%H:%M:%S.%f') for i in hour_data['UTC time']]
    EST = [i.replace(tzinfo=from_zone).astimezone(to_zone) for i in UTC]
    hour_data["date"] = [i.date() for i in EST]
    hour_data["hour"] = [i.hour for i in EST]

    t_diff = np.array(hour_data['timestamp'][1:hour_data.shape[0]])-np.array(hour_data['timestamp'][0:(hour_data.shape[0]-1)])
    t_diff = np.insert(t_diff,0,1)
    index = np.zeros(len(t_diff),dtype=bool)
    index[t_diff>0] = True
    processed_data = hour_data.loc[index == True]
    mag =  np.sqrt(np.array(processed_data["x"])**2+ np.array(processed_data["y"])**2+
           np.array(processed_data["z"])**2).tolist()
    processed_data.insert(3, "magnitude", mag, True)
    del hour_data
    splitted = dest_path.split("/")
    gps_dest_path = splitted[0]
    for i in np.arange(1,len(splitted)):
      if splitted[i] == "accelerometer":
        splitted[i] = "gps"
      gps_dest_path = gps_dest_path + "/" + splitted[i]
    if os.path.exists(gps_dest_path):
      hour_gps = pd.read_csv(gps_dest_path)
      processed_data["latitude"] = np.mean(hour_gps["latitude"])
      processed_data["longitude"] = np.mean(hour_gps["longitude"])
    else:
      processed_data["latitude"] = np.nan
      processed_data["longitude"] = np.nan
  return processed_data


def step_est(data,m,t_seq,par1,par2,par3,par4):
  M = np.array(data['magnitude'])
  h = np.quantile(M,par1)*par2   ## parameter 1
  g = 9.8
  t_seq = t_seq-min(t_seq)
  step = 0
  time=-350
  for i in np.arange(1,len(m)):
    if(m[i]>=h and m[i]-np.quantile(M,par3)>par4*g and t_seq[i]>=time+350):
      step = step + 1
      time = t_seq[i]
  return(step)


## count the steps per day
def hour_step(data,par1,par2,par3,par4,par5):
  step = 0
  sensor_on = 0
  person_on = 0
  t_seq = np.array(data["timestamp"])
  t_diff = t_seq[1:len(t_seq)]-t_seq[0:(len(t_seq)-1)]
  magnitude = np.array(data["magnitude"])
  start_index = 0
  for i in range(len(t_diff)):
    if t_diff[i]>1000*gap or i==len(t_diff)-1:
      end_index = i
      if i==len(t_diff)-1:
        end_index = len(t_diff)
      sub_mag = magnitude[start_index:(end_index+1)]
      sub_t= t_seq[start_index:(end_index+1)]
      sensor_on = sensor_on + (t_seq[end_index] - t_seq[start_index])/60000
      if max(sub_mag)-min(sub_mag) > par5:
        person_on = person_on + (t_seq[end_index] - t_seq[start_index])/60000
        step = step + step_est(data,sub_mag,sub_t,par1,par2,par3,par4)
      start_index = i+1
  return([data["patient"][0],data["date"][0],data["hour"][0],data["latitude"][0],data["longitude"][0],
         sensor_on, person_on, step])


def step_impute(step,person_on,sensor_on,hour,K):
  speed1 = speed1_all[hour]
  speed2 = speed2_all[hour]
  r = np.random.choice(range(len(speed1)),K)
  s1 = speed1[r]
  s2 = speed2[r]
  imputed_steps = step + (sensor_on-person_on)*s1 + (60-sensor_on)*s2
  mean_step = np.mean(imputed_steps)
  var_step = np.var(imputed_steps)
  return mean_step,var_step


### save all the step counts tables as csv
n = len(os.listdir(data_filepath + "/beiwe_accelerometer"))
for m in range(n):
  patient_id = os.listdir(data_filepath + "/beiwe_accelerometer")[m]
  data_folder = data_filepath + "/beiwe_accelerometer/" + patient_id +  "/accelerometer"
  J = len(os.listdir(data_folder))
  step_table = pd.DataFrame({"id":[],"date":[],"hour":[],"latitude":[],"longitude":[],"sensor_on":[],
                             "person_on":[],"step":[],"imp_mean":[],"imp_var":[]})
  for j in range(J):
    sys.stdout.write('\r')
    sys.stdout.write(patient_id + " [%-50s] %d%%" % ('='*int(np.ceil(j/J*50)), np.ceil(j/J*100)))
    sys.stdout.flush()
    data =  data_stream(patient_id,j)
    out =  hour_step(data,par1,par2,par3,par4,par5)
    imp_mean,imp_var = step_impute(out[7],out[6],out[5],out[2],K)
    step_table = step_table.append({"id":out[0],"date":out[1],"hour":out[2],"latitude":out[3],"longitude":out[4],
                     "sensor_on":out[5],"person_on":out[6],"step":out[7],"imp_mean":imp_mean,"imp_var":imp_var},ignore_index=True)

  path = "/home/gl176/results/step_counts/" + str(patient_id)+ ".csv"
  step_table.to_csv(path,index=False)
