# active_time.py
#
# Function calculates active time minutes from raw accelerometer measurements
# collected using smartphones.
#
# The time is assumed as active if summed variance from 3 orthogonal axes is 
# above 0.15g (1g ~= 9.80665 m/s**2).
#
# For relevant publication, see: 
#   Utilizing smartphones to capture novel recovery metrics after cancer surgery, 
#   N Panda, et al.  JAMA Surg. 2020;155(2):123-129. 
#   doi:10.1001/jamasurg.2019.4702 
#
# Example of use:
#   at = active_time(directory, beiwe_ids) 
#
#   Inputs:
#       directory = "C:/users/username/documents/project/data/beiwe/"
#       beiwe_ids = ["abcd1234","efgh5678"]
#
#   Outputs:
#       at = [['2020-01-01', '28.65', '548.48'],
#            ['2020-01-02', '103.23', '477.75']]   
#
# Coded: 2020-04-05
#
# Script author:
# Marcin Straczkiewicz, PhD,
# mstraczkiewicz@hsph.harvard.edu
 

import os
import numpy as np
import pandas as pd

from dateutil import tz
from datetime import datetime, timedelta
from scipy import interpolate

def active_time(directory, beiwe_ids):
    
    '''
     Calculates active time from raw accelerometer data.
     
     Args:
         directory (str):  Path to location of the study with Beiwe IDs.
         beiwe_ids (list of str):  Beiwe IDs for particular individual.
     Returns:
         ac_metric (numpy array): <number of daysx3> active time metric.
             Column 1: date
             Column 2: active time (in minutes)
             Column 3: total time (in minutes)
    ''' 
   
    # set constants
    gravity = 9.80665
    
    # tuning parameters
    activity_threshold = 0.15 # activity threshold (in gravitation units (g))
    fs = 10 # desired sampling rate (sampling frequency) (in Hertz (Hz))

    # empty active time list
    at_metric = []

    # prepare time zone
    fmt = '%Y-%m-%d %H_%M_%S'
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/New_York')

    # assuming that there might be more than 1 Beiwe ID for a given subject,
    # analyse each ID individually, and then append metric
    
    for k in range(len(beiwe_ids)):
        # reach accelerometer folders
        source_folder = directory + '/' + beiwe_ids[k] + '/accelerometer/'
        
        if os.path.exists(source_folder):
            # retrieve files with data
            filelist  = os.listdir(source_folder)
            file_dates = [file.replace('.csv','') for file in filelist]

            # enable time shift to EST zone
            UTC = [datetime.strptime(i, fmt) for i in file_dates]
            EST = [i.replace(tzinfo=from_zone).astimezone(to_zone) for i in UTC]
            EST = [EST[i] - timedelta(hours=EST[i].hour) for i in range(len(EST))]

            start_date = EST[0]
            end_date   = EST[-1]

            # create time vector for metric
            days = pd.date_range(start_date, end_date, freq='D',closed=None) 
            days = pd.DatetimeIndex.tolist(days)
            
            # transform days and EST into string to enable comparison
            days = [days[j].strftime('%Y-%m-%d') for j in range(len(days))]
            EST  = [EST[j].strftime('%Y-%m-%d') for j in range(len(EST))]

            # calculate metric individually for each day of observation
            for d in range(len(days)): 
                # identify positions (indices) of files to open for a given day
                indices = [i for i, x in enumerate(EST) if x == days[d]] 
                
                # initiate temporary metric variables
                temp_A = 0  # active time
                temp_T = 0  # total time
                
                if len(indices) > 0:
                    for i in range(0,len(indices)): # for each file within a day
                        fileToRead = filelist[indices[i]]

                        # load file
                        data = pd.read_csv(source_folder + fileToRead) # load file
                        
                        # separate data vectors
                        t = np.array(data["timestamp"]) # time
                        x = np.array(data["x"]) # x-axis acceleration
                        y = np.array(data["y"]) # y-axis acceleration
                        z = np.array(data["z"]) # z-axis acceleration
                        
                        # calculate vector magnitude vector
                        vm = np.sqrt(x**2+y**2+z**2)
                        
                        # standardize measurement to gravity units (g) if its recorded in m/s**2
                        if np.mean(vm) > 5: 
                            x = x/gravity
                            y = y/gravity
                            z = z/gravity                        
                        
                        # in case there is non-increasing time vector, sort it
                        ind  = np.argsort(t)
                        t    = t[ind];
                                        
                        # find and sort "on"-cycles
                        t_diff   = ((t[1:]-t[:-1])/1000)>5
                        t_breaks = [j for j,val in enumerate(t_diff) if val]
                        t_breaks = [1, len(t)] + t_breaks
                        t_breaks.sort()
                        
                        # calculate active time for all on-cycles
                        for j in range(len(t_breaks)-1):
                            # isolate data from individual bout
                            t_bout = t[t_breaks[j]+1:t_breaks[j+1]]
                            x_bout = x[t_breaks[j]+1:t_breaks[j+1]]
                            y_bout = y[t_breaks[j]+1:t_breaks[j+1]]
                            z_bout = z[t_breaks[j]+1:t_breaks[j+1]]
        
                            if len(t_bout) > fs: # neglect extremely short bouts, e.g., singular observations
                                # interpolate acceleration to unify sampling rate                    
                                t_bout_interp = np.arange(t[t_breaks[j]+1],t[t_breaks[j+1]-1],1000/fs)
                                
                                f = interpolate.interp1d(t_bout, x_bout)
                                x_bout_interp = f(t_bout_interp)
                                
                                f = interpolate.interp1d(t_bout, y_bout)
                                y_bout_interp = f(t_bout_interp)
                                
                                f = interpolate.interp1d(t_bout, z_bout)
                                z_bout_interp = f(t_bout_interp)
                                
                                # number of full seconds of measurements
                                num_seconds = np.floor(len(x_bout_interp)/fs)

                                # trim measurement to full seconds
                                x_bout_interp_trim  = x_bout_interp[:int(num_seconds*fs)]
                                y_bout_interp_trim  = y_bout_interp[:int(num_seconds*fs)]
                                z_bout_interp_trim  = z_bout_interp[:int(num_seconds*fs)]
                                
                                # calculate variance in 3 axes
                                summedVar = np.var(x_bout_interp_trim) + np.var(y_bout_interp_trim) + np.var(z_bout_interp_trim)
                                
                                # add active time if variance exceeds threshold
                                if summedVar>activity_threshold: 
                                    temp_A = temp_A + num_seconds
                                
                                # add time to the entire time of measurement
                                temp_T = temp_T + num_seconds
                    
                    if float(temp_T)>0: # only for positive total time
                        at_metric.append([days[d], float(temp_A)/float(temp_T)*24*60, float(temp_T)/60])
                    else:
                        at_metric.append([days[d], 'NaN', 0])
                else:
                    at_metric.append([days[d], 'NaN', 'NaN'])
        else:
            print('Folder ' + source_folder + ' does not exist')
     
    at_metric = np.array(at_metric)
    return at_metric
