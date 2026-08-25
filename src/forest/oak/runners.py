"""
Original Authors: Marcin Straczkiewicz, Georgios Efstathiadis, Zachary Clement (and probably others)

Maintenance - Eli Jones
"""

import logging
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import tz

from forest.constants import FP64Arr, Frequency
from forest.oak.analysis import find_walking
from forest.oak.preprocess import preprocess_bout, preprocess_dates
from forest.utils import get_ids


logger = logging.getLogger(__name__)


def run_hourly(
    t_hours_pd: pd.Series,
    t_ind_pydate: list | np.ndarray,
    cadence_bout: FP64Arr,
    steps_hourly: FP64Arr,
    walkingtime_hourly: FP64Arr,
    cadence_hourly: FP64Arr,
    frequency: Frequency,
) -> None:
    """Runs hourly metrics computation for steps, walking time, and cadence.
     Updates steps_hourly, walkingtime_hourly, and cadence_hourly in place.
    
    Args:
        t_hours_pd: pd.Series
            timestamp of each measurement
        t_ind_pydate: list
            list of days with hourly resolution
        cadence_bout: ndarray
            cadence of each measurement
        steps_hourly: ndarray
            number of steps per hour
        walkingtime_hourly: ndarray
            number of minutes of walking per hour
        cadence_hourly: ndarray
            average cadence per hour
        frequency: Frequency
            summary statistics format, Frequency class at constants.py
    """
    
    for t_unique in t_hours_pd.unique():
        # get indexes of ranges of dates that contain t_unique
        ind_to_store = -1
        for ind_to_store, t_ind in enumerate(t_ind_pydate):
            if t_ind <= t_unique < t_ind + timedelta(minutes=frequency.value):
                break
        
        cadence_temp = cadence_bout[t_hours_pd == t_unique]
        cadence_temp = cadence_temp[cadence_temp > 0]
        
        # store hourly metrics
        if math.isnan(steps_hourly[ind_to_store].item()):
            steps_hourly[ind_to_store] = int(np.sum(cadence_temp))
            walkingtime_hourly[ind_to_store] = len(cadence_temp)
        else:
            steps_hourly[ind_to_store] += int(np.sum(cadence_temp))
            walkingtime_hourly[ind_to_store] += len(cadence_temp)
    
    for idx in range(len(cadence_hourly)):
        if walkingtime_hourly[idx] > 0:
            cadence_hourly[idx] = steps_hourly[idx] / walkingtime_hourly[idx]


def run(
    study_folder: str,
    output_folder: str,
    tz_str: str | None = None,
    frequency: Frequency = Frequency.DAILY,
    time_start: str | None = None,
    time_end: str | None = None,
    users: list | None = None,
) -> None:
    """Runs walking recognition and step counting algorithm over dataset.
    
    Determine paths to input and output folders, set analysis time frames, subjects' local timezone,
    and time resolution of computed results.
    
    Args:
        study_folder: string
            local repository with beiwe folders (IDs) for a given study
        output_folder: string
            local repository to store results
        tz_str: string
            local time zone, e.g., "America/New_York"
        frequency: Frequency
            summary statistics format, Frequency class at constants.py
        time_start: string
            initial date of study in format: 'YYYY-mm-dd HH_MM_SS'
        time_end: string
            final date of study in format: 'YYYY-mm-dd HH_MM_SS'
        users: list of strings
            beiwe ID selected for computation
    """
    
    # determine timezone shift
    fmt = '%Y-%m-%d %H_%M_%S'
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz(tz_str) if tz_str else from_zone
    
    freq_str = frequency.name.lower()
    
    # create folders to store results
    if frequency == Frequency.HOURLY_AND_DAILY:
        os.makedirs(os.path.join(output_folder, "daily"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "hourly"), exist_ok=True)
    else:
        os.makedirs(os.path.join(output_folder, freq_str), exist_ok=True)
    if users is None:
        users = get_ids(study_folder)
    
    for user in users:
        logger.info("Beiwe ID: %s", user)
        
        # get file list
        source_folder = os.path.join(study_folder, user, "accelerometer")
        file_list = os.listdir(source_folder)
        file_list.sort()
        
        dates_shifted, date_start, date_end = preprocess_dates(
            file_list, time_start, time_end, fmt, from_zone, to_zone
        )
        
        days = pd.date_range(date_start, date_end, freq='D')
        
        steps_daily = np.full((len(days), 1), np.nan)  # allocate memory
        cadence_daily = np.full((len(days), 1), np.nan)
        walkingtime_daily = np.full((len(days), 1), np.nan)
        
        steps_hourly = np.full((1, 1), np.nan)
        cadence_hourly = np.full((1, 1), np.nan)
        walkingtime_hourly = np.full((1, 1), np.nan)
        t_ind_pydate = np.ndarray([], dtype='datetime64[ns]')
        t_ind_pydate_str = None
        
        if frequency != Frequency.DAILY:
            if (frequency == Frequency.HOURLY_AND_DAILY or frequency == Frequency.HOURLY):
                freq = 'h'
            elif frequency == Frequency.MINUTE:
                freq = 'min'
            else:
                freq = str(frequency.value/60) + 'h'
            
            days_hourly = pd.date_range(date_start, date_end + timedelta(days=1), freq=freq)[:-1]
            
            steps_hourly = np.full((len(days_hourly), 1), np.nan)
            cadence_hourly = np.full((len(days_hourly), 1), np.nan)
            walkingtime_hourly = np.full((len(days_hourly), 1), np.nan)
            
            t_ind_pydate = days_hourly.to_pydatetime()
            t_ind_pydate_str = t_ind_pydate.astype(str)
        
        for d_ind, d_datetime in enumerate(days):
            logger.info("Day: %d", d_ind)
            
            # find file indices for this d_ind
            file_ind = [i for i, x in enumerate(dates_shifted) if x == d_datetime]
            
            # check if there is at least one file for a given day
            if len(file_ind) <= 0:
                continue
            
            data = pd.DataFrame()
            
            # load data for a given day
            for f in file_ind:
                logger.info("File: %d", f)
                file_path = os.path.join(source_folder, file_list[f])
                data = pd.concat([data, pd.read_csv(file_path)], axis=0)  # read data
            
            # extract data
            timestamp = np.array(data["timestamp"]) / 1000
            x = np.array(data["x"], dtype="float64")  # x-axis acc.
            y = np.array(data["y"], dtype="float64")  # y-axis acc.
            z = np.array(data["z"], dtype="float64")  # z-axis acc.
            
            # preprocess data fragment
            t_bout_interp, vm_bout = preprocess_bout(timestamp, x, y, z)
            if len(t_bout_interp) == 0:  # no valid data to process here
                continue
            
            cadence_bout = find_walking(vm_bout)  # find walking and estimate cadence
            
            # distribute metrics across hours
            if frequency != Frequency.DAILY:
                # get t as datetimes
                t_datetime = [datetime.fromtimestamp(t_ind) for t_ind in t_bout_interp]
                t_series = pd.Series(t_datetime)  # transform t to full hours
                
                if frequency == Frequency.MINUTE:
                    t_hours_pd = t_series.dt.floor('min')
                else:
                    t_hours_pd = t_series.dt.floor('h')
                # convert t_hours to correct timezone
                t_hours_pd = t_hours_pd.dt.tz_localize(from_zone).dt.tz_convert(to_zone)
                
                run_hourly(
                    t_hours_pd,
                    t_ind_pydate,
                    cadence_bout,
                    steps_hourly,
                    walkingtime_hourly,
                    cadence_hourly,
                    frequency,
                )
            
            cadence_bout = cadence_bout[np.where(cadence_bout > 0)]
            
            # store daily metrics
            steps_daily[d_ind] = int(np.sum(cadence_bout))
            # control for empty slices
            cadence_daily[d_ind] = np.mean(cadence_bout) if len(cadence_bout) > 0 else np.nan
            walkingtime_daily[d_ind] = len(cadence_bout)
            
            # save results depending on "frequency"
            if frequency == Frequency.DAILY or frequency == Frequency.HOURLY_AND_DAILY:
                summary_stats = pd.DataFrame({
                    'date': days.strftime('%Y-%m-%d'),
                    'walking_time': walkingtime_daily[:, -1],
                    'steps': steps_daily[:, -1],
                    'cadence': cadence_daily[:, -1]
                })
                
                output_file = user + ".csv"
                dest_path = os.path.join(output_folder, "daily", output_file)
                summary_stats.to_csv(dest_path, index=False)
            
            if frequency != Frequency.DAILY:
                summary_stats = pd.DataFrame({
                    'date': t_ind_pydate_str,
                    'walking_time': walkingtime_hourly[:, -1],
                    'steps': steps_hourly[:, -1],
                    'cadence': cadence_hourly[:, -1]
                })
                output_file = user + "_gait_hourly.csv"
                
                freq_name = "hourly" if frequency == Frequency.HOURLY_AND_DAILY else freq_str
                dest_path = os.path.join(output_folder, freq_name, output_file)
                summary_stats.to_csv(dest_path, index=False)
