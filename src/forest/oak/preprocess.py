"""
Original Authors: Marcin Straczkiewicz, Georgios Efstathiadis, Zachary Clement (and probably others)

Maintenance - Eli Jones
"""

import logging
from datetime import datetime, timedelta, tzinfo

import numpy as np
from scipy import interpolate

from forest.constants import FP64Arr


logger = logging.getLogger(__name__)


def preprocess_bout(
    t_bout: FP64Arr, x_bout: FP64Arr, y_bout: FP64Arr, z_bout: FP64Arr, fs: int = 10
) -> tuple[FP64Arr, FP64Arr]:
    """Preprocesses accelerometer bout to a common format.
    
    Resample 3-axial input signal to a predefined sampling rate and compute vector magnitude.
    
    Args:
        t_bout: array of floats
            Unix timestamp
        x_bout: array of floats
            X-axis acceleration
        y_bout: array of floats
            Y-axis acceleration
        z_bout: array of floats
            Z-axis acceleration
        fs: integer
            sampling frequency
    
    Returns:
        Tuple of ndarrays:
            - t_bout_interp: resampled timestamp (in Unix)
            - vm_bout_interp: vector magnitude of acceleration
    """
    
    if (len(t_bout) < 2 or len(x_bout) < 2 or len(y_bout) < 2 or len(z_bout) < 2):
        return np.array([]), np.array([])
    
    t_bout_interp = t_bout - t_bout[0]
    t_bout_interp = np.arange(t_bout_interp[0], t_bout_interp[-1], (1/fs))
    t_bout_interp = t_bout_interp + t_bout[0]
    
    f = interpolate.interp1d(t_bout, x_bout)
    x_bout_interp = f(t_bout_interp)
    
    f = interpolate.interp1d(t_bout, y_bout)
    y_bout_interp = f(t_bout_interp)
    
    f = interpolate.interp1d(t_bout, z_bout)
    z_bout_interp = f(t_bout_interp)
    
    x_bout_interp = adjust_bout(x_bout_interp)  # adjust bouts using designated function
    y_bout_interp = adjust_bout(y_bout_interp)
    z_bout_interp = adjust_bout(z_bout_interp)
    
    num_seconds = np.floor(len(x_bout_interp)/fs)  # number of full seconds of measurements
    
    t_bout_interp = t_bout_interp[:int(num_seconds*fs)]  # trim and decimate t
    t_bout_interp = t_bout_interp[::fs]
    
    vm_bout_interp = np.sqrt(x_bout_interp**2 + y_bout_interp**2 + z_bout_interp**2)  # calculate vm
    
    # standardize measurement to gravity units (g) if its recorded in m/s**2
    # Also avoid a runtime warning of taking the mean of an empty slice
    if vm_bout_interp.shape[0] > 0 and np.mean(vm_bout_interp) > 5:
        x_bout_interp = x_bout_interp/9.80665
        y_bout_interp = y_bout_interp/9.80665
        z_bout_interp = z_bout_interp/9.80665
    
    # calculate vm after unit verification
    vm_bout_interp = np.sqrt(x_bout_interp**2 + y_bout_interp**2 + z_bout_interp**2) - 1
    
    return t_bout_interp, vm_bout_interp


def adjust_bout(inarray: FP64Arr, fs: int = 10) -> FP64Arr:
    """Fills observations in incomplete bouts.
    
    For example, if the bout is 9.8s long, add values at its end to make it 10s (results in N%fs=0).
    
    Args:
        inarray: array of floats
            input with one bout of activity
        fs: integer
            sampling frequency
    
    Returns:
        Ndarray with length-adjusted vector magnitude
    """
    # if data is available for 70% of the last second
    if len(inarray) % fs >= 0.7*fs:
        for _ in range(fs-len(inarray) % fs):
            inarray = np.append(inarray, inarray[-1])
    # otherwise, trim the data to the full second
    else:
        inarray = inarray[np.arange(len(inarray)//fs*fs)]
    
    return inarray


def preprocess_dates(
    file_list: list[str],
    time_start: str | None,
    time_end: str | None,
    fmt: str,
    from_zone: tzinfo | None,
    to_zone: tzinfo | None,
) -> tuple[list[datetime], datetime, datetime]:
    """Preprocesses dates of accelerometer files.
    
    Args:
        file_list: list of strings
            list of accelerometer files
        time_start: optional string
            initial date of study in format: 'YYYY-mm-dd HH_MM_SS'
        time_end: optional string
            final date of study in format: 'YYYY-mm-dd HH_MM_SS'
        fmt: string
            python strptime format of dates in file_list
        from_zone: tzinfo
            timezone of dates in file_list
        to_zone: tzinfo
            timezone to localize dates to
    Returns:
        Tuple of ndarrays:
            - dates_shifted: list of datetimes with hours set to 0
            - date_start: datetime of initial date of study
            - date_end: datetime of final date of study
    """
    # transform all files in folder to datelike format, strip all file extensions.
    file_dates = [file.replace("+00_00", "").split(".", 1)[0] for file in file_list]
    
    # process dates
    dates = [datetime.strptime(file, fmt) for file in file_dates]
    dates = [date.replace(tzinfo=from_zone).astimezone(to_zone) for date in dates]
    
    # trim dataset according to time_start and time_end
    if time_start is None or time_end is None:
        dates_filtered = dates
    else:
        time_min = datetime.strptime(time_start, fmt)
        time_min = time_min.replace(tzinfo=from_zone).astimezone(to_zone)
        time_max = datetime.strptime(time_end, fmt)
        time_max = time_max.replace(tzinfo=from_zone).astimezone(to_zone)
        dates_filtered = [date for date in dates if time_min <= date <= time_max]
    
    dates_shifted = [date-timedelta(hours=date.hour) for date in dates]
    
    # create time vector with days for analysis
    if time_start is None:
        date_start = dates_filtered[0]
        date_start = date_start - timedelta(hours=date_start.hour)
    else:
        date_start = datetime.strptime(time_start, fmt)
        date_start = date_start.replace(tzinfo=from_zone).astimezone(to_zone)
        date_start = date_start - timedelta(hours=date_start.hour)
    
    if time_end is None:
        date_end = dates_filtered[-1]
        date_end = date_end - timedelta(hours=date_end.hour)
    else:
        date_end = datetime.strptime(time_end, fmt)
        date_end = date_end.replace(tzinfo=from_zone).astimezone(to_zone)
        date_end = date_end - timedelta(hours=date_end.hour)
    
    return dates_shifted, date_start, date_end
