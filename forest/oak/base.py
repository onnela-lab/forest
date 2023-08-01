"""Step counting method for accelerometer data.

Module is aimed to process raw accelerometer smartphone data collected with
Beiwe Research Platform. Data preprocessing involves signal preproprocesing
(unit standardization and interpolation to 10Hz), transformation using
Continuous Wavelet transform (using ssqueezepy package), and calculation of
steps from the identified walking bouts. Additional gait features calculated
by module are walking time and gait speed (cadence).
Results may be outputted in hourly and daily intervals.
"""

from datetime import datetime, timedelta
import logging
import math
import os

from dateutil import tz
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from ssqueezepy import ssq_cwt

from forest.constants import Frequency
from forest.utils import get_ids

logger = logging.getLogger(__name__)


def preprocess_bout(t_bout: np.ndarray, x_bout: np.ndarray, y_bout: np.ndarray,
                    z_bout: np.ndarray, fs: int = 10) -> tuple:
    """Preprocesses accelerometer bout to a common format.

    Resample 3-axial input signal to a predefined sampling rate and compute
    vector magnitude.

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
    t_bout_interp = np.arange(t_bout[0], t_bout[-1], (1/fs))

    f = interpolate.interp1d(t_bout, x_bout)
    x_bout_interp = f(t_bout_interp)

    f = interpolate.interp1d(t_bout, y_bout)
    y_bout_interp = f(t_bout_interp)

    f = interpolate.interp1d(t_bout, z_bout)
    z_bout_interp = f(t_bout_interp)

    # adjust bouts using designated function
    x_bout_interp = adjust_bout(x_bout_interp)
    y_bout_interp = adjust_bout(y_bout_interp)
    z_bout_interp = adjust_bout(z_bout_interp)

    # number of full seconds of measurements
    num_seconds = np.floor(len(x_bout_interp)/fs)

    # trim and decimate t
    t_bout_interp = t_bout_interp[:int(num_seconds*fs)]
    t_bout_interp = t_bout_interp[::fs]

    # calculate vm
    vm_bout_interp = np.sqrt(x_bout_interp**2 + y_bout_interp**2 +
                             z_bout_interp**2)

    # standardize measurement to gravity units (g) if its recorded in m/s**2
    if np.mean(vm_bout_interp) > 5:
        x_bout_interp = x_bout_interp/9.80665
        y_bout_interp = y_bout_interp/9.80665
        z_bout_interp = z_bout_interp/9.80665

    # calculate vm after unit verification
    vm_bout_interp = np.sqrt(x_bout_interp**2 + y_bout_interp**2 +
                             z_bout_interp**2) - 1

    return t_bout_interp, vm_bout_interp


def adjust_bout(inarray: np.ndarray, fs: int = 10) -> np.ndarray:
    """Fills observations in incomplete bouts.

    For example, if the bout is 9.8s long, add values at its end to make it
    10s (results in N%fs=0).

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
        for i in range(fs-len(inarray) % fs):
            inarray = np.append(inarray, inarray[-1])
    # otherwise, trim the data to the full second
    else:
        inarray = inarray[np.arange(len(inarray)//fs*fs)]

    return inarray


def get_pp(vm_bout: np.ndarray, fs: int = 10) -> npt.NDArray[np.float64]:
    """Calculate peak-to-peak metric in one-second time windows.

    Args:
        vm_bout: array of floats
            vector magnitude with one bout of activity (in g)
        fs: integer
            sampling frequency (in Hz)

    Returns:
        Ndarray with metric

    """
    vm_res_sec = vm_bout.reshape((fs, -1), order="F")
    pp = np.array([max(vm_res_sec[:, i])-min(vm_res_sec[:, i])
                   for i in range(vm_res_sec.shape[1])])

    return pp


def compute_interpolate_cwt(tapered_bout: np.ndarray, fs: int = 10,
                            wavelet: tuple = ('gmw', {'beta': 90,
                                                      'gamma': 3})) -> tuple:
    """Compute and interpolate CWT over acceleration data.

    Args:
        tapered_bout: array of floats
            vector magnitude with one bout of activity (in g)
        fs: integer
            sampling frequency (in Hz)
        wavelet: tuple
            mother wavelet used to compute CWT

    Returns:
        Tuple of ndarrays with interpolated frequency and wavelet coefficients
    """
    # smooth signal on the edges to minimize impact of coin of influence
    window = tukey(len(tapered_bout), alpha=0.02, sym=True)
    tapered_bout = np.concatenate((np.zeros(5*fs),
                                   tapered_bout*window,
                                   np.zeros(5*fs)))

    # compute cwt over bout
    out = ssq_cwt(tapered_bout[:-1], wavelet, fs=10)
    coefs = out[0]
    coefs = np.append(coefs, coefs[:, -1:], 1)

    # magnitude of cwt
    coefs = np.abs(coefs**2)

    # interpolate coefficients
    freqs = out[2]
    freqs_interp = np.arange(0.5, 4.5, 0.05)
    ip = interpolate.interp2d(range(coefs.shape[1]), freqs, coefs)
    coefs_interp = ip(range(coefs.shape[1]), freqs_interp)

    # trim spectrogram from the coi
    coefs_interp = coefs_interp[:, 5*fs:-5*fs]

    return freqs_interp, coefs_interp


def identify_peaks_in_cwt(freqs_interp: np.ndarray, coefs_interp: np.ndarray,
                          fs: int = 10, step_freq: tuple = (1.4, 2.3),
                          alpha: float = 0.6, beta: float = 2.5):
    """Identify dominant peaks in wavelet coefficients.

    Method uses alpha and beta parameters to identify dominant peaks in
    one-second non-overlapping windows in the product of Continuous Wavelet
    Transformation. Dominant peaks need tooccur within the step frequency
    range.

    Args:
        freqs_interp: array of floats
            frequency-domain (in Hz)
        coefs_interp: array of floats
            wavelet coefficients (-)
        fs: integer
            sampling frequency (in Hz)
        step_freq: tuple
            step frequency range
        alpha: float
            maximum ratio between dominant peak below and within
            step frequency range
        beta: float
            maximum ratio between dominant peak above and within
            step frequency range

    Returns:
        Ndarray with dominant peaks
    """
    # identify dominant peaks within coefficients
    dp = np.zeros((coefs_interp.shape[0], int(coefs_interp.shape[1]/fs)))
    loc_min = np.argmin(abs(freqs_interp-step_freq[0]))
    loc_max = np.argmin(abs(freqs_interp-step_freq[1]))
    for i in range(int(coefs_interp.shape[1]/fs)):
        # segment measurement into one-second non-overlapping windows
        x_start = i*fs
        x_end = (i + 1)*fs
        # identify peaks and their location in each window
        window = np.sum(coefs_interp[:, np.arange(x_start, x_end)], axis=1)
        locs, _ = find_peaks(window)
        pks = window[locs]
        ind = np.argsort(-pks)
        locs = locs[ind]
        pks = pks[ind]
        index_in_range = []

        # account peaks that satisfy condition
        for j in range(len(locs)):
            if loc_min <= locs[j] <= loc_max:
                index_in_range.append(j)
            if len(index_in_range) >= 1:
                break
        peak_vec = np.zeros(coefs_interp.shape[0])
        if len(index_in_range) > 0:
            if locs[0] > loc_max:
                if pks[0]/pks[index_in_range[0]] < beta:
                    peak_vec[locs[index_in_range[0]]] = 1
            elif locs[0] < loc_min:
                if pks[0]/pks[index_in_range[0]] < alpha:
                    peak_vec[locs[index_in_range[0]]] = 1
            else:
                peak_vec[locs[index_in_range[0]]] = 1
        dp[:, i] = peak_vec

    return dp


def find_walking(vm_bout: np.ndarray, fs: int = 10, min_amp: float = 0.3,
                 step_freq: tuple = (1.4, 2.3), alpha: float = 0.6,
                 beta: float = 2.5, min_t: int = 3,
                 delta: int = 20) -> npt.NDArray[np.float64]:
    """Finds walking and calculate steps from raw acceleration data.

    Method finds periods of repetitive and continuous oscillations with
    predominant frequency occurring within know step frequency range.
    Frequency components are extracted with Continuous Wavelet Transform.

    Args:
        vm_bout: array of floats
            vector magnitude with one bout of activity (in g)
        fs: integer
            sampling frequency (in Hz)
        min_amp: float
            minimum amplitude (in g)
        step_freq: tuple
            step frequency range
        alpha: float
            maximum ratio between dominant peak below and within
            step frequency range
        beta: float
            maximum ratio between dominant peak above and within
            step frequency range
        min_t: integer
            minimum duration of peaks (in seconds)
        delta: integer
            maximum difference between consecutive peaks (in multiplication of
                                                          0.05Hz)

    Returns:
        Ndarray with identified number of steps per second
    """
    # define wavelet function used in method
    wavelet = ('gmw', {'beta': 90, 'gamma': 3})

    # calculate peak-to-peak
    pp = get_pp(vm_bout, fs)

    # assume the entire bout is of high-intensity
    valid = np.ones(len(pp), dtype=bool)

    # exclude low-intensity periods
    valid[pp < min_amp] = False

    # compute cwt only if valid fragment is sufficiently long
    if sum(valid) >= min_t:
        # trim bout to valid periods only
        tapered_bout = vm_bout[np.repeat(valid, fs)]

        # compute and interpolate CWT
        freqs_interp, coefs_interp = compute_interpolate_cwt(tapered_bout, fs,
                                                             wavelet)

        # get map of dominant peaks
        dp = identify_peaks_in_cwt(freqs_interp, coefs_interp, fs, step_freq,
                                   alpha, beta)

        # distribute local maxima across valid periods
        valid_peaks = np.zeros((dp.shape[0], len(valid)))
        valid_peaks[:, valid] = dp

        # find peaks that are continuous in time (min_t) and frequency (delta)
        cont_peaks = find_continuous_dominant_peaks(valid_peaks, min_t, delta)

        # summarize the results
        cad = np.zeros(valid_peaks.shape[1])
        for i in range(len(cad)):
            ind_freqs = np.where(cont_peaks[:, i] > 0)[0]
            if len(ind_freqs) > 0:
                cad[i] = freqs_interp[ind_freqs[0]]

    else:
        cad = np.zeros(int(vm_bout.shape[0]/fs))

    return cad


def find_continuous_dominant_peaks(valid_peaks: np.ndarray, min_t: int,
                                   delta: int) -> npt.NDArray[np.float64]:
    """Identifies continuous and sustained peaks within matrix.

    Args:
        valid_peaks: nparray
            binary matrix (1=peak,0=no peak) of valid peaks
        min_t: integer
            minimum duration of peaks (in seconds)
        delta: integer
            maximum difference between consecutive peaks (in multiplication of
                                                          0.05Hz)

    Returns:
        Ndarray with binary matrix (1=peak,0=no peak) of continuous peaks
    """
    valid_peaks = np.concatenate((valid_peaks,
                                  np.zeros((valid_peaks.shape[0], 1))), axis=1)
    cont_peaks = np.zeros((valid_peaks.shape[0], valid_peaks.shape[1]))
    for slice_ind in range(valid_peaks.shape[1] - min_t):
        slice_mat = valid_peaks[:, np.arange(slice_ind, slice_ind + min_t)]
        windows = ([i for i in np.arange(min_t)] +
                   [i for i in np.arange(min_t-2, -1, -1)])
        for win_ind in windows:
            pr = np.where(slice_mat[:, win_ind] != 0)[0]
            count = 0
            if len(pr) > 0:
                for i in range(len(pr)):
                    index = np.arange(max(0, pr[i] - delta),
                                      min(pr[i] + delta + 1,
                                          slice_mat.shape[0]
                                          ))
                    if win_ind == 0 or win_ind == min_t - 1:
                        cur_peak_loc = np.transpose(np.array(
                            [np.ones(len(index))*pr[i], index], dtype=int
                            ))
                    else:
                        cur_peak_loc = np.transpose(np.array(
                            [index, np.ones(len(index))*pr[i], index],
                            dtype=int
                            ))

                    peaks = np.zeros((cur_peak_loc.shape[0],
                                      cur_peak_loc.shape[1]), dtype=int)
                    if win_ind == 0:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0],
                                                win_ind]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1],
                                                win_ind + 1]
                    elif win_ind == min_t - 1:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0],
                                                win_ind]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1],
                                                win_ind - 1]
                    else:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0],
                                                win_ind - 1]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1],
                                                win_ind]
                        peaks[:, 2] = slice_mat[cur_peak_loc[:, 2],
                                                win_ind + 1]

                    cont_peaks_edge = cur_peak_loc[np.sum(
                        peaks[:, np.arange(2)], axis=1) > 1, :]
                    cpe0 = cont_peaks_edge.shape[0]
                    if win_ind == 0 or win_ind == min_t - 1:  # first or last
                        if cpe0 == 0:
                            slice_mat[cur_peak_loc[:, 0], win_ind] = 0
                        else:
                            count = count + 1
                    else:
                        cont_peaks_other = cur_peak_loc[np.sum(
                            peaks[:, np.arange(1, 3)], axis=1) > 1, :]
                        cpo0 = cont_peaks_other.shape[0]
                        if cpe0 == 0 or cpo0 == 0:
                            slice_mat[cur_peak_loc[:, 1], win_ind] = 0
                        else:
                            count = count + 1
            if count == 0:
                slice_mat = np.zeros((slice_mat.shape[0], slice_mat.shape[1]))
                break
        cont_peaks[:, np.arange(
            slice_ind, slice_ind + min_t)] = np.maximum(
                cont_peaks[:, np.arange(slice_ind, slice_ind + min_t)],
                slice_mat)

    return cont_peaks[:, :-1]


def run(study_folder: str, output_folder: str, tz_str: str = None,
        frequency: Frequency = Frequency.DAILY, time_start: str = None,
        time_end: str = None, users: list = None) -> None:
    """Runs walking recognition and step counting algorithm over dataset.

    Determine paths to input and output folders, set analysis time frames,
    subjects' local timezone, and time resolution of computed results.

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
    if tz_str is None:
        tz_str = 'UTC'
    to_zone = tz.gettz(tz_str)

    # create folders to store results
    if frequency == Frequency.HOURLY_AND_DAILY:
        os.makedirs(os.path.join(output_folder, "daily"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "hourly"), exist_ok=True)
    else:
        os.makedirs(
            os.path.join(output_folder, frequency.name.lower()), exist_ok=True
        )

    if users is None:
        users = get_ids(study_folder)

    for user in users:
        logger.info("Beiwe ID: %s", user)

        # get file list
        source_folder = os.path.join(study_folder, user, "accelerometer")
        file_list = os.listdir(source_folder)
        file_list.sort()

        # transform all files in folder to datelike format
        if "+00_00.csv" in file_list[0]:
            file_dates = [file.replace("+00_00.csv", "") for file in file_list]
        else:
            file_dates = [file.replace(".csv", "") for file in file_list]

        # process dates
        dates = [datetime.strptime(file, fmt) for file in file_dates]
        dates = [date.replace(tzinfo=from_zone).astimezone(to_zone)
                 for date in dates]

        # trim dataset according to time_start and time_end
        if time_start is not None and time_end is not None:
            time_min = datetime.strptime(time_start, fmt)
            time_min = time_min.replace(tzinfo=from_zone).astimezone(to_zone)
            time_max = datetime.strptime(time_end, fmt)
            time_max = time_max.replace(tzinfo=from_zone).astimezone(to_zone)
            dates = [date for date in dates if time_min <= date <= time_max]

        dates_shifted = [date-timedelta(hours=date.hour) for date in dates]

        # create time vector with days for analysis
        if time_start is None:
            date_start = dates_shifted[0]
            date_start = date_start - timedelta(hours=date_start.hour)
        else:
            date_start = datetime.strptime(time_start, fmt)
            date_start = date_start - timedelta(hours=date_start.hour)

        if time_end is None:
            date_end = dates_shifted[-1]
            date_end = date_end - timedelta(hours=date_end.hour)
        else:
            date_end = datetime.strptime(time_end, fmt)
            date_end = date_end - timedelta(hours=date_end.hour)

        days = pd.date_range(date_start, date_end, freq='D')
        if (
            frequency == Frequency.HOURLY_AND_DAILY
            or frequency == Frequency.HOURLY
        ):
            freq = 'H'
        else:
            freq = str(frequency.value) + 'H'
        days_hourly = pd.date_range(date_start, date_end+timedelta(days=1),
                                    freq=freq)[:-1]

        # allocate memory
        steps_daily = np.full((len(days), 1), np.nan)
        cadence_daily = np.full((len(days), 1), np.nan)
        walkingtime_daily = np.full((len(days), 1), np.nan)

        steps_hourly = np.full((len(days_hourly), 1), np.nan)
        cadence_hourly = np.full((len(days_hourly), 1), np.nan)
        walkingtime_hourly = np.full((len(days_hourly), 1), np.nan)

        for d_ind, d_datetime in enumerate(days):
            logger.info("Day: %d", d_ind)

            # find file indices for this d_ind
            file_ind = [i for i, x in enumerate(dates_shifted)
                        if x == d_datetime]

            # check if there is at least one file for a given day
            if len(file_ind) <= 0:
                continue

            # initiate dataframe
            data = pd.DataFrame()

            # load data for a given day
            for f in file_ind:
                logger.info("File: %d", f)

                # read data
                file_path = os.path.join(source_folder, file_list[f])
                data = pd.concat([data, pd.read_csv(file_path)], axis=0)

            # extract data
            timestamp = np.array(data["timestamp"]) / 1000
            x = np.array(data["x"], dtype="float64")  # x-axis acc.
            y = np.array(data["y"], dtype="float64")  # y-axis acc.
            z = np.array(data["z"], dtype="float64")  # z-axis acc.

            # preprocess data fragment
            t_bout_interp, vm_bout = preprocess_bout(timestamp, x, y, z)

            # get t as datetimes
            t_datetime = [datetime.fromtimestamp(t_ind)
                          for t_ind in t_bout_interp]

            # transform t to full hours
            t_hours = [t_i -
                       timedelta(minutes=t_i.minute) -
                       timedelta(seconds=t_i.second) -
                       timedelta(microseconds=t_i.microsecond)
                       for t_i in t_datetime]
            t_hours = [date.replace(tzinfo=from_zone).astimezone(to_zone)
                       for date in t_hours]

            # find walking and estimate cadence
            cadence_bout = find_walking(vm_bout)

            # distribute metrics across hours
            if frequency != Frequency.DAILY:
                for t_unique in np.unique(np.array(t_hours)):
                    t_ind_pydate = [t_ind.to_pydatetime() for t_ind in
                                    days_hourly]
                    # get indexes of ranges of dates that contain t_unique
                    ind_to_store = -1
                    for ind_to_store, t_ind in enumerate(t_ind_pydate):
                        if (
                            t_ind <= t_unique
                            < t_ind + timedelta(hours=frequency.value)
                        ):
                            break
                    cadence_ind = np.array([time == t_unique for time in
                                            t_hours])
                    cadence_temp = cadence_bout[cadence_ind]
                    cadence_temp = cadence_temp[np.where(cadence_temp > 0)]

                    # store hourly metrics
                    if math.isnan(steps_hourly[ind_to_store]):
                        steps_hourly[ind_to_store] = int(np.sum(cadence_temp))
                        walkingtime_hourly[ind_to_store] = len(cadence_temp)
                    else:
                        steps_hourly[ind_to_store] += int(np.sum(cadence_temp))
                        walkingtime_hourly[ind_to_store] += len(cadence_temp)

                for idx in range(len(cadence_hourly)):
                    if walkingtime_hourly[idx] > 0:
                        cadence_hourly[idx] = (
                            steps_hourly[idx] / walkingtime_hourly[idx]
                        )

            cadence_bout = cadence_bout[np.where(cadence_bout > 0)]

            # store daily metrics
            steps_daily[d_ind] = int(np.sum(cadence_bout))
            if len(cadence_bout) > 0:  # control for empty slices
                cadence_daily[d_ind] = np.mean(cadence_bout)
            else:
                cadence_daily[d_ind] = np.nan
            walkingtime_daily[d_ind] = len(cadence_bout)

            # save results depending on "frequency"
            if (
                frequency == Frequency.DAILY
                or frequency == Frequency.HOURLY_AND_DAILY
            ):
                summary_stats = pd.DataFrame({
                    'date': days.strftime('%Y-%m-%d'),
                    'walking_time': walkingtime_daily[:, -1],
                    'steps': steps_daily[:, -1],
                    'cadence': cadence_daily[:, -1]})
                output_file = user + "_gait_daily.csv"
                dest_path = os.path.join(output_folder, "daily", output_file)
                summary_stats.to_csv(dest_path, index=False)
            if frequency != Frequency.DAILY:
                summary_stats = pd.DataFrame({
                    'date': [date.strftime('%Y-%m-%d %H:%M:%S')
                             for date in days_hourly],
                    'walking_time': walkingtime_hourly[:, -1],
                    'steps': steps_hourly[:, -1],
                    'cadence': cadence_hourly[:, -1]})
                output_file = user + "_gait_hourly.csv"
                if frequency == Frequency.HOURLY_AND_DAILY:
                    freq_name = "hourly"
                else:
                    freq_name = frequency.name.lower()
                dest_path = os.path.join(output_folder, freq_name, output_file)
                summary_stats.to_csv(dest_path, index=False)
