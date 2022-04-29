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
import os
import sys

from dateutil import tz
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import find_peaks, tukey
from ssqueezepy import ssq_cwt


def rle(inarray: np.ndarray) -> tuple:
    """Runs length encoding.

    Args:
        inarray: array of Boolean values
            input for run length encoding
    Returns:
        run_length: tuple
            running length
        start_ind : tuple
            starting index
        val : tuple
            running values
    """
    array_length = len(inarray)
    if array_length == 0:
        return (None, None, None)
    else:
        pairwise_unequal = inarray[1:] != inarray[:-1]  # pairwise unequal
        ind = np.append(np.where(pairwise_unequal),
                        array_length - 1)  # must include last element position
        run_length = np.diff(np.append(-1, ind))  # run lengths
        start_ind = np.cumsum(np.append(0, run_length))[:-1]  # positions
        val = inarray[ind]
        return (run_length, start_ind, val)


def preprocess_bout(t_bout: np.ndarray, x_bout: np.ndarray, y_bout: np.ndarray,
                    z_bout: np.ndarray) -> tuple:
    """Preprocesses accelerometer bout to a common format.

    Resample 3-axial input signal to a predefined sampling rate and compute
    vector magnitude.

    Args:
        t_bout : array of floats
            Unix timestamp
        x_bout : array of floats
            X-axis acceleration
        y_bout : array of floats
            Y-axis acceleration
        z_bout : array of floats
            Z-axis acceleration

    Returns:
        x_bout : array of floats
            Interpolated x-axis acceleration
        y_bout : array of floats
            Interpolated y-axis acceleration
        z_bout : array of floats
            Interpolated z-axis acceleration
        vm_bout : array of floats
            Interpolated, zero-oscillating vector magnitude
    """
    t_bout_interp = np.arange(t_bout[0], t_bout[-1], (1/fs))

    f = interpolate.interp1d(t_bout, x_bout)
    x_bout_interp = f(t_bout_interp)

    f = interpolate.interp1d(t_bout, y_bout)
    y_bout_interp = f(t_bout_interp)

    f = interpolate.interp1d(t_bout, z_bout)
    z_bout_interp = f(t_bout_interp)

    # adjust bouts using designated function
    x_bout_interp = adjust_bout(x_bout_interp, fs)
    y_bout_interp = adjust_bout(y_bout_interp, fs)
    z_bout_interp = adjust_bout(z_bout_interp, fs)

    # number of full seconds of measurements
    num_seconds = np.floor(len(x_bout_interp)/fs)

    # trim measurement to full seconds
    x_bout = x_bout_interp[:int(num_seconds*fs)]
    y_bout = y_bout_interp[:int(num_seconds*fs)]
    z_bout = z_bout_interp[:int(num_seconds*fs)]

    vm_bout = np.sqrt(x_bout**2+y_bout**2+z_bout**2)

    # standardize measurement to gravity units (g) if its recorded in m/s**2
    if np.mean(vm_bout) > 5:
        x_bout = x_bout/gravity
        y_bout = y_bout/gravity
        z_bout = z_bout/gravity

    vm_bout = np.sqrt(x_bout**2+y_bout**2+z_bout**2)-1

    return (x_bout, y_bout, z_bout, vm_bout)


def adjust_bout(vm_bout: np.ndarray, fs: int) -> np.ndarray:
    """Fills observations in incomplete bouts.

    For example, if the bout is 9.8s long, add values at its end to make it
    10s (results in N%fs=0).

    Args:
        vm_bout : array of floats
            vector magnitude with one bout of activity
        fs : integer
            sampling frequency

    Returns:
        vm_bout : array of floats
            vector magnitude with one bout of activity
    """
    if len(vm_bout) % fs >= 0.7*fs:
        for i in range(fs-len(vm_bout) % fs):
            vm_bout = np.append(vm_bout, vm_bout[-1])
    elif len(vm_bout) % fs != 0:
        vm_bout = vm_bout[np.arange(len(vm_bout)//fs*fs)]

    return vm_bout


def find_walking(vm_bout: np.ndarray, fs: int, min_amp: float,
                 step_freq: tuple, alpha: float, beta: float, epsilon: int,
                 delta: int) -> np.ndarray:
    """Finds walking and calculate steps from raw acceleration data.

    Method finds periods of repetetive and continuous oscillations with
    predominant frequency occuring within know step frequency range.
    Frequency components are extracted with Continuous Wavelet Transform.

    Args:
        vm_bout : array of floats
            vector magnAtude with one bout of activity (in g)
        fs : integer
            sampling frequency (in Hz)
        A : float
            minimum amplitude (in g)
        step_freq : tuple
            step frequency range
        alpha : float
            maximum ratio between dominant peak below and within
            step frequency range
        beta : float
            maximum ratio between dominant peak above and within
            step frequency range
        epsilon : integer
            minimum duration of peaks (in seconds)
        delta : integer
            maximum difference between consecutive peaks (in multiplication of
                                                          0.05Hz)

    Returns:
        cad :  array of floats
            gait speed within bout of activity (in Hz or steps/sec)
    """
    # define wavelet function used in method
    wavelet = ('gmw', {'beta': 90, 'gamma': 3})

    # calculate peak-to-peak to exclude low-intensity periods
    vm_res_sec = vm_bout.reshape(fs, -1, order="F")
    pp = np.array([max(vm_res_sec[:, i])-min(vm_res_sec[:, i])
                   for i in range(vm_res_sec.shape[1])])
    valid = np.ones(len(pp), dtype=bool)
    valid[pp < min_amp] = False

    # compute cwt only if valid fragment is suffiently long
    if sum(valid) >= epsilon:
        # trim bout to valid periods only
        tapered_bout = vm_bout[np.repeat(valid, fs)]

        # add some noise in the edges to minimize impact of coin of influence
        win = tukey(len(tapered_bout), alpha=0.02, sym=True)
        tapered_bout = np.concatenate((np.zeros(5*fs),
                                       (tapered_bout)*win,
                                       np.zeros(5*fs)))

        # compute cwt over bout
        out = ssq_cwt(tapered_bout[:-1], wavelet, fs=10)
        coefs = out[0]
        coefs = np.append(coefs, coefs[:, -1:], 1)

        # magnitude of cwt
        coefs = np.abs(coefs**2)

        # interpolate spectrogram
        freqs = out[2]
        freqs_interp = np.arange(0.5, 4.5, 0.05)
        ip = interpolate.interp2d(range(coefs.shape[1]), freqs, coefs)
        coefs_interp = ip(range(coefs.shape[1]), freqs_interp)

        # trim spectrogram from the coi
        coefs_interp = coefs_interp[:, 5*fs:-5*fs]

        # identify dominant peaks with the spectrum
        dp = np.zeros((coefs_interp.shape[0], int(coefs_interp.shape[1]/fs)))
        loc_min = np.argmin(abs(freqs_interp-step_freq[0]))
        loc_max = np.argmin(abs(freqs_interp-step_freq[1]))
        for i in range(int(coefs_interp.shape[1]/fs)):
            # segment measurement into one-second non-overlapping windows
            x_start = i*fs
            x_end = (i+1)*fs
            # identify peaks and their location in each window
            window = np.sum(coefs_interp[:, np.arange(x_start, x_end)],
                            axis=1)
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
                    if pks[0]/pks[index_in_range[0]] < alpha:
                        peak_vec[locs[index_in_range[0]]] = 1
                elif locs[0] < loc_min:
                    if pks[0]/pks[index_in_range[0]] < beta:
                        peak_vec[locs[index_in_range[0]]] = 1
                else:
                    peak_vec[locs[index_in_range[0]]] = 1
            dp[:, i] = peak_vec

        # distribute local maxima across valid periods
        val_peaks = np.zeros((dp.shape[0], len(valid)))
        val_peaks[:, valid] = dp

        # find when peaks are continuous in time and frequency
        cp = find_continuous_dominant_peaks(val_peaks, epsilon, delta)

        # summarize the results
        cad = np.zeros(val_peaks.shape[1])
        for i in range(len(cad)):
            ind_freqs = np.where(cp[:, i] > 0)[0]
            if len(ind_freqs) > 0:
                cad[i] = freqs_interp[ind_freqs[0]]

        # num_steps = int(round(sum(cad),0))
    else:
        # num_steps = 0
        cad = np.zeros(int(vm_bout.shape[0]/fs))

    return cad


def find_continuous_dominant_peaks(val_peaks: np.ndarray, epsilon: int,
                                   delta: int) -> np.ndarray:
    """Identifies continuous and sustained peaks within matrix.

    Args:
        val_peaks : nparray
            binary matrix (1=peak,0=no peak) of valid peaks
        epsilon : integer
            minimum duration of peaks (in seconds)
        delta : integer
            maximum difference between consecutive peaks (in multiplication of
                                                          0.05Hz)

    Returns:
        cp : nparray
            binary matrix (1=peak,0=no peak) of continuous peaks
    """
    val_peaks = np.concatenate((val_peaks, np.zeros((val_peaks.shape[0], 1))),
                               axis=1)
    cp = np.zeros((val_peaks.shape[0], val_peaks.shape[1]))
    for slice_ind in range(val_peaks.shape[1]-epsilon):
        slice_mat = val_peaks[:, np.arange(slice_ind, slice_ind+epsilon)]
        win = [i for i in np.arange(epsilon)] + [i for i in
                                                 np.arange(epsilon-2, -1, -1)]

        for ind in range(len(win)):
            win_ind = win[ind]
            pr = np.where(slice_mat[:, win_ind] != 0)[0]
            count = 0
            if len(pr) > 0:
                for i in range(len(pr)):
                    index = np.arange(max(0, pr[i]-delta),
                                      min(pr[i]+delta+1, slice_mat.shape[0]))
                    if win_ind == 0 or win_ind == epsilon-1:
                        cur_peak_loc = np.transpose(np.array(
                            [np.ones(len(index))*pr[i], index], dtype=int))
                    else:
                        cur_peak_loc = np.transpose(np.array(
                            [index, np.ones(len(index))*pr[i], index],
                            dtype=int))

                    peaks = np.zeros((cur_peak_loc.shape[0],
                                      cur_peak_loc.shape[1]), dtype=int)
                    if win_ind == 0:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0], win_ind]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1], win_ind+1]
                    elif win_ind == epsilon-1:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0], win_ind]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1], win_ind-1]
                    else:
                        peaks[:, 0] = slice_mat[cur_peak_loc[:, 0], win_ind-1]
                        peaks[:, 1] = slice_mat[cur_peak_loc[:, 1], win_ind]
                        peaks[:, 2] = slice_mat[cur_peak_loc[:, 2], win_ind+1]

                    cont_peaks_edge = cur_peak_loc[np.sum(
                        peaks[:, np.arange(2)], axis=1) > 1, :]
                    cpe0 = cont_peaks_edge.shape[0]
                    if win_ind == 0 or win_ind == epsilon-1:  # start or end
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
        cp[:, np.arange(
            slice_ind, slice_ind + epsilon)] = np.maximum(
                cp[:, np.arange(slice_ind, slice_ind + epsilon)], slice_mat)

    cp = cp[:, :-1]
    return cp


def main_function(study_folder: str, output_folder: str, tz_str: str = None,
                  option: str = None, time_start: str = None,
                  time_end: str = None, beiwe_id: list = None) -> None:
    """Runs walking recognition and step counting algorithm over dataset.

    Determine paths to input and output folders, set analysis time frames,
    subjects' local timezone, and time resolution of computed results.

    Args:
        study folder : string
            local repository with beiwe folders (IDs) for a given study
        output folder : string
            local repository to store results
        tz_str : string
            local time zone, e.g., "America/New_York"
        option : string
            summary statistics format (accepts 'both', 'hourly', 'daily')
        time_start : string
            initial date of study in format: 'YYYY-mm-dd HH_MM_SS'
        time_end : string
            final date of study in format: 'YYYY-mm-dd HH_MM_SS'
        beiwe_id : list of strings
            beiwe ID selected for computation
    """
    fmt = '%Y-%m-%d %H_%M_%S'
    from_zone = tz.gettz('UTC')
    if tz_str is None:
        tz_str = 'UTC'
    to_zone = tz.gettz(tz_str)

    # create folders for results
    if option is None or option == 'both' or option == 'daily':
        os.makedirs(output_folder+"/daily", exist_ok=True)
    if option is None or option == 'both' or option == 'hourly':
        os.makedirs(output_folder+"/hourly", exist_ok=True)

    if beiwe_id is None:
        beiwe_id = os.listdir(study_folder)

    for ID in beiwe_id:
        sys.stdout.write('User: ' + ID + '\n')

        source_folder = study_folder + '/' + ID + '/accelerometer/'
        file_list = os.listdir(source_folder)

        # transform all files in folder to datelike format
        if "+00_00.csv" in file_list[0]:
            file_dates = [file.replace("+00_00.csv", "") for file in file_list]
        else:
            file_dates = [file.replace(".csv", "") for file in file_list]

        file_dates.sort()

        dates = [datetime.strptime(file, fmt) for file in file_dates]
        dates = [date.replace(tzinfo=from_zone).astimezone(to_zone)
                 for date in dates]
        dates_shifted = [date-timedelta(hours=date.hour) for date in dates]

        # create time vector with days for analysis
        if time_start is None:
            date_start = dates_shifted[0]
            date_start = date_start - timedelta(hours=date_start.hour)
        else:
            date_start = datetime.strptime(time_start, fmt)

        if time_end is None:
            date_end = dates_shifted[-1]
            date_end = date_end - timedelta(hours=date_end.hour)
        else:
            date_end = datetime.strptime(time_end, fmt)

        days = pd.date_range(date_start, date_end, freq='D')
        days_hourly = pd.date_range(date_start, date_end+timedelta(days=1),
                                    freq='H')

        # activity type - gait
        if option is None or option == 'both' or option == 'daily':
            steps_daily = np.empty((len(days), 1))
            steps_daily.fill(np.nan)

            cadence_daily = np.empty((len(days), 1))
            cadence_daily.fill(np.nan)

            walkingtime_daily = np.empty((len(days), 1))
            walkingtime_daily.fill(np.nan)
        if option is None or option == 'both' or option == 'hourly':
            steps_hourly = np.empty((len(days), 24))
            steps_hourly.fill(np.nan)

            cadence_hourly = np.empty((len(days), 24))
            cadence_hourly.fill(np.nan)

            walkingtime_hourly = np.empty((len(days), 24))
            walkingtime_hourly.fill(np.nan)

        for d_ind, d_datetime in enumerate(days):
            sys.stdout.write('Day: ' + str(d_ind) + '\n')
            # find file indices for this d_ind
            file_ind = [i for i, x in enumerate(dates_shifted)
                        if x == d_datetime]

            # initiate temporal metric
            if option is None or option == 'both' or option == 'daily':
                cadence_temp_daily = list()

            for f in file_ind:
                sys.stdout.write('File: ' + str(f) + '\n')

                # initiate temporal metric
                if option is None or option == 'both' or option == 'hourly':
                    cadence_temp_hourly = list()
                    # hour of the day
                    h_ind = int((dates[f]-dates_shifted[f]).seconds/60/60)

                # read data
                data = pd.read_csv(source_folder + file_list[f])

                try:
                    t = data["UTC time"].tolist()
                    timestamp = np.array(data["timestamp"])
                    x = np.array(data["x"])  # x-axis acceleration
                    y = np.array(data["y"])  # y-axis acceleration
                    z = np.array(data["z"])  # z-axis acceleration

                    t = [t_ind.replace("T", " ") for t_ind in t]
                    t = [datetime.strptime(t_ind, '%Y-%m-%d %H:%M:%S.%f')
                         for t_ind in t]
                    t_shifted = [t_i-timedelta(microseconds=t_i.microsecond)
                                 for t_i in t]

                    # find seconds with enough samples
                    hour_start = t_shifted[0]
                    hour_start = (hour_start -
                                  timedelta(minutes=hour_start.minute) -
                                  timedelta(seconds=hour_start.second))
                    hour_end = hour_start + timedelta(hours=1)
                    t_sec_bins = pd.date_range(hour_start,
                                               hour_end, freq='S').tolist()
                    samples_per_sec, t_sec_bins = np.histogram(t_shifted,
                                                               t_sec_bins)
                    # seconds with enough samples / 9 should be in fact fs
                    samples_enough = samples_per_sec >= (fs - 1)

                    # find bouts with sufficient duration (here, minimum 5s)
                    run_length, start_ind, val = rle(samples_enough)
                    bout_start = start_ind[val & (run_length >= 5)]
                    bout_duration = run_length[val & (run_length >= 5)]

                    for b_ind, b_datetime in enumerate(bout_start):
                        # create a list with second-level timestamps
                        bout_time = pd.date_range(
                            t_sec_bins[bout_start[b_ind]],
                            t_sec_bins[bout_start[b_ind] +
                                       bout_duration[b_ind]],
                            freq='S').tolist()
                        bout_time = bout_time[:-1]
                        bout_time = [t_i.to_pydatetime()
                                     for t_i in bout_time]

                        # find observations in this bout
                        acc_ind = np.isin(t_shifted, bout_time)
                        t_bout = timestamp[acc_ind]/1000
                        x_bout = x[acc_ind]
                        y_bout = y[acc_ind]
                        z_bout = z[acc_ind]

                        if np.sum([np.std(x_bout), np.std(y_bout),
                                   np.std(z_bout)]) > minimum_activity_thr:
                            # interpolate bout to 10Hz and calculate vm
                            vm_bout = preprocess_bout(t_bout, x_bout, y_bout,
                                                      z_bout)[3]
                            # find walking and estimate steps
                            cadence_bout = find_walking(vm_bout, fs, min_amp,
                                                        step_freq, alpha, beta,
                                                        epsilon, delta)
                            cadence_bout = cadence_bout[np.where(cadence_bout
                                                                 > 0)]
                            if (option is None or option == 'both' or
                                    option == 'daily'):
                                cadence_temp_daily.append(cadence_bout)
                            if (option is None or option == 'both' or
                                    option == 'hourly'):
                                cadence_temp_hourly.append(cadence_bout)

                    if (option is None or option == 'both' or
                            option == 'hourly'):
                        cadence_temp_hourly = [item for sublist in
                                               cadence_temp_hourly
                                               for item in sublist]

                        walkingtime_hourly[d_ind, h_ind] = len(
                            cadence_temp_hourly)
                        steps_hourly[d_ind, h_ind] = int(np.sum(
                            cadence_temp_hourly))
                        cadence_hourly[d_ind, h_ind] = np.mean(
                            cadence_temp_hourly)

                except IndexError:
                    print('Empty file')

            if option is None or option == 'both' or option == 'daily':
                cadence_temp_daily = [item for sublist in
                                      cadence_temp_daily
                                      for item in sublist]

                walkingtime_daily[d_ind] = len(cadence_temp_daily)
                steps_daily[d_ind] = int(np.sum(cadence_temp_daily))
                cadence_daily[d_ind] = np.mean(cadence_temp_daily)

        # save results
        if option is None or option == 'both' or option == 'daily':
            summary_stats = pd.DataFrame({'date': days.strftime('%Y-%m-%d'),
                                          'walking_time':
                                              walkingtime_daily[:, -1],
                                          'steps': steps_daily[:, -1],
                                          'cadence': cadence_daily[:, -1]})
            dest_path = output_folder + "/daily/" + ID + "_gait_daily.csv"
            summary_stats.to_csv(dest_path, index=False)

        if option is None or option == 'both' or option == 'hourly':
            summary_stats = pd.DataFrame({'date': days_hourly[:-1].
                                          strftime('%Y-%m-%d %H:%M:%S'),
                                          'walking_time': walkingtime_hourly.
                                          flatten(),
                                          'steps': steps_hourly.flatten(),
                                          'cadence': cadence_hourly.flatten()})
            dest_path = output_folder + "/hourly/" + ID + "_gait_hourly.csv"
            summary_stats.to_csv(dest_path, index=False)


# global variables:
fs = 10  # desired sampling rate (frequency) (in Hertz (Hz))
gravity = 9.80665

# tuning parameters for walking recognition:
min_amp = 0.3  # minimum peak-to-peak amplitude (in gravitational units (g))
step_freq = (1.4, 2.3)  # step frequency (in Hz) - sfr
alpha = 0.6  # maximum ratio between dominant peak below and within sfr
beta = 2.5  # maximum ratio between dominant peak above and within sfr
delta = 20  # maximum change of step frequency between two one-second
# nonoverlapping segments (multiplication of 0.05Hz, e.g., delta=2 -> 0.1Hz)
epsilon = 3  # minimum walking time (in seconds (s))

# other thresholds:
minimum_activity_thr = 0.1  # threshold to qualify act. bout for computation

# study folder (change to your directory)
study_folder = "C:/Users/User1/Documents/project/data"
output_folder = "C:/Users/User1/Documents/project/results"

tz_str = "America/New_York"
time_start = "2018-01-01 00_00_00"
time_end = "2022-01-01 00_00_00"
option = "Both"
beiwe_id = None

# main function
main_function(study_folder, output_folder, tz_str, option,
              time_start, time_end, beiwe_id)
