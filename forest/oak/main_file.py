"""
Quantification of various types of physical
activity using accelerometer data.

To do finished when the paper appears.
"""

import os
import sys
import pandas as pd
import numpy as np
from dateutil import tz
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.signal import find_peaks, tukey
from scipy.interpolate import interp2d

from ssqueezepy import ssq_cwt


def rle(inarray):
    """
    Run length encoding.

    Parameters
    ----------
    inarray: multidatatype
        array for run length encoding

    Returns
    -------
    Runlengths : tuple
        Running length
    Startpositions : tuple
        starting index
    Values : tuple
        Running values

    """

    ia = np.asarray(inarray)                 # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]                # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)    # must include last element posi
        z = np.diff(np.append(-1, i))        # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return(z, p, ia[i])


def standardize_bout(t_bout, x_bout, y_bout, z_bout):
    """
    Interpolate acceleration to unify sampling rate.

    Parameters
    ----------
    t_bout : array of floats
        Unix timestamp
    x_bout : array of floats
        X-axis acceleration
    y_bout : array of floats
        Y-axis acceleration
    z_bout : array of floats
        Z-axis acceleration

    Returns
    -------
    x_bout : array of floats
        Interpolated x-axis acceleration
    y_bout : array of floats
        Interpolated y-axis acceleration
    z_bout : array of floats
        Interpolated z-axis acceleration
    vm_bout : array of floats
        Interpolated vector magnitude

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

    return x_bout, y_bout, z_bout, vm_bout


def adjust_bout(vm_bout, fs):
    """
    Fill observations in incomplete bouts.
    E.g., if the bout is 9.8s long, function fills in values at its
    end to make it 10s (results in N%fs=0).

    Parameters
    ----------
    vm_bout : array of floats
        vector magnitude with one bout of activity
    fs : integer
        sampling frequency

    Returns
    -------
    vm_bout : array of floats
        vector magnitude with one bout of activity

    """

    if len(vm_bout) % fs >= 0.7*fs:
        for i in range(fs-len(vm_bout) % fs):
            vm_bout = np.append(vm_bout, vm_bout[-1])
    elif len(vm_bout) % fs != 0:
        vm_bout = vm_bout[np.arange(len(vm_bout)//fs*fs)]
    return vm_bout


def find_walking(vm_bout, fs, A, step_freq, alpha, beta, epsilon, delta):
    """
    Finds walking periods within raw accelerometery data.
    E.g., if the bout is 9.8s long, function fills in values at it's end to
    make it 10s.

    Parameters
    ----------
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

    Returns
    -------
    cad : gait speed within bout of activity (in Hz or steps/sec)

    """

    # define wavelet function used in method
    wavelet = ('gmw', {'beta': 60, 'gamma': 3})

    # calculate pp to exclude low-intensity periods from further computing
    x = vm_bout.reshape(fs, -1, order="F")
    pp = np.array([max(x[:, i])-min(x[:, i]) for i in range(x.shape[1])])
    valid = np.ones(len(pp), dtype=bool)
    valid[pp < A] = False

    # compute cwt only if valid fragment is suffiently long
    if sum(valid) >= epsilon:
        # trim bout to valid periods only
        tapered_bout = vm_bout[np.repeat(valid, fs)]

        # add some noise in the edges to minimize impact of coin of influence
        w = tukey(len(tapered_bout), alpha=0.02, sym=True)
        tapered_bout = np.concatenate((np.zeros(5*fs),
                                       (tapered_bout)*w,
                                       np.zeros(5*fs)))

        # compute cwt over bout
        try:
            out = ssq_cwt(tapered_bout, wavelet, fs=10)
        except:
            tapered_bout = tapered_bout[:-1]
            out = ssq_cwt(tapered_bout, wavelet, fs=10)

        # magnitude of cwt
        coefs = out[0]
        coefs = np.abs(coefs**2)

        # interpolate spectrogram
        freqs = out[2]
        freqs_interp = np.arange(0.5, 4.5, 0.05)
        ip = interp2d(range(coefs.shape[1]), freqs, coefs)
        coefs_interp = ip(range(coefs.shape[1]), freqs_interp)

        # trim spectrogram from the coi
        coefs_interp = coefs_interp[:, 5*fs:-5*fs]

        # identify dominant peaks with the spectrum
        D = np.zeros((coefs_interp.shape[0], int(coefs_interp.shape[1]/fs)))
        loc_min = np.argmin(abs(freqs_interp-step_freq[0]))
        loc_max = np.argmin(abs(freqs_interp-step_freq[1]))
        for i in range(int(coefs_interp.shape[1]/fs)):
            # segment measurement into one-second non-overlapping windows
            xStart = i*fs
            xFinish = (i+1)*fs
            # identify peaks and their location in each window
            window = np.sum(coefs_interp[:, np.arange(xStart, xFinish)],
                            axis=1)
            locs, _ = find_peaks(window)
            pks = window[locs]
            ind = np.argsort(-pks)
            locs = locs[ind]
            pks = pks[ind]
            index_in_range = []

            # account peaks that satisfy condition
            for j in range(len(locs)):
                if locs[j] >= loc_min and locs[j] <= loc_max:
                    index_in_range.append(j)
                if len(index_in_range) >= 1:
                    break
            x = np.zeros(coefs_interp.shape[0])
            if len(index_in_range) > 0:
                if locs[0] > loc_max:
                    if pks[0]/pks[index_in_range[0]] < alpha:
                        x[locs[index_in_range[0]]] = 1
                elif locs[0] < loc_min:
                    if pks[0]/pks[index_in_range[0]] < beta:
                        x[locs[index_in_range[0]]] = 1
                else:
                    x[locs[index_in_range[0]]] = 1
            D[:, i] = x

        # distribute local maxima across valid periods
        E = np.zeros((D.shape[0], len(valid)))
        E[:, valid] = D

        # find when peaks are continuous in time and frequency
        B = find_continuous_dominant_peaks(E, epsilon, delta)

        # summarize the results
        cad = np.zeros(E.shape[1])
        for i in range(len(cad)):
            ind_freqs = np.where(B[:, i] > 0)[0]
            if len(ind_freqs) > 0:
                cad[i] = freqs_interp[ind_freqs[0]]

        # num_steps = int(round(sum(cad),0))
    else:
        # num_steps = 0
        cad = np.zeros(int(vm_bout.shape[0]/fs))

    return cad


def find_continuous_dominant_peaks(E, epsilon, delta):
    """
    Function that identifies continuous and sustained peaks within matrix.
    Peaks that do not satisfy the desired continuity and stability are cleared.

    Parameters
    ----------
    E : nparray
        binary matrix (1=peak,0=no peak)
    epsilon : integer
        minimum duration of peaks (in seconds)
    delta : integer
        maximum difference between consecutive peaks (in multiplication of
                                                      0.05Hz)

    Returns
    -------
    B : nparray
        binary matrix (1=peak,0=no peak)

    """
    E = np.concatenate((E, np.zeros((E.shape[0], 1))), axis=1)
    B = np.zeros((E.shape[0], E.shape[1]))
    for m in range(E.shape[1]-epsilon):
        A = E[:, np.arange(m, m+epsilon)]
        loop = [i for i in np.arange(epsilon)] + [i for i in
                                                  np.arange(epsilon-2, -1, -1)]
        for t in range(len(loop)):
            s = loop[t]
            pr = np.where(A[:, s] != 0)[0]
            j = 0
            if len(pr) > 0:
                for i in range(len(pr)):
                    index = np.arange(max(0, pr[i]-delta), min(pr[i]+delta+1,
                                                               A.shape[0]))
                    if s == 0 or s == epsilon-1:
                        W = np.transpose(np.array([np.ones(len(index))*pr[i],
                                                   index], dtype=int))
                    else:
                        W = np.transpose(np.array([index,
                                                   np.ones(len(index))*pr[i],
                                                   index], dtype=int))

                    F = np.zeros((W.shape[0], W.shape[1]), dtype=int)
                    if s == 0:
                        F[:, 0] = A[W[:, 0], s]
                        F[:, 1] = A[W[:, 1], s+1]
                    elif s == epsilon-1:
                        F[:, 0] = A[W[:, 0], s]
                        F[:, 1] = A[W[:, 1], s-1]
                    else:
                        F[:, 0] = A[W[:, 0], s-1]
                        F[:, 1] = A[W[:, 1], s]
                        F[:, 2] = A[W[:, 2], s+1]
                    G1 = W[np.sum(F[:, np.arange(2)], axis=1) > 1, :]
                    if s == 0 or s == epsilon-1:
                        if G1.shape[0] == 0:
                            A[W[:, 0], s] = 0
                        else:
                            j = j + 1
                    else:
                        G2 = W[np.sum(F[:, np.arange(1, 3)], axis=1) > 1, :]
                        if G1.shape[0] == 0 or G2.shape[0] == 0:
                            A[W[:, 1], s] = 0
                        else:
                            j = j + 1
            if j == 0:
                A = np.zeros((A.shape[0], A.shape[1]))
                break
        B[:, np.arange(m, m + epsilon)] = np.maximum(B[:,
                                                       np.arange(m, m +
                                                                 epsilon)], A)

    B = B[:, :-1]
    return B


def main_function(study_folder: str, output_folder: str, tz_str: str,
                  option: str, time_start=None, time_end=None, beiwe_id=None):
    """
    Compute walking metrics from raw accelerometer data collected with Beiwe

    Parameters
    ----------
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
    to_zone = tz.gettz('America/New_York')

    # create folders for results
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    if option is None or option == 'both' or option == 'daily':
        if os.path.exists(output_folder+"/daily") is False:
            os.mkdir(output_folder+"/daily")
    if option is None or option == 'both' or option == 'hourly':
        if os.path.exists(output_folder+"/hourly") is False:
            os.mkdir(output_folder+"/hourly")

    if beiwe_id is None:
        beiwe_id = os.listdir(study_folder)

    for ID in beiwe_id:
        sys.stdout.write('User: ' + ID + '\n')

        source_folder = study_folder + '/' + ID + '/accelerometer/'
        file_list = os.listdir(source_folder)

        # transform all files in folder to datelike format
        if "+00_00.csv" in file_list:
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

        # initiate metrics
        # data quality - TO DO
        # compliance - TO DO

        # activity intensity - TO DO

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

        # running
        # TO DO

        for d, d_datetime in enumerate(days):
            sys.stdout.write('Day: ' + str(d) + '\n')
            # find file indices for this day
            file_ind = [i for i, x in enumerate(dates_shifted)
                        if x == d_datetime]

            # initiate temporal metric
            if option is None or option == 'both' or option == 'daily':
                cadence_temp_daily = []

            for f in file_ind:
                sys.stdout.write('File: ' + str(f) + '\n')

                # initiate temporal metric
                if option is None or option == 'both' or option == 'hourly':
                    cadence_temp_hourly = []
                    # hour of the day
                    h = int((dates[f]-dates_shifted[f]).seconds/60/60)

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
                    N, BI, B = rle(samples_enough)
                    bout_start = BI[(B is True) & (N >= 5)]
                    bout_duration = N[(B is True) & (N >= 5)]

                    for b, b_datetime in enumerate(bout_start):
                        # create a list with second-level timestamps
                        bout_time = pd.date_range(t_sec_bins[bout_start[b]],
                                                  t_sec_bins[bout_start[b] +
                                                             bout_duration[b]],
                                                  freq='S').tolist()
                        bout_time = bout_time[:-1]

                        # find observations in this bout
                        acc_ind = np.isin(t_shifted, bout_time)
                        t_bout = timestamp[acc_ind]/1000
                        x_bout = x[acc_ind]
                        y_bout = y[acc_ind]
                        z_bout = z[acc_ind]

                        if np.sum([np.std(x_bout), np.std(y_bout),
                                   np.std(z_bout)]) > minimum_activity_thr:
                            # interpolate bout to 10Hz and calculate vm
                            [x_bout,
                             y_bout,
                             z_bout,
                             vm_bout] = standardize_bout(t_bout,
                                                         x_bout,
                                                         y_bout,
                                                         z_bout)

                            # activity intensity - TO DO
                            # activity types - gait

                            cadence_bout = find_walking(vm_bout, fs, A,
                                                        step_freq, alpha, beta,
                                                        epsilon, delta)

                            cadence_bout = cadence_bout[np.where(cadence_bout
                                                                 > 0)]
                            if (option is None or option == 'both' or
                                    option == 'daily'):
                                cadence_temp_daily = np.concatenate((
                                    cadence_temp_daily, cadence_bout))
                            if (option is None or option == 'both' or
                                    option == 'hourly'):
                                cadence_temp_hourly = np.concatenate((
                                    cadence_temp_hourly, cadence_bout))

                    if (option is None or option == 'both' or
                            option == 'hourly'):
                        walkingtime_hourly[d, h] = len(cadence_temp_hourly)
                        steps_hourly[d, h] = int(np.sum(cadence_temp_hourly))
                        cadence_hourly[d, h] = np.mean(cadence_temp_hourly)

                except IndexError:
                    print('Empty file')

            if option is None or option == 'both' or option == 'daily':
                walkingtime_daily[d] = len(cadence_temp_daily)
                steps_daily[d] = int(np.sum(cadence_temp_daily))
                cadence_daily[d] = np.mean(cadence_temp_daily)

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


# global variables
fs = 10  # desired sampling rate (frequency) (in Hertz (Hz))
gravity = 9.80665

# tuning parameters for walking recognition:
#
# minimum peak-to-peak amplitude (in gravitational units (g))
A = 0.3
# step frequency (in Hz)
step_freq = [1.4, 2.3]
# alpha = maximum ratio between dominant peak below and within step frequency
# range
alpha = 0.6
# beta = maximum ratio between dominant peak above and within step frequency
# range
beta = 2.5
# delta = maximum change of step frequency between two one-second
# nonoverlapping segments (multiplication of 0.05Hz, e.g., delta=2 -> 0.1Hz)
delta = 20
# epsilon = minimum walking time (in seconds (s))
epsilon = 3

# other thresholds
minimum_activity_thr = 0.1

# study folder (CHANGE TO YOUR DIRECTORY)
study_folder = 'C:/Users/mstra/Documents/data/',
'beiwe_test_data/onnela_lab_ios_test2'
output_folder = "C:/Users/mstra/Documents/Python/forest/oak/output"

tz_str = "America/New_York"
time_start = "2018-01-01"
time_end = "2022-01-01"

# main function
main_function(study_folder, output_folder, tz_str, option=None,
              time_start=None, time_end=None, beiwe_id={"sxvpopdz"})
