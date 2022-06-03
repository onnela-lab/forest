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
import os

from dateutil import tz
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate
from scipy.signal import find_peaks, tukey
from ssqueezepy import ssq_cwt

from forest.utils import get_ids

logger = logging.getLogger(__name__)


def rle(inarray: np.ndarray) -> tuple:
    """Runs length encoding.

    Args:
        inarray: array of Boolean values
            input for run length encoding

    Returns:
        Tuple of running length, starting index, and running values

    """
    array_length = len(inarray)
    if array_length == 0:
        return None, None, None
    else:
        pairwise_unequal = inarray[1:] != inarray[:-1]  # pairwise unequal
        ind = np.append(np.where(pairwise_unequal),
                        array_length - 1)  # must include last element position
        run_length = np.diff(np.append(-1, ind))  # run lengths
        start_ind = np.cumsum(np.append(0, run_length))[:-1]  # position
        val = inarray[ind]  # running values
        return run_length, start_ind, val


def preprocess_bout(t_bout: np.ndarray, x_bout: np.ndarray, y_bout: np.ndarray,
                    z_bout: np.ndarray, FS: int = 10) -> tuple:
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
        FS: integer
            sampling frequency

    Returns:
        Tuple of ndarrays with interpolated acceleration in x, y, and z axes,
        as well as their vector magnitude
    """
    t_bout_interp = np.arange(t_bout[0], t_bout[-1], (1/FS))

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
    num_seconds = np.floor(len(x_bout_interp)/FS)

    # trim measurement to full seconds
    x_bout = x_bout_interp[:int(num_seconds*FS)]
    y_bout = y_bout_interp[:int(num_seconds*FS)]
    z_bout = z_bout_interp[:int(num_seconds*FS)]

    vm_bout = np.sqrt(x_bout**2 + y_bout**2 + z_bout**2)

    # standardize measurement to gravity units (g) if its recorded in m/s**2
    if np.mean(vm_bout) > 5:
        x_bout = x_bout/9.80665
        y_bout = y_bout/9.80665
        z_bout = z_bout/9.80665

    vm_bout = np.sqrt(x_bout**2 + y_bout**2 + z_bout**2) - 1

    return x_bout, y_bout, z_bout, vm_bout


def adjust_bout(inarray: np.ndarray, FS: int = 10) -> npt.NDArray[np.float64]:
    """Fills observations in incomplete bouts.

    For example, if the bout is 9.8s long, add values at its end to make it
    10s (results in N%FS=0).

    Args:
        inarray: array of floats
            input with one bout of activity
        FS: integer
            sampling frequency

    Returns:
        Ndarray with length-adjusted vector magnitude
    """
    if len(inarray) % FS >= 0.7*FS:
        for i in range(FS-len(inarray) % FS):
            inarray = np.append(inarray, inarray[-1])
    elif len(inarray) % FS != 0:
        inarray = inarray[np.arange(len(inarray)//FS*FS)]

    return inarray


def find_walking(vm_bout: np.ndarray, FS: int = 10, MIN_AMP: float = 0.3,
                 STEP_FREQ: tuple = (1.4, 2.3), ALPHA: float = 0.6,
                 BETA: float = 2.5, T: int = 3,
                 DELTA: int = 20) -> npt.NDArray[np.float64]:
    """Finds walking and calculate steps from raw acceleration data.

    Method finds periods of repetitive and continuous oscillations with
    predominant frequency occurring within know step frequency range.
    Frequency components are extracted with Continuous Wavelet Transform.

    Args:
        vm_bout: array of floats
            vector magnitude with one bout of activity (in g)
        FS: integer
            sampling frequency (in Hz)
        MIN_AMP: float
            minimum amplitude (in g)
        STEP_FREQ: tuple
            step frequency range
        ALPHA: float
            maximum ratio between dominant peak below and within
            step frequency range
        BETA: float
            maximum ratio between dominant peak above and within
            step frequency range
        T: integer
            minimum duration of peaks (in seconds)
        DELTA: integer
            maximum difference between consecutive peaks (in multiplication of
                                                          0.05Hz)

    Returns:
        Ndarray with identified number of steps per second
    """
    # define wavelet function used in method
    wavelet = ('gmw', {'beta': 90, 'gamma': 3})

    # calculate peak-to-peak to exclude low-intensity periods
    vm_res_sec = vm_bout.reshape((FS, -1), order="F")
    pp = np.array([max(vm_res_sec[:, i])-min(vm_res_sec[:, i])
                   for i in range(vm_res_sec.shape[1])])
    valid = np.ones(len(pp), dtype=bool)
    valid[pp < MIN_AMP] = False

    # compute cwt only if valid fragment is sufficiently long
    if sum(valid) >= T:
        # trim bout to valid periods only
        tapered_bout = vm_bout[np.repeat(valid, FS)]

        # smooth signal on the edges to minimize impact of coin of influence
        window = tukey(len(tapered_bout), alpha=0.02, sym=True)
        tapered_bout = np.concatenate((np.zeros(5*FS),
                                       tapered_bout*window,
                                       np.zeros(5*FS)))

        # compute cwt over bout
        out = ssq_cwt(tapered_bout[:-1], wavelet, FS=10)
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
        coefs_interp = coefs_interp[:, 5*FS:-5*FS]

        # identify dominant peaks with the spectrum
        dp = np.zeros((coefs_interp.shape[0], int(coefs_interp.shape[1]/FS)))
        loc_min = np.argmin(abs(freqs_interp-STEP_FREQ[0]))
        loc_max = np.argmin(abs(freqs_interp-STEP_FREQ[1]))
        for i in range(int(coefs_interp.shape[1]/FS)):
            # segment measurement into one-second non-overlapping windows
            x_start = i*FS
            x_end = (i + 1)*FS
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
                    if pks[0]/pks[index_in_range[0]] < BETA:
                        peak_vec[locs[index_in_range[0]]] = 1
                elif locs[0] < loc_min:
                    if pks[0]/pks[index_in_range[0]] < ALPHA:
                        peak_vec[locs[index_in_range[0]]] = 1
                else:
                    peak_vec[locs[index_in_range[0]]] = 1
            dp[:, i] = peak_vec

        # distribute local maxima across valid periods
        val_peaks = np.zeros((dp.shape[0], len(valid)))
        val_peaks[:, valid] = dp

        # find when peaks are continuous in time and frequency
        cp = find_continuous_dominant_peaks(val_peaks, T, DELTA)

        # summarize the results
        cad = np.zeros(val_peaks.shape[1])
        for i in range(len(cad)):
            ind_freqs = np.where(cp[:, i] > 0)[0]
            if len(ind_freqs) > 0:
                cad[i] = freqs_interp[ind_freqs[0]]

    else:
        cad = np.zeros(int(vm_bout.shape[0]/FS))

    return cad


def find_continuous_dominant_peaks(val_peaks: np.ndarray, T: int,
                                   DELTA: int) -> npt.NDArray[np.float64]:
    """Identifies continuous and sustained peaks within matrix.

    Args:
        val_peaks: nparray
            binary matrix (1=peak,0=no peak) of valid peaks
        T: integer
            minimum duration of peaks (in seconds)
        DELTA: integer
            maximum difference between consecutive peaks (in multiplication of
                                                          0.05Hz)

    Returns:
        Ndarray with binary matrix (1=peak,0=no peak) of continuous peaks
    """
    val_peaks = np.concatenate((val_peaks, np.zeros((val_peaks.shape[0], 1))),
                               axis=1)
    cp = np.zeros((val_peaks.shape[0], val_peaks.shape[1]))
    for slice_ind in range(val_peaks.shape[1] - T):
        slice_mat = val_peaks[:, np.arange(slice_ind, slice_ind + T)]
        windows = ([i for i in np.arange(T)] +
                   [i for i in np.arange(T-2, -1, -1)])
        for win_ind in windows:
            pr = np.where(slice_mat[:, win_ind] != 0)[0]
            count = 0
            if len(pr) > 0:
                for i in range(len(pr)):
                    index = np.arange(max(0, pr[i] - DELTA),
                                      min(pr[i] + DELTA + 1,
                                          slice_mat.shape[0]
                                          ))
                    if win_ind == 0 or win_ind == T - 1:
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
                    elif win_ind == T - 1:
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
                    if win_ind == 0 or win_ind == T - 1:  # first or last
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
            slice_ind, slice_ind + T)] = np.maximum(
                cp[:, np.arange(slice_ind, slice_ind + T)], slice_mat)
    return cp[:, :-1]


def main_function(study_folder: str, output_folder: str, tz_str: str = None,
                  option: str = None, time_start: str = None,
                  time_end: str = None, users: list = None,
                  FS: int = 10) -> None:
    """Runs walking recognition and step counting algorithm over dataset.

    Determine paths to input and output folders, set analysis time frames,
    subjects' local timezone, and time resolution of computed results.

    Args:
        study folder: string
            local repository with beiwe folders (IDs) for a given study
        output folder: string
            local repository to store results
        tz_str: string
            local time zone, e.g., "America/New_York"
        option: string
            summary statistics format (accepts 'both', 'hourly', 'daily')
        time_start: string
            initial date of study in format: 'YYYY-mm-dd HH_MM_SS'
        time_end: string
            final date of study in format: 'YYYY-mm-dd HH_MM_SS'
        users: list of strings
            beiwe ID selected for computation
        FS: integer
            sampling frequency
    """
    fmt = '%Y-%m-%d %H_%M_%S'
    from_zone = tz.gettz('UTC')
    if tz_str is None:
        tz_str = 'UTC'
    to_zone = tz.gettz(tz_str)

    # create folders to store results
    if option is None or option == 'both' or option == 'daily':
        os.makedirs(os.path.join(output_folder, "daily"), exist_ok=True)
    if option is None or option == 'both' or option == 'hourly':
        os.makedirs(os.path.join(output_folder, "hourly"), exist_ok=True)

    if users is None:
        users = get_ids(study_folder)

    for user in users:
        logger.info("Beiwe ID: %s", user)

        source_folder = os.path.join(study_folder, user, "accelerometer")
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
            logger.info("Day: %d", d_ind)

            # find file indices for this d_ind
            file_ind = [i for i, x in enumerate(dates_shifted)
                        if x == d_datetime]

            # initiate temporal metric
            if option is None or option == 'both' or option == 'daily':
                cadence_temp_daily = list()

            for f in file_ind:
                logger.info("File: %d", f)

                # initiate temporal metric
                if option is None or option == 'both' or option == 'hourly':
                    cadence_temp_hourly = list()
                    # hour of the day
                    h_ind = int((dates[f] - dates_shifted[f]).seconds/60/60)

                # read data
                data = pd.read_csv(os.path.join(source_folder, file_list[f]))

                try:
                    t = data["UTC time"].tolist()
                    timestamp = np.array(data["timestamp"])
                    x = np.array(data["x"], dtype="float64")  # x-axis acc.
                    y = np.array(data["y"], dtype="float64")  # y-axis acc.
                    z = np.array(data["z"], dtype="float64")  # z-axis acc.

                except (IndexError, RuntimeError):
                    logger.error('Corrupted file')
                    continue

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
                # seconds with enough samples / 9 should be in fact FS
                samples_enough = samples_per_sec >= (FS - 1)

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

                    # compute only if phone is on the body
                    if np.sum([np.std(x_bout), np.std(y_bout),
                               np.std(z_bout)]) > 0.1:
                        # interpolate bout to 10Hz and calculate vm
                        vm_bout = preprocess_bout(t_bout, x_bout, y_bout,
                                                  z_bout)[3]
                        # find walking and estimate steps
                        cadence_bout = find_walking(vm_bout)
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

            output_file = user + "_gait_daily.csv"
            dest_path = os.path.join(output_folder, "daily", output_file)
            summary_stats.to_csv(dest_path, index=False)

        if option is None or option == 'both' or option == 'hourly':
            summary_stats = pd.DataFrame({'date': days_hourly[:-1].
                                          strftime('%Y-%m-%d %H:%M:%S'),
                                          'walking_time': walkingtime_hourly.
                                          flatten(),
                                          'steps': steps_hourly.flatten(),
                                          'cadence': cadence_hourly.flatten()})

            output_file = user + "_gait_hourly.csv"
            dest_path = os.path.join(output_folder, "hourly", output_file)
            summary_stats.to_csv(dest_path, index=False)
