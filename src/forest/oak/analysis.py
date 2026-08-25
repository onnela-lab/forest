"""
Original Authors: Marcin Straczkiewicz, Georgios Efstathiadis, Zachary Clement (and probably others)

Maintenance - Eli Jones

Step counting method for accelerometer data

Module is aimed to process raw accelerometer smartphone data collected with Beiwe Research Platform.
Data preprocessing involves signal preproprocesing (unit standardization and interpolation to 10Hz),
transformation using Continuous Wavelet transform (using ssqueezepy package), and calculation of
steps from the identified walking bouts. Additional gait features calculated by module are walking
time and gait speed (cadence).

Results may be output in hourly and daily intervals.
"""


import logging

import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from ssqueezepy import ssq_cwt

from forest.constants import FP64Arr


logger = logging.getLogger(__name__)


def get_pp(vm_bout: FP64Arr, fs: int = 10) -> FP64Arr:
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
    return np.ptp(vm_res_sec, axis=0)


def compute_interpolate_cwt(
    tapered_bout: FP64Arr,
    fs: int = 10,
    wavelet: tuple[str, dict[str, int]] = ('gmw', {'beta': 90, 'gamma': 3})
) -> tuple[FP64Arr, FP64Arr]:
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
    tapered_bout = np.concatenate((np.zeros(5 * fs), tapered_bout * window, np.zeros(5 * fs)))
    
    # compute cwt over bout (wavelet is of an accepted type, pyright doesn't like it)
    out: FP64Arr = ssq_cwt(tapered_bout[:-1], wavelet, fs=10)  # type: ignore
    coefs = out[0]
    
    coefs = np.append(coefs, coefs[:, -1:], 1)
    coefs = coefs.astype('complex128')
    
    coefs = np.abs(coefs**2)  # magnitude of cwt
    
    # interpolate coefficients
    freqs = out[2]
    freqs_interp = np.arange(0.5, 4.5, 0.05)
    interpolator = interpolate.RegularGridInterpolator((freqs, range(coefs.shape[1])), coefs)
    grid_x, grid_y = np.meshgrid(freqs_interp, range(coefs.shape[1]), indexing='ij')
    coefs_interp = interpolator((grid_x, grid_y))
    
    coefs_interp = coefs_interp[:, 5*fs:-5*fs]  # trim spectrogram from the coi
    
    return freqs_interp, coefs_interp


def identify_peaks_in_cwt(
    freqs_interp: FP64Arr,
    coefs_interp: FP64Arr,
    fs: int = 10,
    step_freq: tuple = (1.4, 2.3),
    alpha: float = 0.6,
    beta: float = 2.5,
) -> FP64Arr:
    """Identify dominant peaks in wavelet coefficients.
    
    Method uses alpha and beta parameters to identify dominant peaks in one-second non-overlapping
    windows in the product of Continuous Wavelet Transformation. Dominant peaks need tooccur within
    the step frequency range.
    
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
            maximum ratio between dominant peak below and within step frequency range
        beta: float
            maximum ratio between dominant peak above and within step frequency range
    
    Returns:
        Ndarray with dominant peaks
    """
    # identify dominant peaks within coefficients
    num_rows, num_cols = coefs_interp.shape
    num_cols2 = int(num_cols/fs)
    
    dp = np.zeros((num_rows, num_cols2))
    
    loc_min = np.argmin(abs(freqs_interp-step_freq[0]))
    loc_max = np.argmin(abs(freqs_interp-step_freq[1]))
    
    for i in range(num_cols2):
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
        
        index_in_range = None
        
        for j, locs_j in enumerate(locs):  # account peaks that satisfy condition
            if loc_min <= locs_j <= loc_max:
                index_in_range = j
                break
        
        peak_vec = np.zeros(num_rows)
        
        if index_in_range is not None:
            # Check if there are peaks below and above the step frequency range
            peaks_below = pks[locs < loc_min]
            peaks_above = pks[locs > loc_max]
            
            # Calculate max peak magnitudes, handling empty arrays
            max_peak_magnitude_a = (np.max(peaks_below) if len(peaks_below) > 0 else 0)
            max_peak_magnitude_b = (np.max(peaks_above) if len(peaks_above) > 0 else 0)
            
            if (
                max_peak_magnitude_b / pks[index_in_range] < beta
                or max_peak_magnitude_a / pks[index_in_range] < alpha
            ):
                peak_vec[locs[index_in_range]] = 1
        dp[:, i] = peak_vec
    
    return dp


def find_walking(
    vm_bout: FP64Arr,
    fs: int = 10,
    min_amp: float = 0.3,
    step_freq: tuple = (1.4, 2.3),
    alpha: float = 0.6,
    beta: float = 2.5,
    min_t: int = 3,
    delta: int = 20,
) -> FP64Arr:
    """Finds walking and calculate steps from raw acceleration data.
    
    Method finds periods of repetitive and continuous oscillations with predominant frequency
    occurring within know step frequency range. Frequency components are extracted with Continuous
    Wavelet Transform.
    
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
            maximum ratio between dominant peak below and within step frequency range
        beta: float
            maximum ratio between dominant peak above and within step frequency range
        min_t: integer
            minimum duration of peaks (in seconds)
        delta: integer
            maximum difference between consecutive peaks (in multiplication of 0.05Hz)
    
    Returns:
        Ndarray with identified number of steps per second
    """
    
    wavelet = ('gmw', {'beta': 90, 'gamma': 3})  # define wavelet function used in method
    
    pp = get_pp(vm_bout, fs)  # calculate peak-to-peak
    
    valid = np.ones(len(pp), dtype=bool)  # assume the entire bout is of high-intensity
    
    valid[pp < min_amp] = False  # exclude low-intensity periods
    
    # compute cwt only if valid fragment is sufficiently long
    if sum(valid) >= min_t:
        tapered_bout = vm_bout[np.repeat(valid, fs)]  # trim bout to valid periods only
        
        # compute and interpolate CWT
        freqs_interp, coefs_interp = compute_interpolate_cwt(tapered_bout, fs, wavelet)
        
        # get map of dominant peaks
        dp = identify_peaks_in_cwt(freqs_interp, coefs_interp, fs, step_freq, alpha, beta)
        
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


def find_continuous_dominant_peaks(valid_peaks: FP64Arr, min_t: int, delta: int) -> FP64Arr:
    """Identifies continuous and sustained peaks within matrix.
    
    Args:
        valid_peaks: nparray
            binary matrix (1=peak,0=no peak) of valid peaks
        min_t: integer
            minimum duration of peaks (in seconds)
        delta: integer
            maximum difference between consecutive peaks (in multiplication of 0.05Hz)
    
    Returns:
        Ndarray with binary matrix (1=peak,0=no peak) of continuous peaks
    """
    
    num_rows, num_cols = valid_peaks.shape
    
    extended_peaks = np.zeros((num_rows, num_cols + 1), dtype=valid_peaks.dtype)
    extended_peaks[:, :num_cols] = valid_peaks
    
    cont_peaks = np.zeros_like(extended_peaks)
    
    for slice_ind in range(num_cols + 1 - min_t):
        slice_mat = extended_peaks[:, slice_ind:slice_ind + min_t]
        
        windows = list(range(min_t)) + list(range(min_t-2, -1, -1))
        stop = True
        
        for win_ind in windows:
            pr = np.where(slice_mat[:, win_ind] != 0)[0]
            stop = True
            
            for p in pr:
                index = np.arange(max(0, p - delta), min(p + delta + 1, num_rows))
                
                peaks1 = slice_mat[p, win_ind]
                peaks2 = peaks1
                if win_ind == 0:
                    peaks1 += slice_mat[index, win_ind + 1]
                elif win_ind == min_t - 1:
                    peaks1 += slice_mat[index, win_ind - 1]
                else:
                    peaks1 += slice_mat[index, win_ind - 1]
                    peaks2 += slice_mat[index, win_ind + 1]
                
                if win_ind == 0 or win_ind == min_t - 1:
                    if np.any(peaks1 > 1):
                        stop = False
                    else:
                        slice_mat[p, win_ind] = 0
                else:
                    if np.any(peaks1 > 1) and np.any(peaks2 > 1):
                        stop = False
                    else:
                        slice_mat[p, win_ind] = 0
            
            if stop:
                break
        
        if not stop:
            cont_peaks[:, slice_ind:slice_ind + min_t] = slice_mat
    
    return cont_peaks[:, :-1]
