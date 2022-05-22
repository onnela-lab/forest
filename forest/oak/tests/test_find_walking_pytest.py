from datetime import datetime
import os

import numpy as np
import pandas as pd
import pytest
from scipy import interpolate
from scipy.signal import find_peaks, tukey
from ssqueezepy import ssq_cwt

from forest.oak.main import (preprocess_bout, find_walking)


TEST_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def fs():
    return 10


@pytest.fixture(scope="session")
def wavelet():
    return ('gmw', {'beta': 90, 'gamma': 3})


@pytest.fixture(scope="session")
def min_amp():
    return 0.3


@pytest.fixture(scope="session")
def step_freq():
    return (1.4, 2.3)


@pytest.fixture(scope="session")
def alpha():
    return 0.6


@pytest.fixture(scope="session")
def beta():
    return 2.5


@pytest.fixture(scope="session")
def delta():
    return 20


@pytest.fixture(scope="session")
def epsilon():
    return 3


@pytest.fixture(scope="session")
def signal_bout():
    data = pd.read_csv(os.path.join(TEST_DATA_DIR, "test_data_bout.csv"))
    timestamp = np.array(data["timestamp"], dtype="float64")
    t = data["UTC time"].tolist()
    x = np.array(data["x"], dtype="float64")
    y = np.array(data["y"], dtype="float64")
    z = np.array(data["z"], dtype="float64")

    timestamp = timestamp/1000
    t = [t_ind.replace("T", " ") for t_ind in t]
    t = [datetime.strptime(t_ind, '%Y-%m-%d %H:%M:%S.%f')
         for t_ind in t]
    return timestamp, t, x, y, z


def test_reshape(signal_bout, fs):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[3]
    vm_res_sec = vm_bout.reshape(fs, -1, order="F")
    expected_output = (10, 10)
    assert vm_res_sec.shape == expected_output


def test_pp(signal_bout, fs):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[3]
    vm_res_sec = vm_bout.reshape(fs, -1, order="F")
    pp = np.array([max(vm_res_sec[:, i])-min(vm_res_sec[:, i])
                   for i in range(vm_res_sec.shape[1])])
    expected_output = np.array([0.64, 0.71, 1.11, 0.79, 0.37, 0.70, 1.20, 0.61,
                                0.66, 0.48])
    assert np.array_equal(np.round(pp, 2), expected_output)


def test_tukey(signal_bout, fs):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[3]
    win = tukey(len(vm_bout), alpha=0.05, sym=True)
    expected_output = np.array([0, 0.35, 0.91])
    assert np.array_equal(np.round(win[0:3], 2), expected_output)
    assert np.array_equal(np.round(win, 2), np.round(np.flipud(win), 2))


def test_ssq_cwt(signal_bout, fs, wavelet):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[3]
    win = tukey(len(vm_bout), alpha=0.05, sym=True)
    tapered_bout = np.concatenate((np.zeros(5*fs), (vm_bout)*win,
                                   np.zeros(5*fs)))
    out = ssq_cwt(tapered_bout, wavelet, fs=10)
    coefs = out[0]
    coefs = np.abs(coefs**2)
    expected_output_amp = 0.000652
    freqs = out[2]
    expected_output_freqs = np.array([0.0503, 4.9749])
    assert tapered_bout.shape == (200,)
    assert np.round(np.max(coefs), 6) == expected_output_amp
    assert len(freqs) == 153
    assert np.array_equal(np.round(freqs[[0, -1]], 4),
                          expected_output_freqs)


def test_coefs_interp(signal_bout, fs, wavelet):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[3]
    win = tukey(len(vm_bout), alpha=0.05, sym=True)
    tapered_bout = np.concatenate((np.zeros(5*fs), (vm_bout)*win,
                                   np.zeros(5*fs)))
    out = ssq_cwt(tapered_bout, wavelet, fs=10)
    coefs = out[0]
    coefs = np.abs(coefs**2)
    freqs = out[2]
    freqs_interp = np.arange(0.5, 4.5, 0.05)
    ip = interpolate.interp2d(range(coefs.shape[1]), freqs, coefs)
    coefs_interp = ip(range(coefs.shape[1]), freqs_interp)
    coefs_interp = coefs_interp[:, 5*fs:-5*fs]
    expected_output = 0.000482
    assert coefs_interp.shape == (80, 100)
    assert np.array_equal(np.round(np.max(coefs_interp), 6), expected_output)


def test_dominant_peaks(signal_bout, fs, min_amp, step_freq, alpha, beta,
                        wavelet):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[3]
    win = tukey(len(vm_bout), alpha=0.05, sym=True)
    tapered_bout = np.concatenate((np.zeros(5*fs), (vm_bout)*win,
                                   np.zeros(5*fs)))
    out = ssq_cwt(tapered_bout, wavelet, fs=10)
    coefs = out[0]
    coefs = np.abs(coefs**2)
    freqs = out[2]
    freqs_interp = np.arange(0.5, 4.5, 0.05)
    ip = interpolate.interp2d(range(coefs.shape[1]), freqs, coefs)
    coefs_interp = ip(range(coefs.shape[1]), freqs_interp)
    coefs_interp = coefs_interp[:, 5*fs:-5*fs]
    dp = np.zeros((coefs_interp.shape[0], int(coefs_interp.shape[1]/fs)))
    loc_min = np.argmin(abs(freqs_interp-step_freq[0]))
    loc_max = np.argmin(abs(freqs_interp-step_freq[1]))
    for i in range(int(coefs_interp.shape[1]/fs)):
        # segment measurement into one-second non-overlapping windows
        x_start = i*fs
        x_end = (i + 1)*fs
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
                if pks[0]/pks[index_in_range[0]] < beta:
                    peak_vec[locs[index_in_range[0]]] = 1
            elif locs[0] < loc_min:
                if pks[0]/pks[index_in_range[0]] < alpha:
                    peak_vec[locs[index_in_range[0]]] = 1
            else:
                peak_vec[locs[index_in_range[0]]] = 1
        dp[:, i] = peak_vec

    expected_output_val = np.ones(10)
    expected_output_ind = np.array([23, 22, 21, 22, 21, 27, 26, 25, 25, 24])
    assert np.array_equal(np.argmax(dp, axis=0), expected_output_ind)
    assert np.array_equal(np.max(dp, axis=0), expected_output_val)


def test_fing_walking(signal_bout, fs, min_amp, step_freq, alpha, beta,
                      epsilon, delta):
    timestamp, _, x, y, z = signal_bout
    vm_bout = preprocess_bout(timestamp, x, y, z)[3]
    cadence_bout = find_walking(vm_bout, fs, min_amp, step_freq, alpha, beta,
                                epsilon, delta)
    expected_output = np.array([1.65, 1.6, 1.55, 1.6, 1.55, 1.85, 1.8, 1.75,
                                1.75, 1.7])
    assert len(cadence_bout) == 10
    assert np.array_equal(np.round(cadence_bout, 2), expected_output)
