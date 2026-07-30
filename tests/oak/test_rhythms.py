"""Ground-truth tests for forest.oak.rhythms (rest-activity rhythms).

Each test builds a synthetic signal with known RAR properties and checks that the estimators recover
them, plus an end-to-end ``run`` over a small synthetic study folder.

Original Author: Ceyhun Olcan
"""

import numpy as np
import pandas as pd
import pytest

from forest.constants import Frequency
from forest.oak.rhythms import (compute_enmo_epochs, cosinor, interdaily_stability,
    intradaily_variability, rar_metrics, run)


EPOCH_S = 60
EPOCHS_PER_DAY = 86400 // EPOCH_S


@pytest.fixture
def rng():
    return np.random.default_rng(20260709)


def _hours(n_days):
    return np.arange(n_days * EPOCHS_PER_DAY) * (EPOCH_S / 3600.0)


def test_strong_rhythm_recovers_cosinor(rng):
    times = _hours(7)
    omega = 2 * np.pi / 24
    activity = 50 + 40 * np.cos(omega * (times - 14))
    activity = np.clip(activity + rng.normal(0, 2, times.size), 0, None)
    metrics = rar_metrics(activity, EPOCH_S)
    assert metrics["IS"] > 0.9
    assert metrics["IV"] < 0.05
    assert metrics["RA"] > 0.6
    assert abs(metrics["cosinor_mesor"] - 50) < 1.5
    assert abs(metrics["cosinor_amplitude"] - 40) < 1.5
    assert abs(metrics["cosinor_acrophase_h"] - 14) < 0.3
    assert 8 <= metrics["M10_onset_h"] <= 12


def test_white_noise_is_at_floor(rng):
    activity = np.clip(rng.normal(50, 15, 7 * EPOCHS_PER_DAY), 0, None)
    metrics = rar_metrics(activity, EPOCH_S)
    assert abs(metrics["IS"] - 1.0 / 7.0) < 0.05
    assert abs(metrics["IV"] - 2.0) < 0.15
    assert metrics["RA"] < 0.1
    assert metrics["cosinor_r2"] < 0.02


def test_identical_days_give_is_one():
    one_day = np.abs(np.sin(np.linspace(0, 2 * np.pi, EPOCHS_PER_DAY, endpoint=False))) * 30
    activity = np.tile(one_day, 5)
    assert abs(interdaily_stability(activity, EPOCHS_PER_DAY) - 1.0) < 1e-6


def test_cosinor_flat_signal_is_degenerate():
    flat = np.full(3 * EPOCHS_PER_DAY, 5.0)
    result = cosinor(flat, EPOCH_S)
    assert result["cosinor_amplitude"] < 1e-6


def test_intradaily_variability_needs_two_points():
    assert np.isnan(intradaily_variability(np.array([1.0])))


def test_enmo_adapter_separates_rest_and_activity(rng):
    fs, seconds = 10, 3600
    start = pd.Timestamp("2026-03-01 00:00:00", tz="UTC")
    count = fs * seconds
    t_rest = start + pd.to_timedelta(np.arange(count) / fs, unit="s")
    rest = pd.DataFrame({
        "timestamp": t_rest.astype("int64") // 1_000_000,
        "x": rng.normal(0, 0.02, count),
        "y": rng.normal(0, 0.02, count),
        "z": 1 + rng.normal(0, 0.02, count),
    })
    t_active = t_rest + pd.Timedelta(hours=1)
    active = pd.DataFrame({
        "timestamp": t_active.astype("int64") // 1_000_000,
        "x": rng.normal(0, 0.5, count),
        "y": rng.normal(0, 0.5, count),
        "z": 1 + rng.normal(0, 0.5, count),
    })
    raw = pd.concat([rest, active], ignore_index=True)
    epochs = compute_enmo_epochs(raw, EPOCH_S, "UTC", gravity=1.0)
    assert bool(np.all(epochs.dropna() >= 0))
    assert 118 <= int(epochs.notna().sum()) <= 121
    rest_mean = float(np.nanmean(epochs.iloc[:60]))
    active_mean = float(np.nanmean(epochs.iloc[60:120]))
    assert active_mean > 5 * rest_mean


def test_run_end_to_end(tmp_path, rng):
    fs, days = 2, 4
    start = pd.Timestamp("2026-03-01 00:00:00", tz="America/New_York").tz_convert("UTC")
    count = fs * 86400 * days
    times = start + pd.to_timedelta(np.arange(count) / fs, unit="s")
    local_hour = (
        times.tz_convert("America/New_York").hour + times.tz_convert("America/New_York").minute / 60
    ).to_numpy()
    amp = 0.05 + 0.6 * np.clip(np.cos(2 * np.pi * (local_hour - 14) / 24), 0, None)
    frame = pd.DataFrame({
        "timestamp": times.astype("int64") // 1_000_000,
        "x": rng.normal(0, amp),
        "y": rng.normal(0, amp),
        "z": 1 + rng.normal(0, amp),
    })
    acc_dir = tmp_path / "study" / "participantA" / "accelerometer"
    acc_dir.mkdir(parents=True)
    per_day = fs * 86400
    for day in range(days):
        chunk = frame.iloc[day * per_day:(day + 1) * per_day]
        chunk.to_csv(acc_dir / f"2026-03-0{day + 1} 00_00_00.csv", index=False)
    
    out_dir = tmp_path / "out"
    summary = run(
        str(tmp_path / "study"),
        str(out_dir),
        "America/New_York",
        frequency=Frequency.HOURLY_AND_DAILY,
        epoch_seconds=60,
    )
    assert (out_dir / "rar_summary.csv").exists()
    assert (out_dir / "participantA_rar_daily.csv").exists()
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["backend_id"] == "participantA"
    assert abs(row["cosinor_acrophase_h"] - 14) < 0.5
    assert row["IS"] > 0.8
    daily = pd.read_csv(out_dir / "participantA_rar_daily.csv")
    assert len(daily) == days
    assert "IS" not in daily.columns  # IS is recording-level only


def _write_participant(acc_dir, start_local, n_days, fs, rng):
    """Write n_days of synthetic accelerometer CSVs for one participant."""
    acc_dir.mkdir(parents=True)
    start = pd.Timestamp(start_local, tz="America/New_York").tz_convert("UTC")
    count = fs * 86400 * n_days
    times = start + pd.to_timedelta(np.arange(count) / fs, unit="s")
    local_hour = (
        times.tz_convert("America/New_York").hour + times.tz_convert("America/New_York").minute / 60
    ).to_numpy()
    amp = 0.05 + 0.6 * np.clip(np.cos(2 * np.pi * (local_hour - 14) / 24), 0, None)
    frame = pd.DataFrame({
        "timestamp": times.astype("int64") // 1_000_000,
        "x": rng.normal(0, amp),
        "y": rng.normal(0, amp),
        "z": 1 + rng.normal(0, amp),
    })
    per_day = fs * 86400
    for day in range(n_days):
        chunk = frame.iloc[day * per_day:(day + 1) * per_day]
        fname = f"2026-03-{day + 1:02d} 00_00_00.csv"
        chunk.to_csv(acc_dir / fname, index=False)


def test_run_skips_participants_below_min_valid_days(tmp_path, rng):
    """min_valid_days must exclude participants with too little data."""
    study = tmp_path / "study"
    _write_participant(study / "rich" / "accelerometer", "2026-03-01 00:00:00", 5, 2, rng)
    _write_participant(study / "sparse" / "accelerometer", "2026-03-01 00:00:00", 1, 2, rng)
    summary = run(
        str(study),
        str(tmp_path / "out"),
        "America/New_York",
        frequency=Frequency.DAILY,
        epoch_seconds=60,
        min_valid_days=3.0
    )
    ids = set(summary["backend_id"])
    assert "rich" in ids
    assert "sparse" not in ids


def test_run_time_bounds_clip_data(tmp_path, rng):
    """time_start/time_end restrict the window; clipping below min_valid_days then skips the
    participant."""
    study = tmp_path / "study"
    _write_participant(study / "rich" / "accelerometer", "2026-03-01 00:00:00", 5, 2, rng)
    unbounded = run(
        str(study), str(tmp_path / "out_all"), "America/New_York",
        frequency=Frequency.DAILY, epoch_seconds=60, min_valid_days=3.0,
    )
    assert "rich" in set(unbounded["backend_id"])
    bounded = run(
        str(study),
        str(tmp_path / "out_clip"),
        "America/New_York",
        frequency=Frequency.DAILY,
        epoch_seconds=60,
        min_valid_days=3.0,
        time_start="2026-03-01 00_00_00",
        time_end="2026-03-02 12_00_00",
    )
    bounded_ids = set(bounded["backend_id"]) if "backend_id" in bounded.columns else set()
    assert "rich" not in bounded_ids
