"""Ground-truth tests for forest.oak.circadian_rhythms (rest-activity rhythms).

Each test builds a synthetic signal with known RAR properties and checks that the estimators recover
them, plus an end-to-end ``run`` over a small synthetic study folder.

Original Author: Ceyhun Olcan
"""

import numpy as np
import pandas as pd
import pytest

from forest.constants import Frequency
from forest.oak.circadian_rhythms import (
    DAILY_COLUMNS,
    SUMMARY_COLUMNS,
    compute_average_day_profile,
    compute_daily_metrics,
    compute_enmo_intervals,
    compute_l5_m10,
    compute_rhythm_metrics,
    cosinor,
    count_valid_days,
    interdaily_stability,
    intradaily_variability,
    relative_amplitude,
    run,
)

INTERVAL_SECONDS = 60
INTERVALS_PER_DAY = 86_400 // INTERVAL_SECONDS


@pytest.fixture
def rng():
    return np.random.default_rng(20260709)


def _hours(n_days):
    return np.arange(n_days * INTERVALS_PER_DAY) * (INTERVAL_SECONDS / 3600.0)


def test_strong_rhythm_recovers_cosinor(rng):
    times = _hours(7)
    omega = 2 * np.pi / 24
    activity = 50 + 40 * np.cos(omega * (times - 14))
    activity = np.clip(activity + rng.normal(0, 2, times.size), 0, None)
    rows: list[dict[str, float | str]] = []
    compute_rhythm_metrics(rows, "test", activity, INTERVAL_SECONDS)
    metrics = rows[0]
    assert metrics["IS"] > 0.9
    assert metrics["IV"] < 0.05
    assert metrics["RA"] > 0.6
    assert abs(metrics["cosinor_mesor"] - 50) < 1.5
    assert abs(metrics["cosinor_amplitude"] - 40) < 1.5
    assert abs(metrics["cosinor_acrophase_h"] - 14) < 0.3
    assert 8 <= metrics["M10_onset_h"] <= 12


def test_white_noise_is_at_floor(rng):
    activity = np.clip(rng.normal(50, 15, 7 * INTERVALS_PER_DAY), 0, None)
    rows: list[dict[str, float | str]] = []
    compute_rhythm_metrics(rows, "test", activity, INTERVAL_SECONDS)
    metrics = rows[0]
    assert abs(metrics["IS"] - 1.0 / 7.0) < 0.05
    assert abs(metrics["IV"] - 2.0) < 0.15
    assert metrics["RA"] < 0.1
    assert metrics["cosinor_r2"] < 0.02


def test_identical_days_give_is_one():
    one_day = np.abs(np.sin(np.linspace(0, 2 * np.pi, INTERVALS_PER_DAY, endpoint=False))) * 30
    activity = np.tile(one_day, 5)
    assert abs(interdaily_stability(activity, INTERVALS_PER_DAY) - 1.0) < 1e-6


def test_cosinor_flat_signal_is_degenerate():
    flat = np.full(3 * INTERVALS_PER_DAY, 5.0)
    result = cosinor(flat, INTERVAL_SECONDS)
    assert result.amplitude < 1e-6


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
    interval_means = compute_enmo_intervals(raw, INTERVAL_SECONDS, "UTC", gravity=1.0)
    assert bool(np.all(interval_means.dropna() >= 0))
    assert 118 <= int(interval_means.notna().sum()) <= 121
    rest_mean = float(np.nanmean(interval_means.iloc[:60]))
    active_mean = float(np.nanmean(interval_means.iloc[60:120]))
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
        interval_seconds=60,
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
        interval_seconds=60,
        min_valid_days=3.0
    )
    ids = set(summary["backend_id"])
    assert "rich" in ids
    assert "sparse" not in ids


def test_run_time_bounds_clip_data(tmp_path, rng):
    """Clipping a recording below the valid-day threshold skips the participant."""
    study = tmp_path / "study"
    _write_participant(study / "rich" / "accelerometer", "2026-03-01 00:00:00", 5, 2, rng)
    unbounded = run(
        str(study), str(tmp_path / "out_all"), "America/New_York",
        frequency=Frequency.DAILY, interval_seconds=60, min_valid_days=3.0,
    )
    assert "rich" in set(unbounded["backend_id"])
    bounded = run(
        str(study),
        str(tmp_path / "out_clip"),
        "America/New_York",
        frequency=Frequency.DAILY,
        interval_seconds=60,
        min_valid_days=3.0,
        time_start="2026-03-01 00_00_00",
        time_end="2026-03-02 12_00_00",
    )
    bounded_ids = set(bounded["backend_id"]) if "backend_id" in bounded.columns else set()
    assert "rich" not in bounded_ids


def test_intradaily_variability_normalizes_over_valid_pairs():
    """IV uses the number of observed pairs, not all intervals.

    An alternating 0, 2 signal has squared differences of 4
    and variance of 1, giving IV = 4. Removing equal numbers
    of zeros and twos should leave that result unchanged."""
    base = np.array([0.0, 2.0] * 12)          # 24 intervals; even idx 0, odd idx 2
    assert intradaily_variability(base) == pytest.approx(4.0)

    gapped = base.copy()
    gapped[4] = np.nan                         # drops a 0
    gapped[9] = np.nan                         # drops a 2 -> mean unchanged
    assert intradaily_variability(gapped) == pytest.approx(4.0)


def test_average_day_profile_leaves_unobserved_positions_nan():
    """Unobserved clock positions remain NaN in the average-day profile."""
    quiet_then_active = np.concatenate([np.full(8, 1.0), np.full(16, 10.0)])
    gappy = quiet_then_active.copy()
    gappy[3] = np.nan

    profile = compute_average_day_profile(gappy, 24)
    assert np.isnan(profile[3])
    assert int(np.isnan(profile).sum()) == 1


def test_l5_m10_skips_windows_that_overlap_a_gap():
    """L5 and M10 use only fully observed windows.

    With hour 3 missing, the first complete five-hour window
    begins at hour 4. Four quiet hours and one active hour
    give an L5 mean of (1 + 1 + 1 + 1 + 10) / 5 = 2.8."""
    quiet_then_active = np.concatenate([np.full(8, 1.0), np.full(16, 10.0)])

    complete = compute_l5_m10(quiet_then_active, 24)
    assert complete.l5 == pytest.approx(1.0)
    assert complete.m10 == pytest.approx(10.0)
    assert complete.l5_onset_hours == pytest.approx(0.0)

    gappy = quiet_then_active.copy()
    gappy[3] = np.nan
    windows = compute_l5_m10(gappy, 24)
    assert windows.l5 == pytest.approx(2.8)
    assert windows.l5_onset_hours == pytest.approx(4.0)
    assert windows.m10 == pytest.approx(10.0)


def test_l5_m10_undefined_when_no_window_is_complete():
    """Every metric is NaN when the gaps leave no fully observed window."""
    quiet_then_active = np.concatenate([np.full(8, 1.0), np.full(16, 10.0)])
    scattered_gaps = quiet_then_active.copy()
    scattered_gaps[::5] = np.nan

    windows = compute_l5_m10(scattered_gaps, 24)
    assert np.isnan(windows.l5)
    assert np.isnan(windows.m10)
    assert np.isnan(windows.l5_onset_hours)
    assert np.isnan(windows.m10_onset_hours)


def test_count_valid_days_requires_each_day_to_qualify_on_its_own():
    """Each day must independently meet the coverage threshold.

    Six days with 12 observed hours each must not count as
    three fully covered days."""
    intervals_per_day = INTERVALS_PER_DAY

    six_half_days = np.full(6 * intervals_per_day, np.nan)
    for day in range(6):
        day_start = day * intervals_per_day
        six_half_days[day_start:day_start + intervals_per_day // 2] = 1.0

    # what the old check would have produced, for contrast
    assert np.sum(~np.isnan(six_half_days)) / intervals_per_day == pytest.approx(3.0)
    assert count_valid_days(six_half_days, intervals_per_day, 16.0) == 0

    three_full_days = np.full(3 * intervals_per_day, 1.0)
    assert count_valid_days(three_full_days, intervals_per_day, 16.0) == 3


def test_count_valid_days_is_inclusive_at_the_threshold():
    """A day with exactly the required hours counts; just under does not."""
    intervals_per_day = INTERVALS_PER_DAY
    minutes_in_sixteen_hours = 16 * 60

    exactly_sixteen = np.full(3 * intervals_per_day, np.nan)
    just_under = np.full(3 * intervals_per_day, np.nan)
    for day in range(3):
        day_start = day * intervals_per_day
        exactly_sixteen[day_start:day_start + minutes_in_sixteen_hours] = 1.0
        just_under[day_start:day_start + minutes_in_sixteen_hours - 1] = 1.0

    assert count_valid_days(exactly_sixteen, intervals_per_day, 16.0) == 3
    assert count_valid_days(just_under, intervals_per_day, 16.0) == 0


def test_summary_row_uses_summary_columns():
    """Checks that summary fields match the expected column order."""
    hours = _hours(3)
    activity = 1.0 + np.cos(2 * np.pi * hours / 24)

    rows: list[dict[str, float | str]] = []
    compute_rhythm_metrics(rows, "test", activity, INTERVAL_SECONDS)

    assert len(rows) == 1
    assert list(rows[0]) == SUMMARY_COLUMNS


def test_daily_file_keeps_its_columns_when_every_day_is_short():
    """Keeps all daily columns even when every day has no valid data."""
    values = np.full(2 * INTERVALS_PER_DAY, np.nan)
    start = pd.Timestamp("2026-03-01")

    daily = compute_daily_metrics(values, start, INTERVAL_SECONDS)

    assert list(daily.columns) == DAILY_COLUMNS
    assert len(daily) == 2
    assert daily["n_valid_epochs"].tolist() == [0, 0]
    assert daily["date"].tolist() == ["2026-03-01", "2026-03-02"]


def test_is_equals_share_of_pattern_days():
    """Checks IS = k / (k + j) for k patterned and j flat days."""
    hours = _hours(1)
    pattern = 2.0 + np.cos(2 * np.pi * hours / 24)
    flat = np.full(INTERVALS_PER_DAY, 2.0)

    three_days = np.concatenate([pattern, pattern, flat])
    four_days = np.concatenate([pattern, pattern, pattern, flat])

    assert interdaily_stability(
        three_days, INTERVALS_PER_DAY
    ) == pytest.approx(2 / 3)

    assert interdaily_stability(
        four_days, INTERVALS_PER_DAY
    ) == pytest.approx(3 / 4)


def test_iv_matches_cosine_formula():
    """Checks IV = 4 sin²(pi * step_hours / 24) for a daily cosine."""
    minute_hours = _hours(7)
    minute_activity = 5 + 2 * np.cos(2 * np.pi * minute_hours / 24)

    expected_minute = 4 * np.sin(np.pi * (1 / 60) / 24) ** 2

    assert intradaily_variability(minute_activity) == pytest.approx(
        expected_minute, rel=1e-3
    )

    hourly_hours = np.arange(100 * 24)
    hourly_activity = 5 + 2 * np.cos(2 * np.pi * hourly_hours / 24)

    expected_hourly = 4 * np.sin(np.pi / 24) ** 2

    assert intradaily_variability(hourly_activity) == pytest.approx(
        expected_hourly, rel=1e-3
    )


def test_l5_m10_match_cosine_formula():
    """Checks L5, M10, their onsets and RA against a daily cosine."""
    hours = _hours(3)
    mesor = 0.05
    amplitude = 0.04
    peak_hour = 14.0

    activity = mesor + amplitude * np.cos(
        2 * np.pi * (hours - peak_hour) / 24
    )

    windows = compute_l5_m10(activity, INTERVALS_PER_DAY)

    l5_angle = np.pi * 5 / 24
    m10_angle = np.pi * 10 / 24

    expected_l5 = mesor - amplitude * np.sin(l5_angle) / l5_angle
    expected_m10 = mesor + amplitude * np.sin(m10_angle) / m10_angle

    expected_l5_onset = (peak_hour - 12 - 5 / 2) % 24
    expected_m10_onset = (peak_hour - 10 / 2) % 24

    expected_ra = (
        (expected_m10 - expected_l5) / (expected_m10 + expected_l5)
    )

    assert windows.l5 == pytest.approx(expected_l5, abs=1e-5)
    assert windows.m10 == pytest.approx(expected_m10, abs=1e-5)

    assert windows.l5_onset_hours == pytest.approx(
        expected_l5_onset, abs=1 / 60
    )
    assert windows.m10_onset_hours == pytest.approx(
        expected_m10_onset, abs=1 / 60
    )

    assert relative_amplitude(
        windows.l5, windows.m10
    ) == pytest.approx(expected_ra, abs=1e-5)


def test_spring_forward_keeps_missing_hour():
    from forest.oak.circadian_rhythms import align_to_day_grid

    times = pd.to_datetime([
        "2026-03-08 06:59:00+00:00",
        "2026-03-08 07:00:00+00:00",
        "2026-03-08 07:01:00+00:00",
    ])

    raw = pd.DataFrame({
        "timestamp": times.as_unit("ms").astype("int64"),
        "x": [0.0, 0.0, 0.0],
        "y": [0.0, 0.0, 0.0],
        "z": [1.1, 1.2, 1.3],
    })

    activity = compute_enmo_intervals(
        raw, 60, "America/New_York"
    )
    values, start = align_to_day_grid(activity, 60)

    assert start == pd.Timestamp("2026-03-08")
    assert len(values) == 1440

    assert values[119] == pytest.approx(0.1)
    assert np.isnan(values[120:180]).all()
    assert values[180] == pytest.approx(0.2)
    assert values[181] == pytest.approx(0.3)


def test_fall_back_averages_repeated_hour():
    from forest.oak.circadian_rhythms import align_to_day_grid

    times = pd.to_datetime([
        "2026-11-01 05:30:00+00:00",
        "2026-11-01 06:30:00+00:00",
        "2026-11-01 07:00:00+00:00",
    ])

    raw = pd.DataFrame({
        "timestamp": times.as_unit("ms").astype("int64"),
        "x": [0.0, 0.0, 0.0],
        "y": [0.0, 0.0, 0.0],
        "z": [1.1, 1.3, 1.4],
    })

    activity = compute_enmo_intervals(
        raw, 60, "America/New_York"
    )
    values, start = align_to_day_grid(activity, 60)

    assert start == pd.Timestamp("2026-11-01")
    assert len(values) == 1440

    assert values[90] == pytest.approx(0.2)
    assert values[120] == pytest.approx(0.4)
    assert np.count_nonzero(~np.isnan(values)) == 2


def test_cosinor_phase_is_relative_to_recording_start():
    elapsed_hours = _hours(7)
    clock_hours = (elapsed_hours + 6) % 24

    activity = 0.05 + 0.04 * np.cos(
        2 * np.pi * (clock_hours - 14) / 24
    )

    result = cosinor(activity, INTERVAL_SECONDS)

    assert result.mesor == pytest.approx(0.05, abs=1e-6)
    assert result.amplitude == pytest.approx(0.04, abs=1e-6)
    assert result.acrophase_hours == pytest.approx(8.0, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


def test_run_handles_spring_forward_end_to_end(tmp_path):
    tz = "America/New_York"
    start = pd.Timestamp("2026-03-07", tz=tz).tz_convert("UTC")
    end = pd.Timestamp("2026-03-11", tz=tz).tz_convert("UTC")

    times = pd.date_range(start, end, freq="h", inclusive="left")
    local_hours = times.tz_convert(tz).hour.to_numpy()

    activity = 0.05 + 0.04 * np.cos(
        2 * np.pi * (local_hours - 14) / 24
    )

    raw = pd.DataFrame({
        "timestamp": times.as_unit("ms").astype("int64"),
        "x": np.zeros(len(times)),
        "y": np.zeros(len(times)),
        "z": 1.0 + activity,
    })

    acc_dir = tmp_path / "study" / "dst_user" / "accelerometer"
    acc_dir.mkdir(parents=True)
    raw.to_csv(acc_dir / "recording.csv", index=False)

    out_dir = tmp_path / "out"
    summary = run(
        str(tmp_path / "study"),
        str(out_dir),
        tz,
        frequency=Frequency.DAILY,
        interval_seconds=3600,
    )

    assert len(times) == 95
    assert len(summary) == 1
    assert summary.iloc[0]["backend_id"] == "dst_user"

    daily = pd.read_csv(out_dir / "dst_user_rar_daily.csv")

    assert daily["date"].tolist() == [
        "2026-03-07",
        "2026-03-08",
        "2026-03-09",
        "2026-03-10",
    ]

    assert daily["n_valid_epochs"].tolist() == [
        24, 23, 24, 24
    ]

    assert summary.iloc[0]["n_valid_epochs"] == 95
    assert summary.iloc[0]["cosinor_acrophase_h"] == pytest.approx(
        14.0, abs=0.1
    )
