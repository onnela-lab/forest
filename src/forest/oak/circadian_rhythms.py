"""Calculates rest-activity rhythms from Beiwe accelerometer data.

Uses Euclidean Norm Minus One (ENMO) activity to calculate interdaily
stability (IS), intradaily variability (IV), relative amplitude (RA),
the least active 5 hours (L5), the most active 10 hours (M10), and a
24-hour cosinor fit.

Unlike Oak's gait analysis, this module looks at daily activity
patterns rather than walking.

Results are saved in rar_summary.csv, with optional daily files named
<backend_id>_rar_daily.csv.

Activity uses local wall-clock time. Missing spring hours remain NaN,
while repeated fall hours are averaged into the same clock intervals.

References:
    Van Someren EJW, et al. Chronobiol Int 16(4):505-518 (1999).
    Gonçalves BSB, et al. Sleep Sci 7(3):158-164 (2014).
    Cornelissen G. Theor Biol Med Model 11:16 (2014).

Original author: Ceyhun Olcan.

AI assistance:
    OpenAI ChatGPT (GPT-6) and Anthropic Claude
    (Opus 5 and Opus 5.5) assisted with
    documentation and review.

    The author reviewed and verified the implementation
    and takes responsibility for the contributed work.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from os.path import isdir, join as pathjoin

import numpy as np
import pandas as pd

from forest.constants import Frequency, SECONDS_IN_DAY
from forest.utils import get_ids


logger = logging.getLogger(__name__)

ACCELEROMETER_DIR = "accelerometer"

# Daily metrics need at least half the data available.
# This is a reporting cutoff, separate from run()'s valid-day requirement.
MIN_DAY_COVERAGE = 0.5

# Hours used for the least and most active windows.
L5_WINDOW_HOURS = 5
M10_WINDOW_HOURS = 10

# Keep the CSV column order consistent across output files.
SUMMARY_COLUMNS: list[str] = [
    "backend_id", "IS", "IV", "RA", "L5", "M10", "L5_onset_h", "M10_onset_h",
    "cosinor_mesor", "cosinor_amplitude", "cosinor_acrophase_h", "cosinor_r2",
    "n_epochs", "n_valid_epochs", "n_days",
]

DAILY_COLUMNS: list[str] = [
    "date", "IV", "RA", "L5", "M10", "L5_onset_h", "M10_onset_h",
    "cosinor_mesor", "cosinor_amplitude", "cosinor_acrophase_h", "cosinor_r2",
    "n_valid_epochs",
]


def as_float_array(values: np.ndarray) -> np.ndarray:
    """Turns the input into a flat array of floats."""
    arr = np.asarray(values, dtype=float).ravel()

    if arr.size == 0:
        raise ValueError("Array is empty")

    return arr


def interdaily_stability(values: np.ndarray, intervals_per_day: int) -> float:
    """Measures how consistent activity patterns are across days.

    Returns NaN if it can't be calculated.
    """
    arr = as_float_array(values)
    period = int(intervals_per_day)

    count = np.sum(~np.isnan(arr))
    if count == 0:
        return float("nan")

    mean = np.nanmean(arr)

    average_day = compute_average_day_profile(arr, period)

    numerator = count * np.nansum((average_day - mean) ** 2)
    denominator = period * np.nansum((arr - mean) ** 2)

    if denominator == 0:
        return float("nan")

    return float(numerator / denominator)


def intradaily_variability(values: np.ndarray) -> float:
    """Measures activity fragmentation across consecutive intervals.

    Compares successive changes with overall activity variance.
    Returns NaN if there are no valid pairs or no activity variance.
    """
    arr = as_float_array(values)

    if arr.size < 2:
        return float("nan")

    observed = ~np.isnan(arr)
    valid_pairs = observed[1:] & observed[:-1]

    pair_count = np.count_nonzero(valid_pairs)
    observed_count = np.count_nonzero(observed)

    if pair_count == 0 or observed_count < 2:
        return float("nan")

    mean = np.nanmean(arr)
    differences = np.diff(arr)

    mean_squared_difference = np.nansum(differences ** 2) / pair_count
    variance = np.nansum((arr - mean) ** 2) / observed_count

    if variance == 0:
        return float("nan")

    return float(mean_squared_difference / variance)


def compute_average_day_profile(values: np.ndarray, intervals_per_day: int) -> np.ndarray:
    """Calculates the average activity at each position across days.

    Positions with no observed values stay as NaN.
    """
    arr = as_float_array(values)
    period = int(intervals_per_day)

    positions = np.arange(arr.size) % period
    position_means: list[float] = []

    for h in range(period):
        same_time = arr[positions == h]

        if np.any(~np.isnan(same_time)):
            position_means.append(float(np.nanmean(same_time)))
        else:
            position_means.append(np.nan)

    return np.asarray(position_means, dtype=float)


def compute_circular_window_mean(profile: np.ndarray, window: int) -> np.ndarray:
    """Averages values from each starting position, wrapping around the end of the day.

    Returns NaN if a window contains missing values.
    """
    arr = as_float_array(profile)
    window_size = max(1, int(window))

    offsets = np.arange(window_size)
    window_means: list[float] = []

    for start in range(arr.size):
        positions = (start + offsets) % arr.size
        values = arr[positions]

        window_means.append(float(np.mean(values)))

    return np.asarray(window_means, dtype=float)


@dataclass
class ActivityWindows:
    """Stores the least and most active periods of the average day.

    Attributes:
        l5: Average activity during the least active 5 hours.
        m10: Average activity during the most active 10 hours.
        l5_onset_hours: Clock hour when the least active period starts.
            For example, 23.5 means 23:30.
        m10_onset_hours: Clock hour when the most active period starts.
            For example, 8.5 means 08:30.
    """
    l5: float
    m10: float
    l5_onset_hours: float
    m10_onset_hours: float


def compute_l5_m10(values: np.ndarray, intervals_per_day: int) -> ActivityWindows:
    """Finds the least active 5 hours (L5) and most active 10 hours (M10).

    Uses the average day and returns NaN if the results can't be calculated.
    """
    period = int(intervals_per_day)
    intervals_per_hour = period / 24

    average_day = compute_average_day_profile(values, period)

    l5_window = round(L5_WINDOW_HOURS * intervals_per_hour)
    m10_window = round(M10_WINDOW_HOURS * intervals_per_hour)

    l5_means = compute_circular_window_mean(average_day, l5_window)
    m10_means = compute_circular_window_mean(average_day, m10_window)

    if np.all(np.isnan(l5_means)) or np.all(np.isnan(m10_means)):
        missing = float("nan")
        return ActivityWindows(
            l5=missing,
            m10=missing,
            l5_onset_hours=missing,
            m10_onset_hours=missing,
        )

    l5_start = int(np.nanargmin(l5_means))
    m10_start = int(np.nanargmax(m10_means))

    return ActivityWindows(
        l5=float(l5_means[l5_start]),
        m10=float(m10_means[m10_start]),
        l5_onset_hours=float(l5_start / intervals_per_hour),
        m10_onset_hours=float(m10_start / intervals_per_hour),
    )


def relative_amplitude(l5_value: float, m10_value: float) -> float:
    """Measures the contrast between the most and least active periods.

    The score ranges from 0 to 1 for nonnegative activity and returns
    NaN when the total is zero or a value is missing.
    """
    total = m10_value + l5_value

    if total == 0:
        return float("nan")

    return float((m10_value - l5_value) / total)


@dataclass
class CosinorFit:
    """Stores the results of fitting a wave to daily activity patterns.

    Attributes:
        mesor: Average level of the fitted wave.
        amplitude: How far the wave rises above its average level.
        acrophase_hours: Time in hours from the start of the series when
            the wave peaks. This matches local clock time when the series
            starts at local midnight, as run() ensures.
        r_squared: How much of the activity variation the fitted wave
            explains, from 0 to 1.
    """
    mesor: float
    amplitude: float
    acrophase_hours: float
    r_squared: float


def cosinor(
    values: np.ndarray,
    interval_seconds: float,
    period_hours: float = 24.0,
) -> CosinorFit:
    """Fits a cosine wave to activity data and measures how well it matches.

    Peak time is measured from the first value. It matches local clock time
    when run() aligns the series to local midnight.
    Returns all NaN values if fewer than three observations are available.
    """
    missing = float("nan")
    empty_fit = CosinorFit(
        mesor=missing,
        amplitude=missing,
        acrophase_hours=missing,
        r_squared=missing,
    )

    arr = as_float_array(values)
    hours = np.arange(arr.size) * interval_seconds / 3600

    observed = ~np.isnan(arr)
    observed_values = arr[observed]
    observed_hours = hours[observed]

    if observed_values.size < 3:
        return empty_fit

    omega = 2 * np.pi / period_hours

    matrix = np.column_stack([
        np.ones_like(observed_hours),
        np.cos(omega * observed_hours),
        np.sin(omega * observed_hours),
    ])

    coefficients, _, _, _ = np.linalg.lstsq(
        matrix, observed_values, rcond=None
    )

    mesor, beta, gamma = coefficients

    amplitude = np.hypot(beta, gamma)
    peak_time = (np.arctan2(gamma, beta) / omega) % period_hours

    fitted_values = matrix @ coefficients

    residual_sum = np.sum((observed_values - fitted_values) ** 2)
    total_sum = np.sum((observed_values - np.mean(observed_values)) ** 2)

    r_squared = float("nan") if total_sum == 0 else float(1 - residual_sum / total_sum)
    return CosinorFit(
        mesor=float(mesor),
        amplitude=float(amplitude),
        acrophase_hours=float(peak_time),
        r_squared=r_squared,
    )


def compute_rhythm_metrics(
    rows: list[dict[str, float | str]],
    backend_id: str,
    activity: np.ndarray,
    interval_seconds: float,
) -> None:
    """Calculates rhythm metrics and adds one summary row to the results.

    Args:
        rows: List where the new results will be added.
        backend_id: Identifier for the participant.
        activity: Activity values covering whole days, starting at local midnight.
        interval_seconds: Length of each activity interval in seconds.

    Raises:
        ValueError: If the interval length does not divide a day evenly.
    """
    intervals_per_day = SECONDS_IN_DAY / interval_seconds

    if not intervals_per_day.is_integer():
        raise ValueError(
            f"interval_seconds={interval_seconds} does not divide "
            "a 24-hour day evenly"
        )

    period = int(intervals_per_day)
    arr = as_float_array(activity)

    windows = compute_l5_m10(arr, period)
    fit = cosinor(arr, interval_seconds)

    rows.append({
        "backend_id": backend_id,
        "IS": interdaily_stability(arr, period),
        "IV": intradaily_variability(arr),
        "RA": relative_amplitude(windows.l5, windows.m10),
        "L5": windows.l5,
        "M10": windows.m10,
        "L5_onset_h": windows.l5_onset_hours,
        "M10_onset_h": windows.m10_onset_hours,
        "cosinor_mesor": fit.mesor,
        "cosinor_amplitude": fit.amplitude,
        "cosinor_acrophase_h": fit.acrophase_hours,
        "cosinor_r2": fit.r_squared,
        "n_epochs": int(arr.size),
        "n_valid_epochs": int(np.count_nonzero(~np.isnan(arr))),
        "n_days": round(arr.size / period, 3),
    })


def find_axis_columns(data_frame: pd.DataFrame) -> dict[str, str]:
    """Maps cleaned-up column names to the original ones.

    This makes it easier to find columns with different capitalization
    or extra spaces.
    """
    return {
        column.strip().lower(): column
        for column in data_frame.columns
    }


def build_utc_index(data_frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Converts the time column into a UTC timeline.

    Uses the numeric timestamp first because it is more precise than
    parsing text. Raises ValueError if no time column is found.
    """
    columns = find_axis_columns(data_frame)

    if "timestamp" in columns:
        times = pd.to_datetime(
            data_frame[columns["timestamp"]], unit="ms", utc=True
        )
    elif "utc time" in columns:
        times = pd.to_datetime(
            data_frame[columns["utc time"]], utc=True
        )
    else:
        raise ValueError("No timestamp or UTC time column found")

    return pd.DatetimeIndex(times)


def read_accelerometer_file(filepath: str) -> pd.DataFrame:
    """Reads an accelerometer CSV file and checks its required columns.

    Column names are stripped of extra spaces and changed to lowercase.
    Raises ValueError if the file is missing axis or time columns.
    """
    data_frame = pd.read_csv(filepath)
    columns = find_axis_columns(data_frame)

    if not {"x", "y", "z"}.issubset(columns):
        raise ValueError(f"Missing accelerometer axis columns in {filepath}")

    if "timestamp" not in columns and "utc time" not in columns:
        raise ValueError(f"Missing timestamp or UTC time column in {filepath}")

    data_frame.columns = [
        column.strip().lower() for column in data_frame.columns
    ]

    return data_frame


def compute_enmo_intervals(
    data_frame: pd.DataFrame,
    interval_seconds: int,
    tz_str: str,
    gravity: float = 1.0,
) -> pd.Series:
    """Calculates average ENMO for each local clock-time interval.

    ENMO is Euclidean Norm Minus One: acceleration magnitude divided by
    gravity, minus one, with negative values set to zero.

    Uses local wall-clock times so days keep the same clock positions.
    Missing spring hours stay NaN, and repeated fall hours are averaged.

    Args:
        data_frame: Accelerometer data with x, y, and z values.
        interval_seconds: Length of each interval in seconds.
        tz_str: Study's local timezone.
        gravity: Use 1.0 for data in g or about 9.81 for data in m/s².

    Returns:
        Average ENMO indexed by local times without timezone information.
    """
    if data_frame.empty:
        return pd.Series(dtype=float)

    columns = find_axis_columns(data_frame)

    if not {"x", "y", "z"}.issubset(columns):
        raise ValueError("Missing accelerometer axis columns")

    x = data_frame[columns["x"]].to_numpy(dtype=float)
    y = data_frame[columns["y"]].to_numpy(dtype=float)
    z = data_frame[columns["z"]].to_numpy(dtype=float)

    magnitude = np.sqrt(x**2 + y**2 + z**2)
    enmo = np.clip(magnitude / gravity - 1, 0, None)

    times = build_utc_index(data_frame)
    activity = pd.Series(enmo, index=times).sort_index()

    local_activity = activity.tz_convert(tz_str).tz_localize(None)

    return local_activity.resample(f"{int(interval_seconds)}s").mean()


def parse_time_bound(value: str | None, tz_str: str) -> pd.Timestamp | None:
    """Converts a time string into a timestamp in the study's local timezone.

    Expects Forest's format, like "2018-01-01 00_00_00".
    Returns None if no time is provided.
    """
    if value is None:
        return None

    timestamp = pd.to_datetime(value, format="%Y-%m-%d %H_%M_%S")

    return timestamp.tz_localize(tz_str)


def align_to_day_grid(
    activity: pd.Series,
    interval_seconds: int,
) -> tuple[np.ndarray, pd.Timestamp]:
    """Pads activity into complete days starting at local midnight.

    Wall-clock time keeps each day the same length.
    Raises ValueError if the activity data is empty.
    """
    if activity.empty:
        raise ValueError("Activity data is empty")

    times = pd.DatetimeIndex(activity.index)

    start = times[0].normalize()
    end = times[-1].normalize() + pd.Timedelta(days=1)

    grid = pd.date_range(
        start,
        end,
        freq=f"{int(interval_seconds)}s",
        inclusive="left",
    )

    aligned = activity.reindex(grid)

    return aligned.to_numpy(dtype=float), start


def find_clock_change_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz_str: str,
) -> list[str]:
    """Finds dates when clock changes make a day shorter or longer.

    Accepts local wall-clock or timezone-aware timestamps.
    Returns ISO date strings for reporting daylight saving changes.
    """
    start_date = start.tz_convert(tz_str).date() if start.tzinfo else start.date()
    end_date = end.tz_convert(tz_str).date() if end.tzinfo else end.date()

    dates = pd.date_range(
        start_date,
        pd.Timestamp(end_date) + pd.Timedelta(days=1),
        freq="D",
    )

    midnights = dates.tz_localize(tz_str)
    day_lengths = midnights[1:] - midnights[:-1]

    changed_dates: list[str] = []

    for day, length in zip(dates[:-1], day_lengths):
        if length != pd.Timedelta(hours=24):
            changed_dates.append(day.date().isoformat())

    return changed_dates


def load_participant_activity(
    study_folder: str,
    backend_id: str,
    interval_seconds: int,
    tz_str: str,
    gravity: float,
    time_start: str | None,
    time_end: str | None,
) -> pd.Series:
    """Loads a participant's accelerometer data in local wall-clock time.

    Skips unreadable files and applies optional time bounds.
    Warns when daylight saving changes affect the recording.

    Args:
        study_folder: Folder containing the study data.
        backend_id: Identifier for the participant.
        interval_seconds: Length of each activity interval in seconds.
        tz_str: Study's local timezone.
        gravity: Use 1.0 for data in g or about 9.81 for data in m/s².
        time_start: Optional start time in Forest's date format.
        time_end: Optional end time in Forest's date format.

    Returns:
        Activity values indexed by local wall-clock times.
    """
    folder = pathjoin(study_folder, backend_id, ACCELEROMETER_DIR)

    if not isdir(folder):
        return pd.Series(dtype=float)

    frames: list[pd.DataFrame] = []

    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(".csv"):
            continue

        filepath = pathjoin(folder, filename)

        try:
            frame = read_accelerometer_file(filepath)
        except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            logger.warning("skipping %s: %s", filepath, error)
            continue

        frames.append(frame)

    if not frames:
        return pd.Series(dtype=float)

    combined = pd.concat(frames, ignore_index=True)
    activity = compute_enmo_intervals(combined, interval_seconds, tz_str, gravity)

    start_bound = parse_time_bound(time_start, tz_str)
    end_bound = parse_time_bound(time_end, tz_str)

    if start_bound is not None:
        activity = activity[activity.index >= start_bound.tz_localize(None)]

    if end_bound is not None:
        activity = activity[activity.index <= end_bound.tz_localize(None)]

    if not activity.empty:
        changed_dates = find_clock_change_dates(
            pd.Timestamp(activity.index[0]),
            pd.Timestamp(activity.index[-1]),
            tz_str,
        )

        if changed_dates:
            logger.warning(
                "Participant %s has clock changes on %s. "
                "Missing spring hours remain NaN and repeated fall hours are averaged.",
                backend_id,
                ", ".join(changed_dates),
            )

    return activity


def compute_daily_metrics(
    values: np.ndarray,
    start: pd.Timestamp,
    interval_seconds: int,
) -> pd.DataFrame:
    """Calculates activity metrics for each day of the recording.

    Days with less than half their data observed keep only their date
    and valid interval count.

    Args:
        values: Activity values starting at local midnight.
        start: Starting date and time of the recording.
        interval_seconds: Length of each interval in seconds.

    Returns:
        A DataFrame containing the daily activity metrics.
    """
    arr = np.asarray(values, dtype=float).ravel()
    period = int(SECONDS_IN_DAY / interval_seconds)
    whole_days = arr.size // period

    rows: list[dict[str, float | str]] = []

    for day in range(whole_days):
        daily_values = arr[day * period:(day + 1) * period]
        date = (start + pd.DateOffset(days=day)).date().isoformat()

        observed_count = int(np.count_nonzero(~np.isnan(daily_values)))
        coverage = observed_count / period

        if coverage < MIN_DAY_COVERAGE:
            rows.append({
                "date": date,
                "n_valid_epochs": observed_count,
            })
            continue

        windows = compute_l5_m10(daily_values, period)
        fit = cosinor(daily_values, float(interval_seconds))

        rows.append({
            "date": date,
            "IV": intradaily_variability(daily_values),
            "RA": relative_amplitude(windows.l5, windows.m10),
            "L5": windows.l5,
            "M10": windows.m10,
            "L5_onset_h": windows.l5_onset_hours,
            "M10_onset_h": windows.m10_onset_hours,
            "cosinor_mesor": fit.mesor,
            "cosinor_amplitude": fit.amplitude,
            "cosinor_acrophase_h": fit.acrophase_hours,
            "cosinor_r2": fit.r_squared,
            "n_valid_epochs": observed_count,
        })


    return pd.DataFrame(rows, columns=DAILY_COLUMNS)


def get_ids_with_accelerometer_data(study_folder: str) -> list[str]:
    """Finds participants who have accelerometer data.

    Uses get_ids to skip hidden and registry folders, then checks
    which participants have an accelerometer folder.
    """
    participant_ids = get_ids(study_folder)

    ids_with_data = [
        backend_id
        for backend_id in participant_ids
        if isdir(pathjoin(study_folder, backend_id, ACCELEROMETER_DIR))
    ]

    return sorted(ids_with_data)


def count_valid_days(
    values: np.ndarray,
    intervals_per_day: int,
    min_valid_hours_per_day: float,
) -> int:
    """Counts how many days have enough observed activity data.

    Each day is checked separately, so six half-covered days
    aren't combined to count as full days.
    """
    arr = np.asarray(values, dtype=float).ravel()
    period = int(intervals_per_day)

    whole_days = arr.size // period

    if whole_days == 0:
        return 0

    daily_values = arr[:whole_days * period].reshape(whole_days, period)

    observed_per_day = np.count_nonzero(~np.isnan(daily_values), axis=1)
    observed_hours = observed_per_day * 24 / period

    valid_days = np.count_nonzero(observed_hours >= min_valid_hours_per_day)

    return int(valid_days)


def run(
    study_folder: str,
    output_folder: str,
    tz_str: str,
    frequency: Frequency = Frequency.DAILY,
    time_start: str | None = None,
    time_end: str | None = None,
    users: list[str] | None = None,
    interval_seconds: int = 60,
    gravity: float = 1.0,
    min_valid_days: float = 3.0,
    min_valid_hours_per_day: float = 16.0,
) -> pd.DataFrame:
    """Processes accelerometer data and saves rhythm metrics.

    Writes a summary file and optional daily files for each participant.

    Args:
        study_folder: Folder containing the study data.
        output_folder: Folder where results are saved.
        tz_str: Study's local timezone.
        frequency: DAILY and HOURLY_AND_DAILY write daily files.
            Neither option calculates hourly metrics.
        time_start: Optional start time in Forest's YYYY-MM-DD HH_MM_SS format.
        time_end: Optional end time in the same format.
        users: Participant IDs to process, or None to find them automatically.
        interval_seconds: Length of each interval, which must divide a day.
        gravity: Use 1.0 for data in g or about 9.81 for data in m/s².
        min_valid_days: Minimum number of valid days required.
        min_valid_hours_per_day: Minimum observed hours per day, default 16.
            This follows a common wear-time convention and can be adjusted.

    Returns:
        A DataFrame containing the participant-level rhythm metrics.

    Raises:
        ValueError: If the interval length is invalid.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    intervals_per_day = SECONDS_IN_DAY / interval_seconds

    if not intervals_per_day.is_integer():
        raise ValueError(
            f"interval_seconds={interval_seconds} does not divide "
            "a 24-hour day evenly"
        )

    os.makedirs(output_folder, exist_ok=True)

    period = int(intervals_per_day)

    if users is None:
        users = get_ids_with_accelerometer_data(study_folder)

    save_daily = frequency in (Frequency.DAILY, Frequency.HOURLY_AND_DAILY)

    rows: list[dict[str, float | str]] = []

    for backend_id in users:
        activity = load_participant_activity(
            study_folder,
            backend_id,
            interval_seconds,
            tz_str,
            gravity,
            time_start,
            time_end,
        )

        if activity.empty:
            logger.warning("skipping %s: no readable activity data", backend_id)
            continue

        values, start = align_to_day_grid(activity, interval_seconds)

        valid_days = count_valid_days(
            values, period, min_valid_hours_per_day
        )

        if valid_days < min_valid_days:
            logger.warning(
                "skipping %s: %d valid day(s), need at least %g",
                backend_id,
                valid_days,
                min_valid_days,
            )
            continue

        compute_rhythm_metrics(
            rows, backend_id, values, float(interval_seconds)
        )

        if save_daily:
            daily = compute_daily_metrics(values, start, interval_seconds)
            filepath = pathjoin(output_folder, f"{backend_id}_rar_daily.csv")
            daily.to_csv(filepath, index=False)


    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(pathjoin(output_folder, "rar_summary.csv"), index=False)

    return summary
