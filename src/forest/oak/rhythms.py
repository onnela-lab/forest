"""Rest-activity rhythm (RAR) metrics for the accelerometer (Oak) tree.

Derives circadian rest-activity summaries from Beiwe accelerometer data: interdaily stability (IS),
intradaily variability (IV), relative amplitude (RA) with L5/M10, and a single-component 24 h
cosinor fit (MESOR, amplitude, acrophase, R^2).

These complement oak's gait/step outputs: the same raw accelerometer stream, a different (24 h
patterning) question. Activity is summarised per fixed epoch as ENMO -- the Euclidean norm of the
acceleration vector minus one gravitational unit, clipped at zero -- which is the standard
actigraphy proxy for movement intensity.

The public entry point ``run`` mirrors ``forest.oak.base.run``. It writes a recording-level
``rar_summary.csv`` (one row per participant) and, when a daily frequency is requested,
per-participant ``<backend_id>_rar_daily.csv`` files holding the day-decomposable metrics.

References:
    Van Someren EJW, et al. Chronobiol Int 16(4):505-518 (1999).
    Goncalves BSB, et al. Sleep Sci 7(3):158-164 (2014).
    Cornelissen G. Theor Biol Med Model 11:16 (2014).
"""
from __future__ import annotations

import logging
import os
from os.path import join as pathjoin

import numpy as np
import pandas as pd

from forest.constants import Frequency


logger = logging.getLogger(__name__)

ACCELEROMETER_DIR = "accelerometer"
SECONDS_PER_DAY = 86400
MIN_DAY_COVERAGE = 0.5  # fraction of a day required to emit a daily row


def _as_float_array(activity: np.ndarray) -> np.ndarray:
    """Return ``activity`` as a 1-D float array, erroring if empty."""
    arr = np.asarray(activity, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("activity is empty")
    return arr


def interdaily_stability(activity: np.ndarray, epochs_per_day: int) -> float:
    """Interdaily stability (IS), in (0, 1].

    ``IS = n * sum_h (m_h - m)^2 / (p * sum_i (x_i - m)^2)`` where ``p`` is ``epochs_per_day``,
    ``m_h`` the mean over all days at epoch-of-day ``h`` and ``m`` the grand mean. IS approaches 1
    for a perfectly reproducible daily pattern and ``p / n`` (i.e. ``1 / n_days``) for white noise.

    Args:
        activity: Per-epoch activity vector; NaNs are omitted.
        epochs_per_day: Number of epochs in a 24 h day.

    Returns:
        The IS value, or NaN if it is undefined.
    """
    arr = _as_float_array(activity)
    period = int(epochs_per_day)
    n_valid = int(np.sum(~np.isnan(arr)))
    if n_valid == 0:
        return float("nan")
    grand_mean = np.nanmean(arr)
    pos = np.arange(arr.size) % period
    hourly = np.array([
        np.nanmean(arr[pos == h]) if np.any(~np.isnan(arr[pos == h]))
        else np.nan
        for h in range(period)
    ])
    num = n_valid * np.nansum((hourly - grand_mean) ** 2)
    den = period * np.nansum((arr - grand_mean) ** 2)
    return float(num / den) if den > 0 else float("nan")


def intradaily_variability(activity: np.ndarray) -> float:
    """Intradaily variability (IV), typically in [0, ~2].

    ``IV = n * sum (x_i - x_{i-1})^2 / ((n - 1) * sum (x_i - m)^2)``.
    Higher values indicate a more fragmented rhythm; IV approaches ~2 for white noise and ~0 for a
    smooth sinusoid sampled densely.

    Args:
        activity: Per-epoch activity vector; NaNs are omitted.

    Returns:
        The IV value, or NaN if it is undefined.
    """
    arr = _as_float_array(activity)
    n_epochs = arr.size
    if n_epochs < 2:
        return float("nan")
    grand_mean = np.nanmean(arr)
    diffs = np.diff(arr)
    num = n_epochs * np.nansum(diffs ** 2)
    den = (n_epochs - 1) * np.nansum((arr - grand_mean) ** 2)
    return float(num / den) if den > 0 else float("nan")


def _average_day_profile(activity: np.ndarray, epochs_per_day: int) -> np.ndarray:
    """Mean activity at each epoch-of-day, empty positions mean-filled."""
    arr = _as_float_array(activity)
    period = int(epochs_per_day)
    pos = np.arange(arr.size) % period
    profile = np.array([
        np.nanmean(arr[pos == h]) if np.any(~np.isnan(arr[pos == h]))
        else np.nan
        for h in range(period)
    ])
    if np.any(np.isnan(profile)):
        profile = np.where(np.isnan(profile), np.nanmean(profile), profile)
    return profile


def _circular_window_mean(
    profile: np.ndarray, window: int
) -> np.ndarray:
    """Mean over a length-``window`` window at every circular start."""
    width = max(1, int(window))
    extended = np.concatenate([profile, profile[: width - 1]]) if width > 1 else profile
    kernel = np.ones(width) / width
    return np.convolve(extended, kernel, mode="valid")


def l5_m10(activity: np.ndarray, epochs_per_day: int) -> dict:
    """L5 and M10 on the average-day profile, with onset clock hours.

    Args:
        activity: Per-epoch activity vector.
        epochs_per_day: Number of epochs in a 24 h day.

    Returns:
        Dict with ``L5``, ``M10`` (mean activity levels) and their start
        clock hours ``L5_onset_h`` and ``M10_onset_h``.
    """
    period = int(epochs_per_day)
    epochs_per_hour = period / 24.0
    profile = _average_day_profile(activity, period)
    means_5 = _circular_window_mean(profile, round(5 * epochs_per_hour))
    means_10 = _circular_window_mean(profile, round(10 * epochs_per_hour))
    idx_5 = int(np.argmin(means_5))
    idx_10 = int(np.argmax(means_10))
    return {
        "L5": float(means_5[idx_5]),
        "M10": float(means_10[idx_10]),
        "L5_onset_h": float(idx_5 / epochs_per_hour),
        "M10_onset_h": float(idx_10 / epochs_per_hour),
    }


def relative_amplitude(l5_value: float, m10_value: float) -> float:
    """Relative amplitude ``(M10 - L5) / (M10 + L5)``, in [0, 1]."""
    denom = m10_value + l5_value
    if denom == 0:
        return float("nan")
    return float((m10_value - l5_value) / denom)


def cosinor(activity: np.ndarray, epoch_seconds: float, period_hours: float = 24.0) -> dict:
    """Single-component cosinor fit ``M + A cos(w t - phi)`` via OLS.

    Args:
        activity: Per-epoch activity vector; NaNs are omitted.
        epoch_seconds: Epoch length in seconds.
        period_hours: Rhythm period; defaults to 24 h.

    Returns:
        Dict with ``cosinor_mesor``, ``cosinor_amplitude``,
        ``cosinor_acrophase_h`` (clock hour of peak, in [0, period)) and
        ``cosinor_r2``.
    """
    nan_result = {
        "cosinor_mesor": float("nan"),
        "cosinor_amplitude": float("nan"),
        "cosinor_acrophase_h": float("nan"),
        "cosinor_r2": float("nan"),
    }
    arr = _as_float_array(activity)
    hours = np.arange(arr.size) * (float(epoch_seconds) / 3600.0)
    valid = ~np.isnan(arr)
    if int(valid.sum()) < 3:
        return nan_result
    times = hours[valid]
    values = arr[valid]
    omega = 2.0 * np.pi / float(period_hours)
    design = np.column_stack([
        np.ones_like(times), np.cos(omega * times), np.sin(omega * times)
    ])
    coef, _res, _rank, _sv = np.linalg.lstsq(design, values, rcond=None)
    mesor, beta, gamma = coef
    amplitude = float(np.hypot(beta, gamma))
    acrophase = float((np.arctan2(gamma, beta) / omega) % period_hours)
    resid = values - design @ coef
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "cosinor_mesor": float(mesor),
        "cosinor_amplitude": amplitude,
        "cosinor_acrophase_h": acrophase,
        "cosinor_r2": float(r_squared),
    }


def rar_metrics(activity: np.ndarray, epoch_seconds: float) -> dict:
    """Full recording-level RAR metric set.

    Args:
        activity: Per-epoch activity vector spanning whole days.
        epoch_seconds: Epoch length in seconds; ``86400 / epoch_seconds``
            must be a whole number.

    Returns:
        Dict of IS, IV, RA, L5/M10 (+ onsets), cosinor terms and counts.
    """
    epochs_per_day = SECONDS_PER_DAY / float(epoch_seconds)
    if not float(epochs_per_day).is_integer():
        raise ValueError(f"epoch_seconds={epoch_seconds} does not divide a 24 h day")
    epochs_per_day = int(epochs_per_day)
    arr = _as_float_array(activity)
    windows = l5_m10(arr, epochs_per_day)
    metrics = {
        "IS": interdaily_stability(arr, epochs_per_day),
        "IV": intradaily_variability(arr),
        "RA": relative_amplitude(windows["L5"], windows["M10"]),
        **windows,
        **cosinor(arr, epoch_seconds),
        "n_epochs": int(arr.size),
        "n_valid_epochs": int(np.sum(~np.isnan(arr))),
        "n_days": round(arr.size / epochs_per_day, 3),
    }
    return metrics


def _column_map(data_frame: pd.DataFrame) -> dict:
    """Map lower-cased, stripped column names to their originals."""
    return {c.strip().lower(): c for c in data_frame.columns}


def _parse_time_utc(data_frame: pd.DataFrame) -> pd.DatetimeIndex:
    """UTC timestamps from a Beiwe accelerometer frame."""
    columns = _column_map(data_frame)
    if "timestamp" in columns:
        series = pd.to_datetime(data_frame[columns["timestamp"]], unit="ms", utc=True)
    elif "utc time" in columns:
        series = pd.to_datetime(data_frame[columns["utc time"]], utc=True)
    else:
        raise ValueError("no 'timestamp' or 'utc time' column")
    return pd.DatetimeIndex(series)


def _read_accelerometer_file(filepath: str) -> pd.DataFrame:
    """Read one raw accelerometer CSV, validating required columns."""
    data_frame = pd.read_csv(filepath)
    columns = _column_map(data_frame)
    if not {"x", "y", "z"}.issubset(columns):
        raise ValueError(f"{filepath}: missing x/y/z columns")
    if "timestamp" not in columns and "utc time" not in columns:
        raise ValueError(f"{filepath}: missing timestamp column")
    data_frame.columns = [c.strip().lower() for c in data_frame.columns]
    return data_frame


def compute_enmo_epochs(
    data_frame: pd.DataFrame, epoch_seconds: int, tz_str: str, gravity: float = 1.0
) -> pd.Series:
    """Mean ENMO per fixed epoch on a gap-filled, tz-local grid.

    ENMO is ``max(|a| / gravity - 1, 0)`` per sample. Set ``gravity`` to the
    numeric value of 1 g in the data's units (1.0 for g, ~9.81 for m/s^2).

    Args:
        data_frame: Raw Beiwe accelerometer rows (``timestamp`` in ms UTC or
            a ``UTC time`` column, plus ``x``/``y``/``z``; case-insensitive).
        epoch_seconds: Epoch length in seconds.
        tz_str: IANA timezone for localisation, e.g. ``America/New_York``.
        gravity: Value of 1 g in the data's units.

    Returns:
        Activity Series indexed by tz-aware epoch-start timestamps, spanning
        the first to last observed epoch with missing epochs set to NaN.
    """
    if data_frame.empty:
        return pd.Series(dtype=float)
    columns = _column_map(data_frame)
    for axis in ("x", "y", "z"):
        if axis not in columns:
            raise ValueError(f"missing accelerometer axis '{axis}'")
    x = data_frame[columns["x"]].to_numpy(dtype=float)
    y = data_frame[columns["y"]].to_numpy(dtype=float)
    z = data_frame[columns["z"]].to_numpy(dtype=float)
    magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2) / float(gravity)
    enmo = np.clip(magnitude - 1.0, 0.0, None)
    series = pd.Series(enmo, index=_parse_time_utc(data_frame)).sort_index()
    series = series.tz_convert(tz_str)
    rule = f"{int(epoch_seconds)}s"
    epoched = series.resample(rule).mean()
    grid = pd.date_range(epoched.index.min(), epoched.index.max(), freq=rule, tz=tz_str)
    return epoched.reindex(grid)


def _parse_bound(value: str | None, tz_str: str) -> pd.Timestamp | None:
    """Parse a Beiwe ``%Y-%m-%d %H_%M_%S`` bound into a tz-aware stamp."""
    if value is None:
        return None
    naive = pd.to_datetime(value, format="%Y-%m-%d %H_%M_%S")
    return naive.tz_localize(tz_str)


def _align_to_day_grid(activity: pd.Series, epoch_seconds: int) -> tuple[np.ndarray, pd.Timestamp]:
    """Pad to local-midnight-aligned whole days for epoch-of-day folding."""
    epochs_per_day = SECONDS_PER_DAY // int(epoch_seconds)
    index = activity.index
    start = index[0].normalize()
    end = index[-1].normalize() + pd.Timedelta(days=1)
    rule = f"{int(epoch_seconds)}s"
    grid = pd.date_range(start, end, freq=rule, tz=index.tz, inclusive="left")
    values = activity.reindex(grid).to_numpy(dtype=float)
    usable = (values.size // epochs_per_day) * epochs_per_day
    return values[:usable], start


def _load_participant_activity(
    study_folder: str,
    backend_id: str,
    epoch_seconds: int,
    tz_str: str,
    gravity: float,
    time_start: str | None,
    time_end: str | None,
) -> pd.Series:
    """Read, concatenate and epoch one participant's accelerometer data."""
    acc_dir = pathjoin(study_folder, backend_id, ACCELEROMETER_DIR)
    if not os.path.isdir(acc_dir):
        return pd.Series(dtype=float)
    files = sorted(f for f in os.listdir(acc_dir) if f.endswith(".csv"))
    frames = []
    for filename in files:
        try:
            frames.append(_read_accelerometer_file(pathjoin(acc_dir, filename)))
        except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            logger.warning("skipping unreadable file %s", filename)
    if not frames:
        return pd.Series(dtype=float)
    raw = pd.concat(frames, ignore_index=True)
    activity = compute_enmo_epochs(raw, epoch_seconds, tz_str, gravity)
    lower = _parse_bound(time_start, tz_str)
    upper = _parse_bound(time_end, tz_str)
    if lower is not None:
        activity = activity[activity.index >= lower]
    if upper is not None:
        activity = activity[activity.index <= upper]
    return activity


def _daily_frame(values: np.ndarray, start: pd.Timestamp, epoch_seconds: int) -> pd.DataFrame:
    """Per-day day-decomposable metrics (IS is recording-level only)."""
    epochs_per_day = SECONDS_PER_DAY // int(epoch_seconds)
    n_days = values.size // epochs_per_day
    rows = []
    for day in range(n_days):
        chunk = values[day * epochs_per_day:(day + 1) * epochs_per_day]
        date = (start + pd.Timedelta(days=day)).date().isoformat()
        valid_fraction = np.sum(~np.isnan(chunk)) / epochs_per_day
        if valid_fraction < MIN_DAY_COVERAGE:
            rows.append({"date": date, "n_valid_epochs": int(np.sum(~np.isnan(chunk)))})
            continue
        windows = l5_m10(chunk, epochs_per_day)
        row = {
            "date": date,
            "IV": intradaily_variability(chunk),
            "RA": relative_amplitude(windows["L5"], windows["M10"]),
            **windows,
            **cosinor(chunk, epoch_seconds),
            "n_valid_epochs": int(np.sum(~np.isnan(chunk))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _discover_users(study_folder: str) -> list:
    """List backend IDs that have an accelerometer subfolder."""
    return sorted(
        entry for entry in os.listdir(study_folder)
        if os.path.isdir(pathjoin(study_folder, entry, ACCELEROMETER_DIR))
    )


def run(
    study_folder: str,
    output_folder: str,
    tz_str: str,
    frequency: Frequency = Frequency.DAILY,
    time_start: str | None = None,
    time_end: str | None = None,
    users: list | None = None,
    epoch_seconds: int = 60,
    gravity: float = 1.0,
    min_valid_days: float = 3.0,
) -> pd.DataFrame:
    """Compute rest-activity rhythm summaries over a Beiwe study folder.

    Args:
        study_folder: Path with one subfolder per backend ID, each holding
            an ``accelerometer`` directory of raw CSVs.
        output_folder: Directory for outputs (created if absent).
        tz_str: IANA timezone of the study site.
        frequency: Output resolution; ``Frequency.DAILY`` or
            ``Frequency.HOURLY_AND_DAILY`` additionally writes per-day files.
        time_start: Optional lower bound, ``%Y-%m-%d %H_%M_%S``.
        time_end: Optional upper bound, ``%Y-%m-%d %H_%M_%S``.
        users: Optional explicit backend IDs; otherwise all are used.
        epoch_seconds: Epoch length; must divide 86400 evenly.
        gravity: Value of 1 g in the data's units (1.0 = data already in g).
        min_valid_days: Participants with fewer valid days are skipped.

    Returns:
        The recording-level summary DataFrame (also written to disk).
    """
    os.makedirs(output_folder, exist_ok=True)
    if users is None:
        users = _discover_users(study_folder)
    epochs_per_day = SECONDS_PER_DAY // int(epoch_seconds)
    write_daily = frequency in (Frequency.DAILY, Frequency.HOURLY_AND_DAILY)
    rows = []
    for backend_id in users:
        activity = _load_participant_activity(
            study_folder, backend_id, epoch_seconds, tz_str, gravity, time_start, time_end
        )
        if activity.empty:
            continue
        values, start = _align_to_day_grid(activity, epoch_seconds)
        valid_days = np.sum(~np.isnan(values)) / epochs_per_day
        if valid_days < min_valid_days:
            logger.warning(
                "skipping %s: %.1f valid days (< %.1f)", backend_id, valid_days, min_valid_days
            )
            continue
        rows.append({"backend_id": backend_id, **rar_metrics(values, epoch_seconds)})
        if write_daily:
            daily = _daily_frame(values, start, epoch_seconds)
            daily.to_csv(pathjoin(output_folder, f"{backend_id}_rar_daily.csv"), index=False)
    summary = pd.DataFrame(rows)
    if summary.empty:
        # No participant met the coverage threshold; still return a frame with the identifier column
        # so callers can rely on its presence.
        summary = pd.DataFrame(columns=["backend_id"])
    summary.to_csv(pathjoin(output_folder, "rar_summary.csv"), index=False)
    return summary
