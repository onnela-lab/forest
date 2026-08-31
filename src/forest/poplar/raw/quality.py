"""Raw-data quality and coverage report for Beiwe study folders.

Sits alongside ``poplar.raw.doc`` (which *describes* raw streams) and
``poplar.raw.readers`` (which *reads* them): this module *validates* them.
For every participant and every data stream it summarises presence, volume,
temporal coverage, ordering/duplication issues, unreadable files, and a
schema check against the documented headers.

Stream knowledge is not hard-coded. The list of streams and their
per-platform availability comes from ``poplar.raw.doc.STREAMS``; expected
column headers come from ``poplar.raw.doc.HEADERS``. The report therefore
tracks whatever Beiwe adds and never drifts from the packaged documentation.

The public entry point ``run`` writes a single long-format
``data_quality.csv`` with one row per ``(backend_id, stream)``.

Note: this is a coverage/quality inventory, not statistical anomaly
detection on the signal values -- that is a natural follow-on.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from forest.constants import Frequency
from forest.poplar.raw.doc import HEADERS, STREAMS
from forest.utils import get_ids

logger = logging.getLogger(__name__)

IDENTIFIERS_STREAM = "identifiers"
MS_PER_HOUR = 3_600_000
NS_PER_MINUTE = 60_000_000_000

# Column order for the output report.
REPORT_COLUMNS = [
    "backend_id", "device_os", "stream", "stream_type", "expected",
    "present", "n_files", "size_mb", "n_rows", "first_observation",
    "last_observation", "n_days_spanned", "days_with_data", "coverage",
    "resolution", "largest_gap_h", "n_duplicate_rows", "n_out_of_order",
    "n_unreadable_files", "schema_status", "missing_columns",
    "extra_columns",
]

_READ_ERRORS = (
    pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, OSError,
)


def _bin_minutes(frequency: Frequency) -> int:
    """Coverage bin size in minutes; the two-resolution sentinel is daily."""
    if frequency == Frequency.HOURLY_AND_DAILY:
        return int(Frequency.DAILY.value)
    return int(frequency.value)


def _resolution_label(frequency: Frequency) -> str:
    """Human-readable name of the coverage resolution."""
    if frequency == Frequency.HOURLY_AND_DAILY:
        return "DAILY"
    return frequency.name


def _dir_size_mb(stream_dir: str, ndigits: int = 2) -> float:
    """Total size in megabytes of the files directly in ``stream_dir``."""
    total = 0
    for name in os.listdir(stream_dir):
        path = os.path.join(stream_dir, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return round(total / (2 ** 20), ndigits)


def _read_concat(stream_dir: str, files: list) -> tuple:
    """Concatenate readable CSVs, counting unreadable ones.

    Returns ``(data_frame_or_none, n_unreadable)``.
    """
    frames = []
    n_unreadable = 0
    for name in files:
        try:
            frames.append(pd.read_csv(os.path.join(stream_dir, name)))
        except _READ_ERRORS:
            n_unreadable += 1
            logger.warning("unreadable file %s", name)
    if not frames:
        return None, n_unreadable
    return pd.concat(frames, ignore_index=True), n_unreadable


def _device_os(study_folder: str, backend_id: str) -> str:
    """Read the identifiers stream for the participant's OS.

    Returns ``"Android"``, ``"iOS"`` or ``"unknown"``. If the participant
    re-enrolled, the most recent identifiers row wins.
    """
    id_dir = os.path.join(study_folder, backend_id, IDENTIFIERS_STREAM)
    if not os.path.isdir(id_dir):
        return "unknown"
    files = sorted(
        f for f in os.listdir(id_dir)
        if f.endswith((".csv", ".csv.zst"))
    )
    device_os = "unknown"
    for name in files:
        try:
            frame = pd.read_csv(os.path.join(id_dir, name))
        except _READ_ERRORS:
            continue
        columns = {c.strip().lower(): c for c in frame.columns}
        if "device_os" in columns and len(frame):
            value = str(frame[columns["device_os"]].iloc[-1]).strip()
            if value:
                device_os = value
    return device_os


def _is_expected(stream: str, device_os: str) -> bool:
    """Whether ``stream`` should exist given the participant's OS."""
    info = STREAMS.get(stream)
    if info is None:
        return False
    if device_os in ("Android", "iOS"):
        return bool(info[device_os])
    return bool(info["Android"] or info["iOS"])


def _expected_columns(stream: str, device_os: str):
    """Documented header for ``stream``: a column list, ``"To do"``, or None.

    When the OS is unknown, prefer a documented list from either platform.
    """
    entry = HEADERS.get(stream) if HEADERS is not None else None
    if entry is None:
        return None
    if device_os in ("Android", "iOS"):
        return entry.get(device_os)
    for candidate in (entry.get("iOS"), entry.get("Android")):
        if isinstance(candidate, list):
            return candidate
    return entry.get("iOS") if entry.get("iOS") is not None \
        else entry.get("Android")


def _schema_check(actual: list, expected) -> tuple:
    """Compare actual columns to the documented header.

    Returns ``(status, missing_columns, extra_columns)`` where status is
    ``"ok"``, ``"mismatch"``, ``"undocumented"`` (header is ``"To do"``) or
    ``"na"`` (no documented header).
    """
    if isinstance(expected, list):
        missing = [c for c in expected if c not in actual]
        extra = [c for c in actual if c not in expected]
        status = "ok" if not missing and not extra else "mismatch"
        return status, missing, extra
    if expected == "To do":
        return "undocumented", [], []
    return "na", [], []


def _parse_bound(value: str | None, tz_str: str) -> int | None:
    """Parse a ``%Y-%m-%d %H_%M_%S`` bound into a UTC millisecond stamp."""
    if value is None:
        return None
    naive = pd.to_datetime(value, format="%Y-%m-%d %H_%M_%S")
    localized = pd.Timestamp(naive).tz_localize(tz_str)
    return int(localized.value // 1_000_000)


def _apply_time_filter(
    data_frame: pd.DataFrame, tz_str: str,
    time_start: str | None, time_end: str | None,
) -> pd.DataFrame:
    """Restrict rows to the ``[time_start, time_end]`` window if given."""
    columns = {c.strip().lower(): c for c in data_frame.columns}
    if "timestamp" not in columns:
        return data_frame
    lower = _parse_bound(time_start, tz_str)
    upper = _parse_bound(time_end, tz_str)
    if lower is None and upper is None:
        return data_frame
    stamps = pd.to_numeric(data_frame[columns["timestamp"]],
                           errors="coerce")
    mask = pd.Series(True, index=data_frame.index)
    if lower is not None:
        mask &= stamps >= lower
    if upper is not None:
        mask &= stamps <= upper
    return data_frame[mask]


def _quality_metrics(
    data_frame: pd.DataFrame, tz_str: str, bin_minutes: int
) -> dict:
    """Row-level quality and coverage metrics keyed off ``timestamp``."""
    columns = {c.strip().lower(): c for c in data_frame.columns}
    n_rows = int(len(data_frame))
    metrics = {
        "n_rows": n_rows,
        "n_duplicate_rows": int(n_rows - len(data_frame.drop_duplicates())),
        "n_out_of_order": 0,
        "first_observation": "",
        "last_observation": "",
        "n_days_spanned": 0,
        "days_with_data": 0,
        "coverage": float("nan"),
        "largest_gap_h": float("nan"),
    }
    if "timestamp" not in columns:
        return metrics
    stamps = pd.to_numeric(data_frame[columns["timestamp"]],
                           errors="coerce").dropna()
    values = stamps.astype("int64").to_numpy()
    if values.size == 0:
        return metrics
    metrics["n_out_of_order"] = int(np.sum(np.diff(values) < 0))
    ordered = np.sort(values)
    if ordered.size > 1:
        metrics["largest_gap_h"] = round(
            int(np.max(np.diff(ordered))) / MS_PER_HOUR, 3
        )
    local = (
        pd.to_datetime(ordered, unit="ms", utc=True)
        .tz_convert(tz_str).tz_localize(None)
    )
    metrics["first_observation"] = local.min().isoformat()
    metrics["last_observation"] = local.max().isoformat()
    metrics["days_with_data"] = int(pd.Index(local.date).nunique())
    span = (local.max().normalize() - local.min().normalize()).days + 1
    metrics["n_days_spanned"] = int(span)
    bins = local.astype("int64") // (bin_minutes * NS_PER_MINUTE)
    observed = int(np.unique(bins).size)
    expected = int(bins.max() - bins.min()) + 1
    if expected > 0:
        metrics["coverage"] = round(observed / expected, 4)
    return metrics


def _blank_row(
    backend_id: str, device_os: str, stream: str, frequency: Frequency
) -> dict:
    """A report row for an absent or empty stream."""
    return {
        "backend_id": backend_id,
        "device_os": device_os,
        "stream": stream,
        "stream_type": STREAMS.get(stream, {}).get("type", ""),
        "expected": _is_expected(stream, device_os),
        "present": False,
        "n_files": 0,
        "size_mb": 0.0,
        "n_rows": 0,
        "first_observation": "",
        "last_observation": "",
        "n_days_spanned": 0,
        "days_with_data": 0,
        "coverage": float("nan"),
        "resolution": _resolution_label(frequency),
        "largest_gap_h": float("nan"),
        "n_duplicate_rows": 0,
        "n_out_of_order": 0,
        "n_unreadable_files": 0,
        "schema_status": "na",
        "missing_columns": "",
        "extra_columns": "",
    }


def _stream_row(
    study_folder: str, backend_id: str, device_os: str, stream: str,
    frequency: Frequency, tz_str: str,
    time_start: str | None, time_end: str | None,
) -> dict:
    """Build the full report row for one participant-stream pair."""
    row = _blank_row(backend_id, device_os, stream, frequency)
    stream_dir = os.path.join(study_folder, backend_id, stream)
    if not os.path.isdir(stream_dir):
        return row
    files = sorted(f for f in os.listdir(stream_dir)
                   if f.endswith((".csv", ".csv.zst")))
    if not files:
        return row
    row["present"] = True
    row["n_files"] = len(files)
    row["size_mb"] = _dir_size_mb(stream_dir)
    data_frame, n_unreadable = _read_concat(stream_dir, files)
    row["n_unreadable_files"] = n_unreadable
    if data_frame is None or data_frame.empty:
        return row
    data_frame = _apply_time_filter(
        data_frame, tz_str, time_start, time_end
    )
    if data_frame.empty:
        return row
    row.update(
        _quality_metrics(data_frame, tz_str, _bin_minutes(frequency))
    )
    status, missing, extra = _schema_check(
        list(data_frame.columns), _expected_columns(stream, device_os)
    )
    row["schema_status"] = status
    row["missing_columns"] = ";".join(missing)
    row["extra_columns"] = ";".join(extra)
    return row


def _streams_to_report(
    study_folder: str, backend_id: str, device_os: str
) -> list:
    """Union of streams expected for the OS and streams found on disk."""
    if device_os in ("Android", "iOS"):
        expected = {s for s, v in STREAMS.items() if v[device_os]}
    else:
        expected = {
            s for s, v in STREAMS.items() if v["Android"] or v["iOS"]
        }
    found: set = set()
    participant_dir = os.path.join(study_folder, backend_id)
    if os.path.isdir(participant_dir):
        found = {
            name for name in os.listdir(participant_dir)
            if name in STREAMS
            and os.path.isdir(os.path.join(participant_dir, name))
        }
    return sorted(expected | found)


def run(
    study_folder: str,
    output_folder: str,
    tz_str: str,
    frequency: Frequency = Frequency.DAILY,
    time_start: str | None = None,
    time_end: str | None = None,
    users: list | None = None,
) -> pd.DataFrame:
    """Build a raw-data quality report for a Beiwe study folder.

    Args:
        study_folder: Path with one subfolder per backend ID, each holding
            one subfolder per data stream of raw CSVs.
        output_folder: Directory for ``data_quality.csv`` (created if
            absent).
        tz_str: IANA timezone of the study site.
        frequency: Resolution of the coverage metric; ``Frequency.DAILY``
            gives day coverage, ``Frequency.HOURLY`` hour coverage, and so
            on. ``Frequency.HOURLY_AND_DAILY`` is treated as daily.
        time_start: Optional lower bound, ``%Y-%m-%d %H_%M_%S``.
        time_end: Optional upper bound, ``%Y-%m-%d %H_%M_%S``.
        users: Optional explicit backend IDs; otherwise all are used.

    Returns:
        The report DataFrame (also written to disk), one row per
        ``(backend_id, stream)``.
    """
    os.makedirs(output_folder, exist_ok=True)
    if users is None:
        users = get_ids(study_folder)
    rows = []
    for backend_id in users:
        device_os = _device_os(study_folder, backend_id)
        for stream in _streams_to_report(
            study_folder, backend_id, device_os
        ):
            rows.append(_stream_row(
                study_folder, backend_id, device_os, stream,
                frequency, tz_str, time_start, time_end,
            ))
    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    report.to_csv(
        os.path.join(output_folder, "data_quality.csv"), index=False
    )
    return report
