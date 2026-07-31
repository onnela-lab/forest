"""Tests for forest.poplar.raw.quality (raw-data quality report).

Builds a synthetic study folder that deliberately contains a clean stream,
a stream with a schema mismatch plus a gap, duplicate and out-of-order
rows plus an unreadable file, and an expected-but-absent stream.
"""
import numpy as np
import pandas as pd
import pytest

from forest.constants import Frequency
from forest.poplar.raw.quality import (
    REPORT_COLUMNS,
    _bin_minutes,
    _schema_check,
    run,
)

TZ = "America/New_York"


def _ms(timestamp):
    return int(pd.Timestamp(timestamp).value // 1_000_000)


def _write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    pid = "participantA"

    _write(
        root / pid / "identifiers" / "2026-03-01 00_00_00.csv",
        pd.DataFrame({
            "timestamp": [_ms("2026-03-01T00:00:00Z")],
            "UTC time": ["2026-03-01T00:00:00.000"],
            "patient_id": [pid],
            "device_os": ["iOS"],
        }),
    )

    for day in ("2026-03-01", "2026-03-02"):
        base = pd.Timestamp(day + "T00:00:00Z")
        stamps = [base + pd.Timedelta(hours=h) for h in range(24)]
        _write(
            root / pid / "accelerometer" / f"{day} 00_00_00.csv",
            pd.DataFrame({
                "timestamp": [_ms(t) for t in stamps],
                "UTC time": [t.isoformat() for t in stamps],
                "accuracy": ["unknown"] * 24,
                "x": np.zeros(24),
                "y": np.zeros(24),
                "z": np.ones(24),
            }),
        )

    rows = []
    day0 = pd.Timestamp("2026-03-01T00:00:00Z")
    for h in range(24):
        t = day0 + pd.Timedelta(hours=h)
        rows.append([_ms(t), t.isoformat(), 43.0 + 0.001 * h, -74.0, 12.0])
    day3 = pd.Timestamp("2026-03-04T00:00:00Z")  # two-day gap
    for h in range(24):
        t = day3 + pd.Timedelta(hours=h)
        rows.append([_ms(t), t.isoformat(), 43.5, -74.1, 15.0])
    rows.append(list(rows[0]))  # exact duplicate
    early = day0 + pd.Timedelta(hours=5)
    rows.append([_ms(early), early.isoformat(), 43.9, -74.2, 20.0])  # OOO
    gps = pd.DataFrame(
        rows,
        columns=["timestamp", "UTC time", "latitude", "longitude",
                 "accuracy"],  # 'altitude' deliberately missing
    )
    _write(root / pid / "gps" / "2026-03-01 00_00_00.csv", gps)
    (root / pid / "gps" / "empty.csv").write_text("")  # unreadable

    return root


def test_run_report(study, tmp_path):
    out = tmp_path / "out"
    report = run(str(study), str(out), TZ, frequency=Frequency.DAILY)

    assert (out / "data_quality.csv").exists()
    assert list(report.columns) == REPORT_COLUMNS
    assert (report["device_os"] == "iOS").all()

    acc = report[report["stream"] == "accelerometer"].iloc[0]
    assert acc["present"]
    assert acc["expected"]
    assert acc["schema_status"] == "ok"
    assert acc["n_rows"] == 48
    assert acc["n_unreadable_files"] == 0
    assert acc["coverage"] == 1.0

    gps = report[report["stream"] == "gps"].iloc[0]
    assert gps["schema_status"] == "mismatch"
    assert "altitude" in gps["missing_columns"]
    assert gps["n_duplicate_rows"] >= 1
    assert gps["n_out_of_order"] >= 1
    assert gps["largest_gap_h"] > 24
    assert gps["n_unreadable_files"] >= 1

    power = report[report["stream"] == "power_state"].iloc[0]
    assert power["expected"]
    assert not power["present"]


def test_schema_check_states():
    assert _schema_check(["a", "b"], ["a", "b", "c"]) == (
        "mismatch", ["c"], [])
    assert _schema_check(["a", "b"], ["a", "b"]) == ("ok", [], [])
    assert _schema_check(["a"], "To do") == ("undocumented", [], [])
    assert _schema_check(["a"], None) == ("na", [], [])


def test_bin_minutes():
    assert _bin_minutes(Frequency.DAILY) == 1440
    assert _bin_minutes(Frequency.HOURLY) == 60
    assert _bin_minutes(Frequency.HOURLY_AND_DAILY) == 1440


def test_run_discovers_zst_files(tmp_path):
    """Compressed .csv.zst streams must be discovered and read, matching
    the sycamore fix in #320. Without the extension fix, discovery filters
    to .csv only and the stream is reported absent."""
    root = tmp_path / "study"
    pid = "participantZ"

    _write(
        root / pid / "identifiers" / "2026-03-01 00_00_00.csv",
        pd.DataFrame({
            "timestamp": [_ms("2026-03-01T00:00:00Z")],
            "UTC time": ["2026-03-01T00:00:00.000"],
            "patient_id": [pid],
            "device_os": ["Android"],
        }),
    )

    base = pd.Timestamp("2026-03-01T00:00:00Z")
    stamps = [base + pd.Timedelta(hours=h) for h in range(24)]
    frame = pd.DataFrame({
        "timestamp": [_ms(t) for t in stamps],
        "UTC time": [t.isoformat() for t in stamps],
        "accuracy": ["unknown"] * 24,
        "x": np.zeros(24),
        "y": np.zeros(24),
        "z": np.ones(24),
    })
    # Write the accelerometer stream as a single compressed .csv.zst file.
    acc_dir = root / pid / "accelerometer"
    acc_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(acc_dir / "2026-03-01 00_00_00.csv.zst", index=False)

    out = tmp_path / "out"
    report = run(str(root), str(out), TZ, frequency=Frequency.DAILY)

    acc = report[report["stream"] == "accelerometer"].iloc[0]
    assert acc["present"]
    assert acc["n_files"] == 1
    assert acc["n_rows"] == 24
    assert acc["n_unreadable_files"] == 0
