import pandas as pd
import numpy as np

from forest.constants import Frequency
from forest.willow.log_stats import (
    comm_logs_summaries,
    get_call_reciprocity,
    get_mean_responsiveness,
)

STAMP_START = 1453837206
STAMP_END = 1454634000
TZ_STR = "America/New_York"
OPTION = Frequency.DAILY


def test_comm_log_summaries_with_empty_data():
    text_data = pd.DataFrame.from_dict({})
    call_data = pd.DataFrame.from_dict({})
    stats_pdframe = comm_logs_summaries(
        text_data, call_data, STAMP_START, STAMP_END, TZ_STR, OPTION
    )
    assert isinstance(stats_pdframe, pd.DataFrame)


def test_comm_log_summaries_with_empty_data_hourly():
    text_data = pd.DataFrame.from_dict({})
    call_data = pd.DataFrame.from_dict({})
    stats_pdframe = comm_logs_summaries(
        text_data, call_data, STAMP_START, STAMP_END, TZ_STR, Frequency.HOURLY
    )
    assert isinstance(stats_pdframe, pd.DataFrame)


def test_comm_log_summaries_with_empty_text_data():
    text_data = pd.DataFrame.from_dict({})
    call_data = pd.DataFrame.from_dict(
        {
            "timestamp": {0: 1454428647649},
            "UTC time": {0: "2016-02-02T15:57:27.649"},
            "hashed phone number": {
                0: "ZlGtb-SRRIgOcHLBD02d2_F049naF0YZbCx_CeP7jss="
            },
            "call type": {0: "Missed Call"},
            "duration in seconds": {0: 0},
        }
    )
    stats_pdframe = comm_logs_summaries(
        text_data, call_data, STAMP_START, STAMP_END, TZ_STR, OPTION
    )
    assert isinstance(stats_pdframe, pd.DataFrame)


def test_get_call_reciprocity():
    full_reciprocity = get_call_reciprocity(
        {
            "incoming": ["a", "a", "b", "c"],
            "outgoing": ["a", "a", "b", "c"],
        }
    )
    assert np.abs(full_reciprocity - 1) < 1e-10

    half_reciprocity = get_call_reciprocity(
        {
            "incoming": ["a", "a", "a"],
            "outgoing": ["a"],
        }
    )

    assert np.abs(half_reciprocity - 0.5) < 1e-10

    no_reciprocity = get_call_reciprocity(
        {
            "incoming": ["a", "a", "b", "c"],
            "outgoing": [],
        }
    )

    assert np.abs(no_reciprocity - 0) < 1e-10


def test_no_data_returns_na():
    df = pd.DataFrame(
        columns=["hashed phone number", "timestamp", "direction"]
    )
    result = get_mean_responsiveness(df, "direction", ["received"], ["sent"])
    assert pd.isna(result)


def test_no_sent_messages_returns_na():
    df = pd.DataFrame(
        {
            "hashed phone number": ["A", "A"],
            "timestamp": [1000, 2000],
            "direction": ["received", "received"],
        }
    )
    result = get_mean_responsiveness(df, "direction", ["received"], ["sent"])
    assert pd.isna(result)


def test_no_received_messages_returns_na():
    df = pd.DataFrame(
        {
            "hashed phone number": ["A", "A"],
            "timestamp": [1000, 2000],
            "direction": ["sent", "sent"],
        }
    )
    result = get_mean_responsiveness(df, "direction", ["received"], ["sent"])
    assert pd.isna(result)


def test_simple_case():
    # One received and one sent for same number
    df = pd.DataFrame(
        {
            "hashed phone number": ["A", "A"],
            "timestamp": [1000, 160000],  # 1000 ms and 160000 ms
            "direction": ["received", "sent"],
        }
    )
    # Expected diff: (160000 - 1000) = 159000 ms = 2.65 minutes
    result = get_mean_responsiveness(df, "direction", ["received"], ["sent"])
    assert np.isclose(result, 2.65, atol=0.01)


def test_multiple_received_and_sent():
    # Received at 1000 and 2000 ms; sent at 2500 and 3000 ms
    df = pd.DataFrame(
        {
            "hashed phone number": ["A", "A", "A", "A"],
            "timestamp": [1000, 3000, 2000, 4000],
            "direction": ["received", "received", "sent", "sent"],
        }
    )

    result = get_mean_responsiveness(df, "direction", ["received"], ["sent"])
    # Each sent message is 1000 ms after a received message, so the result
    # should be 1 second
    assert np.isclose(result, 1 / 60, atol=0.001)


def test_multiple_phone_numbers():
    df = pd.DataFrame(
        {
            "hashed phone number": ["A", "A", "B", "B"],
            "timestamp": [1000, 2000, 1000, 3000],
            "direction": ["received", "sent", "received", "sent"],
        }
    )
    # For A: diff = (2000 - 1000) = 1000 ms = 1/60 min
    # For B: diff = (3000 - 1000) = 2000 ms = 2/60 min
    expected_mean = (1 / 60 + 2 / 60) / 2
    result = get_mean_responsiveness(df, "direction", ["received"], ["sent"])
    assert np.isclose(result, expected_mean, atol=0.001)


def test_sent_before_received_filtered_out():
    df = pd.DataFrame(
        {
            "hashed phone number": ["A", "A"],
            "timestamp": [2000, 1000],
            "direction": ["received", "sent"],
        }
    )
    # sent timestamp is before received timestamp, should be filtered out,
    # no pairs left
    result = get_mean_responsiveness(df, "direction", ["received"], ["sent"])
    assert pd.isna(result)
