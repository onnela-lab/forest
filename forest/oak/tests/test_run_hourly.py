import numpy as np
import pandas as pd
import pytest

from forest.oak.base import run_hourly
from forest.constants import Frequency


@pytest.fixture()
def sample_run_input(signal_bout):
    t_hours_pd = pd.Series(pd.to_datetime([
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00",
        "2020-02-25 08:00:00-05:00"
    ], utc=True).tz_convert('US/Eastern'))
    days_hourly = pd.date_range(
        start='2020-02-24 00:00:00',
        end='2020-02-25 23:00:00',
        freq='H',
        tz='US/Eastern'
    )
    cadence_bout = np.array(
        [1.65, 1.6, 1.55, 1.6, 1.55, 1.85, 1.8, 1.75, 1.75, 1.7]
    )
    steps_hourly = np.full((48, 1), np.nan)
    cadence_hourly = np.full((48, 1), np.nan)
    walkingtime_hourly = np.full((48, 1), np.nan)

    return (
        t_hours_pd,
        days_hourly,
        cadence_bout,
        steps_hourly,
        walkingtime_hourly,
        cadence_hourly,
    )


def test_run_hourly_one_hour_data(sample_run_input):
    run_hourly(*sample_run_input, Frequency.HOURLY)
    steps_hourly, cadence_hourly, walkingtime_hourly = sample_run_input[3:]

    assert len(steps_hourly) - np.sum(np.isnan(steps_hourly)) == 1
    assert len(cadence_hourly) - np.sum(np.isnan(cadence_hourly)) == 1
    assert len(walkingtime_hourly) - np.sum(np.isnan(walkingtime_hourly)) == 1


def test_run_hourly_accuracy(sample_run_input):
    run_hourly(*sample_run_input, Frequency.HOURLY)
    steps_hourly, cadence_hourly, walkingtime_hourly = sample_run_input[3:]
    index = np.where(~np.isnan(steps_hourly))[0]
    # get non-nan indices
    assert steps_hourly[index][0] == 16
    assert cadence_hourly[index][0] == 10
    assert walkingtime_hourly[index][0] == 1.6
