from datetime import datetime, timedelta
from dateutil import tz
import numpy as np
import pandas as pd
import pytest

from forest.oak.base import run_hourly, find_walking, preprocess_bout
from forest.constants import Frequency


@pytest.fixture(scope="module")
def test_input(signal_bout):
    timestamp, _, x, y, z = signal_bout

    t_bout_interp, vm_bout = preprocess_bout(timestamp, x, y, z)

    cadence_bout = find_walking(vm_bout)

    t_datetime = [datetime.fromtimestamp(t_ind) for t_ind in t_bout_interp]

    t_series = pd.Series(t_datetime)
    t_hours_pd = t_series.dt.floor("H")

    from_zone = tz.gettz("UTC")
    to_zone = tz.gettz("America/New_York")

    # convert t_hours to correct timezone
    t_hours_pd = t_hours_pd.dt.tz_localize(from_zone).dt.tz_convert(to_zone)

    date_start = datetime(2020, 2, 24, 0, 0, tzinfo=to_zone)
    date_end = datetime(2020, 2, 25, 0, 0, tzinfo=to_zone)
    days_hourly = pd.date_range(
        date_start, date_end + timedelta(days=1), freq="h"
    )[:-1]

    steps_hourly = np.full((len(days_hourly), 1), np.nan)
    cadence_hourly = np.full((len(days_hourly), 1), np.nan)
    walkingtime_hourly = np.full((len(days_hourly), 1), np.nan)

    return (
        t_hours_pd,
        days_hourly,
        cadence_bout,
        steps_hourly,
        walkingtime_hourly,
        cadence_hourly,
    )


def test_run_hourly_one_hour_data(test_input):
    run_hourly(*test_input, Frequency.HOURLY)

    steps_hourly, cadence_hourly, walkingtime_hourly = test_input[3:]

    assert len(steps_hourly) - np.sum(np.isnan(steps_hourly)) == 1
    assert len(cadence_hourly) - np.sum(np.isnan(cadence_hourly)) == 1
    assert len(walkingtime_hourly) - np.sum(np.isnan(walkingtime_hourly)) == 1


def test_run_hourly_accuracy(test_input):
    run_hourly(*test_input, Frequency.HOURLY)

    steps_hourly, cadence_hourly, walkingtime_hourly = test_input[3:]

    indice = np.where(~np.isnan(steps_hourly))[0]

    # get non-nan indices
    assert steps_hourly[indice][0] == 32
    assert cadence_hourly[indice][0] == 3.2
    assert walkingtime_hourly[indice][0] == 10
