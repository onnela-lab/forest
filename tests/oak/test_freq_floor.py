"""Regression test for issue #308: deprecated pandas frequency aliases.

Pandas 2.2 deprecated 'H' (hour) and 'T' (minute) in favor of 'h' and 'min',
and removed them in pandas 3.0. The Series.dt.floor calls inside
``forest.oak.base.run`` must use the new aliases.
"""
from datetime import datetime
import inspect

import pandas as pd

from forest.oak import base as oak_base


def _exercise_floor_calls():
    """Mirror the dt.floor invocations used in oak_base.run."""
    t_datetime = [datetime(2020, 2, 25, 8, 30, 15)]
    t_series = pd.Series(t_datetime)
    minute = t_series.dt.floor('min')
    hour = t_series.dt.floor('h')
    return minute.iloc[0], hour.iloc[0]


def test_floor_uses_pandas3_compatible_aliases():
    minute, hour = _exercise_floor_calls()
    assert minute == pd.Timestamp("2020-02-25 08:30:00")
    assert hour == pd.Timestamp("2020-02-25 08:00:00")


def test_oak_run_source_has_no_deprecated_freq_aliases():
    src = inspect.getsource(oak_base.run)
    assert "floor('H')" not in src
    assert 'floor("H")' not in src
    assert "floor('T')" not in src
    assert 'floor("T")' not in src
