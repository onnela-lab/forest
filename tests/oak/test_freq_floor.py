"""
Regression test for issue #308: deprecated pandas frequency aliases.

Pandas 2.2 deprecated 'H' (hour) and 'T' (minute) in favor of 'h' and 'min', and removed them in
pandas 3.0. The Series.dt.floor calls inside ``forest.oak.base.run`` must use the new aliases.

"""
from datetime import datetime
from pathlib import Path

import pandas as pd


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


# timedelta aliases deprecated in 2.2 and enforced in 3.0, applies to floor, ceil, and round
REMOVED_ALIASES = ("H", "T", "S", "L", "U", "N")
FLR_CEIL_RND = ("floor", "ceil", "round")
BAD_STRINGS = [f"{f}('{alias}')" for f in FLR_CEIL_RND for alias in REMOVED_ALIASES]
# and the " instead of a ' variants
BAD_STRINGS += [s.replace("'", '"') for s in BAD_STRINGS if s not in BAD_STRINGS]


def test_source_has_no_deprecated_freq_aliases():
    # this test was expanded from just H atd T in floor
    repo_root = Path(__file__).resolve().parents[2]
    candidate_files = [
        path for path in repo_root.rglob("*") if path.is_file()
        and path.suffix in (".py", ".md", ".rst", ".ipynb", )
    ]
    candidate_files.remove(Path(__file__))  # don't test this file
    
    # go through files looking for these retired aliases
    for path in candidate_files:
        text = path.read_text(errors="ignore")
        relative_path = path.relative_to(repo_root)
        
        for bad_string in BAD_STRINGS:
            assert bad_string not in text, f"Found deprecated alias {bad_string} in {relative_path}"
