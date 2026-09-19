# Data Quality

## Executive Summary:
Use `poplar.raw.quality` to build a raw-data quality and coverage report
across every Beiwe data stream. For each participant and stream it
summarises presence, volume, temporal coverage, ordering and duplication
issues, unreadable files, and a schema check against the documented
headers. This sits beside `poplar.raw.doc` (which *describes* streams) and
`poplar.raw.readers` (which *reads* them): it *validates* them.

Stream knowledge is not hard-coded: the streams and their per-platform
availability come from `poplar.raw.doc.STREAMS`, and the expected column
headers from `poplar.raw.doc.HEADERS`, so the report tracks whatever Beiwe
adds.

## Installation Instruction
For instructions on how to install forest, please visit
[here](https://github.com/onnela-lab/forest).
`from forest.poplar.raw import quality`

## Usage:
```
from forest.poplar.raw.quality import run
from forest.constants import Frequency


# Determine study folder and output folder
study_folder = "project/data"
output_folder = "project/results"

# Determine study timezone and time frames for data analysis
tz_str = "America/New_York"
time_start = "2018-01-01 00_00_00"
time_end = "2022-01-01 00_00_00"

# Resolution of the coverage metric. Frequency.DAILY gives day coverage,
# Frequency.HOURLY hour coverage, etc. Frequency.HOURLY_AND_DAILY is treated
# as daily. See forest.constants.Frequency:
# https://github.com/onnela-lab/forest/blob/develop/forest/constants.py
frequency = Frequency.DAILY
users = None

run(study_folder, output_folder, tz_str, frequency,
    time_start, time_end, users)
```

## Coverage and schema

Coverage is the fraction of `frequency`-sized bins that contain at least one
observation, over the span from the first to the last observation. Every
stream shares a `timestamp` column (milliseconds since the Unix epoch, UTC),
which is used for all temporal metrics; timestamps are localised to `tz_str`
before day/hour binning.

The schema check compares each stream's actual columns to the documented
header for the participant's device OS (read from the `identifiers` stream).
It reports one of four states: `ok`, `mismatch` (with the missing and extra
columns listed), `undocumented` (the header is not yet documented), or `na`
(no documented header for that stream/OS).

## List of summary statistics

The output is written to `data_quality.csv` with one row per participant per
data stream. The following variables are created:

| Variable | Type | Description of Variable |
|---|---|---|
| backend_id | str | Beiwe backend identifier of the participant |
| device_os | str | Device OS from the identifiers stream (Android, iOS, or unknown) |
| stream | str | Name of the data stream |
| stream_type | str | passive or survey |
| expected | bool | Whether the stream is available for the participant's OS |
| present | bool | Whether the stream folder exists with at least one CSV |
| n_files | int | Number of CSV files in the stream folder |
| size_mb | float | Total size of the stream folder, in megabytes |
| n_rows | int | Total number of rows across all files |
| first_observation | str | Earliest observation, local time (ISO 8601) |
| last_observation | str | Latest observation, local time (ISO 8601) |
| n_days_spanned | int | Number of calendar days from first to last observation |
| days_with_data | int | Number of distinct local dates with at least one observation |
| coverage | float | Fraction of frequency-sized bins containing data |
| resolution | str | Name of the coverage resolution (e.g. DAILY) |
| largest_gap_h | float | Largest gap between consecutive observations, in hours |
| n_duplicate_rows | int | Number of exactly duplicated rows |
| n_out_of_order | int | Number of rows whose timestamp is earlier than the previous row |
| n_unreadable_files | int | Number of CSV files that could not be read |
| schema_status | str | ok, mismatch, undocumented, or na |
| missing_columns | str | Documented columns absent from the data (semicolon-separated) |
| extra_columns | str | Columns present in the data but not documented (semicolon-separated) |
