# Rest-Activity Rhythms

## Overview
Use `oak.circadian_rhythms` to compute circadian rest-activity rhythm (RAR) summary
statistics from Beiwe accelerometer data. This complements the gait/step
output of `oak` (see [oak](oak.md)): the same raw accelerometer stream,
summarised as 24-hour activity patterning rather than walking.

## Usage
```python
from forest.oak.circadian_rhythms import run
from forest.constants import Frequency


# Determine study folder and output folder
study_folder = "project/data"
output_folder = "project/results"

# Determine study timezone and time frames for data analysis
tz_str = "America/New_York"
time_start = "2018-01-01 00_00_00"
time_end = "2022-01-01 00_00_00"

# Write the recording-level summary and a daily file for each participant.
frequency = Frequency.DAILY
users = None

# Call the main function
run(study_folder, output_folder, tz_str, frequency,
    time_start, time_end, users)
```
`Frequency.DAILY` is the default. `Frequency.HOURLY_AND_DAILY` is accepted
as an alias for these same two outputs; it does not produce hourly metrics.
Other `Frequency` values write only the recording-level summary. This module
uses the shared Forest enum, whose names describe other analyses too.

### Default tuning parameters

```python
interval_seconds = 60           # Seconds per interval; must divide 86,400
gravity = 1.0                   # Use ~9.81 for acceleration in m/s²
min_valid_days = 3              # Minimum number of qualifying days
min_valid_hours_per_day = 16.0  # Observed hours required per day
```

Each day must independently meet the 16-hour threshold. For example, six
days with 12 hours of data each do not count as three valid days.

These are configurable coverage thresholds, not validated smartphone
wear-time criteria. Studies with different recording protocols may need
different thresholds and sensitivity checks.

Daily output uses a separate 12-hour cutoff (`MIN_DAY_COVERAGE = 0.5`).
Days below this cutoff still appear in the daily file, but their metric
values are left blank.

Once a participant meets the inclusion requirement, recording-level
metrics use all available intervals, including observations from days
that did not individually reach 16 hours.

## Activity metric

Per sample, movement intensity is summarised as ENMO (the Euclidean norm of
the acceleration vector minus one gravitational unit, clipped at zero):
`ENMO = max(sqrt(x^2 + y^2 + z^2) / gravity - 1, 0)`. ENMO is averaged into
fixed intervals on a timezone-local grid, with missing intervals held as `NaN`.
The result is expressed in gravitational units after subtracting gravity:
`0.02` means an average excess magnitude of 0.02 g. It is a movement proxy;
it does not identify sleep, device wear, or physical activity categories.

An interval is observed if it has at least one valid sample. The coverage
count therefore measures availability of intervals, not continuous sampling
or verified wear time. A sparsely sampled minute counts the same as a densely
sampled minute. Inspect raw sample counts and gaps before using these summaries.

### Platform checks

The module does not infer operating system or acceleration units from the CSV.
Before pooling iOS and Android recordings, check units and resting magnitude,
sample rates, within-interval sample counts, missing intervals by clock time,
and ENMO distributions separately for each OS, app version, and OS version.
Use `gravity=1.0` for data in g and approximately `9.81` for data in m/s²;
process groups with different units separately. Synthetic unit-scaling tests
verify the arithmetic, but real iOS/Android equivalence has not been validated
by this PR. A phone left on a desk can produce an apparently regular rest
period that says little about the participant's activity.

## Output

### Recording-level summary

The recording-level output is written to `rar_summary.csv` with one row per
participant. The following variables are created:

| Variable | Type | Description of Variable |
| --- | --- | --- |
| backend_id | str | Beiwe backend identifier of the participant |
| IS | float | Interdaily stability, in (0, 1]; higher means a more reproducible daily pattern |
| IV | float | Intradaily variability; higher values indicate more rapid activity changes |
| RA | float | Relative amplitude, (M10 - L5) / (M10 + L5) |
| L5 | float | Mean activity over the least-active 5 hours of the average day |
| M10 | float | Mean activity over the most-active 10 hours of the average day |
| L5_onset_h | float | Clock hour at which the L5 window starts |
| M10_onset_h | float | Clock hour at which the M10 window starts |
| cosinor_mesor | float | MESOR (rhythm-adjusted mean) of the 24 h cosinor fit |
| cosinor_amplitude | float | Amplitude of the 24 h cosinor fit |
| cosinor_acrophase_h | float | Clock hour of the fitted peak, in [0, 24) |
| cosinor_r2 | float | Coefficient of determination of the cosinor fit |
| n_epochs | int | Number of intervals on the day-aligned grid |
| n_valid_epochs | int | Number of non-missing intervals |
| n_days | float | Number of days spanned by the recording |

### Daily summary

When `frequency` is `Frequency.DAILY` or `Frequency.HOURLY_AND_DAILY`, a
per-participant `<backend_id>_rar_daily.csv` is also written with the
day-decomposable metrics (IS is recording-level only and is omitted):

| Variable | Type | Description of Variable |
| --- | --- | --- |
| date | str | Day of observation (yyyy-mm-dd) |
| IV | float | Intradaily variability for the day |
| RA | float | Relative amplitude for the day |
| L5 | float | Least-active 5-hour mean for the day |
| M10 | float | Most-active 10-hour mean for the day |
| L5_onset_h | float | Clock hour at which the L5 window starts |
| M10_onset_h | float | Clock hour at which the M10 window starts |
| cosinor_mesor | float | MESOR of the day's cosinor fit |
| cosinor_amplitude | float | Amplitude of the day's cosinor fit |
| cosinor_acrophase_h | float | Clock hour of the day's fitted peak |
| cosinor_r2 | float | Coefficient of determination of the day's cosinor fit |
| n_valid_epochs | int | Number of non-missing intervals in the day |

### Worked output example

This example is synthetic, with three identical, completely observed days at
one-minute intervals. Activity follows `0.05 + 0.04*cos(2*pi*(hour-14)/24)`.
The following excerpt from `rar_summary.csv` is rounded for display:

```text
backend_id,IS,IV,RA,L5,M10,L5_onset_h,M10_onset_h,cosinor_mesor,cosinor_amplitude,cosinor_acrophase_h,cosinor_r2,n_epochs,n_valid_epochs,n_days
example,1.000000,0.000019,0.722781,0.012795,0.079516,23.5,9.0,0.05,0.04,14.0,1.0,4320,4320,3.0
```

IS is 1 because every day repeats exactly. The fitted curve peaks at 14:00,
with a mean of 0.05 g and an amplitude of 0.04 g. L5 begins at 23:30 and
wraps across midnight; it is a five-hour average, so it exceeds the single
lowest point of the curve. M10 begins at 09:00. The low IV follows from the
smooth curve sampled every minute, not from a clinical cutoff.

For each of these three dates the daily file contains the same metric values
except IS and recording-level counts, which are omitted. Its count is 1440
rather than 4320. For example, selected daily columns read:

```text
date,IV,RA,L5,M10,L5_onset_h,M10_onset_h,cosinor_acrophase_h,n_valid_epochs
2026-03-01,0.000019,0.722781,0.012795,0.079516,23.5,9.0,14.0,1440
```

`n_epochs` counts intervals on the aligned recording grid, while
`n_valid_epochs` counts intervals with observed activity. These are
counts, whereas `interval_seconds` specifies the duration of each
interval.

`n_days` counts the calendar days spanned by the aligned recording,
including days with missing data or daylight-saving changes. It is
not the number of days that passed the inclusion threshold.

The existing CSV field names and `rar` filenames are retained
pending maintainer feedback.

## Reading the output

Interdaily stability (IS) measures how consistently the same activity
pattern repeats across days. Identical daily patterns give IS = 1.
Independent random activity tends toward approximately `1/n_days`
when the days are complete.

Intradaily variability (IV) measures changes between consecutive
observed intervals relative to overall activity variance. Smooth
rhythms have low IV, while independent random noise approaches 2.
Rapidly alternating activity can produce values above 2, including
IV = 4 for a regularly alternating series. IV also depends on the
sampling interval.

Relative amplitude (RA) compares the most active ten-hour period
(M10) with the least active five-hour period (L5) of the average day.
Their onset times describe recorded activity, not necessarily
sleep and wake times.

The cosinor fit provides the fitted mean (MESOR), amplitude,
peak time (acrophase), and R-squared. When R-squared is low or
amplitude is close to zero, the estimated peak time may not be
meaningful.

Missing data can affect all these estimates, particularly when
gaps occur at the same clock times each day. Undefined metrics
appear as blank cells in the CSV output, not zeros. Check coverage,
sampling intervals, and acceleration units before comparing
participants.

These metrics have been studied in relation to mortality in prospective
cohorts. Xu et al. found that only one of the two held up in NHANES (n=7,252,
US adults, 674 deaths): IV associated with higher all-cause mortality, HR 1.13
(1.01-1.26) per quartile, fully adjusted, while IS was not significant. Wendt
et al. found the opposite split in a Brazilian cohort of older adults (n=965,
80 deaths): IS associated with lower mortality, HR 0.74 (0.55-0.98) per
standard deviation, while IV was not, and neither association survived
adjustment for physical activity. So the direction is consistent, more
disrupted rhythms track with worse outcomes, but which specific metric reaches
significance depends on the population and the covariates, and physical
activity can absorb the effect entirely. Treat IS and IV as descriptive rhythm
measures, not validated standalone mortality predictors.

### Data-problem signals

These metrics describe recorded phone movement, not confirmed device wear
or sleep. Check the raw data and processing logs when:

- **Activity is flat or nearly stationary.** A phone left in one place may
  produce little movement or mostly sensor noise. Constant activity makes
  IS and IV undefined, while an all-zero signal also makes RA undefined.
  Noise can produce high IV and low IS without reflecting actual activity.

- **A participant is missing from `rar_summary.csv`.** Check the processing
  logs and coverage requirements. By default, participants need at least
  three days with 16 observed hours each.

- **Metric values are blank.** These indicate unavailable or undefined
  estimates, not zero activity. Daily metrics require 12 observed hours.
  L5 and M10 also require fully observed candidate windows.

- **Gaps occur at similar times each day.** Missing nighttime data may
  exclude the quietest activity window or shift the estimated L5 onset.
  Each position in the average-day profile needs at least one observation,
  but the number of contributing days can differ between positions.

- **Coverage looks high despite sparse recording.** An interval counts as
  observed even if it contains only one valid accelerometer sample.
  Coverage alone does not confirm continuous recording or device wear.

- **Results differ unexpectedly across recordings.** Check acceleration
  units, the `gravity` setting, sampling intervals, and missing-data
  patterns. Compare iOS and Android recordings separately before
  interpreting platform differences as behavioral differences.

A low cosinor R² does not necessarily indicate bad data. It may simply
mean that a single 24-hour cosine is a poor fit for the recorded pattern.

## Assumptions and limitations

- **Daylight-saving changes.** The module uses local clock time
  and keeps a fixed 24-hour grid for each day. When clocks move
  forward, the missing hour remains `NaN`. When clocks move
  backward, observations from the repeated hour are averaged.
  Forest logs a warning when it detects these changes.

  DST dates are not automatically excluded. They follow the
  same coverage rules as other days, although missing or repeated
  hours can affect the resulting metrics. These dates should be
  checked when interpreting the results.

- **Missing observations.** IV uses only consecutive observed
  pairs and calculates variance from the available values.
  It does not bridge gaps. Each clock position in the average-day
  profile needs at least one observation across the recording.
  Positions without observations remain `NaN`.

  L5 and M10 are calculated only from windows where every clock
  position has data. If no complete window is available, the
  affected metrics are undefined. Missing observations can also
  shift the selected windows, especially when missingness is
  related to activity.

- **Local-midnight alignment.** Recording-level metrics assume
  that the series starts at local midnight. The `run()` function
  handles this alignment automatically. Anyone calling the
  metric functions directly must ensure the data is aligned
  correctly (see the `cosinor` docstring).

## Validation against existing packages

We compared Forest's calculations with ActCR 0.4.0 and CosinorPy 3.1
using six synthetic activity patterns. These included regular and noisy
cosine rhythms, a square wave, a square wave with naps, and white noise.
Each recording covered seven days and was tested at one-minute and
hourly resolution.

We used ActCR to check IS, IV, L5, M10, and RA, and CosinorPy to check
the four cosinor metrics.

Of the 108 comparisons, 100 agreed within our chosen tolerances.
Six differences came from how the packages handle windows crossing
midnight: Forest allows them, while ActCR searches within a single
day. We considered two white-noise acrophase results separately because
peak timing is difficult to interpret when there is little rhythmic
signal.

IS and IV agreed almost exactly. Cosinor mesor, amplitude, and R-squared
also matched closely, with a maximum acrophase difference of about
1.2 minutes.

These results support the calculations on complete synthetic data.
They do not establish how well the metrics perform with missing
observations, daylight-saving transitions, or real-world differences
in smartphone use.


## Implementation and provenance

The methods in this module build on Ceyhun Olcan's earlier
[actrhythm library](https://github.com/ceyhunolcan/actrhythm).
That library implements interdaily stability (IS), intradaily
variability (IV), relative amplitude (RA), the least active five
hours (L5), the most active ten hours (M10), and related activity
fragmentation measures.

These methods were also applied in a
[Research Square preprint](https://doi.org/10.21203/rs.3.rs-9830931/v1)
examining olfactory dysfunction and rest-activity patterns in the
2013-2014 National Health and Nutrition Examination Survey (NHANES).
The [analysis code](https://github.com/ceyhunolcan/od-activity-rhythm)
is available separately.

The NHANES study uses Monitor-Independent Movement Summary (MIMS)
values derived from wrist accelerometry. This Forest module uses
Euclidean Norm Minus One (ENMO) calculated from smartphone
accelerometer data. Although the rhythm formulas are related,
these activity measures are not directly interchangeable.

The calculations are implemented directly in Oak rather than
adding new runtime dependencies. The metric functions are small
enough to test independently, while ActCR and CosinorPy provide
external references for checking the numerical results.

Data preparation, metric calculations, and output handling are
kept in separate functions. They could be split into separate
files later if needed. ENMO processing also remains independent
of Oak's existing gait detection pipeline.

The validation covers synthetic activity recordings. It does
not establish clinical validity or measurement equivalence
between iOS and Android devices.

## References

- Van Someren EJW, et al. Bright light therapy: improved sensitivity to its
  effects on rest-activity rhythms in Alzheimer patients by application of
  nonparametric methods. *Chronobiology International* 16(4):505-518 (1999).
- Gonçalves BSB, et al. Nonparametric methods in actigraphy: an update.
  *Sleep Science* 7(3):158-164 (2014).
- Cornelissen G. Cosinor-based rhythmometry. *Theoretical Biology and
  Medical Modelling* 11:16 (2014).
- Xu Y, Su S, Li X, Mansuri A, McCall WV, Wang X. Blunted rest-activity
  circadian rhythm increases the risk of all-cause, cardiovascular disease and
  cancer mortality in US adults. *Scientific Reports* 12:20665 (2022).
  https://doi.org/10.1038/s41598-022-24894-z
- Wendt A, Bielemann RM, Wehrmeister FC, et al. Is rest-activity rhythm
  prospectively associated with all-cause mortality in older people regardless
  of sleep and physical activity level? The 'Como Vai?' Cohort study.
  *PLoS ONE* 19(2):e0298031 (2024).
  https://doi.org/10.1371/journal.pone.0298031
