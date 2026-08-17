# Rest-Activity Rhythms

## Executive Summary:
Use `oak.rhythms` to compute circadian rest-activity rhythm (RAR) summary
statistics from Beiwe accelerometer data. This complements the gait/step
output of `oak` (see [oak](oak.md)): the same raw accelerometer stream,
summarised as 24-hour activity patterning rather than walking.

## Installation Instruction
For instructions on how to install forest, please visit
[here](https://github.com/onnela-lab/forest).
`from forest.oak import rhythms`

## Usage:
```
from forest.oak.rhythms import run
from forest.constants import Frequency


# Determine study folder and output folder
study_folder = "project/data"
output_folder = "project/results"

# Determine study timezone and time frames for data analysis
tz_str = "America/New_York"
time_start = "2018-01-01 00_00_00"
time_end = "2022-01-01 00_00_00"

# Determine output resolution. Frequency.DAILY or Frequency.HOURLY_AND_DAILY
# additionally writes a per-participant daily file; any other value writes
# only the recording-level summary. See forest.constants.Frequency here:
# https://github.com/onnela-lab/forest/blob/develop/forest/constants.py
frequency = Frequency.HOURLY_AND_DAILY
users = None

# Call the main function
run(study_folder, output_folder, tz_str, frequency,
    time_start, time_end, users)
```
### Default tuning parameters:
```
# epoch length in seconds; must divide a 24-hour day evenly - epoch_seconds
# (this is the length of one epoch, not the count per day: the number of
# epochs in a 24-hour day is 86_400 / epoch_seconds, e.g. 1440 at 60 s)
epoch_seconds = 60

# numeric value of 1 g in the data's units (1.0 if the accelerometer is
# already in g, consistent with oak; ~9.81 if the data are in m/s^2) - gravity
gravity = 1.0

# participants with fewer valid days than this are skipped - min_valid_days
min_valid_days = 3
```

## Activity metric

Per sample, movement intensity is summarised as ENMO (the Euclidean norm of
the acceleration vector minus one gravitational unit, clipped at zero):
`ENMO = max(sqrt(x^2 + y^2 + z^2) / gravity - 1, 0)`. ENMO is averaged into
fixed epochs on a timezone-local grid, with missing epochs held as `NaN`.

## List of summary statistics

The recording-level output is written to `rar_summary.csv` with one row per
participant. The following variables are created:

|     Variable            	|     Type     	|     Description of Variable                                                            	|
|-------------------------	|--------------	|----------------------------------------------------------------------------------------	|
|     backend_id          	|       str     	|     Beiwe backend identifier of the participant                                        	|
|     IS                  	|      float    	|     Interdaily stability, in (0, 1]; higher means a more reproducible daily pattern    	|
|     IV                  	|      float    	|     Intradaily variability, typically in [0, ~2]; higher means more fragmentation      	|
|     RA                  	|      float    	|     Relative amplitude, (M10 - L5) / (M10 + L5)                                         	|
|     L5                  	|      float    	|     Mean activity over the least-active 5 hours of the average day                     	|
|     M10                 	|      float    	|     Mean activity over the most-active 10 hours of the average day                     	|
|     L5_onset_h          	|      float    	|     Clock hour at which the L5 window starts                                           	|
|     M10_onset_h         	|      float    	|     Clock hour at which the M10 window starts                                          	|
|     cosinor_mesor       	|      float    	|     MESOR (rhythm-adjusted mean) of the 24 h cosinor fit                               	|
|     cosinor_amplitude   	|      float    	|     Amplitude of the 24 h cosinor fit                                                  	|
|     cosinor_acrophase_h 	|      float    	|     Clock hour of the fitted peak, in [0, 24)                                          	|
|     cosinor_r2          	|      float    	|     Coefficient of determination of the cosinor fit                                    	|
|     n_epochs            	|       int     	|     Number of epochs on the day-aligned grid                                           	|
|     n_valid_epochs      	|       int     	|     Number of non-missing epochs                                                       	|
|     n_days              	|      float    	|     Number of days spanned by the recording                                            	|

When `frequency` is `Frequency.DAILY` or `Frequency.HOURLY_AND_DAILY`, a
per-participant `<backend_id>_rar_daily.csv` is also written with the
day-decomposable metrics (IS is recording-level only and is omitted):

|     Variable            	|     Type     	|     Description of Variable                                                	|
|-------------------------	|--------------	|----------------------------------------------------------------------------	|
|     date                	|       str     	|     Day of observation (yyyy-mm-dd)                                        	|
|     IV                  	|      float    	|     Intradaily variability for the day                                     	|
|     RA                  	|      float    	|     Relative amplitude for the day                                         	|
|     L5                  	|      float    	|     Least-active 5-hour mean for the day                                   	|
|     M10                 	|      float    	|     Most-active 10-hour mean for the day                                   	|
|     L5_onset_h          	|      float    	|     Clock hour at which the L5 window starts                               	|
|     M10_onset_h         	|      float    	|     Clock hour at which the M10 window starts                              	|
|     cosinor_mesor       	|      float    	|     MESOR of the day's cosinor fit                                         	|
|     cosinor_amplitude   	|      float    	|     Amplitude of the day's cosinor fit                                     	|
|     cosinor_acrophase_h 	|      float    	|     Clock hour of the day's fitted peak                                    	|
|     cosinor_r2          	|      float    	|     Coefficient of determination of the day's cosinor fit                  	|
|     n_valid_epochs      	|       int     	|     Number of non-missing epochs in the day                               	|

## References

- Van Someren EJW, et al. Bright light therapy: improved sensitivity to its
  effects on rest-activity rhythms in Alzheimer patients by application of
  nonparametric methods. *Chronobiology International* 16(4):505-518 (1999).
- Gonçalves BSB, et al. Nonparametric methods in actigraphy: an update.
  *Sleep Science* 7(3):158-164 (2014).
- Cornelissen G. Cosinor-based rhythmometry. *Theoretical Biology and
  Medical Modelling* 11:16 (2014).
