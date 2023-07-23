# Oak

## Executive Summary: 
Use `oak` to calculate number of steps using Beiwe accelerometer data.

## Installation Instruction
For instructions on how to install forest, please visit [here](https://github.com/onnela-lab/forest). 
`from forest import oak`

## Usage:
```
from forest.oak.base import run
from forest.constants import Frequency


# Determine study folder and output_folder
study_folder = "project/data"
output_folder = "project/results"

# Determine study timezone and time frames for data analysis
tz_str = "America/New_York"
time_start = "2018-01-01 00_00_00"
time_end = "2022-01-01 00_00_00"

# Determine window for analysis. Frequency of the summary stats (resolution for summary statistics) e.g. Frequency.HOURLY, Frequency.DAILY, etc. see forest.constants.Frequency here: https://github.com/onnela-lab/forest/blob/develop/forest/constants.py

frequency = Frequency.HOURLY_AND_DAILY
beiwe_id = None

# Call the main function
run(study_folder, output_folder, tz_str, frequency,
              time_start, time_end, beiwe_id)
```
### Default tuning parameters for walking recognition and step counting:
```
# minimum peak-to-peak amplitude (in gravitational units (g))
min_amp = 0.3  

# step frequency (in Hz) - sfr
step_freq = (1.4, 2.3)

# maximum ratio between dominant peak below and within sfr
alpha = 0.6

# maximum ratio between dominant peak above and within sfr
beta = 2.5

# maximum change of step frequency between two one-second non-overlapping
# segments (expressed in multiplication of 0.05Hz, e.g., delta=2 -> 0.1Hz)
delta = 20

# minimum walking time (in seconds (s))
min_t = 3
```