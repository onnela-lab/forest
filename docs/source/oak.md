#### Authors: Marcin Straczkiewicz, Jukka-Pekka Onnela  

#### Executive Summary: 
Use `oak` to calculate number of steps using Beiwe accelerometer data.


#### Installation

`from forest.oak import base.py`  


#### Usage:  
**Determine study folder and output_folder**
study_folder = "project/data"
output_folder = "project/results"

**Determine study timezone and time frames for data analysis**
tz_str = "America/New_York"
time_start = "2018-01-01 00_00_00"
time_end = "2022-01-01 00_00_00"

**Determine window for analysis. Available opts: "Hourly", "Daily", "both".**
option = "both"
beiwe_id = None

**Call the main function**
main_function(study_folder, output_folder, tz_str, option,
              time_start, time_end, beiwe_id)


## Default tuning parameters for walking recognition and step counting
# minimum peak-to-peak amplitude (in gravitational units (g))
min_amp = 0.3  

# step frequency (in Hz) - sfr
step_freq = (1.4, 2.3)

# maximum ratio between dominant peak below and within sfr
alpha = 0.6

# maximum ratio between dominant peak above and within sfr
beta = 2.5

# maximum change of step frequency between two one-second
# nonoverlapping segments (expressed in multiplication of 0.05Hz, e.g.,
# delta=2 -> 0.1Hz)
delta = 20

# minimum walking time (in seconds (s))
epsilon = 3

## other thresholds:
# threshold to qualify activity bout for computation
minimum_activity_thr = 0.1