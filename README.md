![alt text](https://github.com/onnela-lab/forest/blob/master/forest-logo-color.png)

<p align="center">
  <img width="460" height="300" src="https://github.com/onnela-lab/forest/blob/master/forest-logo-color.png">
</p>

# forest (Based on Python 3.6.0)

For more detailed info on specific subpackages, see our [Wiki](https://github.com/onnela-lab/forest/wiki). Please note that Forest uses Python 3.6.

## Description

Description of how beiwe data looks (folder structure + on/off cycles)

Input: typically raw data from smartphones
Output: typically summary files
-	Creating synthetic data
    - Want to try out our methods, but don't have smartphone data at hand? Use **bonsai**
-	Data preparation
    - Identifying time zones and unit conversion: use **poplar**
    - Collate Beiwe survey data into .csvs per participant or per study: use **sycamore**
-	Data imputation
    - State-of-the-art GPS imputation: use **jasmine**
-	Data summarizing (see tables below for summary metrics)
    - Mobility metrics from GPS data: use **jasmine**
    - Daily summaries of call & text metadata: use **willow**
    - Survey completion time from survey metadata: use **sycamore**
    
## Usage
To install, clone this repository to a local directory and then:
```
pip install path/to/forest
```
Alternatively, [install directly from github](https://pip.pypa.io/en/stable/reference/pip_install/#git) with `pip`. As long as the repo is private, it may prompt you to login.
```
pip install git+https://github.com/onnela-lab/forest
```

Currently, all imports from `forest` must be explicit.  For example:
```
from forest.jasmine.traj2stats import gps_stats_main
from forest.poplar.functions.io import read_json
```

To immediately test out forest, adapt the filepaths in the code below and run:
```
# Import forest
import forest

# 1. If you don't have any smartphone data (yet) you can generate fake data
path_to_synthetic_gps_data = "ENTER/PATH1/HERE"
path_to_synthetic_log_data = "ENTER/PATH2/HERE"
path_to_gps_summary = "ENTER/PATH/TO/DESIRED/OUTPUT/FOLDER1/HERE"
path_to_log_summary = "ENTER/PATH/TO/DESIRED/OUTPUT/FOLDER2/HERE"

# Generate fake call and text logs 
forest.bonsai.sim_log_data(path_to_synthetic_log_data)

# Generate synthetic gps data and communication logs data as csv files
# Define parameters for generating the data
# To save smartphone battery power, we typically collect location data intermittently: e.g. during an on-cycle of 3 minutes, followed by an off-cycle of 12 minutes. We'll generate data in this way
cycle = 15 # Length of off-cycle + length of on-cycle in minutes
p = 0.8 # Length off-cycle / (length off-cycle + length on-cycle)
forest.bonsai.sim_GPS_data(cycle, p, path_to_synthetic_log_data)

# 2. Specify parameters for imputation 
# See https://github.com/onnela-lab/forest/wiki/Jasmine-documentation#input for details
tz_str = "America/New_York" # time zone where the study took place (assumes that all participants were always in this time zone)
option = "daily" # Generate summary metrics "hourly", "daily" or "both"
save_traj = False # Save imputed trajectories?
time_start = None 
time_end = None
beiwe_id = None
parameters = None
all_memory_dict = None
all_BV_set= None

# 3. Impute location data and generate mobility summary metrics using the simulated data above
gps_stats_main(path_to_synthetic_gps_data, path_to_gps_summary, tz_str, option, save_traj, time_start, time_end, beiwe_id, parameters, all_memory_dict, all_BV_set)

# 4. Generate daily summary metrics for call/text logs
log_stats_main(path_to_synthetic_log_data, path_to_log_summary, tz_str, option, time_start, time_end, beiwe_id)
```


## For contributors

## More info

[Beiwe platform for smartphone data collection](https://www.beiwe.org/)
[Onnela lab](https://www.hsph.harvard.edu/onnela-lab/)

## Version 
Forest 1.0
