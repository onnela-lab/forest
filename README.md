# forest (Based on Python 3.6.0)

For more detailed info on specific subpackages, see our Wiki.

**Please note that Forest uses Python 3.6.**

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

## For contributors

## References

Onnela lab:

Beiwe platform for smartphone data collection:

### Papers describing forest

### Papers using forest

## Version 
Forest 0.1
