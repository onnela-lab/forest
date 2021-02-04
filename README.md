# forest (Based on Python 3.6.0)

For more detailed info on specific subpackages, see our Wiki.

**Please note that Forest uses Python 3.6.**

## Description

Description of how beiwe data looks (folder structure + on/off cycles)

Input: typically raw data from smartphones
Output: typically summary files
-	Data preparation
    - Identifying time zones
    - Unit conversion
-	Data imputation
-	Data summarizing

### Poplar

### Jasmine

### Willow

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



## References

Onnela lab:

Beiwe platform for smartphone data collection:

### Papers describing forest

### Papers using forest

## Version 
Forest 0.1
