# forest 

For more detailed info on specific subpackages, see our Wiki

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
Alternatively, [install directly from github](https://pip.pypa.io/en/stable/reference/pip_install/#git) with `pip`.

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
