Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
examples
===

This directory contains sample code for the `beiwetools` package and sample Beiwe configuration files.  The contents are as follows.

## 1. `configuration_files`

This directory contains six Beiwe study configuration files.  Five of these files correspond to studies from the publically available Beiwe data sets located at [`https://zenodo.org/record/1188879#.XcDUyHWYW02`](https://zenodo.org/record/1188879#.XcDUyHWYW02).


| **Configuration File** | **Public Data Directory** |
| -------------------- | ----------------------- | 
| `generic_study.json`| *None* |
| `study_A.json` | `onnela_lab_gps_testing` |
| `study_B.json` | `passive_data_high_sampling` |
| `study_C.json` | `onnela_lab_ios_test1` |
| `study_D.json` | `onnela_lab_ios_test2` |
| `study_E.json` | `onnela_lab_test1` |


## 2. `configread_example.ipynb`

This notebook provides code for three example tasks:
1. Create a registry for a single directory of raw data,
2. Create a registry for multiple directories,
3. Query a registry for study settings, device parameters, and available file paths.

In addition to creating registries, these tasks demonstrate how to:  
* Manage user names and object names,
* Review numerical summaries of raw user data,
* Save and reload a raw data registry.


## 3. `manage_example.ipynb`