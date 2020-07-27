Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
fitrep
===

The `fitrep` package provides basic tools for working with [Fitabase](https://www.fitabase.com/) data sets from Fitbit devices.  This document gives a brief description of each module.

Note that `fitrep` requires the `beiwetools` package, located in the `forest/poplar` repository.  

To install with pip:

```bash
pip install /path/to/fitrep
```

Example imports:

```python
import fitrep
import fitrep.functions as ff
from fitrep import parse_filename
```


___
## Table of Contents
1.  [Version Notes](#version)  
2.  [Overview](#overview)  
3.  [Fitabase Data](#data)  
    * [Download Instructions](#instructions)  
    * [File Names](#names)
    * [File Types](#types)
    * [File Contents & Variable Names](#contents)  
4.  [Modules](#modules)
    * [`headers`](#headers)  
	* [`functions`](#functions)  
	* [`classes`](#classes)  
	* [`summary`](#summary)
	* [`sync`](#sync)
	* [`format`](#format)
5.  [Examples](#examples)  
6.  [Cautions & Notes](#cautions)  

___
## 1. Version Notes <a name="version"/>

This is version 0.0.1 of `fitrep`.  This package was developed with Python 3.8.1 on PCs running Manjaro Kyria 19.0.2.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository. 

___
## 2. Overview <a name="overview"/>

Participants in some Beiwe studies may also use Fitbit activity monitors.  The [Fitabase platform](https://www.fitabase.com/) may be used to handle delivery of Fitbit data, such as step counts and sleep classifications.

This package converts Fitabase data (which are reported in local time) to a format that can be integrated with Beiwe data streams (which are reported in UTC time).

___
## 3.  Fitabase Data <a name="data"/>

#### Download Instructions <a name="instructions"/>

1. Log in to the Fitabase dashboard.

2. Select a project and click on the "batch export" link.

3. Click on the button labeled "Create new batch download."

4. Select start/end dates.  Change the download name and select tags if desired.

5. Select desired file types with checkboxes.  Note that this package does not handle any data formatted as "Minute (Wide)."

6. **Be sure to check the box for "Sync Events."**

7. **Make sure that "Individual Files" is highlighted.**

8. Click "Create" and wait until files are written and archived.

9. Click the "Download" button.

10. Extract the archive.

Fitabase's batch export documentation is located [here](https://www.fitabase.com/resources/knowledge-base/exporting-data/the-batch-export-tool/).


___
#### File Names <a name="names"/>

Raw individual fitabase files are named with the following convention:

```
<fitabase_id>_<file_type>_<start_date>_<end_date>.csv
```

* ```<fitabase_id>```:  The unique alphanumeric string that identifies the Fitbit device, as it appears in the "Name" column on the project dashboard.

* ```<file_type>```:  A description of data that are reported in the file.  See below for details.

* ```<start_date>```:  The start date for the batch download, formatted as `%Y%m%d`.

* ```<end_date>```: The end date for the batch download, formatted as `%Y%m%d`.

___
#### File Types <a name="types"/>

Each file type has a slightly different name according to where it is mentioned.  **Note that `fitrep` refers to file types as they appear in file names.**  This package handles the following seven file types:

|Location|1|2|3|
|-----|-----|-----|-----|
|*File Name* | `heartrate_1min` | `minuteCaloriesNarrow` | `minuteIntensitiesNarrow` |
|[*Batch Export Options (Row / Column)*](https://www.fitabase.com/resources/knowledge-base/exporting-data/the-batch-export-tool/) | Heart Rate / Minute | Calories / Minute | Intensities /			 Minute |
|*Dashboard Export Details* | Heart Rate 1 Min Avg | Calories Minutes | Intensity Minutes |
|[*Data Dictionary  (Category / Subcategory)*](https://www.fitabase.com/media/1748/fitabasedatadictionary.pdf)| Heart Rate / 1 Minute | Calories / Minute (narrow) | Intensity / Minute (narrow) |

|Location|4|5|6|7|
|-----|-----|-----|-----|-----|
|*File Name* | `minuteMETsNarrow` | `minuteSleep` | `minuteStepsNarrow` | `syncEvents` |
|[*Batch Export Options (Row / Column)*](https://www.fitabase.com/resources/knowledge-base/exporting-data/the-batch-export-tool/) | METs / Minute | Sleep / Minute | Steps / Minute | Sync Events / Other |
|*Dashboard Export Details* | METs Minutes | Sleep Minutes | Steps Minutes | Sync Events |
|[*Data Dictionary  (Category / Subcategory)*](https://www.fitabase.com/media/1748/fitabasedatadictionary.pdf)| Intensity / METs - Minute (narrow) | Sleep / Classic Sleep Log (1 minute) | Steps / Minute (narrow) |  Sync Data / Sync Events |

___
#### File Contents & Variable Names <a name="contents"/>

See `/fitrep/headers.py` or the [Fitabase Data Dictionary](https://www.fitabase.com/media/1748/fitabasedatadictionary.pdf) for documentation of file headers.

Minute-by-minute Fitabase files generally have a column with datetime strings in each participant's local time; this column may be named `Time`, `ActivityMinute`, or `date`.  **Note that Fitabase datetimes do not contain leading zeros.**  A second column contains corresponding values reported by Fitbit, which may be numeric (e.g. step counts) or categorical (e.g. codes for sleep state).  

Fitbit sync event logs are associated with two datetimes.  The first is the participant's local time (`DateTime`) and the second is the corresponding UTC time (`SyncDateUTC`).  This package uses the offset between local and UTC time to convert all local datetimes to UTC timestamps.


___
## 4. Modules <a name="modules"/>

#### `headers` <a name="headers"/>
This module contains headers for `csv` files that are handled by `fitrep`.  Documentation for Fitabase file contents and variables is taken from the [Fitabase Data Dictionary](https://www.fitabase.com/media/1748/fitabasedatadictionary.pdf).
___
#### `functions` <a name="functions"/>
This module provides functions for various tasks, such as handling Fitabase time formats, parsing Fitabase file names, and summarizing raw Fitabase files.
___
#### `classes` <a name="classes"/>




___
#### `summary` <a name="summary"/>
Wrappers for functions that summarize Fitabase file contents.

___
#### `sync` <a name="sync"/>
Wrappers for functions that process Fitabase `syncEvents` files, including capture of UTC offsets.

___
#### `format` <a name="format"/>
Wrappers for functions that reformat Fitabase data by (1) converting datetimes to Beiwe format and (2) re-synchronizing minute-by-minute Fitbit sleep classifications.

___
## 5. Examples <a name="examples"/>

The script `fitrep/examples/fitrep_example.py` provides a sample workflow for using this package.

___
## 6. Cautions & Notes <a name="cautions"/>

* Fitabase records the originating device model at sync time.  These logs are found in the `DeviceName` column of `syncEvents` files.  Prior to January 2018, device models may not have been logged, and this column will be incomplete.

* Fitabase data sets may include mingled information from both a Fitbit device and the Fitbit smartphone app.  Synchronization with an app instance is logged in the `syncEvents` file with `MobileTrack` as the `DeviceName`.  **This package does not provide tools for identifying whether a particular Fitbit observation originated from a device or an app instance.**

* This package was developed with a limited sample of Fitabase data collected from fifty-three Fitbit Charge 2 devices between April 2017 and March 2020.

* This package does not handle Fitabase data obtained from multiple downloads, e.g. that cover different or overlapping follow-up windows.  For such situations, it may be best to download a new data set that includes the entire time period of interest.  Alternatively, CSVs for corresponding variables can be merged, with care taken to drop duplicate records, before using `fitrep`.