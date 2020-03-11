Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
fitrep
===

The `fitrep` package provides basic tools for working with data from Fitbit devices that have been downloaded from [Fitabase](https://www.fitabase.com/).  This document gives a brief description of each module.

Note that `fitrep` requires the `beiwetools` package, located in the `forest/poplar` repository.  

To install with pip:

```bash
pip install /path/to/fitrep
```

Example imports:

```python
import fitrep
import fitrep.functions as ff
from fitrep.functions import xx
```


___
## Table of Contents
1.  [Version Notes](#version)  
2.  [Overview](#overview)  
3.  [Modules](#modules)
    * [`headers`](#headers)  
	* [`functions`](#functions)  
4.  [Downloading Data](#download)  
5.  [Examples](#examples)  
6.  [Cautions & Notes](#cautions)  

___
## 1. Version Notes <a name="version"/>

This is version 0.0.1 of `fitrep`.  This package was developed with Python 3.8.1 on PCs running Manjaro Kyria 19.0.2.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository. 

___
## 2. Overview <a name="overview"/>


___
## 3. Modules <a name="modules"/>


#### `headers` <a name="headers"/>
This module contains headers for `csv` files that are handled by `fitrep`.  Each variable is 
described with a brief comment.


Raw Fitabase headers were extracted from files downloaded on xx

These files included data collected from Fitbit Charge 2 and Zip devices between xx/xx/xx and xx/xx/xx.


___
#### `functions` <a name="functions"/>
This module provides functions for xx.


___
## 4. Downloading Data <a name="download"/>

#### Instructions

1. Log in to your Fitabase dashboard.

2. Select your project and click on the "batch export" link.

3. Click on the button labeled "Create new batch download."

4. Select start/end dates.  Change the download name and select tags if desired.

5. Select desired file types with checkboxes.  Note that this package does not handle any data formatted as "Minute (Wide)."

6. **Be sure to check "Sync Events."**

7. **Make sure that "Individual Files" is highlighted.**

8. Click "Create" and wait until files are written and archived.

9. Click the "Download" button.

10. Extract the archive.

Fitabase's batch export documentation is located [here](https://www.fitabase.com/resources/knowledge-base/exporting-data/the-batch-export-tool/).


___
#### File Names

Raw individual fitabase files are named with the following convention:

```
<fitabase_id>_<file_type>_<start_date>_<end_date>.csv
```

* ```<fitabase_id>```:  The unique alphanumeric string that identifies the Fitbit device, as it appears in the "Name" column on the project dashboard.

* ```<file_type>```:  A description of data that are reported in the file.  See below for some details.

* ```<start_date>```:  The start date for the batch download, formatted as `%Y%m%d`.

* ```<end_date>```: The end date for the batch download, formatted as `%Y%m%d`.

___
#### File Types

Each file type has a slightly different name according to where it is mentioned.  Some examples:

|Location|1|2|3|
|-----|-----|-----|-----|
|File Name | `` | `` | `` |
|Batch Export Options | `` | `` | `` |
|Dashboard Export Records | `` | `` | `` |
|[Data Dictionary](https://www.fitabase.com/media/1748/fitabasedatadictionary.pdf)| `` | `` | `` |

**Note that this package refers to file types as they appear in file names.**  See `/fitrep/names.json` for a list of all file types supported by `fitrep`.

___
#### File Contents & Variable Names

headers


[activity logs](https://www.fitabase.com/resources/knowledge-base/learn-about-fitbit-data/activity-logs/)



___
## 5. Examples <a name="examples"/>


`fitabase_id`: A unique alphanumeric identifier on the Fitabase platform.

`fitabase_dir`: The directory containing raw fitabase CSV files.


___
## 6. Cautions & Notes <a name="cautions"/>


This package was developed with a limited sample of Fitabase data xx

This package does not handle Fitabase data obtained from multiple downloads, e.g. that cover different or overlapping follow up windows.  For such situations, it may be best to download a new data set that includes the entire time period of interest.  Alternatively, CSVs for corresponding variables can be merged, with care taken to drop duplicate records, before using `fitrep`.


