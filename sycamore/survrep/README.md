Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
survrep
===

The `survrep` package provides some limited tools for processing metadata from Beiwe tracking surveys.  This document gives an overview of survey metadata on the Beiwe platform, followed by a brief description of each module.

Note that `survrep` requires the `beiwetools` package, located in the `forest/poplar` repository.  

To install with pip:

```bash
pip install /path/to/survrep
```

Example imports:

```python
import survrep
import survrep.functions as sf
from survrep import xx
```

___
## Table of Contents
1.  [Version Notes](#version)  
2.  [Overview](#overview)  
3.  [iOS Survey Timings](#ios)
4.  [Android Survey Timings](#android)
5.  [Modules](#modules)
    * [`headers`](#headers)      
	* [`functions`](#functions)  
	* [`summary`](#summary)  
	* [`compatibility`](#compatibility)  	
	* [`meta`](#meta)  	
6.  [Examples](#examples)  
7.  [Cautions & Notes](#cautions)  


___
## 1. Version Notes <a name="version"/>  

This is version 0.0.1 of `survrep`.  This package was developed with Python 3.8.1 on PCs running Manjaro Juhraya 18.1.5.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository.

___
## 2. Overview <a name="version"/>

Tracking survey metadata ("survey timings") are collected and organized by the Beiwe backend according to standard conventions for raw Beiwe data.  See `beiwetools/README.md` for more information about Beiwe time formats, file-naming conventions, and directory structure.

These metadata include event records of when tracking survey questions were answered, and when surveys were submitted.  Files are organized into directories named after tracking survey identifiers.  

**Important notes:**

* **Event records may be mingled.  Metadata for a particular tracking survey may be found in a directory named for a different survey.**

* **Event records for a given survey submission may span multiple files.**


The format and content of metadata files differ for the iPhone and Android versions of the Beiwe app.  Headers for these files are documented in `survrep/headers.py`.  See below for some additional details.

___
## 3. iOS Survey Timings <a name="ios"/>

On the Beiwe iOS app, tracking survey metadata is handled by a [TrackingSurveyPresenter]() object.  Event records for survey notifications and expirations originate from a [StudyManager]() object.  The following events are logged in the `events` column of an iOS survey timings file:

| **Event** | **Interpretation** |
|-----------|-----------|
| `notified` | |
| `expired` | |
| `present` | |
| `changed` | |
| `unpresent` | |
| `submitted` | |









___
## 4. Android Survey Timings <a name="android"/>

**Important notes:**

* **Additional survey metadata may be found in `app_log` files.  This package does not handle metadata from these files.**

`Survey first rendered and displayed to user`

`User hit submit`


___
## 5. Modules <a name="modules"/>

#### `headers` <a name="headers"/>
This module contains headers for `csv` files that are handled by `survrep`, including raw Beiwe survey timings files.  Each variable is described with a brief comment.


___
#### `functions` <a name="functions"/>

___
#### `summary` <a name="summary"/>


___
#### `compatibility` <a name="compatibility"/>

___
#### `meta` <a name="meta"/>

___
## 6. Examples <a name="examples"/>

___
## 7. Cautions & Notes <a name="cautions"/>

* Note that this package only permits assignment of one configuration file to each participant.  This may be inadequate for some situations.

