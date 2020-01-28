Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
beiwetools
===

The `beiwetools` package provides classes and functions for working with Beiwe data sets and Beiwe study configurations.  

This document gives a brief description of each sub-package, along with some background about Beiwe studies, configuration files and raw data.  Some guidelines for maintaining this package can be found [here](#maintenance).

There are four sub-packages:

* `helpers`:  Functions for handling common scenarios, such as converting time formats, summarizing sensor sampling rates, and plotting timestamps,
* `configread`:  Tools for querying Beiwe configuration files and generating study documentation,
* `manage`:  Tools for organizing and summarizing directories of raw Beiwe data,
* `localize`:  Classes and functions for incorporating each user's time zone into the analysis of processed Beiwe data.

To install this package with pip:

```bash
pip install /path/to/beiwetools
```

Example imports:

```python
import beiwetools.configread as configread
from beiwetools.helpers import Timer, sort_by
from beiwetools.helpers.time import to_timestamp
```


___
## Table of Contents
1.  [Version Notes](#version)  
2.  [Overview](#overview)  
3.  [Time Formats](#time)  
4.  [Directory Structure](#directory)  
5.  [`beiwetools.helpers`](#helpers)  
6.  [`beiwetools.configread`](#configread)  
7.  [`beiwetools.manage`](#manage)  
8.  [`beiwetools.localize`](#localize)  
9.  [Examples](#examples)
10. [Maintenance](#maintenance)

___
## 1. Version Notes <a name="version"/>

This is version 0.0.1 of `beiwetools`.  This package was developed with Python 3.7.4 on PCs running Manjaro Illyria 18.0.4.

#### 1.1 Requirements
Among the package requirements, the following are not in the Python Standard Library:

* `holidays`
* `humanize`
* `pandas`
* `pytz`
* `seaborn`
* `timezonefinder`

#### 1.2 Compatibility
The modules in this package were written with the intention of preserving compatibility with previous versions of Python 3.  For example, it is generally desirable to preserve key insertion order when reading JSON files into dictionaries.  Since Python 3.6, dictionaries do preserve insertion order.  However, for compatibility with previous versions, we use ordered dictionaries (`collections.OrderedDict`) instead.

Note that sub-packages are currently collected in a [native namespace package](https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages).  This structure is supported only by Python 3.3 and later.

___
## 2. Overview <a name="overview"/>

#### 2.1 Beiwe Studies
A **Beiwe study** corresponds to a collection of surveys and device settings.  These study parameters determine the app's behavior when it is installed on a user's phone, including:

* How and when surveys are delivered,
* How often data are uploaded,
* Which sensors are sampled for passive data collection.

Each Beiwe user is assigned an alphanumeric string (**user ID**), when he or she is registered in a study.  Each Beiwe study is assigned both a hex-string identifier (**study ID**) and a human-readable name.  

Using the above identifiers, researchers can download raw user data from the Beiwe backend.  This can be done manually (e.g. from `studies.beiwe.org`) or with tools from the `mano` package.


#### 2.2 Configuration Files <a name="configuration"/>
The parameters of a Beiwe study can be exported to a **configuration file** in JSON format.  Configuration file names may have the following format:

```
The_Name_of_the_Study_surveys_and_settings.json
```

Such a file contains a serialized representation all the study's **device settings**, including:

* On/off cycle periods for passive data collection,
* Text that is displayed to users,
* How and when data are uploaded to the backend.

A configuration file also contains details about the study's **surveys**, such as:

* Survey delivery schedule,
* Question types and and content,
* Branching logic (or "skip" logic) for delivery of questions.

Note that configuration files do *not* contain the following information:

* Individual user IDs,
* Human-readable names for studies, surveys, or other objects.

The Beiwe backend assigns a human-readable name to each study.  However, this name appears only in the name of the configuration file.  For this reason, configuration files should usually not be renamed.

In this package, lists of attributes found in Beiwe configuration files are provided in three JSON files:

```bash
/beiwetools/configread/
	study_settings.json
	survey_settings.json
	question_settings.json
```

<a name="format"/>Note that there are three slightly different formats for configuration files:

* **MongoDB Extended JSON format.** In older Beiwe configuration files, objects are identified only with  an `_id` attribute.  Objects in these configuration files do not have a `deleted` attribute, and in some cases there may be other minor differences across attributes.
* **Current format V1.**  Since May 2018, Beiwe configuration files identify objects (such as surveys) with both an `id` (integer) and an `object_id` (hex string).  From September 2018 communication with developer:  
`id` is a database-table-specific key that may be subject to change,  
`object_id` is equivalent to the old `_id` identifier.

 All objects also have an additional Boolean attribute `deleted`.
* **Current format V2.** In 2019, the following device settings were added:  
    `call_clinician_button_enabled`  
    `call_research_assistant_button_enabled`  
    `use_gps_fuzzing`  

Lastly, note that the Beiwe study identifier that appears in a configuration file may not match the study ID found elsewhere on the Beiwe backend.  (This is a concern only for study IDs.  Configuration files do appear to provide correct identifiers for surveys and questions.)

___
## 3. Time Formats <a name="time"/>

Several time formats are used in Beiwe data and configuration files.  With few exceptions, these formats correspond to times in Coordinated Universal Time (UTC).  Various formats can be found in `helpers.time_constants`.

#### 3.1 Timestamps
In raw Beiwe data, each observation is associated with a timestamp corresponding to the number of milliseconds that have elapsed since the Unix epoch, January 1, 1970 00:00:00 Coordinated Universal Time (UTC).


#### 3.2 File Names
A raw Beiwe data set contains `csv` files organized according to the directory structure described [below](#directory).  Most files correspond to one hour of observations.  If no file exists for a particular hour, then no data were observed from that stream during that hour.

The name of each file is the UTC time corresponding to the beginning of the hour in which data were collected.  The filename format is: 
`<%Y-%m-%d %H_00_00>.csv`

Exceptions are found in these directories:

* `survey_answers`: Responses to tracking surveys are named with the time of submission.  The filename format is:  
`<%Y-%m-%d %H_%M_%S>.csv`
* `audio_recordings`: Responses to audio surveys are also named with the time of submission.  Extensions correspond to audio formats, such as `mp4` or `wav`.


#### 3.3 Raw Data <a name="files"/>
Most raw data files have columns labeled `timestamp` and `UTC time`.  These contain the millisecond timestamp and human-readable UTC time (`%Y-%m-%dT%H:%M:%S.%f`) for the observations in the corresponding row.


#### 3.4 Configuration Files
Each survey in a Beiwe study has a `timings` attribute that indicates which days and times the survey is delivered.  This attribute is a list of seven lists of integers.  For example:

```
[[],[37800],[],[37800, 50400],[],[50400],[]]
```

Position in the top list indicates day of the week, starting with Sunday.  (Note that this differs from the usual Python day order, which starts with Monday.)  Integers correspond to the user's local time, in seconds from midnight (see `helpers.time.convert_seconds()`).  In the above example, the survey is delivered at 10:30AM on Mondays & Wednesdays, and at 2:00PM on Wednesdays & Fridays.


#### 3.4 Local Time
Some modules in this package report local times for various purposes:

* File names or directory names may include the researcher's local date/time, formatted as: `%Y-%m-%d_%H:%M:%S_%Z`

* Log files may include the researcher's local date/time, formatted as: `%Y-%m-%d %H:%M:%S %Z`

* Except for timestamps, the `localize` sub-package always reports date/times in the user's local timezone (which may change during the follow-up period).  Formats are:  
`'%Y-%m-%d'`  
`'%H:%M:%S'`  
`'%Y-%m-%d %H:%M:%S'`  


___
## 4. Directory Structure <a name = "directory"/>
  
Raw Beiwe data may be downloaded from the backend and extracted to a directory chosen by the researcher.  The `beiwetools` package assumes the following directory structure:

```
<raw data directory>/

	<Beiwe User ID #1>/

		identifiers/
		<passive data stream #1>/
		<passive data stream #2>/
		.
		.
		.

		audio_recordings/
			<audio survey identifier #1>/
			<audio survey identifier #2>/
			.
			.
			.

		survey_answers/
			<tracking survey identifier #1>/
			<tracking survey identifier #2>/
			.
			.
			.

		survey_timings/
			<tracking survey identifier #1>/
			<tracking survey identifier #2>/
			.
			.
			.

	<Beiwe User ID #2>/
		.
		.
		.
```

Note that `<raw data directory>` is typically a location chosen by the researcher.  This directory's contents are determined by the study's specific data collection settings.

Each user's data are found in a folder labeled with the user's Beiwe ID.  Multiple users enrolled in the same Beiwe study may have data folders in the same raw data directory.

Each user's data directory will include a folder labeled `identifiers`; this contains files with records of the user's device.  Passive data folder names may be `accelerometer`, `calls`, `gps`, etc.

Raw survey data are organized as follows:

* Audio survey recordings:  
`<Beiwe User ID>/audio_recordings/<audio survey identifier>/`  
Note: Prior to Summer 2017, raw audio data from multiple surveys may have been delivered directly to `<Beiwe User ID>/audio_recordings`.

* Item responses from tracking surveys:  
`<Beiwe User ID>/survey_answers/<tracking survey identifier>/`

* Survey metadata:  
`<Beiwe User ID>/survey_timings/<tracking survey identifier>/`

___
## 5. `beiwetools.helpers` <a name="helpers"/>

This sub-package provides classes and functions that are used by `beiwetools` and also by Beiwe reporting packages (`accrep`, `gpsrep`, `survrep`).  Below is an overview of each module.

#### `classes` & `functions`
These modules provide general-purpose tools for tasks that often arise when working with Beiwe configurations and raw data.

#### `colors`
Some color maps and functions for generating Color Brewer palettes.

#### `decorators`
Some decorators to use when defining functions that may be used with `helpers.proc_template`.

#### `plot`
Functions for handling basic data visualization tasks, such as plotting timestamps and generating axis labels for longitudinal data.

#### `proc_template`
This module provides a wrapper for functions that process raw Beiwe data.

#### `process`
The tools in this module assume the Beiwe [file naming conventions](#files) and [directory structure](#directory) that are described in this document.  

#### `time` & `time_constants`
The `time` module provides functions for working with the various [time formats](#time) found in Beiwe data.  Commonly used timezones and date-time formats are provided in `time_constants`.

#### `trackers`
Classes for online calculation of summary statistics during data processing tasks.


___
## 6. `beiwetools.configread` <a name="configread"/>

The `configread` sub-package provides classes for representing information in Beiwe configuration files, with methods for generating documentation.  These modules have been tested on 32 configuration files from nine Beiwe studies.

Review [this section](#configuration) for information about configuration files.

#### 6.1 Study Attributes
To ensure compatibility across formats, `configread` looks for all known study attributes, regardless of file format.  Therefore, due to differences across configuration file formats, it is not unusual to see many missing values.

Documented study attributes can be found in `study_settings.json`, `survey_settings.json`, and `question_settings.json`.  Any undocumented attributes are logged when a configuration file is read into a `BeiweConfig` instance.

#### 6.2 Identifiers
A `BeiweConfig` instance will attach identifiers to each Beiwe survey and question object.  These correspond to the unique identifiers found in raw survey response data, and can be used to query the content of surveys and questions.  
 
Each identifier is either the `object_id` or the `_id` assigned to the survey or question, depending on the format of the configuration file.

#### 6.3 Study Documentation
The `export()` method generates configuration documentation for a `BeiweConfig` object.  Documentation from a `BeiweConfig` instance is organized as follows:

```
configuration_documentation_from_<local time>/

	The_Name_of_the_Study/

		overview.txt
		warnings.txt
		
		records/			
			paths.json
			raw.json
			names.json

		settings/					
			general_settings.txt
			display_settings.txt
			passive_data_settings.txt
						
		audio_surveys/
			<Name_of_Audio_Survey_#1>.txt
			<Name_of_Audio_Survey_#2>.txt		
			.
			.
			.

		tracking_surveys/
			<Name_of_Tracking_Survey_#1>.txt
			<Name_of_Tracking_Survey_#2>.txt		
			.
			.
			.
			
		other_surveys/
			<Name_of_Other_Survey_#1>.txt
			<Name_of_Other_Survey_#2>.txt		
			.
			.
			.
```

The file `warnings.txt` provides a record of any undocumented settings or objects found in the configuration file. A common undocumented object is the "dummy" survey type, which is probably assigned to surveys that have been deleted.

The `records` folder contains everything needed to recreate an identical `BeiweConfig` instance.  To do this, either:  

1. Provide paths to `raw.json` and `names.json` as input, or  
2. Provide a path to the folder labeled with the name of the study.  

Note that `raw.json` is just a "pretty-printed" copy of the original Beiwe configuration file.

Other files with the `.txt` extension contain human-readable summaries of the contents of the Beiwe configuration.  In these files, an attribute that is "Not found" probably belongs to a different format of configuration file.  It is normal to see "Not found" in many places.

#### 6.4 Naming
By default, `configread` assigns human-readable names to each study survey and question.  Default names look like this:  

* `Survey 01`, `Survey 02`, ...
* `Survey 01 - Question 01`, `Survey 01 - Question 02`, ...  

Note that these names are assigned in the order in which objects appear in the corresponding JSON file, and may not agree with names found on the Beiwe backend.

For convenience, it may be desirable to assign a descriptive name to each study survey and survey item.  Assigned names can be exported and reloaded for use in the future.  See `configread_example.ipynb` for an illustration of this feature.

#### 6.5 Scoring
In Beiwe questionnaires (called "tracking surveys"), responses to checkbox and radio button items are assigned a numeric score.  This score is the zero-based index of the response in the corresponding list of answers.  For example, if possible answers are `['High', 'Medium', 'Low']` then the corresponding scores are 0, 1, 2.

#### 6.6 Limitations & Cautions
1. This sub-package does not parse branching logic settings used for conditional delivery of tracking survey items.  A question's logic configuration, if any, is stored in the `logic` attribute of the corresponding  `TrackingQuestion` instance.

2. Note that only tracking surveys and audio surveys are represented by dedicated classes (`TrackingSurvey`, `AudioSurvey`).  Other survey types (e.g. image surveys) are represented with the generic `BeiweSurvey` class.  [Here](#newsurvey) are guidelines for implementing additional survey types.

3. Lastly, note that comparison of `configread` objects is intended to be somewhat flexible.  This is to accommodate the possibility that the same study configuration may be duplicated or serialized in different formats.  Therefore, some caution should be used when checking equality.


___
## 7. `beiwetools.manage` <a name="manage"/>

This sub-package provides classes and functions for managing raw Beiwe data.  These tools are intended for use when processing data locally, e.g. on a PC with data that have been downloaded from the Beiwe backend.

#### 7.1 Device attributes
A new set of "identifiers" is created whenever the Beiwe app is installed or re-installed.  These files provide a partial record of changes to each user's device, including phone model, operating system, and Beiwe app version.  Information in these files is managed by instances of the `DeviceInfo` class.

#### 7.2 Raw data registries
The `UserData` class is a tool for managing an individual user's raw Beiwe data.  This includes maintaining a registry of locations for files from each survey and data stream.

#### 7.3 Using the `BeiweProject` class
The `BeiweProject` class is intended to assist with implementation and reproducibility of Beiwe data analysis.  The main purpose of this class is to create a registry of available raw data files for a set of Beiwe users over a fixed follow-up period.  These records can then be exported and reloaded for use in the future.

A `BeiweProject` instance includes:

* `DeviceInfo` and `UserData` instances for each user,

* Optionally, one or more `BeiweConfig` instances for each user.

A `BeiweProject` instance can handle management of a single raw data directory, as well as merging of multiple raw data directories.  The latter may be useful under some circumstances:

* Raw data from the same study may have been downloaded to multiple locations corresponding to different subsets of users or to different time ranges.  It may be desirable to create a single project that pools all users and time ranges.

* Research participants may be organized into multiple arms, with smartphone data collection in each arm implemented with a different Beiwe study.  (This strategy might be used when different arms receive different surveys.)  In this case, it may be desirable to pool all users for analysis of common data streams and for preservation of blinding.

See the notebook `examples/manage_example.ipynb` for sample code for the following tasks:

* 

*

*

*

#### 7.5 Cautions
1. In the past, some raw audio data may have been delivered directly to this folder:  
`<raw data directory>/<Beiwe User ID>/audio_recordings`  
Since these audio files are not attached to a particular audio survey identifier, they will not be registered by `BeiweProject` objects.

2. Certain raw data from Android and iPhone devices are formatted differently, so researchers should use caution if data are collected for the same user ID with phones of different types.  In this unusual situation, the user's data should be carefully divided according to phone type and analyzed separately.  Such users are identified with various flags and warnings by `BeiweProject` and `UserData` objects. 

3. Researchers can use a `BeiweProject` instance to associate configuration files with a Beiwe user's raw data.  Before attaching multiple configuration files to a project, object name assignments should be manually updated to avoid confusion. Otherwise, it's likely that the same name (e.g. `Survey_01`) will be assigned to more than one object.


___
## 8. `beiwetools.localize` <a name="localize"/>

This sub-package provides tools for localizing processed Beiwe data to the time zone of the user.

The `Localize` class is used to identify timezones for timestamps, given a dictionary of a user's timezone transitions generated with the `gpsrep` package.  

The `ProcData` class provides a framework for partitioning processed data into 24-hour periods that are consistent with the user's local time.  Variables of interest can be "loaded" into a `ProcData` object as 2-D arrays.  Arrays of processed data can then be exported to text files or reshaped into a feature matrix.

Additional modules:

* `plot`: Some functions for generating simple visualizations of longitudinal data from `ProcData` objects,
* `fitabase`: Functions for loading fitabase data sets into a `ProcData` object.


___
## 9. Examples <a name="examples"/>

Public Beiwe data sets can be downloaded here:

[`https://zenodo.org/record/1188879#.XcDUyHWYW02`](https://zenodo.org/record/1188879#.XcDUyHWYW02)

The example data were collected from five different Beiwe studies.  The corresponding configuration files are located in:

`examples/configuration_files`

The following code samples (iPython notebooks) are also located in the `examples` folder:

* `configread_example.ipynb`
* `manage_example.ipynb`
* `gpsrep_summary_example.ipynb`
* `accrep_summary_example.ipynb`
* `survrep_example.ipynb`
* `localize_example.ipynb` 

___
## 10. Maintenance <a name="maintenance"/>

Future updates and improvements to the Beiwe platform may require some changes to this package.  

In general, new or undocumented study settings and study objects are flagged in several ways:

* With `logging` warnings,
* In the `warnings` attribute of a `BeiweConfig` instance,
* In the `warnings.txt` file generated by `BeiweConfig`'s `export()` method.

Note that an unknown setting can still be queried with `BeiweConfig.settings.other`, and an unknown survey will be represented as a generic `BeiweSurvey`.  Documentation will be generated in the "Other/Unknown Settings" section of `general_settings.txt` and in the `other_surveys` folder, respectively.  

Below are some guidelines for updating this package to accommodate new settings and objects.  After making changes, edit this file (`beiwetools/README.md`) with notes [here](#format) and elsewhere as needed.

#### 10.1 New settings
Study settings are organized alphabetically under several headings in `configread/study_settings.json`.  The headings are arbitrary, so a new setting key can be added to this file wherever it fits best.

Survey attributes are found in `configread/survey_settings.json`.  A new setting that is common to all survey objects can go in the first entry.  New attributes for a specific survey type should be added to the corresponding entry.

Question settings are in `configread/question_settings.json`.  As for surveys, the first entry is for shared attributes and the remaining entries are for specific types of question objects. 

#### 10.2 New tracking question type
To implement a new type of question:

1. Add an entry for the new question type here:  
`configread/question_settings.json`  
The key is the question type (as it appears in a JSON file) and the value is a list of unique attributes for that type of question.

2. Define a class for the new question that inherits from this class:   
`configread.questions.TrackingQuestion`

3. After creating the class, add it to this dictionary:  
`configread.questions.tracking_questions`

For examples, see:  
`configread.questions.RadioButton`  
`configread.questions.Slider`  

#### 10.3 New survey type <a name="newsurvey"/>
To implement a new survey type:

1. Add an entry for the new survey type here:  
`beiwetools/configread/survey_settings.json`  
The key is the survey type and the value is a list of specific attributes for the survey type.  Don't include attributes that are already listed under the key `common_survey_info`.
2. Define a class for the new survey that inherits from this class:  
`configread.surveys.BeiweSurvey`  
Re-define `__init__()` to indicate where the survey should be documented.
If necessary, re-define the following methods:  
    * `get_settings()`: Which type-specific survey settings to look for in the configuration file.
    * `get_content()`: What are the survey attributes, how should attributes be printed to a summary, and which attributes are used for object comparison.  
    * `update_names()`: How to rename the survey and any nested objects.  Only redefine this method if the new survey has named attributes (such as question objects).

3. Add the class to this dictionary:   
`configread.surveys.survey_classes`

4. Identify the output folder(s) for raw data from the new survey type and update this file:  
`/manage/data_streams.json`

5. Verify that `beiwetools.manage.BeiweProject` correctly handles registries of raw data from the new survey type.  This should not be a problem if the new survey delivers data to the expected location:  
`<raw data directory>/<Beiwe User ID>/<survey type>/<survey identifier>/`

#### 10.4 New passive data stream
Add new passive data streams here:  
`/manage/data_streams.json`

The new entry should indicate whether the data stream is specific to iPhones or Android phones.