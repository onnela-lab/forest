# Naming conventions

## 1. Introduction

The purpose of this document is to establish best practices for Forest development.  **This document is a work-in-progress.  Changes, edits, and updates are welcome.**
 
General guidelines for Python naming conventions are found in [PEP 8](https://www.python.org/dev/peps/pep-0008/).  For some additional perspectives, see:

* [Good coding practices - Selecting “good” variable names.](https://geo-python.github.io/site/notebooks/L1/gcp-1-variable-naming.html)
* [Data Scientists: Your Variable Names Are Awful. Here’s How to Fix Them.](https://towardsdatascience.com/data-scientists-your-variable-names-are-awful-heres-how-to-fix-them-89053d2855be)
* [How To Write Unmaintainable Code.](https://cs.fit.edu/~kgallagher/Schtick/How%20To%20Write%20Unmaintainable%20Code.html)

For legibility & convenience, we'd like to establish a common naming framework for use in Forest modules.  Our goals are to:

* Enhance human-readability of Forest modules,
* Simplify the process of writing & reviewing code,
* Create the basis for a user-friendly API,
* Clarify the mechanics of the Beiwe platform with accurate documentation,
* Respect conventions established by the developers of Beiwe's apps and backend, whenever possible.

Because we expect Forest to be implemented by diverse end-users, we should also:

* Avoid overly-prescriptive names,
* Avoid names that may conflict with typical use-cases,
* Respect the conventions of diverse research cultures.

## 2. Usage Guidelines

### Naming Collections

We'll often use Python data structures to collect multiple objects of the same type.  The naming convention for a specific type of object can be extended to a name for a collection.  There are two reasonable strategies:

* **Pluralize**:  For example, refer to a list of timestamp objects as `timestamps`.  
* **Descriptive suffix**: For example, refer to a list or array of timestamp objects as `timestamp_list` or `timestamp_array`, respectively.

___
### Long & Short Variable Names
In some cases it's useful to have both a long and short variable name for a particular type of object.

The long name is generally preferable for function arguments.  For example:

```
def some_function(question_id, **kwargs):
	.
	.
	.
```

For control variables, it's sometimes easier to work with a short variable name.  For example:

```
for qid in question_ids:
	.
	.
	.
```

### Recycling Names

We suggest recycling variable names whenever it does not create confusion.  When possible, use consistent naming conventions for variables, keywords, headers, file names, etc.  For example, `question_id` may be used as a variable name, a key in a `dict` or `OrderedDict`, the column label in a `pandas.DataFrame`, and so on.

## 3. Organizing Raw Data

### Overview

The Beiwe platform enables collection of raw smartphone data from multiple sources and multiple individuals.  Several identifiers and keywords are used to organize and attribute these data.

### Variable Names

|**Long Name** | **Short Name** | **Data Type** | **Details**|
|----|----|----|----|
|`backend_id` | `bid` | string | A short alphanumeric string (~8 characters).  Referred to as a `patient_id` on the Beiwe backend.  The basic identifier for organizing raw Beiwe data.  Note that a single individual may contribute raw data under multiple `backend_ids`.  Also, note that a single `backend_id` may correspond to data from multiple devices. |
|`person_id`| `pid` | string | An identifier from a non-Beiwe framework, such as the end-user's identification system.  A single `person_id` corresponds to exactly one individual. Raw data for a `person_id` may be bundled under one or more `backend_ids`.|
|`data_stream`| `stream` | string | A name for a Beiwe data stream, e.g. `gyro` or `survey_answers`.  Note that some data streams are available only on iPhones (e.g. `magnetometer`) or Android phones (e.g. `texts`).  See `data_streams.json` for details. |

## 4. Directories and Files

### Overview

Raw Beiwe data are organized into a specific directory structure.  The output of Forest modules may also be delivered to directories with a common structure.  In general, the terminology for file names and directory names is not standardized across operating systems.  Therefore, it's useful for Forest developers to agree on a common framework for referring to these locations.

Also, it's important to note that conventions for path syntax differ across operating systems.  Some of these discrepancies are handled natively by Python.  However, the best practice is to use either the `os` package or the `pathlib` package for all path manipulation tasks.   For example:

|| Python Code|
|----|----|
|**Not safe:**|`filepath = dirpath + '/' + filename`
|**Safe:**|`filepath = os.path.join(dirpath, filename)`|
|**Safe:**|`filepath = pathlib.Path(dirpath)/filename`|

### Variable Names

|**Long Name** | **Short Name** | **Data Type** | **Details**|
|----|----|----|----|
|`filename`| `f`, `fn`| string |  The base name of a file path, including the file extension, e.g. `data.csv`.
|`filepath`| `f`, `fp`| string | The full path to a file, e.g. `absolute/path/to/data.csv`.
|`dirname`| `d`, `dn`| string | The name of a directory or folder, e.g. `data`.
|`dirpath`| `d`, `dp`| string | The full path to a directory, e.g. `absolute/path/to/data`.

### Special Paths

For absolute paths to special directories and files, use the suffixes `_dir` and `_file`, respectively.  For example:

|**Name** | **Description**|
|----|----|
|`raw_dir`| Full path to a directory containing raw data, which are organized into folders named after backend identifiers.|
|`log_dir`| Full path to an output directory where log records should be written.|
|`config_file` | Full path to a study configuration file in JSON format.|

## 5. Time

### Overview

Several different time formats appear in raw Beiwe data and configuration files.  Additional time formats may be used in output from Forest modules.

### Variable Names

For time formats found on the Beiwe platform:

|**Long Name** | **Short Name** | **Data Type** | **Details**|
|----|----|----|----|
`filename_datetime`|`fdt`| string | A UTC datetime string formatted as `%Y-%m-%d %H_%M_%S`, as found in the names of files containing raw Beiwe data.|
|`data_datetime`| `ddt` | string | A UTC datetime string formatted as `%Y-%m-%dT%H:%M:%S.%f`, as found in the `UTC time` column of raw Beiwe files.|
|`timestamp` | `t`| integer | A millisecond timestamp, as found in the `timestamp` column of raw Beiwe files. The number of elapsed milliseconds since the Unix epoch. |

In some cases, it may be necessary to work with isolated dates or times without specifying a timezone or UTC offset.  If necessary, use the following prefixes for clarification:

|**Prefix** | **Description**|
|----|----|
|`utc_`| To identify a date or time in UTC.|
|`local_`|To identify a date or time that has been localized to the user's timezone. |

### Special Times

When processing raw Beiwe data, it's often necessary to refer to specific times.  Use the following prefixes:

|**Prefix** | **Description**|
|----|----|
|`start_`, `end_`|To identify a specific time period, e.g. a particular followup duration or a particular window of observations. |
|`first_`, `last_`| To identify the first and last observations in a data set. |

## 6. Configuration Settings

### Overview

Beiwe configuration parameters determine app behavior, including collection of raw sensor data and delivery of surveys.  These parameters are organized using keywords and identifiers.

### Variable Names

|**Long Name** | **Short Name** | **Data Type** | **Details**|
|----|----|----|----|
|`config_name`|| string | The Beiwe backend's human-readable name for the configuration.  Note that this name does not appear in a configuration file, but may appear in the name of a configuration file. |
|`config_id`|| string or integer | The hex string that uniquely identifies the configuration.  Note that some configuration files may disagree with the backend's identifier. |
|`survey_id`| `sid` | string | A hex string that uniquely identifies each Beiwe survey.|
|`question_id`|`qid`| string | A string that uniquely identifies each tracking survey question.  The format of this identifier is five hex strings separated by dashes.|

## 7. Device Parameters

### Overview

Information about a participant's smartphone is collected whenever the Beiwe app is installed.  This information appears in each participant's `identifiers` data stream directory.

### Variable Names

Variable names for device parameters are taken from the header of `identifiers` CSV files, for example:

|**Long Name** | **Short Name** | **Data Type** | **Details**|
|----|----|----|----|
|`device_os` | | string |   The operating system of a smartphone.  Either `iOS` or `Android`.
|`os_version`|| string | The version number of the operating system. |

## 8. Packages & Modules

## 9. Functions & Classes

## 10. Summary statistics

- snake case
- descriptive
- check if it doesn't already exist
