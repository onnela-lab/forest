# Poplar  

## Usage:  
Universal information about raw data stream structures, constants, and data cleaning procedures for all Beiwe data. Includes legacy code and its dependencies. 

## Data:   
Methods can be used on all data streams.

## Installation Instruction: 
`from forest import poplar`
___
## Functions
1.  [`poplar.raw`](#1-poplarraw)  
2.  [`poplar.constants`](#2-poplarconstants)
3.  [`poplar.functions`](#3-poplarfunctions)
4.  [`poplar.classes`](#4-poplarclasses)
5.  [`poplar.legacy`](#5-poplarlegacy)
6.  [Maintenance Notes](#6-maintenance-notes)

___
## 1. `poplar.raw`

* **`poplar.raw.readers`**:  Functions for reading unencrypted raw Beiwe CSV files.  Functions are based on `pandas.read_csv` and return a cleaned `pandas.DataFrame`.  To read encrypted files, replace `pandas.read_csv` with an alternate file reader.

* **`poplar.raw.doc`**: Provides access to documentation from the following files:
	* **`data_streams.csv`**:  Table of Beiwe passive data streams, identifying which streams are available on Android and iOS devices.

	* **`headers.json`**:  Dictionary of headers found in files containing raw Beiwe data.
	
	* **`question_type_names.json`**:  Concordance for question type names.  Keys are names that are used in Beiwe configuration files.  Values give the corresponding names that appear in tracking survey files.  Raw iOS data use the same naming conventions as configuration files, but raw Android data use different names (e.g. `Open Response Question` instead of `free_response`).

	* **`power_events.json`**: Basic framework for organizing events reported in raw Bewe `power_state` files.


___
## 2. `poplar.constants`

* **`poplar.constants.time`**:  Formats of dates and times found in raw Beiwe data; conversion factors for commonly used time units. 

* **`poplar.constants.misc`**:  Miscellaneous constants used when processing raw Beiwe data.


___
## 3. `poplar.functions`

* **`poplar.functions.io`**:  Functions for reading/writing JSON files, and for writing to CSV files.  May need to be supplemented with corresponding functions for S3 buckets.

* **`poplar.functions.log`**: Functions for formatting and exporting logging messages.

* **`poplar.functions.helpers`**: Tools for common data processing tasks.

* **`poplar.functions.time`**: Functions for working with Beiwe time formats.

* **`poplar.functions.timezone`**:  Tools for extracting timezone information from GPS data.  Isolated in a separate module to avoid unnecessary imports of the `timezonefinder` package.

* **`poplar.functions.holidays`**:  Tools for identifying dates that are holidays.  Isolated in a separate module to avoid unnecessary imports of the `holidays` package.


___
## 4. `poplar.classes`

These empty modules are placeholders for future software.

* **`poplar.classes.template`**:  We anticipate the need for classes that will enable object & data persistence.  This module is a proposed location for template(s) for such classes.

* **`poplar.classes.registry`**:  Future location of identifier management tools and directory management tools.

* **`poplar.classes.history`**:  Future location of tools for handling longitudinal tracking of categorical variables  (e.g. timezones, device power states).

* **`poplar.classes.trackers`**:  Future location of tools for handling online calculation of summary statistics, such as an implementation of Welford's algorithm.


___
## 5. `poplar.legacy`
The `legacy` directory contains untested modules that may or may not be essential for other Forest subpackages.  The contents of these modules should be either (1) deleted or (2)  reviewed, updated, and moved to a suitable module elsewhere.

* **`poplar.legacy.common_funcs`**





___
## 6. Maintenance Notes

Maintenance notes are in progress!  

## References  

## Contact information for questions: 
[Email the Onnela Lab](mailto:onnela.lab@gmail.com)
