# Poplar example code

## Table of Contents
1.  [`poplar.raw`](#1-poplarraw)  
2.  [`poplar.constants`](#2-poplarconstants)
3.  [`poplar.functions`](#3-poplarfunctions)
4.  [`poplar.classes`](#4-poplarclasses)
5.  [`poplar.legacy`](#5-poplarlegacy)

## 1. `poplar.raw`

### **`poplar.raw.readers`**

### **`poplar.raw.doc`**
This module provides access to documentation of basic features of the Beiwe platform.  Documentation can be imported as dictionaries, for example:
```
from poplar.raw.doc import HEADERS
HEADERS[][]
>>>
```
Alternatively, documentation can be accessed in `CSV` or `JSON` format:
```
from poplar.raw.doc import DOCPATHS
print(DOCPATHS.keys())
>>>
DOCPATHS['headers.json']
>>>
```

___
## 2. `poplar.constants`

### **`poplar.constants.time`**

Formats of dates and times found in raw Beiwe data; conversion factors for commonly used time units. 

### **`poplar.constants.misc`**

Miscellaneous constants used when processing raw Beiwe data.


___
## 3. `poplar.functions`

### **`poplar.functions.io`**

Functions for reading/writing JSON files, and for writing to CSV files.  May need to be supplemented with corresponding functions for S3 buckets.

### **`poplar.functions.log`**

Functions for formatting and exporting logging messages.

### **`poplar.functions.helpers`**

Tools for common data processing tasks.

### **`poplar.functions.time`**

Functions for working with Beiwe time formats.

### **`poplar.functions.timezone`**

Tools for extracting timezone information from GPS data.  Isolated in a separate module to avoid unnecessary imports of the `timezonefinder` package.

### **`poplar.functions.holidays`**

Tools for identifying dates that are holidays.  Isolated in a separate module to avoid unnecessary imports of the `holidays` package.


___
## 4. `poplar.classes`

These empty modules are placeholders for future software.

### **`poplar.classes.template`**

### **`poplar.classes.registry`**

### **`poplar.classes.history`**

### **`poplar.classes.trackers`**


## 5. `poplar.legacy`

### **`poplar.legacy.common_funcs`**
