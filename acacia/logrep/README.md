Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
logrep
===

The `logrep` package provides tools for processing raw Beiwe app log data from Android devices.  This document gives an overview of app log data collection on the Beiwe platform, followed by a brief description of each module.

Note that `logrep` requires the `beiwetools` package, located in the `forest/poplar` repository.  

To install with pip:

```bash
pip install /path/to/logrep
```

Example imports:

```python
import logrep
import logrep.functions as lf
from logrep.classes import xx
```

___
## Table of Contents
1.  [Version Notes](#version)  
2.  [Overview](#overview)  
3.  [Modules](#modules)
    * [`headers`](#headers)      
	* [`functions`](#functions)  
	* [`classes`](#classes)  
	* [`summary`](#summary)  
	* [`plot`](#plot)  	
4.  [Examples](#examples)  
5.  [Cautions & Notes](#cautions)  


___
## 1. Version Notes <a name="version"/>  


This is version 0.0.1 of `logrep`.  This package was developed with Python 3.8.1 on PCs running Manjaro Juhraya 18.1.5.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository.

___
## 2. Overview <a name="version"/>

Android app log data are collected and organized by the Beiwe backend according to standard conventions for raw data.  See `beiwetools/README.md` for more information about Beiwe time formats, file-naming conventions, and directory structure.








___
## 3. Modules <a name="modules"/>

#### `headers` <a name="headers"/>
This module contains headers for `csv` files that are handled by `logrep`, including raw Beiwe app log files.  Each variable is described with a brief comment.

___
#### `functions` <a name="functions"/>

___
#### `classes` <a name="classes"/>

___
#### `summary` <a name="summary"/>

___
#### `plot` <a name="plot"/>

___
## 4. Examples <a name="examples"/>

___
## 5. Cautions & Notes <a name="cautions"/>



