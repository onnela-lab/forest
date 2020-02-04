Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
gpsrep
===

The `gpsrep` package provides tools for some basic processing tasks involving raw Beiwe GPS data.  This document gives a brief description of each module.

Note that `gpsrep` requires the `beiwetools` package, located in the `forest/poplar` repository.  

To install with pip:

```bash
pip install /path/to/gpsrep
```

Example imports:

```python
import gpsrep
import gpsrep.functions as gf
from gpsrep.functions import read_gps
```


___
## Table of Contents
1.  [Version Notes](#version)  
2.  [Overview](#overview)  
3.  [Modules](#modules)
    * [`headers`](#headers)  
	* [`functions`](#functions)  
4.  [Examples](#examples)  
5.  [Cautions & Notes](#cautions)  

___
<a name="version"/>
## 1. Version Notes

This is version 0.0.1 of `gpsrep`.  This package was developed with Python 3.7.4 on PCs running Manjaro Illyria 18.0.4.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository. 

___
<a name="overview"/>
## 2. Overview


___
<a name="modules"/>
## 3. Modules

<a name="headers"/>
### `headers`
This module contains headers for `csv` files that are handled by `gpsrep`, including raw Beiwe GPS files.  Each variable is described with a brief comment.

___
<a name="functions"/>
### `functions`
This module provides functions for loading raw Beiwe GPS files.




___
<a name="examples"/>
## 4. Examples

___
<a name="cautions"/>
## 5. Cautions & Notes







