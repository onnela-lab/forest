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
## 1. Version Notes <a name="version"/>

This is version 0.0.1 of `gpsrep`.  This package was developed with Python 3.7.4 on PCs running Manjaro Illyria 18.0.4.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository. 

___
## 2. Overview <a name="overview"/>


___
## 3. Modules <a name="modules"/>


#### `headers` <a name="headers"/>
This module contains headers for `csv` files that are handled by `gpsrep`, including raw Beiwe GPS files.  Each variable is described with a brief comment.

___
#### `functions` <a name="functions"/>
This module provides functions for loading raw Beiwe GPS files.




___
## 4. Examples <a name="examples"/>

___
## 5. Cautions & Notes <a name="cautions"/>







