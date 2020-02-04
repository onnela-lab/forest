Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
accrep
===

The `accrep` package provides tools for processing raw Beiwe accelerometer data.  This document gives an overview of accelerometer data collection on the Beiwe platform, followed by a brief description of each module.

Note that `accrep` requires the `beiwetools` package, located in the `forest/poplar` repository.  

To install with pip:

```bash
pip install /path/to/accrep
```

Example imports:

```python
import accrep
import accrep.functions as af
from accrep.functions import gee_to_mps2
```

___
## Table of Contents
1.  [Version Notes](#version)  
2.  [Overview](#overview)  
3.  [Modules](#modules)
    * [`headers`](#headers)  
	* [`functions`](#functions)  
	* [`summary`](#summary)  
	* [`calibration`](#calibration)  
	* [`proximity`](#proximity)  
	* [`spectral`](#spectral)  
	* [`plot`](#plot)  	
4.  [Examples](#examples)  
5.  [Cautions & Notes](#cautions)  


___
## 1. Version Notes <a name="version"/>  


This is version 0.0.1 of `accrep`.  This package was developed with Python 3.8.1 on PCs running Manjaro Juhraya 18.1.5.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository.  Among other package requirements, only `astropy` is not in the Python Standard Library.

___
## 2. Overview <a name="version"/>

Accelerometer data from wearable devices have been used in the study of human behavior and physical activity for over twenty years.  The `accrep` package implements some commonly used strategies for accelerometer data processing, along with methods developed at the Onnela Lab.

The Beiwe app is configured to collect raw smartphone accelerometer data according to three study settings.  These settings are found in the study's configuration file (see `beiwetools.configread`).  The first setting is `accelerometer`, a Boolean indicating whether or not the accelerometer is sampled.  The other two settings are integers that describe the app's alternating observation pattern:

* `accelerometer_off_duration_seconds`
* `accelerometer_on_duration_seconds`

When designing a Beiwe study, researchers choose values for these two parameters in order to obtain maximum accelerometer data coverage within battery and storage constraints.  Typical values for these parameters are under one minute, e.g. 10 or 20 seconds.  Note that the resulting sensor observation periods are not synchronized to clock time.

A single triaxial accelerometer observation consists of a millisecond timestamp along with three accelerations corresponding to measurements in the *x*, *y*, and *z* axes.  The *x* and *y* measurements are parallel to the horizontal and vertical axes of the device screen when it is held in the usual orientation.  The *z* measurement is perpendicular to the screen of the device.  Note that units differ according to operating system:  Android devices report accelerations in meters/s^2 while the iPhone acceleration unit is *g*.

During accelerometer-off periods, no observations are recorded.  During accelerometer-on periods, observations are collected at a rate determined by the user's device.  Beiwe accelerometer data from iPhones are characterized by a sampling rate that is quite close to 10 Hz.  Sampling rates in Android devices are much more heterogeneous, and over time may vary by several orders of magnitude.

Accelerometer observations are collected and organized by the Beiwe backend according to standard conventions for raw data.  See `beiwetools/README.md` for more information about Beiwe time formats, file-naming conventions, and directory structure.

___
## 3. Modules <a name="modules"/>

#### `headers` <a name="headers"/>
This module contains headers for `csv` files that are handled by `accrep`, including raw Beiwe accelerometer files.  Each variable is described with a brief comment.

___
#### `functions` <a name="functions"/>
This module provides functions for loading raw Beiwe accelerometer files, along with implementations of commonly used transformations of triaxial accelerometer data, including Vector Magnitude and ENMO. Overall Dynamic Body Acceleration (Wilson et al. 2006) and the absolute and relative versions of the Activity Index (Bai et al. 2016) are not implemented; these will be provided in a subsequent version of the package.

___
#### `summary` <a name="summary"/>


___
#### `calibration` <a name="calibration"/>


___

### `proximity` <a name="proximity"/>


___
#### `spectral` <a name="spectral"/>


Tudor-Locke et al. 2019 - cutoffs of 100 and 130 steps per minute.

Hansen et al. 2017 - 
70.6  strides per minute ~ 2.35 Hz - transition from walk to run (1 stride = 2 steps)

Oberg et al. 1993
1.5 (slow gait) to 2.6 (fast gait)

Verghese et al. 2014
Neurology - motoric cognitive risk syndrome, slow gait

___
#### `plot` <a name="plot"/>

___
## 4. Examples <a name="examples"/>

___
## 5. Cautions & Notes <a name="cautions"/>

#### accuracy
accuracy variable is not handled


