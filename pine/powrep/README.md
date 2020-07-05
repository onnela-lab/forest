Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
powrep
===

The `powrep` package provides tools for processing raw Beiwe power state data.  This document gives an overview of power state data collection on the Beiwe platform, followed by a brief description of each module.

Note that `powrep` requires the `beiwetools` package, located in the `forest/poplar` repository.  

To install with pip:

```bash
pip install /path/to/powrep
```

Example imports:

```python
import powrep
import powrep.functions as powf
from powrep.classes import xx
```

___
## Table of Contents
1.  [Version Notes](#version)  
2.  [Overview](#overview)  
3.  [iOS Events](#ios)      
4.  [Android Events](#android)  
5.  [Modules](#modules)
    * [`headers`](#headers)      
	* [`functions`](#functions)  
	* [`classes`](#classes)  
	* [`summary`](#summary)  
	* [`plot`](#plot)  	
6.  [Examples](#examples)  
7.  [Cautions & Notes](#cautions)  


___
## 1. Version Notes <a name="version"/>  

This is version 0.0.1 of `powrep`.  This package was developed with Python 3.8.1 on PCs running Manjaro Juhraya 18.1.5.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository.

___
## 2. Overview <a name="version"/>

The iOS and Android versions of the Beiwe app can be configured to collect various operating system messages related to a phone's power state.  These are collected and organized by the Beiwe backend according to standard conventions for raw data.  However, note that the raw file format and contents differ between iOS and Android.  The following two sections provide details.

See `beiwetools/README.md` for more information about Beiwe time formats, file-naming conventions, and directory structure.

___
## 3. iOS Events <a name="ios"/>


On Apple devices, the Beiwe `power_state` data stream is handled by:

`beiwe-ios/Beiwe/Managers/PowerStateManager.swift`




https://developer.apple.com/documentation/uikit/uidevice/1620051-batterystate?language=objc
https://developer.apple.com/documentation/uikit/uidevice/1620042-batterylevel?language=objc

iPhone lock/unlock logs 
https://developer.apple.com/documentation/uikit/uiapplicationdelegate/1623019-applicationprotecteddatawillbeco
https://developer.apple.com/documentation/uikit/uiapplicationdelegate/1623044-applicationprotecteddatadidbecom

___
## 4. Android Events <a name="android"/>

On Android phones, the Beiwe `power_state` data stream reports operating system ["intents"](https://developer.android.com/reference/android/content/Intent) corresponding to transitions in the device's interactive state and power states.  Reporting is handled by an instance of [this class](`beiwe-android/app/src/main/java/org/beiwe/app/listeners/PowerStateListener.java`).  

Transitions between power states are handled by Android's [`DeviceIdleController`](https://github.com/aosp-mirror/platform_frameworks_base/blob/nougat-release/services/core/java/com/android/server/DeviceIdleController.java).
[This article](https://medium.com/@tsungi/android-doze-tweaks-83dadb5b4a9a) provides an informal overview, with state diagrams.
___
#### Interactive State

**Note:  The names of these events and intents are misleading.  These events actually refer to whether the phone is in an "interactive" state.  Earlier in the history of Android, this was the same as the screen power state.  However, as of Android xx, the screen may be powered on or off regardless of whether the phone is interactive.**


| **Beiwe Event** | **Android Intent**|
|-----------|-----------|
|`Screen turned off` | [ACTION\_SCREEN\_OFF](https://developer.android.com/reference/android/content/Intent.html#ACTION\_SCREEN\_OFF)|
|`Screen turned on` | [ACTION\_SCREEN\_ON](https://developer.android.com/reference/android/content/Intent.html#ACTION\_SCREEN\_ON)|

https://developer.android.com/reference/android/os/PowerManager.html#isInteractive()



___
#### External Power

| **Beiwe Event** | **Android Intent**|
|-----------|-----------|
|`Power connected` | [ACTION\_POWER\_CONNECTED]()|
|`Power disconnected` | [ACTION\_POWER\_DISCONNECTED]()|

___
#### Power Save

[ACTION\_POWER\_SAVE\_MODE\_CHANGED]()

[isPowerSaveMode()]()



| **Beiwe Event** | ****|
|-----------|-----------|
|`Power Save Mode state change signal received; device in power save state.`||
|`Power Save Mode change signal received; device not in power save state.`||


___
#### Doze

Doze was originally introduced in [Android 6.0](https://developer.android.com/about/versions/marshmallow/android-6.0-changes).  This original implementation appears in all subsequent versions of Android, and is sometimes referred to as "Deep Doze."  It is a power-saving state initiated when these three conditions have been met for some period of time:

1. The device is unplugged,
2. The device is stationary,
3. The screen is off.

A second, less restrictive version of Doze has been implemented since [Android 7.0](https://developer.android.com/about/versions/nougat/android-7.0-changes).  This state is sometimes called "Light Doze," and it may be initiated when only conditions (1) and (3) are met.  For example, a phone may enter Light Doze if it is carried in a pocket while the user is walking.


[ACTION\_DEVICE\_IDLE\_MODE\_CHANGED]()

[isDeviceIdleMode()]()

| **Beiwe Event** | ****|
|-----------|-----------|
|`Device Idle (Doze) state change signal received; device in idle state.`||
|`Device Idle (Doze) state change signal received; device not in idle state.`||




___
#### Device Turned Off

| **Beiwe Event** | **Android Intent**|
|-----------|-----------|
|`Device shut down signal received`| [ACTION\_SHUTDOWN]()|
|`Device reboot signal received`| [ACTION\_REBOOT]()|









___
## 5. Modules <a name="modules"/>

#### `headers` <a name="headers"/>
This module contains headers for `csv` files that are handled by `powrep`, including raw Beiwe power state files.  Each variable is described with a brief comment.

___
#### `functions` <a name="functions"/>

___
#### `classes` <a name="classes"/>

___
#### `summary` <a name="summary"/>

___
#### `plot` <a name="plot"/>

___
## 6. Examples <a name="examples"/>

___
## 7. Cautions & Notes <a name="cautions"/>

For both iPhone and Android, `power_state` data are relatively sparse (xx approx MB per month).  Storage and processing is comparably cheap.  Therefore, the code in this package does not implement any special strategies for optimization or memory management.

