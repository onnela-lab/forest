Josh Barback  
barback@fas.harvard.edu  
Onnela Lab, Harvard T. H. Chan School of Public Health

___
powrep
===

The `powrep` package provides some limited tools for processing raw Beiwe power state data.  This document gives an overview of power state data collection on the Beiwe platform, links to Apple and Android documentation, and a brief description of each module in the package.

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
3.  [iOS Power State Events](#ios)      
4.  [Android Power State Events](#android)  
5.  [Modules](#modules)
    * [`headers`](#headers)      
	* [`functions`](#functions)  
	* [`summary`](#summary)  
	* [`extract`](#extract)  
6.  [Examples](#examples)  
7.  [Cautions & Notes](#cautions)  

___
## 1. Version Notes <a name="version"/>  

This is version 0.0.1 of `powrep`.  This package was developed with Python 3.8.1 on PCs running Manjaro Juhraya 18.1.5.

This package requires the `beiwetools` package, which can be found in the `forest/poplar` repository.

___
## 2. Overview <a name="version"/>

The iOS and Android versions of the Beiwe app can be configured to collect various operating system messages related to a phone's power state.  These are collected and organized by the Beiwe backend according to the usual conventions for raw Beiwe data.  However, note that the raw file format and contents differ between iOS and Android.  The following two sections provide details.

See `beiwetools/README.md` for more information about Beiwe time formats, file-naming conventions, and directory structure.

___
## 3. iOS Power State Events <a name="ios"/>


On Apple devices, the Beiwe `power_state` data stream is handled by an instance of [this class](https://github.com/onnela-lab/beiwe-ios/blob/master/Beiwe/Managers/PowerStateManager.swift).  There are six iOS power state events, documented below.  These events correspond to object properties and notifications from Apple's [UIKit](https://developer.apple.com/documentation/uikit).

In addition to a timestamp, date-time, and event description, each observation also includes a measure of the device's [battery level](https://developer.apple.com/documentation/uikit/uidevice/1620042-batterylevel?language=objc), which is between  0.0 (empty) and 1.0 (fully charged).

___
#### Availability of Encrypted Data

Encrypted iPhone files may be available or unavailable according to whether the device is unlocked or locked.  The `Unlocked` and `Locked` power state events correspond to transitions between these two states; see Apple's UIKit documentation for details.
		

| **Beiwe Event** | **UIKit Notification**|
|-----------|-----------|
|`Unlocked` | [applicationProtectedDataDidBecomeAvailable(_:)](https://developer.apple.com/documentation/uikit/uiapplicationdelegate/1623044-applicationprotecteddatadidbecom)|
|`Locked` | [applicationProtectedDataWillBecomeUnavailable(_:)](https://developer.apple.com/documentation/uikit/uiapplicationdelegate/1623019-applicationprotecteddatawillbeco)|


___
#### Battery State

Battery state events correspond to values of the [`batteryState`](https://developer.apple.com/documentation/uikit/uidevice/1620051-batterystate?language=objc) property of a [`UIDevice`](https://developer.apple.com/documentation/uikit/uidevice?language=objc) object.  There are four possible events; see Apple's UIKit documentation for details.

| **Beiwe Event** | **UIDevice Property**|
|-----------|-----------|
|`Unknown` | [UIDeviceBatteryStateUnknown](https://developer.apple.com/documentation/uikit/uidevicebatterystate/uidevicebatterystateunknown?language=objc)|
|`Unplugged` | [UIDeviceBatteryStateUnplugged](https://developer.apple.com/documentation/uikit/uidevicebatterystate/uidevicebatterystateunplugged?language=objc	)|
|`Charging` | [UIDeviceBatteryStateCharging](https://developer.apple.com/documentation/uikit/uidevicebatterystate/uidevicebatterystatecharging?language=objc)|
|`Full` | [UIDeviceBatteryStateFull](https://developer.apple.com/documentation/uikit/uidevicebatterystate/uidevicebatterystatefull?language=objc)|



___
## 4. Android Power State Events <a name="android"/>

On Android phones, the Beiwe `power_state` data stream reports operating system ["intents"](https://developer.android.com/reference/android/content/Intent) corresponding to transitions in the device's interactive state and power states.  Reporting is handled by an instance of [this class](`https://github.com/onnela-lab/beiwe-android/blob/master/app/src/main/java/org/beiwe/app/listeners/PowerStateListener.java`).  

Below are brief descriptions of the ten Android power state events; refer to Android documentation for details.

___
#### Interactive State

**Important note:  The names of these events and intents are misleading.  These events actually refer to whether the phone is transitioning to or from an ["interactive"](https://developer.android.com/reference/android/os/PowerManager#isInteractive\(\)) state.  Earlier in the history of Android, this was the same as the screen power state.  However, for recent Android versions, the screen may be powered on or off regardless of whether the phone is interactive.**


| **Beiwe Event** | **Android Intent**|
|-----------|-----------|
|`Screen turned off` | [ACTION\_SCREEN\_OFF](https://developer.android.com/reference/android/content/Intent.html#ACTION\_SCREEN\_OFF)|
|`Screen turned on` | [ACTION\_SCREEN\_ON](https://developer.android.com/reference/android/content/Intent.html#ACTION\_SCREEN\_ON)|


___
#### External Power

These events are logged when the device is connected or disconnected from an external power source.

| **Beiwe Event** | **Android Intent**|
|-----------|-----------|
|`Power connected` | [ACTION\_POWER\_CONNECTED](https://developer.android.com/reference/android/content/Intent.html#ACTION_POWER_CONNECTED)|
|`Power disconnected` | [ACTION\_POWER\_DISCONNECTED](https://developer.android.com/reference/android/content/Intent.html#ACTION_POWER_DISCONNECTED)|

___
#### Shut Down & Reboot

These events are logged when the device prepares to shut down or reboot.


| **Beiwe Event** | **Android Intent**|
|-----------|-----------|
|`Device shut down signal received`| [ACTION\_SHUTDOWN](https://developer.android.com/reference/android/content/Intent.html#ACTION_SHUTDOWN)|
|`Device reboot signal received`| [ACTION\_REBOOT](https://developer.android.com/reference/android/content/Intent.html#ACTION_REBOOT)|

___
#### Power Save Mode

Android's [PowerManager](https://developer.android.com/reference/android/os/PowerManager) was introduced in version 5.0; this enables the device to enter a low-power, battery-conserving state.  Transitions to and from Android's ["power save mode"](https://developer.android.com/reference/android/os/PowerManager#isPowerSaveMode\(\)) are signaled by the same intent: [ACTION\_POWER\_SAVE\_MODE\_CHANGED](https://developer.android.com/reference/android/os/PowerManager#ACTION_POWER_SAVE_MODE_CHANGED).


| **Beiwe Event** |
|-----------|
|`Power Save Mode state change signal received; device in power save state.`|
|`Power Save Mode change signal received; device not in power save state.`|

___
#### Doze (Idle Mode)

The [Doze feature](https://developer.android.com/reference/android/os/PowerManager#isDeviceIdleMode\(\)) was originally introduced in [Android 6.0](https://developer.android.com/about/versions/marshmallow/android-6.0-changes).  The original implementation appears in all subsequent versions of Android, and is sometimes referred to as "Deep Doze."  It is a power-saving state initiated when three conditions have been met for some period of time:

1. The device is unplugged,
2. The device is stationary,
3. The screen is off.

A second, less restrictive version of Doze has been implemented since [Android 7.0](https://developer.android.com/about/versions/nougat/android-7.0-changes).  This state is sometimes called "Light Doze," and it may be initiated when only conditions (1) and (3) are met.  For example, a phone may enter Light Doze if it is carried in a pocket while the user is walking.


Transitions among these power states are handled by Android's [`DeviceIdleController`](https://github.com/aosp-mirror/platform_frameworks_base/blob/nougat-release/services/core/java/com/android/server/DeviceIdleController.java).
[This article](https://medium.com/@tsungi/android-doze-tweaks-83dadb5b4a9a) provides an informal overview, with state diagrams.  

Transitions in and out of Doze are signaled by the same intent: [ACTION\_DEVICE\_IDLE\_MODE\_CHANGED](https://developer.android.com/reference/android/os/PowerManager#ACTION_DEVICE_IDLE_MODE_CHANGED).  Beiwe power state events do not distinguish between Deep Doze and Light Doze; however, it may be possible to infer the correct level of Doze based on recent sensor observations.


| **Beiwe Event** |
|-----------|
|`Device Idle (Doze) state change signal received; device in idle state.`|
|`Device Idle (Doze) state change signal received; device not in idle state.`|

___
## 5. Modules <a name="modules"/>

#### `headers` <a name="headers"/>
This module contains headers for `csv` files that are handled by `powrep`, including raw Beiwe power state files.  Each variable is described with a brief comment.

___
#### `functions` <a name="functions"/>

___
#### `summary` <a name="summary"/>

___
#### `extract` <a name="extract"/>


___
## 6. Examples <a name="examples"/>

The script powrep/examples/powrep_example.py provides a sample workflow for summarizing `power_state` data and extracting iOS events.
___
## 7. Cautions & Notes <a name="cautions"/>

For both iPhone and Android, `power_state` data are relatively sparse.  Storage and processing is comparably cheap.  Therefore, the code in this package does not implement any special strategies for optimization or memory management.

