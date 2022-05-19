# Passive Data

This Wiki Has Moved and this page is probably out of date. You can find the new wiki page here: [Updated Passive-Data](https://github.com/onnela-lab/beiwe-backend/wiki/[Researchers]-Passive-Data)



## Accelerometer Data
The app records the phone's accelerometer data, which indicate movement of the phone and can be used as an estimate of how long a participant sat still, when a participant was walking, how many steps the participant took while walking, etc.
```{list-table}
:header-rows: 1
* - timestamp
  - UTC time
  - accuracy
  - x
  - y
  - z
* - 1.5E+12
  - 2017-06-01T23:07:13.940
  - unknown
  -  -0.03641
  -  -0.60576
  -  -0.78812
```

## GPS Data
The Beiwe app can record the phone's GPS location in latitude and longitude, as well as the precision of that measure. The GPS is often accurate to within about 10-20 meters. It can be used to construct a map of where a participant traveled and when a participant was at different places, although it cannot identify the mode of travel. The rate of GPS sampling is customizable to each study, or GPS sampling can be disabled for a particular study if GPS data is not part of the research questions/goals.
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - latitude
  - longitude
  - altitude
  - accuracy
* - 1.5E+12
  - 2017-06-15T14:15:18.675
  - 43.33558
  -  -74.1021
  -  -4.3
  - 19.669
```

## Power State
Apple iOS: Beiwe reports when the participant’s phone screen is locked/unlocked and if the phone is charging, unplugged, the percentage level of available battery, when the battery is full or if there is an unknown charging state.

Android: Beiwe reports screen on/off. Researchers can use Power State data to infer phone usage and if missing data is related to a depleted battery.

### Android Power State
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - event
* - 1.5E+12
  - 2017-06-16T13:22:10.642
  - Screen turned on
```

### iOS Power State
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - event
  - level
* - 1.5E+12
  - 2017-06-05T02:00:00.321
  - Unlocked
  - 0.68
* - 1.5E+12
  - 2017-06-05T02:34:02.662
  - Unplugged
  - 0.67
* - 1.5E+12
  - 2017-06-05T02:36:45.742
  - Locked
  - 0.67
```

## Phone/Screen Usage Data
The app records when the participant turns on or off the phone's screen. For iOS, these events are referred to as the phone being locked/unlocked and for Android it’s screen on/off in the Power State data stream. Tracking when the screen is on provides a proxy for when the participant is using his/her phone. For example, if a participant wakes up at 3:44am, checks his/her phone for 10 seconds, and then goes back to sleep, Beiwe will record that the phone screen turned on for 10 seconds at 3:44am.

## Identifiers
The identifiers file is created when a study participant enrolls in a study. The file only appears in the data on the day of enrollment. If for any reason a study participant un-enrolls/re-enrolls in the study (ie they get a new phone) a new identifiers file will be created. This file includes the patient_id, phone number and MAC (if available), device OS type and version, phone manufacturer and model and Beiwe version.
### Updated Android Identifier File
```{list-table} 
:header-rows: 1

* - timestamp
  - UTC time
  - patient_id
  - MAC
  - phone_number
  - device_id
  - device_os
  - os_version
  - product
  - brand
  - hardware_id
  - manufacturer
  - model
* - 2021-04-08T21:12:31.000
  - xsivxe9u
  - -uJZmeWcsTkmCTWBjmqggyWVeWAAK2d5SYJSE7Et8LMhRAim1KCd_7CjF4j6esZf2tEuwrKea5x9T8C0FLSe-w==
  - iHTAfHe9wxdjJnBaKb7T4LvGseyZw8nYX9tqCczU15FH0ouYbQTyGNoLV5FZ5WR-SeO6xXLWAucXBeEvhdBbWQ==
  - 0o3HT7OFSgicoFKFvX8EOW2U-fDrLBhoszZ68aK1qTk=
  - Android
  - 9
  - cv1s_g
  - lge
  - cv1s
  - LGE
  - LM-X220
  - googlePlayStore-3.0.3
```

### Android Identifier File
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - patient_id
  - MAC
  - phone_number
  - device_id
  - device_os
  - os_version
  - product
  - brand
  - hardware_id
  - manufacturer
  - model
  - beiwe_version
* - 1.5E+12
  - 2017-06-15T14:14:57.000
  - 43l6b8dp
  - jGR0bVANoSACCD0IynXHyHM5p
  - TS_8prhqsgtr57hDjCUEjIiJY=
  - _xjeWARoRevoDoL9OK=
  - Android
  - 5.0.1
  - jfltevzw
  - Verizon
  - qcom
  - samsung
  - SCH-I545
  - 16
```
### iOS Identifier File
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - patient_id
  - MAC
  - phone_number
  - device_id
  - device_os
  - os_version
  - product
  - brand
  - hardware_id
  - manufacturer
  - model
  - beiwe_version
* - 1.49E+12
  - 2017-05-15T17:00:34.000
  - ggsj4e7e
  - none
  - NOT_SUPPLIED
  - EE6KJHN19EB6-EAFB-4241-9EFD-A17B49
  - iOS
  - 10.3.1
  - iPhone
  - apple
  - none
  - apple
  - iPhone8,1
  - 1.2.0.12
```

## Wi-Fi Router Data (Android Only)
The Beiwe app can record a list of all Wi-Fi routers with which the phone can communicate, and the signal strength of each of those routers.ii This serves as a proxy for location; when the phone has a very strong, clear signal from a Wi-Fi router, it is probably in the same room as that router. The Wi-Fi scans cover both the 2.4 GHz and the 5 GHz frequencies. The Beiwe app records the hashed MAC address of the Wi-Fi router (using the industry-standard SHA-256 hashing algorithm), described in section 1.4.

Hashed MAC addresses of Wi-Fi routers are necessary to collect in certain pilot studies (can be disabled for any study if this is not part of the scientific question) because we are interested in indoor mobility of patients (such as those recovering from surgery). For some patient groups, it is of interest to monitor movement inside buildings, and GPS is not reliable to monitor indoor movement.

The MAC addresses are hashed, so it is impossible to work backward from the hashed MAC address to the real address. Wi-Fi routers change in signal strength with mobility patterns, so the collection of Wi-Fi signals allows researchers to learn about '''changes in a person’s movement in a building,''' as opposed to the exact location within the building. From our testing at HSPH, hundreds of Wi-Fi signals are available in the Longwood area. One of the goals of the Beiwe Research Platform is to determine the clinical relevance of certain data collected from smartphones, so it is necessary to collect to ultimately analyze a person’s indoor mobility to determine if this may be clinically important for specific studies. If indoor mobility is not a critical part of the research question for that study, '''the collection of this data stream can be disabled prior to the start of the study''' within the Beiwe Research Administration website by unchecking a box in that specific study’s device settings section.
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - hashed MAC
  - frequency
  - RSSI
* - 1.5E+12
  - 2017-06-15T17:00:21.464
  - MNMdDDePZ5now3vP4ZtDZFvhhjVbz4l4kQWTklr-TDo=
  - 2437
  -  -72
```

## Bluetooth Data (Android Only)
In some instances, the Beiwe app can record the hashed MAC address of nearby Bluetooth devices. Because the Bluetooth MAC address is hashed, it is not possible to learn the identities of devices from the Bluetooth data (see Data Anonymity Section of this document). Beiwe can record Bluetooth in the standard 2.4 to 2.485 GHz frequency band. Due to limitations imposed in Bluetooth on Android 6+, Beiwe will no longer report the device's true MAC address although some devices will continue to report it anyway. Instead, all devices now return the same default address when queried for the MAC address in the Identifiers data stream. When possible, Beiwe will report the correct MAC address for the Identifiers data stream.

## Phone Call Log Data (Android Only)
Beiwe records metadata on all incoming, outgoing, and missed calls to and from the phone. It records the time of each call, the length of the call in seconds, and the '''hash of the phone number''' on the other end. It does not record the audio of the calls, and does not record the identity (or actual phone number) of the person called.
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - hashed phone number
  - call type
  - duration in seconds
* - 1.5E+12
  - 2017-07-03T16:54:40.961
  - tE6XkkmChr4Md3mypsZ1wqGMrndhDK4mGqs0kPrfXE4=
  - Incoming Call
  - 489
```

## Text Message Log Data (Android Only)
Beiwe records metadata on all text messages sent from and received by the phone. It records the time each message was sent or received, the length of the message (in number of characters), and the '''hash of the phone number''' the message was to or from. It does not record the content of text messages, and does not record the identity (or actual phone number) of the person called.
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - hashed phone number
  - sent vs received
  - message length
  - time sent
* - 1.5E+12
  - 2017-06-15T14:19:03.843
  - 3WqwQJ9SBvSQaq8jmVLAQy9n4YryXfFLJFLJhY2WZ0k=
  - sent SMS
  - 18
  - 
* - 1.5E+12
  - 2017-06-15T14:19:04.979
  - 3WqwQJ9SBvSQaq8jmVLAQy9n4YryXfFLJFLJhY2WZ0k=
  - sent SMS
  - 18
  - 
```

## Proximity (Apple iOS Only)
Beiwe reports proximity from iOS phones. Proximity tells when the device is "near" the user, specifically intended to indicate the phone thinks it is near or not near the users’ ear. This information could be used together with other information as a proxy for the study participant’s use of the phone to make phone calls.
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - event
* - 1.51E+12
  - 2017-12-01T15:45:56.969
  - NearUser
* - 1.51E+12
  - 2017-12-01T15:45:58.741
  - NotNearUser
```

## Gyro (Apple iOS Only)
Beiwe reports rotation or twist on a phone via the gyroscope (gyro). This data can be used together with other collected data to determine various activities. The sign follows the right hand rule: If the right hand is wrapped around the axis such that the tip of the thumb points toward positive, a positive rotation is one toward the tips of the other four fingers.
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - x
  - y
  - z
* - 1.51E+12
  - 2017-12-04T19:01:54.442
  - 0.007534
  -  -0.00846
  - 0.00107
```

## Magnetometer (Apple iOS Only)
Beiwe can be configured to collect the uncalibrated magnetic field of the phone in each direction. This data can be used together with accelerometer data to determine the orientation of the phone.
```{list-table} 
:header-rows: 1
* - timestamp
  - UTC time
  - x
  - y
  - z
* - 1.49E+12
  - 2017-04-17T12:08:36.637
  -  -118.817
  -  -103.967
  - 28.25653
```

## DeviceMotion (Apple iOS Only)
Bewie collects this calculated information to determine information about the motion of the participants’ phone. The acceleration, magnetometer, and gyro sensors influence each other and are used to determine things such as what part of the phones acceleration is due to gravity, and what part is due to the user exerting force on the phone.

```{list-table}
:header-rows: 1
* - timestamp
  - UTC time
  - roll
  - pitch
  - yaw
  - rotation_rate_x
  - rotation_rate_y
  - rotation_rate_z
  - gravity_x
  - gravity_y
  - gravity_z
  - user_accel_x
  - user_accel_y
  - user_accel_z
  - magnetic_field_calibration_accuracy
  - magnetic_field_x
  - magnetic_field_y
  - magnetic_field_z
* - 1.49E+12
  - 2017-04-17T20:01:44.178
  - -1.80147
  - 0.18687
  - 0.235159
  - 0.004084
  - 0.026402
  - -0.06429
  - -0.95656
  - -0.18578
  - 0.224656
  - 0.015539
  - -0.00918
  - -0.00345
  - uncalibrated
  - 0
  - 0
  - 0
```

## Reachability (Apple iOS Only)
Beiwe reports whether and how the phone is connected to the internet, either via Wi-Fi or a cellular data connection or if the Internet is not reachable.
```{list-table}
:header-rows: 1
* - timestamp
  - UTC time
  - event
* - 1.5E+12
  - 2017-06-04T04:48:55.927
  - wifi
* - 1.5E+12
  - 2017-06-04T02:21:48.870
  - unreachable
* - 1.5E+12
  - 2017-06-04T02:21:56.482
  - cellular
```
