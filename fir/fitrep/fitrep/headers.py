'''
Headers for selected fitabase CSVs.
'''

minute_headers = {}

hour_headers = {}

day_headers = {}

other_headers = {}




raw_header = { # Column names for fitabase files handled by fitrep
'30secondSleepStages': ['LogId', 'Time', 'Level', 'ShortWakes', 'SleepStage'],

'activitylogs': ['Date', 'StartTime', 'Duration', 'Activity', 'ActivityType', 
                 'LogType', 'Steps', 'Distance', 'ElevationGain', 'Calories',
                 'SedentaryMinutes', 'LightlyActiveMinutes', 'FairlyActiveMinutes',
                 'VeryActiveMinutes', 'AverageHeartRate', 'OutOfRangeHeartRateMinutes',
                 'FatBurnHeartRateMinutes', 'CardioHeartRateMinutes',
                 'PeakHeartRateMinutes'],

'battery': ['DateTime', 'DeviceName', 'BatteryLevel', 'LastSync'],

'dailyActivity': ['ActivityDate', 'TotalSteps', 'TotalDistance', 'TrackerDistance',
                  'LoggedActivitiesDistance', 'VeryActiveDistance',
                  'ModeratelyActiveDistance', 'LightActiveDistance',
                  'SedentaryActiveDistance', 'VeryActiveMinutes',
                  'FairlyActiveMinutes', 'LightlyActiveMinutes',
                  'SedentaryMinutes', 'Calories', 'Floors', 'CaloriesBMR',
                  'MarginalCalories', 'RestingHeartRate'],

'dailyCalories': ['ActivityDay', 'Calories'],

'dailyIntensities': ['ActivityDay', 'SedentaryMinutes', 'LightlyActiveMinutes',
                      'FairlyActiveMinutes', 'VeryActiveMinutes',
                      'SedentaryActiveDistance', 'LightActiveDistance',
                      'ModeratelyActiveDistance', 'VeryActiveDistance'],

'dailySteps': ['ActivityDay', 'StepTotal'],

'heartrate_15min': ['Time', 'Value'],

'heartrate_1min': ['Time', 'Value'],

'heartrate_seconds': ['Time', 'Value'],

'hourlyCalories': ['ActivityHour', 'Calories'],

'hourlyIntensities': ['ActivityHour', 'TotalIntensity', 'AverageIntensity'],

'hourlySteps': ['ActivityHour', 'StepTotal'],

'minuteCaloriesNarrow': ['ActivityMinute', 'Calories'],

'minuteIntensitiesNarrow': ['ActivityMinute', 'Intensity'],

'minuteMETsNarrow': ['ActivityMinute', 'METs'],

'minuteSleep': ['date', 'value', 'logId'],

'minuteStepsNarrow': ['ActivityMinute', 'Steps'],

'sleepDay': ['SleepDay', 'TotalSleepRecords', 'TotalMinutesAsleep',
             'TotalTimeInBed'],

'sleepLogInfo': ['LogId', 'StartTime', 'Duration', 'Efficiency', 'IsMainSleep',
                 'MinutesAfterWakeup', 'MinutesAsleep', 'MinutesToFallAsleep',
                 'TimeInBed', 'AwakeCount', 'AwakeDuration', 'RestlessCount',
                 'RestlessDuration'],

'sleepStageLogInfo': ['LogId', 'StartTime', 'Duration', 'Efficiency',
                      'IsMainSleep', 'SleepDataType', 'MinutesAfterWakeUp',
                      'MinutesAsleep', 'MinutesToFallAsleep', 'TimeInBed',
                      'ClassicAsleepCount', 'ClassicAsleepDuration',
                      'ClassicAwakeCount', 'ClassicAwakeDuration',
                      'ClassicRestlessCount', 'ClassicRestlessDuration',
                      'StagesWakeCount', 'StagesWakeDuration',
                      'StagesWakeThirtyDayAvg', 'StagesLightCount',
                      'StagesLightDuration', 'StagesLightThirtyDayAvg',
                      'StagesDeepCount', 'StagesDeepDuration',
                      'StagesDeepThirtyDayAvg', 'StagesREMCount',
                      'StagesREMDuration', 'StagesREMThirtyDayAvg'],

'sleepStagesDay': ['SleepDay', 'TotalSleepRecords', 'TotalMinutesAsleep',
                   'TotalTimeInBed', 'TotalTimeAwake', 'TotalMinutesLight',
                   'TotalMinutesDeep', 'TotalMinutesREM'],
 
'syncEvents': ['DateTime', 'SyncDateUTC', 'Provider', 'DeviceName'],

'weightLogInfo': ['Date', 'WeightKg', 'WeightPounds', 'Fat', 'BMI',
                  'IsManualReport', 'LogId']
}


raw_header_other = { # Column names for fitabase files NOT handled by fitrep
    
    
}


raw_header = [ # Column names for raw Beiwe GPS files.
  'timestamp', # Millisecond timestamp.
  'UTC time', # Human-readable UTC date-time formatted as '%Y-%m-%dT%H:%M:%S.%f'.
  'latitude', 'longitude', 
  'altitude', # Is this relative to mean sea level or to WGS84 reference ellipsoid?
  'accuracy'  # Probably not standardized across different manufacturers and models.
  ]

raw_keep_header = [ # Same notes as above.
  'timestamp', 
  'accuracy', 
  'latitude', 'longitude', 
  'altitude'
  ]

file_summary = [
  'user_id', # Beiwe User ID.
  'n_files', # Number of raw GPS files for this user.
  'first_file', 'last_file', # Basenames of first and last hourly files.
  'approximate_frequency_Hz' # An estimate of sampling frequency when GPS is active.
  ]

process_summary = [
  'local_start_time', 'local_finish_time', # Local summary start/end times.
  'total_time_minutes' # Time taken to summarize this user's GPS data.
  ]

location_ranges = [ # min/max for each GPS coordinate
  'latitude_min',  'latitude_max',
  'longitude_min', 'longitude_max',
  'altitude_min',  'altitude_max'
  ]

accuracy_summary = [ # accuracy summary statistics
  'accuracy_min', 'accuracy_max',
  'accuracy_mean', 'accuracy_std'
  ]

timezone_summary = [
  'n_timezones', # Number of distinct timezones.
  'n_timezone_transitions' # Number of timezone transitions.
  ]  

# Column names for gpsrep.summary output:
summary_records_header = file_summary + process_summary + location_ranges + \
                         accuracy_summary + timezone_summary
