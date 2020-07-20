'''
Headers for selected fitabase CSVs.
'''

summary_header = [ # Column names for summary output files
'fitabase_id',       # Fitabase identifier.
'first_observation', # Datetime of first observation.
'last_observation',  # Datetime of last observation.
'n_observations',    # Number of rows (observations).
]


extra_summary_header = { # Extra column names for summary output files
'minuteSleep': [
    'n_offsets', # Number of observations synced with 30-second offset.
    'p_offsets'  # Proportion of observations synced with 30-second offset.
    ],
'syncEvents': [
    'provider', # Name of sync service.
    'device'    # Name of device model.  
    ]
}


sync_records_header = [
'fitabase_id', # Fitabase identifier.
'first_sync',  # Datetime of first sync, formatted as 


'%Y-%m-%dT%H:%M:%S.%f'.


'last_sync',   # Datetime of last sync,  formatted as 


'%Y-%m-%dT%H:%M:%S.%f'.



'n_syncs',     # Number of rows (syncs).
'app_syncs',   # Number of syncs with the Fitbit smartphone apps.
'provider',    # Name of sync service.
'device',      # One of: 
               #    - the name of the device model (e.g. 'Charge 2') 
               #    - 'multiple'
               #    - 'missing'
]


datetime_header = { # Names for Fitabase columns that contain local datetimes
'heartrate_1min': 'Time',
'minuteCaloriesNarrow': 'ActivityMinute',
'minuteIntensitiesNarrow': 'ActivityMinute',
'minuteMETsNarrow': 'ActivityMinute',
'minuteSleep': 'date',
'minuteStepsNarrow': 'ActivityMinute',
'syncEvents': 'DateTime'
}


raw_header = { # Column names for fitabase files handled by fitrep
# Documentation is from:
# https://www.fitabase.com/media/1748/fitabasedatadictionary.pdf
'heartrate_1min': [
    '''
    Heart rate values recorded by the Fitbit device. 
    Note 1: A variable sampling technique controls the frequency at which 
    heart rate is recorded.Devices will sample heart rate every 5 to 
    15 seconds on average.
    Note 2: For all 15min, 5min, and 1min data sets Fitabase uses the seconds 
    data to generate a mean value and reports the floor of that value for 
    the specified interval. For example, if the calculated mean for the 
    interval is 156.79, we report 156.
    '''
    'Time', # Date and hour value in mm/dd/yyyy hh:mm:ss format.
    'Value' # Mean heart rate value.
    ],
'minuteCaloriesNarrow': [
    '''
    Estimated energy expenditure.
    Note​: Fitbit uses the ​gender, age, height, and weight ​data entered into 
    the user profile to calculate basal metabolic rate (BMR). The estimated 
    energy expenditure that Fitbit provides takes into account the user’s BMR, 
    the activity recorded by the device, and any manually logged activities.
    '''
    'ActivityMinute', # Date and time value in mm/dd/yyyy hh:mm:ss format.
    'Calories'        # Total number of estimated calories burned.
    ],
'minuteIntensitiesNarrow': [
    '''
    Time spent in one of four intensity categories.
    Note​: The cut points for intensity classifications and METs are not 
    determined by Fitabase, but by proprietary algorithms from Fitbit.
    '''
    'ActivityMinute', # Date and time value in mm/dd/yyyy hh:mm:ss format.
    'Intensity'       # Intensity value.
                      # 0 = Sedentary
                      # 1 = Light
                      # 2 = Moderate
                      # 3 = Very Active
    ],
'minuteMETsNarrow': [
    'ActivityMinute', # Date and time value in mm/dd/yyyy hh:mm:ss format.
    'METs'            # MET value for the given minute.
                      # Important:​ All MET values exported from Fitabase 
                      # are multiplied by 10. Please divide by 10 to get 
                      # accurate MET values
    ],
'minuteSleep': [
    '''
    Data from each tracked sleep event.
    Notes​: Sleep durations are either specified by Fitbit wearer (interacting 
    with the device or Fitbit.com profile), or are automatically detected on 
    supported models (Charge, Alta, Alta HRCharge HR, Flex, Blaze, Charge 2, 
    Flex 2, Ionic, and Surge). Sleep Stages are supported bythe Alta HR, 
    Charge 2, Blaze, and Ionic. All other devices support the ​Classic ​sleep algorithm.
    '''
    'date', # Date and minute of that day within a defined sleep period 
            # in ​mm/dd/yy hh:mm:ss​ format
            # Note​: sleep minute data is commonly exported with :30 sec. 
            # In this case, the “floor” of the time value can be used to 
            # convert to whole minutes.
    'value', # Value indicating the sleep state.
             # 1 = asleep, 2 = restless, 3 = awake
    'logId'  # The unique log id in Fitbit’s system for the sleep record.
    ],
'minuteStepsNarrow': [
    '''
    Steps tracked by the activity tracker or entered by participant 
    for the given period.
    '''
    'ActivityMinute', # Date and time value in mm/dd/yyyy hh:mm:ss format.
    'Steps'           # Total number of steps taken.
    ],
'syncEvents': [
    '''
    A record of device sync events as reported by Fitbit.
    Sync events are recorded by Fitbit when the device syncs with the 
    Fitbit service through either the Fitbit mobile app or the Fitbit 
    Connect application.
    '''
    'DateTime',    # Date and time value in mm/dd/yyyy hh:mm:ssAM/PM format.
    'SyncDateUTC', # Date and time value in mm/dd/yyyy hh:mm:ssAM/PM format.
    'Provider',  # Name of the sync service.
    'DeviceName' # Name of the device model.
    ],
}


raw_header_other = { # Column names for fitabase files NOT handled by fitrep
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

'heartrate_seconds': ['Time', 'Value'],

'hourlyCalories': ['ActivityHour', 'Calories'],

'hourlyIntensities': ['ActivityHour', 'TotalIntensity', 'AverageIntensity'],

'hourlySteps': ['ActivityHour', 'StepTotal'],

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

'weightLogInfo': ['Date', 'WeightKg', 'WeightPounds', 'Fat', 'BMI',
                  'IsManualReport', 'LogId']
}

