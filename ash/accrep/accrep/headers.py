'''Headers for CSVs handled by accrep.

'''

raw_header = [ # column names for raw Beiwe accelerometer files
  'timestamp', # Millisecond timestamp.
  'UTC time',  # Human-readable UTC date-time formatted as '%Y-%m-%dT%H:%M:%S.%f'.
  'accuracy',  # Unknown variable.  
               # Appears to always be 'unknown' for iPhone data and integer (e.g. 2 or 3) for Android data.
  'x', 'y', 'z' # Acceleration in each axis. 
                # Units are m/s^2 for Android and G for iPhones.
  ]

keep_header = [ # same notes as above
  'timestamp', 
  'x', 'y', 'z'
  ]

summary_records_header = [ # column names for accrep.summary output
  'user_id', # Beiwe User ID.
  'os',   # 'Android' or 'iOS'.
  'n_files', # Number of raw accelerometer files for this user.
  'first_file', 'last_file', # Basenames of first and last hourly files.
  'unique_accuracy_values', # Number of unique accuracy entries.
                            # Probably only 1 for iPhone, unknown for Android.
  'approximate_frequency_Hz', # An estimate of sampling frequency when the accelerometer is active.
  'local_start_time', 'local_finish_time', # When the summary process began/ended.
  'total_time_minutes' # Time taken to summarize this user's accelerometer data.
   ]

spectral_records_header = [
  'user_id',
  'os',   # 'Android' or 'iOS'.
  'n_windows', # Number of processed windows.


  'local_start_time', 'local_finish_time', # Local summary start/end times.
  'total_time_minutes' # Time taken to summarize this user's accelerometer data.
   ]

spectral_analysis_header = [
  'timestamp',
        
        
  ]
