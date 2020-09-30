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
  'n_files', # Number of raw accelerometer files for this user.
  'n_observations', # Total number of observations for this user.
  'approximate_frequency_Hz', # An estimate of sampling frequency when the 
                              # accelerometer is active.
  'min_x', 'max_x',
  'min_y', 'max_y',
  'min_z', 'max_z',  
    ]

user_summary_header = [
    'filepath',       # Path to the raw file.
    'timestamp',      # Timestamp corresponding to the file's basename.
    'n_observations', # Number of rows in the file.
    'elapsed_s',      # Elapsed time between first and last observation in the file.
    ]

spectral_records_header = [
  'user_id',
  'os',   # 'Android' or 'iOS'.
  'n_windows', # Number of processed windows.
   ]

spectral_analysis_header = [
  'timestamp',
        
        
  ]
