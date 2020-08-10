'''Headers for CSVs handled by powrep.

'''

raw_header = {
    'iOS': [ # Column names for raw Beiwe power state files from iPhones.
        'timestamp', # Millisecond timestamp.
        'UTC time',  # Human-readable UTC date-time formatted as '%Y-%m-%dT%H:%M:%S.%f'.        
        'event', # Name of the event.  See README.md for documentation.
        'level'  # Battery level.
        ],
    'Android': [ # Column names for raw Beiwe power state files from Android phones.
        'timestamp', # Millisecond timestamp.
        'UTC time',  # Human-readable UTC date-time formatted as '%Y-%m-%dT%H:%M:%S.%f'.
        'event' # Name of the event.  See README.md for documentation.
        ]
    }


keep_header = {
    'iOS': [
        'timestamp', # Millisecond timestamp.
        'event', # Name of the event.  See README.md for documentation.
        'level'  # Battery level.
        ],
    'Android': [ # column names for raw Beiwe power state files from Android phones
        'timestamp', # Millisecond timestamp.
        'event' # Name of the event.  See README.md for documentation.
        ]
    }


    
    
summary_records_header = [ # column names for powrep.summary output
  'user_id', # Beiwe User ID.
  'os',   # 'Android' or 'iOS'.
  'n_files', # Number of raw power state files for this user.
  'first_file', 'last_file', # Basenames of first and last hourly files.
  
  


  'local_start_time', 'local_finish_time', # When the summary process began/ended.
  'total_time_minutes' # Time taken to summarize this user's power state data.
   ]

power_records_header = [
  'user_id',
  'os',   # 'Android' or 'iOS'.

  'local_start_time', 'local_finish_time', # Local summary start/end times.
  'total_time_minutes' # Time taken to summarize this user's power state data.
   ]

power_analysis_header = [
  'timestamp',
  ]
