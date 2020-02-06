'''Headers for CSVs handled by powrep.

'''

raw_header = [ # column names for raw Beiwe power state files
  'timestamp', # Millisecond timestamp.
  'UTC time',  # Human-readable UTC date-time formatted as '%Y-%m-%dT%H:%M:%S.%f'.
  ]

keep_header = [ # same notes as above
  'timestamp', 
  ]

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
