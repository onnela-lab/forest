'''Headers for CSVs handled by logrep.

'''

raw_header = [ # column names for raw Beiwe app log files
  'timestamp', # Millisecond timestamp.
  'UTC time',  # Human-readable UTC date-time formatted as '%Y-%m-%dT%H:%M:%S.%f'.
  ]

keep_header = [ # same notes as above
  'timestamp', 
  ]

summary_records_header = [ # column names for logrep.summary output
  'user_id', # Beiwe User ID.
  'os',   # 'Android' or 'iOS'.
  'n_files', # Number of raw app log files for this user.
  'first_file', 'last_file', # Basenames of first and last hourly files.



  'local_start_time', 'local_finish_time', # When the summary process began/ended.
  'total_time_minutes' # Time taken to summarize this user's app log data.
   ]

log_records_header = [
  'user_id',

  'local_start_time', 'local_finish_time', # Local summary start/end times.
  'total_time_minutes' # Time taken to summarize this user's app log data.
   ]

log_analysis_header = [
  'timestamp',
  ]
