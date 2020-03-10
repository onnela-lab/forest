'''
Headers for CSVs handled by fitrep.
'''

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
