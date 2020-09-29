'''
Headers for CSVs handled by gpsrep.
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

user_summary = [
  'user_id', # Beiwe User ID.
  'n_files', # Number of raw GPS files for this user.
  'first_file', 'last_file', # Basenames of first and last hourly files.
  'n_observations', # Number of GPS observations for this user.
  'approximate_frequency_Hz' # An estimate of sampling frequency when GPS is active.
  ]

variable_ranges = [ # min/max for each variable
  'latitude_min',  'latitude_max',
  'longitude_min', 'longitude_max',
  'altitude_min',  'altitude_max',
  'accuracy_min',  'accuracy_max'
  ]

timezone_summary_header = [
  'user_id', # Beiwe User ID.
  'n_timezones', # Number of distinct timezones.
  'n_timezone_transitions' # Number of timezone transitions.
  'n_offsets', # Number of distinct offsets.
  'n_offset_transitions' # Number of offset transitions.
  ]  

