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
    'Android': [
        'timestamp', # Millisecond timestamp.
        'event' # Name of the event.  See README.md for documentation.
        ]
    }


summary_header = [ # Column names for powrep.summary output.
  'user_id', # Beiwe User ID.
  'os',      # 'Android' or 'iOS'.
  'n_files', # Number of raw power state files for this user.
  'first_file', 'last_file', # Basenames of first and last hourly files.
  'unknown_headers', # Number of files with unrecognized headers.
  'unknown_events'   # Number of unknown event categories.
   ]


