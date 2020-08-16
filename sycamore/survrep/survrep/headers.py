''' Headers for CSVs handled by survrep.

'''


raw_header = {
'iOS': [ # Column names for raw iPhone Beiwe survey timings files.
    'timestamp',   # Millisecond timestamp.
    'UTC time',    # Human-readable UTC date-time formatted as '%Y-%m-%dT%H:%M:%S.%f'.
    'question id', # Hex-string identifier for the question.
    'survey id',   # Hex-string identifier for the survey.
    'question type', 
    'question text', 
    'question answer options',
    'answer', 
    'event'
    ],
'Android': [ # Column names for raw Android Beiwe survey timings files.
    'timestamp',   # Millisecond timestamp.
    'UTC time',    # Human-readable UTC date-time formatted as '%Y-%m-%dT%H:%M:%S.%f'.
    'question id', # Hex-string identifier for the question.
    'survey id',   # Hex-string identifier for the survey.
    'question type',
    'question text',
    'question answer options',
    'answer'    
    ]
}


summary_header = [ # Column names for summary output.
  'user_id', # Beiwe User ID.
  'opsys',   # 'Android' or 'iOS'.
  'first_file', 'last_file', # Basenames of first and last hourly files.
  'n_files',        # Number of survey timings files for this user.
  'n_events',       # Number of event records for this user.
  'unknown_headers',# Number of files with unrecognized headers.
  'unknown_events', # Number of unique unrecognized event labels.
  'unknown_question_types', # Number of unique unrecognized question types.
  'foreign_surveys' # Number of unique surveys with events incorrectly logged 
                    # in this folder.
  ]


compatibility_header = [ # Column names for configuration compatibility output.
  'user_id', # Beiwe User ID.
  'opsys',   # 'Android' or 'iOS'.
  'absent_surveys',         # Number of unique survey ids not found in the configuration.
  'absent_questions',       # Number of unique question ids not found in the configuration.                      
  'disagree_question_type', # Number of unique question ids with different type than in configuration.
  'disagree_question_text', # Number of unique question ids with inconsistent text.
  'disagree_answer_options' # Number of unique question ids with inconsistent answer options.
   ]
