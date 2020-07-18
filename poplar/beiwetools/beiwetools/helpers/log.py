'''Code for fomatting and exporting logging messages.

'''
import logging
from .functions import setup_csv


logger = logging.getLogger(__name__)


# Simple format for logging messages:
basic_log_attributes = ['%(created)f',
                        '%(levelname)s',
                        '%(name)s',
                        '%(funcName)s',
                        '%(lineno)d',                        
                        '%(message)s',
                        '%(pathname)s'
                        ]
basic_log_format = ','.join(basic_log_attributes)

# Header for csv containing messages formatted as basic_log_format:
# (For details:  https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes)
basic_log_header = [
    'created',   # Unix timestamp (seconds since epoch).
    'levelname', # Message level, one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    'name',      # Name of originating logger.
    'funcName',  # Originating function.
    'lineno',    # Source line number, if available.
    'message',   # Logged message.
    'pathname',  # Path to originating file.
    ]


def log_to_csv(log_dir, level = logging.DEBUG, 
               log_name = 'log', 
               log_format = basic_log_format,
               log_header = basic_log_header):
    '''
    Configure the logging system to write messages to a csv.
    Overwrites previous logging configuration
    
    Args:
        log_dir (str): Path to a directory where log messages should be written.
        level (int):  An integer between 0 and 50.  
            Set level = logging.DEBUG to log all messages.
        log_name (str): Name for the log file.
        log_format (str): The format argument for logging.basicConfig.
            For available attributes and formatting instructions, see:
            https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes)
        log_header (list): Header for the csv.

    Returns:
        None
    '''
    try:
        # initialize csv
        filepath = setup_csv(name = log_name, directory = log_dir, header = log_header)
        # configure logging output
        logging.basicConfig(format = log_format,
                            filename = filepath,
                            level = level, force = True)
        # success message
        logger.info('Writing log messages to %s.csv...' % log_name)
    except:
        logger.warning('Unable to write logging messages.')
