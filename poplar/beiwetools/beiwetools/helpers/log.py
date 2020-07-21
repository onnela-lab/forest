'''Code for fomatting and exporting logging messages.

'''
import logging
from .functions import setup_csv


logger = logging.getLogger(__name__)


# Dictionary of log record attributes:
# (For details:  https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes)
log_attributes = {
'asctime': '%(asctime)s',     # Human-readable time.
'created': '%(created)f',     # Unix timestamp (seconds since epoch).
'filename': '%(filename)s',   # Filename portion of pathname.
'funcName': '%(funcName)s',   # Originating function.
'levelname': '%(levelname)s', # Message level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.  
'levelno': '%(levelno)s',     # Numeric message level.
'lineno': '%(lineno)d',       # Source line number, if available.
'message': '%(message)s',     # Logged message.
'module': '%(module)s',       # Module name.
'msecs': '%(msecs)d',         # Millisecond portion of timestamp.
'name': '%(name)s',              # Name of originating logger.
'pathname': '%(pathname)s',      # Path to originating file.
'process': '%(process)d',        # Process id, if available.
'processName': '%(processName)s',# Process name, if available.
'relativeCreated': '%(relativeCreated)d', # Milliseconds since logging was loaded.
'thread': '%(thread)d',          # Thread id, if available.
'threadName': '%(threadName)s',  # Thread name, if available.
 }


def attributes_to_csv(attribute_list):
    '''
    Given a list of attributes (keys of log_attributes), returns a logging 
    format with header for writing records to CSV.
    '''
    try:
        log_format = type('logging_format', (), {})()
        attributes = [log_attributes[a] for a in attribute_list]
        log_format.attributes = ','.join(attributes)
        log_format.header = attribute_list        
    except:
        logger.warning('Unable to assemble logging format.')        
    return(log_format)


# Simple format for logging messages:
basic_format = attributes_to_csv([
    'created',   
    'asctime',
    'levelname', 
    'module',
    'funcName',  
    'message',   
    ])


# More comprehensive format for logging messages, including traceback info:
traceback_format = attributes_to_csv([
    'created',   
    'asctime',
    'levelname', 
    'module',
    'funcName',  
    'message',       
    'lineno',
    'pathname'    
    ])

    
def log_to_csv(log_dir, level = logging.DEBUG, 
               log_name = 'log', 
               log_format = basic_format.attributes,
               log_header = basic_format.header):
    '''
    Configure the logging system to write messages to a csv.
    Overwrites any existing logging handlers and configurations.
    
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
