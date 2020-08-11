''' Logging Guidelines & Snippets for Forest Developers

This script provides some guidelines and sample code for using features of 
the Python's logging package in Beiwe's Forest modules.

Overview:
    - The logging package from Python's Standard Library provides several 
    simple tools for handling errors, debugging code, and notating events.
	- The logging package is GREAT for:
		- Easy implementation,
		- Human readability,
		- Flexible output.
	- The logging package is NOT as good for:
		- Machine readability,
		- Flexible input.
	- We want to take advantage of the strengths of the logging package.
    - We'd like to implement logging for every Forest module that is intended
    for import.
    - Successful logging happens in two steps.  
        1. Implement loggers in all of our modules.  
        2. Implement handlers to collect the records emitted by those loggers.
    - For end users, our goal is to format basic log messages in a CSV that 
    can be easily reviewed with pandas or Excel.
    - For development purposes, we may wish to review more comprehensive logs,
    and we may benefit from greater flexibility in message handling.    
    - This script provides code snippets for the above tasks.
	- log.py provides some basic tools for handling log records.

Key Recommendations:
    - The logging package should be imported for most modules.  
    - Most modules should initialize a logger object before any definitions 
    or statements.  Exceptions may include:
        - Modules that contain mostly constants.
        - Snippet scripts that provide sample code, such as this script.
        - Scripts that document workflow.    
    - Use logging to document the following events:
         - Failure of a task.
         - Checkpoints during a time-consuming task.
         - The beginning & conclusion of a time-consuming task.
         - Any unusual situations that the end user should know about.
         - Events of interest for development and testing.
    - Assume that Forest log messages will be written to a CSV.  Therefore, 
    log messages SHOULD NOT contain commas or line breaks.

Snippets in this script:
    - How to implement logging in a module,
    - How to insert log messages into definitions,
    - Two examples showing how to handle log records at runtime.

Resources:
    - Documentation for the logging package:
        https://docs.python.org/3/library/logging.html
    - Logging tutorial:
        https://docs.python.org/3/howto/logging.html
    - Log record attributes:
        https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes    
'''

###############################################################################
# How to implement logging in a module
###############################################################################

# First, include logging among the module imports:
import logging

# Next, we will initialize a logger object.  
#   - The name of the logger is arbitrary.
#   - For convenience, we can name the logger after the module in which it 
#     appears, unless there is a good reason to choose another name.
#   - To initialize a logger, insert the following code AFTER imports but 
#     BEFORE any definitions or statements:
logger = logging.getLogger(__name__)


###############################################################################
# How to insert log messages into definitions
###############################################################################

# To document an event in a procedure:
def f(x):
    logger.info('We are going to do some arithmetic.')
    y = 1/x
    return(y)
    
# To document an event with context:
def f(x):
    logger.info('We are going to do some arithmetic with x = %s.' % x)
    y = 1/x
    return(y)

# To document a potential problem:
def f(x):
    if x == 0:
        logger.warning('There may be a problem with this value of x.')
    y = 1/x
    return(y)

# To document the runtime outcome of a try...except statement:
def f(x):
    try:
        y = 1/x
        logger.info('We did some arithmetic.') # Optional success message.
    except:
        y = None
        logger.warning('There may be a problem with x = %s.' % x)
    return(y)
    
# The default logging levels, in order of severity, are:
#   - DEBUG
#   - INFO
#   - WARNING
#   - ERROR
#   - CRITICAL
# The names are arbitrary.  For basic Forest tasks, we may just need two 
# levels, e.g. INFO and WARNING.  Be sure to document the interpretation of 
# additional levels, if implemented.  For example:
logger.critical('This is a critical event!') # Something horrible happened.


###############################################################################
# Example Task 1:  
#   - Import and run a Forest function.  Collect all the log messages.
#   - For example, the end user is importing & using tools from a Forest 
#     module, or a Forest developer is debugging a function.
###############################################################################

# If we don't monitor log records, the script looks like this:
from forest_module import f
f(x)

# For basic logging written to CSV, make three changes:
from log import log_to_csv            # 1. Import this function.
from forest_module import f
log_dir = 'path/to/output/directory'  # 2. Choose where to write log records.
log_to_csv(log_dir)                   # 3. Initialize log file and handler.
f(x)

# For comprehensive logging written to CSV, including traceback information:
from log import traceback_format, log_to_csv
from forest_module import f
log_dir = 'path/to/output/directory'
log_to_csv(log_dir, 
           log_format = traceback_format.attributes,
           log_header = traceback_format.header)
f(x)
    
# For custom logging written to CSV:
from log import attributes_to_csv, log_to_csv
from forest_module import f
log_dir = 'path/to/output/directory'  
custom_format = attributes_to_csv(['asctime',  # Just show a human-readable
                                   'message']) # time and the log message.
log_to_csv(log_dir, 
           log_format = custom_format.attributes,
           log_header = custom_format.header)
f(x)
    
# For custom logging sent to console:
import logging
from forest_module import f
custom_format = '%(asctime)s %(levelname)-8s %(message)s'
logging.basicConfig(format = custom_format, # See links above for format documentation.
                    level = logging.DEBUG,  # Ensure that all messages are printed.
                    force = True) # Overwrite pre-existing handlers & configurations.
f(x)
    

###############################################################################
# Example Task 2:  
#   - Write a wrapper for Forest functions that handles its own logging.
#   - For example, a developer may write a single function that processes 
#     Beiwe data by calling other Forest functions.
#   - Advantages:  The end user doesn't have to deal explicity with logging, 
#     and the developer can specify exactly how logs are recorded.
#   - Disadvantage:  Less flexible for a savvy end user who wants fine
#     control over logging.
###############################################################################

# Without log handling, the module looks like this:
import logging
from some_other_forest_module import f, g, h

logger = logging.getLogger(__name__)

def wrapper(x, y, z):
    logger.info('Getting started...')
    a = f(x)
    b = g(y)
    c = h(z)
    logger.info('All done.')
    return(a, b, c)

# What happens when this wrapper is called?
#   - The wrapper has a logger that emits log records.  
#   - Also, the imported functions originate log records since they're from a 
#     Forest module which should have a logger.  
#   - So at runtime, the wrapper will emit records from multiple sources.  
# Example 1 shows how an end user can handle all these records when 
# calling the wrapper.  The end user would have to do this at runtime:
import logging                        
from log import log_to_csv
from forest_module import wrapper
log_dir = 'path/to/output/directory'
log_to_csv(log_dir)
a, b, c = wrapper(x, y, z)

# But it might be useful for the wrapper to handle these records by itself.
# That way, the end user simply runs the wrapper and automatically gets a CSV 
# of log records for free.  To do this, make three changes in the module:
import logging
from log import log_to_csv              # 1. Import this function.
from some_other_forest_module import f, g, h

logger = logging.getLogger(__name__)

def wrapper(x, y, z, log_dir):          # 2. Add an output directory argument.
    log_to_csv(log_dir)                 # 3. Initialize log file & handler.
    logger.info('Getting started...')
    a = f(x)
    b = g(y)
    c = h(z)
    logger.info('All done.')
    return(a, b, c)

# Now the end user simply does this at runtime:
from forest_module import wrapper
a, b, c = wrapper(x, y, z, 'path/to/output/directory')


