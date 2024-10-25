# Logging

## 1. Introduction

### Overview

* The `logging` package provides several simple tools for notating runtime events.
* End-users can use log records to identify data anomalies and to verify task completion.
* Developers can use log records to locate bugs and to collect performance benchmarks.
* With the `logging` package, it's easy to send human-readable output to the console or to a text file.
* Note that the `logging` package is NOT the best option for generating machine-readable files, or for writing non-chronological output.

### How it works

* Events are associated with a `LogRecord` that includes a timestamp, traceback information, and a brief message.
* `Logger` objects are associated with modules.  A `Logger` *emits* a record whenever an event occurs.
* A `Handler` object is initialized at runtime.  The `Handler` is responsible for *collecting*, *formatting* and *delivering*  event records.

### Recommendations for Forest contributors

* We'd like to implement `logging` for every Forest module that is intended for import.
* For end users, our goal is to deliver basic log records to a CSV that can be easily reviewed with pandas or Excel.  (Therefore, we should avoid using commas and line breaks in log messages.)
* For development purposes, we may wish to review more comprehensive logs, and we may benefit from greater flexibility in handling records. 
* In general, we should use `logging` to document the following events:
	* Failure of a task.
	* Checkpoints during a time-consuming task.
	* The beginning & conclusion of a time-consuming task.
	* Any unusual situations that the end user should know about.
	* Events of interest for development and testing.

## 2. How to associate a `Logger` with a module

First, include `logging` among the module imports.  Then initialize a `Logger` instance before any definitions or statements.  The name of the `Logger` instance doesn't matter too much, but it's convenient to name it after the module with which it is associated.  We can do it like this:

```
import logging
logger = logging.getLogger(__name__)
```

## 3. How to insert log messages into definitions

Basic `logging` messages:

```
logger.info('Here is some information.')
logger.warning('This is a warning message!')
```

To document an event in a procedure:
 
``` 
def f(x):
    logger.info('We are going to do some arithmetic.')
    y = 1/x
    return(y)
```
    
To document an event with context:

```
def f(x):
    logger.info('We are going to do some arithmetic with x = %s.' % x)
    y = 1/x
    return(y)
```

To document a potential problem:

```
def f(x):
    if x == 0:
        logger.warning('There may be a problem with this value of x.')
    y = 1/x
    return(y)
```

To document the runtime outcome of a `try...except` statement:

```
def f(x):
    try:
        y = 1/x
        logger.info('We did some arithmetic.') # Optional success message.
    except:
        y = None
        logger.warning('There may be a problem with x = %s.' % x)
    return(y)
```

## 4.  How to log exceptions

To document the warning or the error from the system output instead of a self-defined message:

```
def f(x):
    try:
        y = 1/x
    except:
        y = None
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))
    return(y)
```

### Advice from Alvin for try except errors

Guidelines for throwing errors / try except clauses
•	Define your own Exception classes for expected errors
•	Try to minimize the amount of code inside the try block
Both of these will help keep the code more maintainable, separate out the handling of expected and unexpected error cases, and allow for more precise logging behavior.

Instead of doing this:

```
# Don't do this
try:

    # Do a whole bunch of stuff

except:

    # Log that something went wrong
```

Python allows you create custom Exception classes, with which you can be more precise with your try/except blocks:
```
class FileNotFound(Exception): pass

class InvalidDataFormat(Exception): pass


try:

    # Do a whole bunch of stuff

except FileNotFound:

    # Skip this analysis

except InvalidDataFormat:

    # Try to fix data

except Exception as e:

    # Note: You generally shouldn't use bare except clauses unless you

    #       truly want to catch all exceptions and prevent that from

    #       stopping execution. Instead, if you need to log something,

    #       you can do it this way where you re-raise the exception.

    logging.error()

    raise e
```

It's also better to minimize what's inside of a try whenever possible; otherwise you'll end up in situations where your try catches a legitimate error in your code that really should break execution. For example, here's a contrived way you could miss a potential error in the code:

```
def get_data():

    if some_condition:

        raise InvalidDataFormat

    return {"distance": distance}



# Don't do this

try:

    data = get_data()

    value = data["distence"]

    # More lines of code

except:

    value = 0



# Do this instead

try:

    data = get_data()

except InvalidDataFormat:

    data = {"distance": 0}

value = data["distence"]

# More lines of code
```

1.	By only caching `InvalidDataFormat` exception, if a different exception were to occur (or if someone else built additional exception types into `get_data`), you would know immediately that you need to modify your code to account for this
2.	By minimizing the size of the try block to the smallest possible, you notice that there is a typo ("distence") that would have caused a silent bug in your original code.
So, in summary:
•	Define your own Exception classes for expected errors
•	Try to minimize the amount of code inside the try block
Both of these will help keep the code more maintainable, separate out the handling of expected and unexpected error cases, and allow for more precise logging behavior.

## 5.  Log levels

The default log levels, in order of severity, are:
* `DEBUG`
* `INFO`
* `WARNING`
* `ERROR`
* `CRITICAL`

For basic Forest tasks, we may just need two levels, e.g. `INFO` and `WARNING`.  Be sure to document the interpretation of additional levels, if implemented.  For example:

```
logger.critical('This is a critical event!') # Something horrible happened.
```

## 6. Example Task 1

* Write a script that imports and runs a Forest function.  Collect all the log messages.
* This corresponds to a typical use case, e.g. the end-user is importing & using tools from a Forest module, or a Forest developer is debugging a function.

___


If we don't monitor log records, the script looks like this:

```
from forest_module import f    # Import the function.
f(x)                           # Run the function.
```

For basic log records written to CSV, make three changes:

```
from trunk.log import log_to_csv          # 1. Import this function.
from forest_module import f
log_dir = 'path/to/log/output/directory'  # 2. Choose where to write log records.
log_to_csv(log_dir)                       # 3. Initialize log file and handler.
f(x)
```

For comprehensive logging written to CSV, including traceback information:

```
from log import TRACEBACK_CSV_LOG, log_to_csv
from forest_module import f
log_dir = 'path/to/log/output/directory'
log_to_csv(log_dir, 
           log_format = TRACEBACK_CSV_LOG.attributes,
           log_header = TRACEBACK_CSV_LOG.header)
f(x)
```

For custom logging written to CSV:

```
from log import attributes_to_csv, log_to_csv
from forest_module import f
log_dir = 'path/to/log/output/directory'  
CUSTOM_FORMAT = attributes_to_csv(['created',  # Just show a timestamp
                                   'message']) # and the log message.
log_to_csv(log_dir, 
           log_format = CUSTOM_FORMAT.attributes,
           log_header = CUSTOM_FORMAT.header)
f(x)
```
    

For custom logging sent to console:

```
import logging
from forest_module import f
CUSTOM_FORMAT = '%(created)f %(levelname)-8s %(message)s'
logging.basicConfig(format = CUSTOM_FORMAT, # See links above for format documentation.
                    level = logging.DEBUG,  # Ensure that all messages are printed.
                    force = True) # Overwrite pre-existing handlers & configurations.
f(x)
```    

## 7. Example Task 2
   - Write a wrapper for Forest functions that handles its own logging.
   - For example, a developer may write a single function that processes Beiwe data by calling other Forest functions.
   - Advantages:  The end-user doesn't have to deal explicitly with logging, and the developer can specify exactly how logs are recorded.
   - Disadvantage:  Less flexible for an end-user who wants fine control over logging.

___

Without log handling, the module looks like this:

```
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
```

When this wrapper is called:

* The wrapper's module has an associated `Logger` instance that emits log records.  
* Also, the imported functions come from a Forest module which should have an associated `Logger` instance that also emits log records.  
* So at runtime, there will be event records emitted by multiple sources.

[](#6-example-task-1)
 shows how an end-user can handle all these records when calling the wrapper.  The end-user would do something like this at runtime:

```
from log import log_to_csv
from forest_module import wrapper
log_dir = 'path/to/log/output/directory'
log_to_csv(log_dir)
a, b, c = wrapper(x, y, z)
```

But it might be useful for the wrapper to handle these records by itself.  That way, the end user simply runs the wrapper and automatically gets a CSV of log records for free.  To do this, make three changes in the module:

```
import logging
from log import log_to_csv                   # 1. Import this function.
from some_other_forest_module import f, g, h

logger = logging.getLogger(__name__)

def wrapper(x, y, z, log_dir):               # 2. Add an output directory argument.
    log_to_csv(log_dir)                      # 3. Initialize log file & handler.
    logger.info('Getting started...')
    a = f(x)
    b = g(y)
    c = h(z)
    logger.info('All done.')
    return(a, b, c)
```

Now the end user simply does this at runtime:

```
from forest_module import wrapper
a, b, c = wrapper(x, y, z, 'path/to/log/output/directory')
```

## 8. Resources

* [The Python Standard Library's documentation for `logging`](https://docs.python.org/3/library/logging.html)
* Vinay Sajip's [*Logging HOWTO*](https://docs.python.org/3/howto/logging.html)
* [`LogRecord` attributes](https://docs.python.org/3.11/library/logging.html#logrecord-attributes)
