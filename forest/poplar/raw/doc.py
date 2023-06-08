""" Access to documentation from the following files:

    - data_streams.csv
    - headers.json
    - question_type_names.json
    - power_events.csv

"""
import os
from logging import getLogger
from pkg_resources import resource_filename
from ..functions.io import read_json


logger = getLogger(__name__)


# Get paths to files:
DOCPATHS = {}
docnames = [
    "data_streams.csv",
    "headers.json",
    "question_type_names.json",
    "power_events.csv",
]
for d in docnames:
    DOCPATHS[d] = resource_filename(__name__, os.path.join("noncode", d))


"""Dictionary of data streams.
    Keys are names of data streams, e.g. 'gps'.
    STREAMS[<data_stream>]['type']: Either 'passive' or 'survey'.
    STREAMS[<data_stream>]['Android']: True if available on Android phones.
    STREAMS[<data_stream>]['iOS']: True if available on iPhones.
"""
STREAMS = {}
with open(DOCPATHS["data_streams.csv"], encoding="utf-8") as f:
    next(f)  # skip header row
    for line in f:
        line = line.replace("\n", "")  # get rid of line breaks
        stream_name, stream_type, android, ios = line.split(",")
        temp_dict = {
            "type": stream_type,
            "Android": android == "True",
            "iOS": ios == "True",
        }
        STREAMS[stream_name] = temp_dict


"""Dictionary of headers for raw Beiwe data.
    For a given <data_stream> and <device_os>, look up the header with:
        HEADERS[<data_stream>][<device_os>]
    Possible values:
        List of column labels if <data_stream> is available for <device_os>,
        Or the string `To do` if it hasn't been documented yet,
        Or None if the data stream isn't available for <device_os>.
"""
HEADERS = read_json(DOCPATHS["headers.json"])


"""Dictionary of question type names.
    Keys are names of survey question types, as used in Beiwe configuration
    files.  Values provide concordance for different naming conventions used
    in raw data from Android and iOS platforms.
    For example:
        >>> QUESTION_TYPES['checkbox']['Android']
        'Checkbox Question'
        >>> QUESTION_TYPES['checkbox']['iOS']
        'checkbox'
"""
QUESTION_TYPES = read_json(DOCPATHS["question_type_names.json"])


"""Dictionary of power state events.
    Keys are ['Android', 'iOS'].
    POWER_EVENTS[<device_os>][<event>] is a tuple (<variable_name>, <code>).
    Note that <variable_name> does not appear in raw Beiwe data.  Events are
    bundled under a single variable name that corresponds to common reporting
    processes.
    The numeric <code> does not appear in raw Beiwe data.  These are intended
    to provide a convenient & consistent encoding for variable levels.
"""
POWER_EVENTS = {"Android": {}, "iOS": {}}  # type: ignore
with open(DOCPATHS["power_events.csv"], encoding="utf-8") as f:
    next(f)  # skip header row
    for line in f:
        line = line.replace("\n", "")  # get rid of line breaks
        event, device_os, variable, code = line.split(",")
        POWER_EVENTS[device_os][event] = (variable, int(code))
