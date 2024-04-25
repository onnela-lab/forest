""" Code for fomatting & exporting logging messages.

- Will require some modifications to direct logging output to S3, see:
    https://stackoverflow.com/questions/51070891/how-can-i-write-logs-directly-to-aws-s3-from-memory-without-first-writing-to-std

"""
import logging
from typing import List

from .io import setup_csv


logger = logging.getLogger(__name__)


# Dictionary of available log record attributes:
# For details, see:
# https://docs.python.org/3.11/library/logging.html#logrecord-attributes
AVAILABLE_ATTRIBUTES = {
    "asctime,msecs": "%(asctime)s",  # Human-readable time with milliseconds.
    "created": "%(created)f",  # Unix timestamp (seconds since epoch).
    "filename": "%(filename)s",  # Filename portion of pathname.
    "funcName": "%(funcName)s",  # Originating function.
    "levelname": "%(levelname)s",  # Message level, one of:
    #   DEBUG, INFO, WARNING, ERROR, CRITICAL
    "levelno": "%(levelno)s",  # Numeric message level.
    "lineno": "%(lineno)d",  # Source line number, if available.
    "message": "%(message)s",  # Logged message.
    "module": "%(module)s",  # Module name.
    "msecs": "%(msecs)d",  # Millisecond portion of timestamp.
    "name": "%(name)s",  # Name of originating logger.
    "pathname": "%(pathname)s",  # Path to originating file.
    "process": "%(process)d",  # Process id, if available.
    "processName": "%(processName)s",  # Process name, if available.
    "relativeCreated": "%(relativeCreated)d",  # Milliseconds
    # since logging was loaded.
    "thread": "%(thread)d",  # Thread id, if available.
    "threadName": "%(threadName)s",  # Thread name, if available.
}


class ExtendedLogFormat:
    """
    Class for assembling logging format and header for writing records to CSV.
    """
    def __init__(self, attribute_list: list, available_attributes: dict):
        """
        Args:
            attribute_list (list): List of keys from available_attributes.
            available_attributes (dict): Dictionary of available log record
                attributes (see above).
        """
        self.attributes = ",".join(
            [available_attributes[attr] for attr in attribute_list]
        )
        self.header = []
        for attr in attribute_list:
            if "," in attr:
                self.header += attr.split(",")  # hack for asctime
            else:
                self.header.append(attr)


def attributes_to_csv(attribute_list: list) -> ExtendedLogFormat:
    """
    Given a list of attributes (keys of AVAILABLE_ATTRIBUTES), returns a
    logging format with header for writing records to CSV.

    Args:
        attribute_list (list): List of keys from AVAILABLE_ATTRIBUTES.

    Returns:
        extended_format (object):
            extended_format.attributes (str): format for logging.basicConfig.
            extended_format.header (list): header for the corresponding csv.

    """
    try:
        extended_format = ExtendedLogFormat(
            attribute_list, AVAILABLE_ATTRIBUTES
        )
        return extended_format
    except Exception:
        logger.warning("Unable to assemble logging format and header.")
        return ExtendedLogFormat(
            [], AVAILABLE_ATTRIBUTES
        )


BASIC_CSV_LOG = attributes_to_csv(
    [
        # simple format for logging messages
        "created",
        "asctime,msecs",
        "levelname",
        "module",
        "funcName",
        "message",
    ]
)


TRACEBACK_CSV_LOG = attributes_to_csv(
    [
        # more comprehensive format for logging messages
        # (includes traceback info)
        "created",
        "asctime,msecs",
        "levelname",
        "module",
        "funcName",
        "message",
        "lineno",
        "pathname",
    ]
)


def log_to_csv(
    log_dir: str,
    level: int = logging.DEBUG,
    log_name: str = "log",
    log_format: str = BASIC_CSV_LOG.attributes,
    header: List[str] = BASIC_CSV_LOG.header,
) -> None:
    """
    Configure the logging system to write messages to a csv.
    Overwrites any existing logging handlers and configurations.

    Args:
        log_dir (str): Path to directory where log messages should be written.
        level (int):  An integer between 0 and 50.
            Set level = logging.DEBUG to log all messages.
        log_name (str): Name for the log file.
        log_format (str): The format argument for logging.basicConfig.
            For available attributes and formatting instructions, see:
            https://docs.python.org/3.11/library/logging.html#logrecord-attributes
        header (list): Header for the csv.

    Returns:
        None
    """
    try:
        # initialize csv
        filepath = setup_csv(name=log_name, dirpath=log_dir, header=header)
        # configure logging output
        logging.basicConfig(
            format=log_format, filename=filepath, level=level, force=True
        )
        # success message
        logger.info("Writing log messages to %s.csv...", log_name)
    except Exception:
        logger.warning("Unable to write logging messages.")
