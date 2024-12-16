"""Tools for identifying dates that are holidays."""
from logging import getLogger
import datetime
import holidays
from ..constants.time import DATE_FORMAT


logger = getLogger(__name__)


US_HOLIDAYS = holidays.UnitedStates()  # type: ignore[attr-defined]


def is_us_holiday(date, date_format=DATE_FORMAT):
    """Identify dates that are US holidays.
    There is probably a better way to do this with pandas.

    Args:
        date (str): Date string.
        date_format (str): Format of date.

    Returns:
        is_holiday (bool): True if the date is a US holiday.
    """
    try:
        datetime_date = datetime.datetime.strptime(date, date_format)
        is_holiday = datetime_date in US_HOLIDAYS
        return is_holiday
    except ValueError:
        logger.warning("Unable to determine holiday status: %s.", date)
        return False
