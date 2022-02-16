from ..legacy.common_funcs import datetime2stamp, stamp2datetime

# EASTERN = "America/New_York"
EASTERN = "US/Eastern"


def test_datetime2stamp_before_dst():
    assert datetime2stamp([2021, 11, 7, 0, 0, 0], EASTERN) == 1636257600


def test_datetime2stamp_during_dst():
    assert datetime2stamp([2021, 11, 7, 23, 0, 0], EASTERN) == 1636344000


def test_datetime2stamp_after_dst():
    assert datetime2stamp([2021, 11, 8, 23, 0, 0], EASTERN) == 1636430400


def test_stamp2datetime_before_dst():
    assert stamp2datetime(1636257600, EASTERN) == [2021, 11, 7, 0, 0, 0]


def test_stamp2datetime_during_dst():
    assert stamp2datetime(1636344000, EASTERN) == [2021, 11, 7, 23, 0, 0]


def test_stamp2datetime_after_dst():
    assert stamp2datetime(1636430400, EASTERN) == [2021, 11, 8, 23, 0, 0]
