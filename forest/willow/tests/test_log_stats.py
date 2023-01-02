import pandas as pd

from forest.constants import Frequency
from forest.willow.log_stats import comm_logs_summaries

ID = "6b38vskd"
STAMP_START = 1453837206
STAMP_END = 1454634000
TZ_STR = "America/New_York"
OPTION = Frequency.DAILY


def test_comm_log_summaries_with_empty_data():
    text_data = pd.DataFrame.from_dict({})
    call_data = pd.DataFrame.from_dict({})
    stats_pdframe = comm_logs_summaries(ID, text_data, call_data, STAMP_START,
                                        STAMP_END, TZ_STR, OPTION)
    assert isinstance(stats_pdframe, pd.DataFrame)


def test_comm_log_summaries_with_empty_text_data():
    text_data = pd.DataFrame.from_dict({})
    call_data = pd.DataFrame.from_dict(
        {'timestamp': {0: 1454428647649},
         'UTC time': {0: '2016-02-02T15:57:27.649'},
         'hashed phone number':
             {0: 'ZlGtb-SRRIgOcHLBD02d2_F049naF0YZbCx_CeP7jss='},
         'call type': {0: 'Missed Call'},
         'duration in seconds': {0: 0}}
    )
    stats_pdframe = comm_logs_summaries(ID, text_data, call_data, STAMP_START,
                                        STAMP_END, TZ_STR, OPTION)
    assert isinstance(stats_pdframe, pd.DataFrame)
