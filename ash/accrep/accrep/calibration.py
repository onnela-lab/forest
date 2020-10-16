'''Get reference periods from summarized Beiwe accelerometer data.

    - Do this AFTER accrep.summary.Summary.do().

'''
import os
from logging import getLogger
import numpy as np
import pandas as pd


from beiwetools.helpers.time import (local_now, to_timestamp, 
                                     filename_time_format, hour_ms, day_ms)
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import setup_directories, write_json


logger = getLogger(__name__)


def user_calibration(user_id, data_path, timestamp_range = None,
                     epoch_days = 10, force_breaks = None,
                     minimize = None, minimum_s = None, minimum_obs = None):
    '''
    user_id (str)
    data_path (str)
    timestamp_range (list or tuple)
    epoch_days (int)
    force_breaks (list)
    minimize (list)
    minimum_s (int) 
    minimum_obs (int)
    '''    
    filepath = os.path.join(data_path, user_id + '.csv')
    data = pd.read_csv(filepath)
    epoch_ms = epoch_days * day_ms
    if timestamp_range is None:
        timestamp_range = (data.timestamp[0], 
                           data.timestamp[len(data)-1])
    if not force_breaks is None:
        pass # what to do if there are multiple devices
    if minimize is None:
        minimize = list(data.columns)[4:]
    if minimum_s is None: minimum_s = 1
    if minimum_obs is None: minimum_obs = 1    
    data = data.loc[(data.elapsed_s >= minimum_s) & 
                    (data.n_observations >= minimum_obs)]
    data.set_index(np.arange(len(data)), inplace = True)
    breaks = np.arange(timestamp_range[0], timestamp_range[1], 
                       epoch_ms).tolist()[:-1]
    breaks.append(timestamp_range[1]+hour_ms)
    [(breaks[i+1] - breaks[i])/day_ms for i in range(len(breaks)-1)]
    reference_periods = dict(zip(minimize, [{} for m in minimize]))
    for i in range(len(breaks)-1):
        temp = data.loc[(data.timestamp >= breaks[i]) & 
                        (data.timestamp < breaks[i+1])]
        for m in minimize:
            try: 
                index = temp[m].idxmin()
                filepath = temp['filepath'][index]
                reference_periods[m][breaks[i]] = filepath
            except:
                logger.warning('Unable to find reference period for epoch: ' +\
                               '%s - %s' % (breaks[i], breaks[i+1]))
    return(reference_periods)


# def setup_calibration():

#     logger.info('Finished setting up output directories.')
#     pass
    

# def do_calibration(user_ids):

#     log_to_csv()
#     setup_calibration()

#     for user_id in user_ids:
#         try:
#             cal_dict = user_calibration()
#             write_json(cal_dict)
#         except:
#             logger.warning('Unable to do calibration for user: %s' % user_id)
#     logger.info('Finished.')
#     pass

    