'''Extract events from raw Beiwe power state data files.

'''
import os
import logging
import numpy as np
import pandas as pd
from collections import OrderedDict
from beiwetools.helpers.functions import read_json
from beiwetools.helpers.process import clean_dataframe, stack_frames
from beiwetools.helpers.trackers import CategoryTracker
from .functions import *


logger = logging.getLogger(__name__)


# load events
this_dir = os.path.dirname(__file__)
events = read_json(os.path.join(this_dir, 'events.json'))


# organize events into powrep variables
def organize_events():
    categories = {}
    for opsys in events.keys():
        categories[opsys] = {}
        for k in events[opsys].keys():
            cat, value = events[opsys][k]
            if not cat in categories[opsys]:    
                categories[opsys][cat] = {}
            categories[opsys][cat][value] = k
    return(categories)
variables = organize_events()            
