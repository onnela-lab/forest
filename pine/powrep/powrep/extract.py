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



