#!/usr/bin/env python

"""
Run Willow algorithm on raw device data
"""

import argparse
import os

from forest.willow.log_stats import log_stats_main

parser = argparse.ArgumentParser()
parser.add_argument("raw_data_path", help="path to the folder with raw data",
                    type=str)
parser.add_argument("summary_data_path", help="path to the output folder",
                    type=str)
args = parser.parse_args()

summary_data_path = os.path.join(args.summary_data_path, "log")
# time zone where the study took place
# (assumes that all participants were always in this time zone)
tz_str = "America/New_York"
option = "daily"  # generate summary metrics "hourly", "daily" or "both"
time_start = None
time_end = None
beiwe_id = None

log_stats_main(args.raw_data_path, summary_data_path, tz_str, option,
               time_start, time_end, beiwe_id)
