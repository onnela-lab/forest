#!/usr/bin/env python

"""
Run Jasmine algorithm on raw device data
"""

import argparse
import os

from forest.jasmine.traj2stats import gps_stats_main

parser = argparse.ArgumentParser()
parser.add_argument("raw_data_path", help="path to the folder with raw data",
                    type=str)
parser.add_argument("summary_data_path", help="path to the output folder",
                    type=str)
args = parser.parse_args()

summary_data_path = os.path.join(args.summary_data_path, "gps")
# time zone where the study took place
# (assumes that all participants were always in this time zone)
tz_str = "America/New_York"
option = "daily"  # generate summary metrics "hourly", "daily" or "both"
save_traj = False  # Save imputed trajectories?
time_start = None
time_end = None
beiwe_id = None
parameters = None
all_memory_dict = None
all_BV_set = None

os.makedirs(summary_data_path, exist_ok=True)
gps_stats_main(args.raw_data_path, summary_data_path, tz_str, option,
               save_traj, time_start, time_end, beiwe_id, parameters,
               all_memory_dict, all_BV_set)
