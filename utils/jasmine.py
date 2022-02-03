#!/usr/bin/env python

"""
Run Jasmine algorithm on raw device data
"""

import argparse
import os

from forest.jasmine.traj2stats import Frequency, gps_stats_main

parser = argparse.ArgumentParser()
parser.add_argument("raw_data_path", type=str,
                    help="path to the folder with raw data")
parser.add_argument("summary_data_path", type=str,
                    help="path to the output folder")
parser.add_argument(
    "--quality_threshold", type=float, default=0.05,
    help="a quality threshold for data worth reporting, value between 0 and 1"
)
args = parser.parse_args()

summary_data_path = os.path.join(args.summary_data_path, "gps")
os.makedirs(summary_data_path, exist_ok=True)

gps_stats_main(study_folder=args.raw_data_path,
               output_folder=summary_data_path, tz_str="Etc/GMT-1",
               frequency=Frequency.DAILY, save_traj=False,
               quality_threshold=args.quality_threshold)
