#!/usr/bin/env python

"""Compare output of summary API with output of Forest and print the result to
stdout in CSV format
"""

import argparse
import collections
import csv
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("api", help="path to summary API output file", type=str)
parser.add_argument("forest", help="path to Forest output file", type=str)
parser.add_argument("participant_id", help="study participant ID", type=str)
args = parser.parse_args()

variable_mapping = collections.OrderedDict([
    ("distance_diameter", "diameter"),
    ("distance_from_home", "max_dist_home"),
    ("distance_traveled", "dist_traveled"),
    ("flight_distance_average", "av_flight_length"),
    ("flight_distance_standard_deviation", "sd_flight_length"),
    ("flight_duration_average", "av_flight_duration"),
    ("flight_duration_standard_deviation", "sd_flight_duration"),
    ("gps_data_missing_duration", "missing_time"),
    ("home_duration", "home_time"),
])

forest_output = {}

with open(args.forest) as forest_file:
    for row in csv.DictReader(forest_file):
        date = f"{int(float(row['year']))}-" \
               f"{int(float(row['month'])):02}-" \
               f"{int(float(row['day'])):02}"
        forest_output[date] = row

diff_output = csv.writer(sys.stdout)
diff_output.writerow(["date"] + list(variable_mapping.keys()))

with open(args.api) as api_file:
    api_output = json.load(api_file)
    for api_entry in api_output:
        if api_entry["participant_id"] != args.participant_id:
            continue
        date = api_entry["date"]
        forest_entry = forest_output[date]
        output_row = [date]
        for json_name, csv_name in variable_mapping.items():
            try:
                output_row.append(
                    api_entry[json_name] - float(forest_entry[csv_name])
                )
            except KeyError:
                # ignore missing variables in Forest output for now
                pass
        diff_output.writerow(output_row)
