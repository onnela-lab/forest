import argparse
import sys

from forest.sycamore.base import compute_survey_stats

parser = argparse.ArgumentParser(description='Run Sycamore Locally')

parser.add_argument('--output_dir', default=None, type=str,
                    help="Directory to write Output .csv files")
parser.add_argument(
    '--beiwe_ids', default=None, type=str, nargs='+',
    help="IDs of participants to run sycamore on. If not specified, sycamore "
         "analyzes all IDs included in study_folder")
parser.add_argument(
    '--study_folder', default=None, type=str,
    help="Folder where survey_timings data is downloaded. The subdirectories "
         "of this folder should be the Beiwe IDs which will be analyzed."
)
parser.add_argument('--time_start', default=None, type=str,
                    help="Start Date of Analysis in YYYY-MM-DD")
parser.add_argument('--time_end', default=None, type=str,
                    help="End Date of Analysis in YYYY-MM-DD")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path to survey config file downloaded from website")
parser.add_argument(
    '--tz_str', default="UTC", type=str,
    help="Time Zone to use to generate local times. Default is 'UTC'"
         "See https://en.wikipedia.org/wiki/List_of_tz_database_time_zones"
)
parser.add_argument(
    '--interventions_filepath', default=None, type=str,
    help="Path to interventions json file downloaded from website"
)
parser.add_argument(
    '--augment_with_answers', default=None, type=str,
    help="whether to augment survey_timings data with survey_answers data when"
         "data appears to be missing."
)

args = parser.parse_args()

if args.study_folder is None or args.output_folder is None:
    parser.print_help()
    sys.exit()

compute_survey_stats(output_folder=args.output_dir,
                     study_folder=args.study_folder,
                     tz_str=args.tz_str, beiwe_ids=args.beiwe_ids,
                     time_start=args.time_start, time_end=args.time_end,
                     config_path=args.config_path,
                     interventions_filepath=args.interventions_filepath,
                     augment_with_answers=args.augment_with_answers)
