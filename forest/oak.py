from forest.oak.base import run
from forest import constants
import logging

def main():
    logging.getLogger().setLevel(logging.INFO)
    # Determine study folder and output_folder
    study_folder = "/Volumes/One Touch/cnoc/study1_decrypt"
    output_folder = "/Volumes/One Touch/cnoc/oak_output"

    # Determine study timezone and time frames for data analysis
    tz_str = "America/New_York"
    time_start = None
    time_end = None

    # Determine window for analysis. Available opts: "Hourly", "Daily", "both".
    frequency = constants.Frequency.DAILY
    users = None

    # Call the main function
    run(study_folder, output_folder, tz_str=tz_str, frequency=frequency, time_start=time_start, time_end=time_end, users=users)

if __name__ == '__main__':
    main()



