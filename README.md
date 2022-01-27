[![build](https://github.com/onnela-lab/forest/actions/workflows/build.yml/badge.svg)](https://github.com/onnela-lab/forest/actions/workflows/build.yml)

<img width="264" height="99" src="forest-logo-color.png" alt="Forest logo">

# Forest (Python 3.8)

The Onnela Lab at the Harvard T.H. Chan School of Public Health has developed the Forest library to analyze smartphone-based high-throughput digital phenotyping data. The main intellectual challenge in smartphone-based digital phenotyping has moved from data collection to data analysis. Our research focuses on the development of mathematical and statistical methods for analyzing intensive high-dimensional data. We are actively developing the Forest library for analyzing smartphone-based high-throughput digital phenotyping data collected with the [Beiwe](https://github.com/onnela-lab/beiwe-backend) platform. Forest will implement our methods for analyzing Beiwe data as a Python 3.8 package and is released under the BSD-3 open-source license. The Forest library will continue to grow over the coming years as we develop new analytical methods.

Forest can be run locally but is also integrated into the Beiwe back-end on AWS, consistent with the preferred big-data computing paradigm of moving computation to the data. Integrated with Beiwe, Forest can be used to generate on-demand analytics, most importantly daily or hourly summary statistics of collected data, which are stored in a relational database on AWS. The system also implements an API for Tableau, which supports the creation of customizable workbooks and dashboards to view data summaries and troubleshoot any issues with data collection. Tableau is commercial software but is available under free viewer licenses and may be free to academic users for the first year (see Tableau for more information).

For more detailed info on specific subpackages, see our [Wiki](https://github.com/onnela-lab/forest/wiki). Please note that Forest uses Python 3.8.

# Description

Description of how beiwe data looks (folder structure + on/off cycles)

Input: typically raw data from smartphones
Output: typically summary files

- Creating synthetic data
  - Want to try out our methods, but don't have smartphone data at hand? Use **bonsai**
- Data preparation
  - Identifying time zones and unit conversion: use **poplar**
  - Collate Beiwe survey data into .csvs per participant or per study: use **sycamore**
- Data imputation
  - State-of-the-art GPS imputation: use **jasmine**
- Data summarizing (see tables below for summary metrics)
  - Mobility metrics from GPS data: use **jasmine**
  - Daily summaries of call & text metadata: use **willow**
  - Survey completion time from survey metadata: use **sycamore**

# Usage

To install, clone this repository to a local directory and then:

```console
pip install path/to/forest
```

Alternatively, [install directly from github](https://pip.pypa.io/en/stable/reference/pip_install/#git) with `pip`. As the repo is public, it won't prompt you to login. If you've used forest in the past, it might be prudent to do a '''pip uninstall forest''' first.

```console
pip install git+https://github.com/onnela-lab/forest
```

Currently, all imports from `forest` must be explicit.  For the below example you need to import:

```python
from forest.bonsai.simulate_log_data import sim_log_data
from forest.bonsai.simulate_gps_data import sim_gps_data
from forest.jasmine.traj2stats import gps_stats_main
from forest.willow.log_stats import log_stats_main
```

To immediately test out forest, adapt the filepaths in the code below and run:

```python
# Run the above 4 imports
# In future, it would be great to have all functions import automatically
import datetime

from forest.jasmine.traj2stats import Frequency, Hyperparameters

# 1. If you don't have any smartphone data (yet) you can generate fake data
path_to_synthetic_gps_data = "ENTER/PATH1/HERE"
path_to_synthetic_log_data = "ENTER/PATH2/HERE"
path_to_gps_summary = "ENTER/PATH/TO/DESIRED/OUTPUT/FOLDER1/HERE"
path_to_log_summary = "ENTER/PATH/TO/DESIRED/OUTPUT/FOLDER2/HERE"

# Generate fake call and text logs 
# Because of the explicit imports, you don't have to precede the functions with forest.subpackage.
sim_log_data(path_to_synthetic_log_data)

# Generate synthetic gps data and communication logs data as csv files
# Define parameters for generating the data
# To save smartphone battery power, we typically collect location data intermittently: e.g. during an on-cycle of 3 minutes, followed by an off-cycle of 12 minutes. We'll generate data in this way
n_persons = 1 # number of persons to generate
location = "GB/Bristol" # location of person to generate format: Country_2_letter_ISO_code/City_Name
start_date = datetime.date(2021, 10, 1) # start date of generated trajectories
end_date = datetime.date(2021, 10, 5) # end date of trajectories
api_key = "mock_api_key" # api key for openroute service, generated from https://openrouteservice.org/
cycle = 15 # Length of off-cycle + length of on-cycle in minutes
percentage = 0.8 # Length off-cycle / (length off-cycle + length on-cycle)
attributes_dict = None # dictionary of personallity attributes for each user, set to None if random, check Attributes class for usage in simulate_gps_data module. Example of dictionary of user attributes:
attributes_dict = {
    "User 1":
    {
        "main_employment": "none", 
        "vehicle" : "car",
        "travelling_status": 10,
        "active_status": 0
    },

    "Users 2-4":
    {
        "main_employment": "university",
        "vehicle" : "bicycle",
        "travelling_status": 8,
        "active_status": 8,
        "active_status-16": 2 
    },

    "User 5":
    {
        "main_employment": "office",
        "vehicle" : "foot",
        "travelling_status": 9,
        "travelling_status-20": 1,
        "preferred_exits": ["cafe", "bar", "cinema"] 
    }
}
sample_gps_data = sim_gps_data(n_persons, location, start_date, end_data, api_key, cycle, percentage, attributes_dict)

# 2. Specify parameters for imputation 
# See https://github.com/onnela-lab/forest/wiki/Jasmine-documentation#input for details
tz_str = "America/New_York" # time zone where the study took place (assumes that all participants were always in this time zone)
frequency = Frequency.DAILY # Generate summary metrics Frequency.HOURLY, Frequency.DAILY or Frequency.BOTH
save_traj = False # Save imputed trajectories?
parameters = None # Hyperparameters class for imputation (default leave None)
places_of_interest = ['cafe', 'bar', 'hospital'] # list of locations to track if visited, leave None if don't want these summary statistics
save_log = False # True if want to save a log of all locations and attributes of those locations visited
threshold = 15 # threshold of time spent in a location to count as being in that location, in minutes
split_day_night = False # split summary statistics to day and nighttime statistics, only if frequency == Frequency.DAILY
person_point_radius = 2 # meters around person to count as being in a place
place_point_radius = 7.5 # meters around location to count as POI
time_start = None 
time_end = None
participant_ids = None
all_memory_dict = None
all_bv_set= None
quality_threshold = 0.05 # minimum percentage of data being valid needed to perform the analysis

# 3. Impute location data and generate mobility summary metrics using the simulated data above
gps_stats_main(path_to_synthetic_gps_data, path_to_gps_summary, tz_str, frequency, parameters, places_of_interest, save_log, threshold, split_day_night, person_point_radius, place_point_radius, time_start, time_end, participant_ids. all_memory_dict, all_bv_set, quality_threshold)

# 4. Generate daily summary metrics for call/text logs
log_stats_main(path_to_synthetic_log_data, path_to_log_summary, tz_str, option, time_start, time_end, beiwe_id)
```

## For contributors

## More info

[Beiwe platform for smartphone data collection](https://www.beiwe.org/)
[Onnela lab](https://www.hsph.harvard.edu/onnela-lab/)

## Version

Forest 1.0
