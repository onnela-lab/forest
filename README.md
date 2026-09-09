[![build](https://github.com/onnela-lab/forest/actions/workflows/build.yml/badge.svg)](https://github.com/onnela-lab/forest/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/forest-docs/badge/)](https://forest.beiwe.org/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/beiwe-forest.svg)](https://pypi.org/project/beiwe-forest/)
[![status](https://joss.theoj.org/papers/98ea37f95e00c6a3f515b03a3214571b/status.svg)](https://joss.theoj.org/papers/98ea37f95e00c6a3f515b03a3214571b)

<!--- using a URL to display logo on PyPI --->
<img width="264" height="99" src="https://raw.githubusercontent.com/onnela-lab/forest/main/forest-logo-color.png" alt="Forest logo">

The Onnela Lab at the Harvard T.H. Chan School of Public Health has developed the Forest library to analyze smartphone-based high-throughput digital phenotyping data. The main intellectual challenge in smartphone-based digital phenotyping has moved from data collection to data analysis. Our research focuses on the development of mathematical and statistical methods for analyzing intensive high-dimensional data. We are actively developing the Forest library for analyzing smartphone-based high-throughput digital phenotyping data collected with the [Beiwe](https://github.com/onnela-lab/beiwe-backend) platform. Forest will implement our methods for analyzing Beiwe data as a Python package and is released under the BSD-3 open-source license. The Forest library will continue to grow over the coming years as we develop new analytical methods.

Forest can be run locally but is also integrated into the Beiwe back-end on AWS, consistent with the preferred big-data computing paradigm of moving computation to the data. Integrated with Beiwe, Forest can be used to generate on-demand analytics, most importantly daily or hourly summary statistics of collected data, which are stored in a relational database on AWS. The system also implements an API for Tableau, which supports the creation of customizable workbooks and dashboards to view data summaries and troubleshoot any issues with data collection. Tableau is commercial software but is available under free viewer licenses and may be free to academic users for the first year (see Tableau for more information).

For more detailed info on specific subpackages, see our [Documentation](https://forest.beiwe.org).

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

## Recent Additions and Improvements

### Optimizations
- Jasmine's PCR (Physical Circadian Rhythm) feature, which is highly computationally intensive, has been substantially optimized. It is roughly 13x faster and no longer spikes in memory usage.
- Other components of Jasmine also benefitted from these optimizations but were not as highly tracked.
- File read-in was found to use a slow approach is at least one scenario, this has been fixed and is about 2x faster.

### Features
- Forest now supports `.zst` compressed data files! These files can be downloaded directly from The Beiwe Platform, and also our in-development version of Mano. They are transparently consumed if present, and take up roughly 1/5th the space. (Decompression of these is on the order of gigabytes-per-second, so tend to be _faster_ to import than uncompressed .csv files, especially off slower storage devices.)

# Usage

### Installation

Forest requires Python version 3.12 or greater. <details> <summary>click for more Python version details</summary>

Python version 3.15 is not currently enabled in the `pyproject.toml` file because dependencies were not available at time of writing. To check if Forest is compatible with 3.15+, clone the repo (details below) and edit the `requires-python` line in the pyproject.toml file to a include a higher version number, then install in editable mode. At time of writing this process fails, but packages will be available soon, and we expect no incompatibilities.

Python free-threaded builds are not actively tested, however compatibility is not expected to require changes.

</details>

To install:

```bash
pip install beiwe-forest
# or, if you want support and our most up-to-date fixes, you can install our active development branch with:
pip install git+https://github.com/onnela-lab/forest
```

_Note that if you swap between versions of Forest (usually because we may not increment the package version number on our development branch) you might need run an uninstall of the old version first, and you may need to bypass the pip cache when you install._

```bash
# To do this crun these commands:
pip uninstall beiwe-forest
pip install --no-cache-dir git+https://github.com/onnela-lab/forest
```

### Installing Forest for Debugging

If you want to view, debug, or just swap between branches with lower friction, you can clone the repo and install that specific folder using Pip's `--editable` flag as your own local copy of the Forest package. This makes debugging much easier, and lets you view the code in a full IDE, like VS Code or PyCharm. If you are submitting a bug report we will request that you install the package in this way.

```bash
git clone https://github.com/onnela-lab/forest.git
# or, if you have GitHub SSH access configured, `git clone git@github.com:onnela-lab/forest.git`
# then cd into the forest folder you just cloned and install the package with development dependencies
cd forest
pip install --editable .\[dev]
# the \ above is for compatibility across commandline shells, it is required on the default shell, `zsh`, on Mac.
```

### Live Reloading Forest

If you find yourself in a situation where you are tweaking or developing Forest with data already loaded, frequently from within a Jupyter Notebook cell, you will find you have an older version of the Forest package loaded. You can reload the package manually like this:

```python
# Note that you have to reload the module containing your target function, and that further
# imports won't cascade, only the target _file_ will be reloaded.
import importlib
importlib.reload("forest.jasmine.traj2stats")
```

You can also try Jupyter Notebook's `%autoreload` magic command. This should watch files for changes.

```ipython
# Put these lines in your notebook cell and then just execute it normally.
# The first line loads the feature, the second line sets it to run universally.
# (If you want to see when reloads happen you will need to put print statements in the module you are reloading.)
%load_ext autoreload
%autoreload 2

# to disable it execute this in your notebook cell:
%autoreload 0
```

### To immediately test out Forest, adapt the filepaths in the code below and run:

```python
# Currently, all imports from `forest` must be explicit.  For the below example you need to import the following
# In the future, it would be great to have all functions import automatically
import datetime

from forest.bonsai.simulate_log_data import sim_log_data
from forest.bonsai.simulate_gps_data import sim_gps_data, gps_to_csv
from forest.jasmine.traj2stats import Frequency, gps_stats_main
from forest.willow.log_stats import log_stats_main

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
# number of persons to generate
n_persons = 1
# location of person to generate format: Country_2_letter_ISO_code/City_Name
location = "GB/Bristol"
# start date of generated trajectories
start_date = datetime.date(2021, 10, 1)
# end date of trajectories
end_date = datetime.date(2021, 10, 5)
# api key for openroute service, generated from https://openrouteservice.org/
api_key = "mock_api_key"
# Length of off-cycle + length of on-cycle in minutes
cycle = 15
# Length off-cycle / (length off-cycle + length on-cycle)
percentage = 0.8
# dictionary of personal attributes for each user, set to None if random, check Attributes class for usage in simulate_gps_data module.
personal_attributes = {
    "User 1":
    {
        "main_employment": "none", 
        "vehicle" : "car",
        "travelling_status": 10,
        "active_status": 7
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
sample_gps_data = sim_gps_data(n_persons, location, start_date, end_date, cycle, percentage, api_key, personal_attributes)
# save data in format of csv files
gps_to_csv(sample_gps_data, path_to_synthetic_gps_data, start_date, end_date)

# 2. Specify parameters for imputation 
# See https://forest.beiwe.org/en/latest/jasmine.html for details
# time zone where the study took place (assumes that all participants were always in this time zone)
tz_str = "Etc/GMT-1"
# Generate summary metrics e.g. Frequency.HOURLY, Frequency.DAILY or Frequency.HOURLY_AND_DAILY (see Frequency class in constants.py)
frequency = Frequency.DAILY
# Save imputed trajectories?
save_traj = False
# Hyperparameters class for imputation (default leave None), from forest.jasmine.traj2stats import Hyperparameters
parameters = None
# list of locations to track if visited, leave None if don't want these summary statistics
places_of_interest = ['cafe', 'bar', 'hospital']
# list of OpenStreetMap tags to use for identifying locations, leave None to default to amenity and leisure tagged locations or if you don't want to use OSM (see OSMTags class in constants.py)
osm_tags = None

# 3. Impute location data and generate mobility summary metrics using the simulated data above
gps_stats_main(
    study_folder = path_to_synthetic_gps_data,
    output_folder = path_to_gps_summary,
    tz_str = tz_str,
    frequency = frequency,
    save_traj = save_traj,
    parameters = parameters,
    places_of_interest = places_of_interest,
    osm_tags = osm_tags,
)

# 4. Generate daily summary metrics for call/text logs
option = Frequency.DAILY
time_start = None 
time_end = None
participant_ids = None

log_stats_main(path_to_synthetic_log_data, path_to_log_summary, tz_str, option, time_start, time_end, participant_ids)
```

## More info
* [Beiwe platform for smartphone data collection](http://www.beiwe.org/)
* [Onnela lab](https://www.hsph.harvard.edu/onnela-lab/)

## Publications
* Straczkiewicz, M., Huang, E.J., and Onnela, JP. A “one-size-fits-most” walking recognition method for smartphones, smartwatches, and wearable accelerometers. _npj Digit. Med._ **6**, 29 (2023) [![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41746--022--00745--z-blue)](https://doi.org/10.1038/s41746-022-00745-z) [Open Access](https://rdcu.be/c6dGV)
* Huang E, Yan K, and Onnela JP. Smartphone-Based Activity Recognition Using Multistream Movelets Combining Accelerometer and Gyroscope Data. Sensors 22 (7), 2618 (2022) [![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs22072618-blue)](https://doi.org/10.3390/s22072618)
* Onnela JP, Dixon C, Griffin K, Jaenicke T, Minowada L, Esterkin S, Siu A, Zagorsky J, and Jones E. Beiwe: A data collection platform for high-throughput digital phenotyping. Journal of Open Source Software, 6(68), 3417 (2021) [![DOI](https://joss.theoj.org/papers/10.21105/joss.03417/status.svg)](https://doi.org/10.21105/joss.03417)
* Liu G and Onnela JP. Bidirectional imputation of spatial GPS trajectories with missingness using sparse online Gaussian Process. Journal of the American Medical Informatics Association 28(8), 1777 (2021) [![DOI](https://img.shields.io/badge/DOI-10.1093%2Fjamia%2Focab069-blue)](https://doi.org/10.1093/jamia/ocab069)
* Barnett I and Onnela JP. Inferring mobility measures from GPS with missing data. Biostatistics 21:2, e98, 2020 [![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbiostatistics%2Fkxy059-blue)](https://doi.org/10.1093/biostatistics/kxy059) [Open Access](https://academic.oup.com/biostatistics/article/21/2/e98/5145908?guestAccessKey=0e3baa8c-2a80-405e-a7b4-1444099f48a2)
* Huang E and Onnela JP. Augmented movelet method for activity classification using smartphone gyroscope and accelerometer data. Sensors 20(13), 3706, 2020 [![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs20133706-blue)](https://doi.org/10.3390/s20133706) Dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3925679.svg)](https://doi.org/10.5281/zenodo.3925679)
