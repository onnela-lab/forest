[![build](https://github.com/onnela-lab/forest/actions/workflows/build.yml/badge.svg)](https://github.com/onnela-lab/forest/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/forest-docs/badge/)](https://forest.beiwe.org/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/beiwe-forest.svg)](https://pypi.org/project/beiwe-forest/)
[![status](https://joss.theoj.org/papers/98ea37f95e00c6a3f515b03a3214571b/status.svg)](https://joss.theoj.org/papers/98ea37f95e00c6a3f515b03a3214571b)

<!--- using a URL to display logo on PyPI --->
<img width="264" height="99" src="https://raw.githubusercontent.com/onnela-lab/forest/main/forest-logo-color.png" alt="Forest logo">

The Onnela Lab at the Harvard T.H. Chan School of Public Health has developed the Forest library to analyze smartphone-based high-throughput digital phenotyping data. The main intellectual challenge in smartphone-based digital phenotyping has moved from data collection to data analysis. Our research focuses on the development of mathematical and statistical methods for analyzing intensive high-dimensional data. We are actively developing the Forest library for analyzing smartphone-based digital phenotyping data collected with [The Beiwe Platform](https://github.com/onnela-lab/beiwe-backend). Forest implements our methods for analyzing Beiwe data as a Python package and is released under the BSD-3 open-source license. The Forest library will continue to grow over the coming years as we develop new analytical methods.

Forest can be run locally but is also integrated into the Beiwe back-end on AWS, consistent with the preferred big-data computing paradigm of moving computation to the data. Integrated with Beiwe, Forest can be used to generate on-demand analytics, most importantly daily or hourly summary statistics of collected data, which are stored in a relational database on AWS. The system also implements an API for our data-download tool Mano, and one for the Tableau platform.

#### For more detailed info on specific Forest Tree submodules, [please see our Documentation]at https://forest.beiwe.org(https://forest.beiwe.org).

#### If you are downloading bulk or live data we recommend our sister package [Mano](https://github.com/onnela-lab/mano), which you can use to automate data download from and file management for The Beiwe Platform.

## Recent Additions and Improvements

### Optimizations
- Jasmine's PCR (Physical Circadian Rhythm) feature, which is highly computationally intensive, has been substantially optimized. It is roughly 13x faster and no longer spikes in memory usage.
- Other components of Jasmine also benefitted from these optimizations but were not as highly tracked.
- File read-in was found to use a slow approach is at least one scenario, this has been fixed and is about 2x faster.

### Features
- Forest now supports `.zst` compressed data files! These files can be downloaded directly from The Beiwe Platform, and also our in-development version of Mano. They are transparently consumed if present, and take up roughly 1/5th the space. (Decompression of these is on the order of gigabytes-per-second, so tend to be _faster_ to import than uncompressed .csv files, especially off slower storage devices.)

# Usage

### Installation

Forest requires Python version 3.12 or greater.

<details> <summary><i><b>Click Here</b> for more Python version details.</i></summary>

Python version 3.15 is not currently enabled in the `pyproject.toml` file because dependencies were not available at time of writing. To check if Forest is compatible with 3.15+, clone the repo (details below) and edit the `requires-python` line in the pyproject.toml file to a include a higher version number, then install in editable mode. At time of writing this process fails, but packages will be available soon, and we expect no incompatibilities.

Python free-threaded builds are not actively tested, however compatibility is not expected to require changes.

----

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

<details> <summary><i><b>Click Here</b> for a useful tip on how to live-reload Forest in your Jupyter Notebook file </i> </summary>

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

---

</details>

# The Forest Tree Submodules

The source data for all Forest Trees is typically the raw data from smartphones, as collected and formatted by the
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



## Trying Out Forest

To quickly try out Forest, adapt the filepaths in the code below. You may want to use Jupyter Notebook, in which case you may want to separate this code into multiple cells.  This simple example is a walkthrough of how to generate synthetic data, impute GPS trajectories, and generate summary metrics for both GPS and call/text logs.

Forest makes use of the OpenRouteService API from [https://openrouteservice.org/](https://openrouteservice.org/). To run the code below you will need to generate a key. If you do not have an account the easiest option is to use your GitHub account to sign in.  The service is free and open source.

```python
# We use the Python datetime library and these 4 Forest Trees.
from datetime import date
from forest.bonsai.simulate_log_data import sim_log_data
from forest.bonsai.simulate_gps_data import sim_gps_data, gps_to_csv
from forest.constants import Frequency
from forest.jasmine.traj2stats import Frequency, gps_stats_main
from forest.willow.log_stats import log_stats_main

# Since we don't have any smartphone data (yet) we can generate fake data
path_to_synthetic_gps_data = "ENTER/PATH1/HERE"
path_to_synthetic_log_data = "ENTER/PATH2/HERE"
path_to_gps_summary = "ENTER/PATH/TO/DESIRED/OUTPUT/FOLDER1/HERE"
path_to_log_summary = "ENTER/PATH/TO/DESIRED/OUTPUT/FOLDER2/HERE"

# Generate call and text logs
sim_log_data(path_to_synthetic_log_data)

# To generate synthetic GPS data and communication logs data as csv files
# define these  parameters for generating the data
# Location data is collected intermittently on The Beiwe Platform, e.g. you have an
# on-cycle of 3 minutes, followed by an off-cycle of 12 minutes.
# We'll generate data to match this pattern.

number_persons = 1

# location for which we will generate data. This uses those standard 2-character country codes,
# which have the somewhat clunky designation "ISO 3166-1 alpha-2 Country Codes",
# followed by a city name. We will use Bristol in the UK, which uses "GB", for "Great Britain".
# You can find them here: https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
location = "GB/Bristol"

# Setstart date and end-date for our generated trajectories
start_date = date(2021, 10, 1)
end_date = date(2021, 10, 5)

# api key for openroute service, generated from https://openrouteservice.org/
api_key = "fill_me_in!"

# We use a full duration and a fractional value to represent our on and off recording cycles.
# (Note that real world data is always more complicated than this number represents.)
full_duration = 15
fraction_disabled = 12/15  # 12-of-15 minutes off, so 3 minutes on. (equal to 0.8)

# Dictionary of personal attributes for each of our generated participants.
# Set values to None to generate random values.  See the Attributes class
# in `forest.bonsai.simulate_gps_data` for more information on simulating these values.
personal_attributes = {
    "Participant 1":
    {
        "main_employment": "none",
        "vehicle" : "car",
        "travelling_status": 10,
        "active_status": 7
    },
    
    "Participants 2-4":
    {
        "main_employment": "university",
        "vehicle" : "bicycle",
        "travelling_status": 8,
        "active_status": 8,
        "active_status-16": 2
    },
    
    "Participant 5":
    {
        "main_employment": "office",
        "vehicle" : "foot",
        "travelling_status": 9,
        "travelling_status-20": 1,
        "preferred_exits": ["cafe", "bar", "cinema"]
    }
}

# now we can generate our synthetic data...
sample_gps_data = sim_gps_data(
  number_persons,
  location,
  start_date,
  end_date,
  full_duration,
  fraction_disabled,
  api_key,
  personal_attributes
)

# ... and save it in some csv files
gps_to_csv(sample_gps_data, path_to_synthetic_gps_data, start_date, end_date)


# 2) Specify parameters for imputation
# See https://forest.beiwe.org/en/latest/jasmine.html for details on exactly what these parameters do.

# set a time zone to center around, usually the home location a study. We will use a random fixed
# timezone to avoid daylight savings for our example purposes.
tz_str = "Etc/GMT-1"

# Use the Frequency class (defined in forest.constants) to specify the granularity of your summary
# metrics. The most common options are Frequency.DAILY and Frequency.HOURLY.
frequency = Frequency.DAILY

# Save imputed trajectories
save_traj = False

# We aren't going to set any Hyperparameters for now, but we would create a Hyperparameters instance
# (`from forest.jasmine.jasmine_common import Hyperparameters`) with defined values if were.
# See the larger Jasmine Tree documentation for details on what those values are.
hyper_parameters = None

# a list of nearby location types to for tracking in our output
# (leave None if don't want these summary statistics, they can make the openrouteservice API calls
# take longer, or if they are too large they may return errors. That shouldn't occur in this example.)
places_of_interest = ['cafe', 'bar', 'hospital']

# A list of OpenStreetMap tags to use for further identifying of locations.
# We will leave this as None, which defaults to the "amenity and leisure" tagged location types.
# For more information see the OpenStreetMap wiki: https://wiki.openstreetmap.org/wiki/Map_features
# and use our OSMTags class in forest.jasmine.jasmine_common for options.
osm_tags = None

# 3) Impute Location Data!
# This will generate mobility summary metrics using the simulated data above
# Note: since we aren't saving the trajectories you will get different data every time you run this.
gps_stats_main(
    study_folder = path_to_synthetic_gps_data,
    output_folder = path_to_gps_summary,
    tz_str = tz_str,
    frequency = frequency,
    save_traj = save_traj,
    parameters = hyper_parameters,
    places_of_interest = places_of_interest,
    osm_tags = osm_tags,
)

# 4) Generate daily summary metrics for call/text logs
logs_frequencies = Frequency.DAILY
logs_time_start = None
logs_time_end = None
participant_ids = None

log_stats_main(
    path_to_synthetic_log_data,
    path_to_log_summary,
    tz_str,
    logs_frequencies,
    logs_time_start,
    logs_time_end,
    participant_ids
)
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
