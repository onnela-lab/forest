
################################################################################
# Overview of Poplar
################################################################################

# Contains the beiwetools package
	pip install /path/to/poplar/beiwetools
	pipenv install /path/to/poplar/beiwetools

# About 15 modules organized into sub-packages:
#		beiwetools.helpers
#		beiwetools.configread
#		beiwetools.manage
#		beiwetools.localize (in progress)


# Functions for common tasks (e.g. time format conversions),
# Common data management tasks,
# Functions for localization (timezone conversions),
# Documentation of contents of raw data files & configuration files.
# README, examples in ipython notebooks, lots of comments & docstrings.


################################################################################
# Data management with Poplar tools
################################################################################

# Limitations:
# Designed for handling static data sets.
#	- e.g. Fixed subset of days of participant data,
#	- Fixed followup period for a particular study. 
# For online Forest algorithms, we would like online data management.
#	- Poplar data representations need to be rebuilt to accomodate new data.
# 	- Poplar data structures & classes don't have an "update" method!
# More data management issues as followup period gets longer: 
#	- More likely that phones are replaced, 
#	- Possibility of edits to study settings,
#	- Accumulation of different timezones (travel & DST).


# Goals:
# Identify elements necessary for analyzing Beiwe data.
# What do we need to go from raw files to patient-level summaries?
#	- e.g. raw accelerometer data -> daily step counts
# At least four elements:
#	1. Locations of raw data files,
#	2. Study settings (sampling rates & survey contents),
#	3. Phone information (e.g. iOS or Android?),
#	4. Timezone transitions for localization.
# How do we deliver these elements efficiently to a pipeline function?
#	- Each element corresponds to a bookkeeping problem.
# 	- Poplar's solution is a separate set of data structures/classes for each element.
#	- May be better solutions for an online data management framework.


# 1. Locations of raw data files:
 	 from beiwetools.manage.classes import UserData
 	 udata = UserData(<raw file directories>)
	 udata.passive['files']['accelerometer']
	 udata.passive['bytes']['accelerometer']


# 2. Study settings (sampling rates & survey contents):
	 from beiwetools.configread import BeiweConfig
	 config = BeiweConfig(path/to/configuration/file)
	 config.settings.passive['accelerometer_off_duration_seconds']
	 config.settings.passive['accelerometer_on_duration_seconds']

		
# 3. Device information (e.g. iOS or Android?):
	 from beiwetools.manage.classes import DeviceInfo
	 dinfo = DeviceInfo(<paths to raw identifier files>)
	 dinfo.history('device_os')


# 4. Timezone transitions for localization:
#	 - First extract timezone transitions:
	 from gpsrep.coarse import Coarse
	 Coarse.do(udata.passive['files']['gps'],
			   <some other arguments>)
#	 - Then lookup local datetimes for UTC timestamps:
	 from beiwetools.localize.classes import Localizer
	 loc = Localizer(path/to/timezone/dictionary)
	 loc.get(<UTC timestamp>)


# 5. A single class that handles 1, 2, 3:
	 from beiwetools.manage.classes import BeiweProject
	 p = BeiweProject.create(<raw file directories>)
	 p.data[<user_id>] 		  # the participant's UserData object	 
	 p.data[<user_id>].device # the participant's DeviceInfo object
	 p.configurations # dictionary of BeiweConfig objects

