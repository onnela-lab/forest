# forest

- This readme file can function as our current TODO list and a place for sharing info
- We'll want to look into https://readthedocs.org/ for documenting the library
- Let's make sure to loop in Josh B as soon as possible
- We should also look into continous integration; this look's especially promising: https://travis-ci.org/
- And for much later, more on Apache Spark here: https://spark.apache.org/


# plumtree 
Plumtree is a repo of code to summarize communication logs from Beiwe data. The fucntion "comm_logs" in "comm_logs.py" takes the path of the data folder and the path of the output folder as inputs, and generates individual hourly communucation logs as a CSV file for all the users in the given folder as outputs. The summary statistics includes the number of incoming calls, the number of outgoing calls, the number of missed calls, the number of callers, the number of callees, the number of unique missed calls, the total length of incoming calls, the total length of outgoing calls, the number of sent messages, the number of received messages, the number of senders, the number of receivers, the total number of words in sent messages, the total number of words in received messages, the number of sent MMS and the number of received MMS. 

# pagoda
Pagoda is a repo to deal with accelerometer data. The function "hourly_step_count" in "step_count_impute_stream.py" takes the path of the data folder and the path of the output folder, together with parameters hz, q, c, k, h as inputs, and generates individual hourly step count as a CSV file for all the users in the given folder as outputs. 
- hz: the sampling frequency you want to smooth over the raw data. For example, if you want a measure of acceleration every 0.1 second, then set hz to 10, the algorithm will calculate an average acceleration every that amount of time.
- q: A quantile threshold. The magnitudes should exceed q percentile during walking. q=75 is recommended.
- c: Another threshold. The magnitudes should exceed c\*g during walking. c=1.05 is recommended.
- k: In the algorithm, we need to impute the step counts when the sensor is off. It is done by sampling a step count from an empirical distribution derived from a fitbit dataset. The distribution is characterized by the hour of the day and the activity status in the nearby observed time windows (when the sensor is on). K is the parameter to determine how far we want to go as nearby time windows, for example, if k=10, then we need to check if there exists any walk 10 mins before and after the current timestamp. k=60 is recommended.
- h: A person should walk at least h steps in a minute to be considered as "walking". h=60 is recommended.

# sweet osmanthus
Sweet osmanthus is created for GPS data. The file "imputeGPS.py" can return hourly GPS summary statistics and full trajectories as CSV files once you specify the path of the data folder and the path of the output folder. To illustrate the long code, I decompose it to three parts in Jupyter Notebooks with outputs and graphs from each step. "GPS2MobMat.ipynb" is about how to covert longitude/lattitude to coordinates on a 2D plane, how to smooth the raw data over time, and how to summarize those raw data into trajectories (linear flights and pauses) using the rectangle method. "ImputeMobMat.ipynb" shows how to apply sparse online Gaussian Process to select representative flights and pauses, and then how to use those flights and pauses to impute the missing trajectories so this is the key part of the code. Finally, "Mat2Stats.ipynb" tells you how to get summary statistics from the imputed trajectories. 

# sweet olive
Sweet olive is same as sweet osmanthus, so is the code. When we applied the code in sweet osmanthus on the real data, we observed abnormal distances traveled if the user had long-distance flights. The reason is the projection of the latitude and longitude to the 2D plane will distort the coordinates near the edges. To fix this problem, sweet olive does every thing on the 3D sphere instead of the 2D plane. The usage of the function stays the same. Finally, "GetStats" function (feed full trajectories as inputs) has a "daily" or "hourly" option to choose, and "daily" option will give you more statistics like "number of significant places visited", "entropy", etc.

# coconut_tree
Coconut trees are pretty, so the code here generates good-looking graphs. The code is written in R, the R notebook is a good way to see what the functions do in the repo. The functions are (1) heatmaps for GPS quality check (2) Integrate all summary statistics from GPS, accelerometer and communication logs from the same user together as one big dataset (3) convert the hourly output to daily level for some of the statistics (4) reshape clinical events data, this one is ad-hoc, very specific to a dataset, not useful for general datasets (5) heatmaps for all the summary statistics from a user during the follow-up (6) linecharts for some statistics with the transparency reflecting the confidence we have in the estimates.

# poplar
This repo contains two directories.  The first directory is the `beiwetools` package.  Install this package with `pip install /path/to/beiwetools` or similar.

The second directory contains ipython notebooks with example code, and also some sample Beiwe study configuration files.  These configuration files correspond to studies found in the [public data sets](https://zenodo.org/record/1188879#.XcDUyHWYW02).

There are currently three sub-packages in `beiwetools`:

* `helpers`: Functions and classes for handling common scenarios, such as converting time formats, summarizing sensor sampling rates, and plotting timestamps,
* `configread`: Tools for querying Beiwe configuration files and generating study documentation,
* `manage`: Tools for organizing and summarizing directories of raw Beiwe data.

A fourth sub-package, `localize`, will be added shortly.  This sub-package  provides some classes and functions for incorporating each user's time zone into the analysis of processed Beiwe data.

The example notebooks and `beiwetools/README.md` provide an overview of the package.  More details are documented in the various modules.

Some features of text files from the Beiwe platform are documented in specific locations:

* The module `beiwetools/helpers/time_constants.py` includes common Beiwe time formats.
* Some raw data features are described in `data_streams.json` and `headers.py`, both found in `beiwetools/manage`.
* The contents of configuration settings are documented in three `JSON` files in `beiwetools/configread`.

