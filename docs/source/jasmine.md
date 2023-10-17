# Jasmine

## Usage
Jasmine provides a forest implementation of GPS trajectory imputation as well as hourly and daily summarization. 

## Installation Instruction
For instructions on how to install forest, please visit [here](https://github.com/onnela-lab/forest). 
`from forest import jasmine`

## Data

### Input

When using jasmine, you should call function `gps_stats_main(study_folder, output_folder, tz_str, frequency, save_traj, places_of_interest = None, osm_tags = None, time_start = None, time_end = None, participant_ids = None, parameters = None, all_memory_dict = None, all_bv_set = None)` in the `traj2stats` module and specify:
   - `study_folder`, string, the path of the study folder. The study folder should contain individual participant folder with a subfolder `gps` inside
   - `output_folder`, string, the path of the folder where you want to save results

Furthermore, if you want to use jasmine for some participants only or for some time only, you can specify:
   - `participant_ids`: a list of beiwe IDs. If it is set to None (default), then it is a list of all available beiwe IDs in your study folder.
   - `time_start`, `time_end` are starting time and ending time of the window of interest.  
     The time should be a list of integers with format [year, month, day, hour, minute, second] (default: None).  
     If `time_start` is None and `time_end` is None: then it reads all the available files.  
     If `time_start` is None and `time_end` is given, then it reads all the files before the given time.  
     If `time_start` is given and `time_end` is None, then it reads all the files after the given time.   

In addition, the main function takes four arguments that provide further flexibility:  
   - `tz_str`, string, the timezone where the study is/was conducted. Please use "`pytz.all_timezones`" to check all options. For example, "America/New_York".
   - `frequency`, Frequency class, the frequency of the summary stats (resolution for summary statistics) e.g. Frequency.HOURLY, Frequency.DAILY, etc.
   - `save_traj`, bool, True if you want to save the trajectories as a csv file, False if you don't (default: False).
   - `places_of_interest`, a list of places of interest, by default it is set to None. The details are as used in openstreetmaps
   - `osm_tags`, list of OSMTags class, a list of tags to filter the places of interest, by default it is set to None. The details are as used in openstreetmaps. Avoid using a lot of them if large area is covered.
   - `parameters`, a list of parameters, by default it is set to None. The details are as below.
   - `all_memory_dict` and `all_bv_set` are dictionaries from previous run (none if it's the first time).

You can also tweak the parameters that change the assumptions of the imputation and summary statistics. The parameters are 

(1) `l1`: the scale parameter in the abs function in the daily kernel;   
(2) `l2`: the scale parameter in the abs function in the weekly kernel;    
(3) `l3`: the scale parameter in the geographical kernel if only latitude or longitude is used;   
(4) `g`: the scale parameter in the geographical kernel if both latitude and longitude are used;   
(5) `a1`: the scale parameter in the sin function in the daily kernel;   
(6) `a2`: the scale parameter in the sin function in the weekly kernel;   
(7) `b1`: the weight of daily kernel in the final kernel;   
(8) `b2`: the weight of weekly kernel in the final kernel;   
(9) `b3`: the weight of geographical kernel in the final kernel;    
(10) `d`: the number of basis vectors for flights and pauses using latitude/longitude as X in the kernel function. If N is specified here, there will be 4N basis vectors in total;   
(11) `sigma2`: the variance parameter in sparse online gaussian process;   
(12) `tol`: the tolerance/threshold of the residual to add the current observation to the basis vector set;   
(13) `switch`: the number of binary variables we want to generate in fucntion `I_flight`, which controls the difficulty to change the status from flight to pause or from pause to flight;   
(14) `num`: If specified as K, we will use top K trajectories in terms of the similarity to the current time and location in fucntion `I_flight`(to avoid the cumulative effect of many low prob trajs);   
(15) `linearity`: a scalar that controls the smoothness of a trajectory: a large linearity tends to have a more linear traj from starting point toward destination, a small one tends to have more random directions;   
(16) `method`: it should be 'TL', or 'GL' or 'GLC' (corresponding to temporal kernel only, geographical kernel only and combined kernel);   
(17) `itrvl`: the window size of moving average,  unit is second;   
(18) `accuracylim`: we filter out GPS record with accuracy higher than this threshold.    
(19) `r`: the maximum radius of a pause;   
(20) `w`: a threshold for distance, if the distance to the great circle is greater than this threshold, we consider there is a knot;   
(21) `h`: a threshold of distance, if the movement between two timestamps is less than h, consider it as a pause and a knot   
(22) `save_osm_log`: bool, True if you want to output a log of locations visited and their tags(default: False).   
(23) `log_threshold`: int, time spent in a pause needs to exceed the threshold to be placed in the log   
(24) `split_day_night`: bool, True if you want to split all metrics to datetime and nighttime patterns (only for Frequency.DAILY)   
(25) `person_point_radius`: float, radius of the person's circle when discovering places near him in pauses (default: 2)   
(26) `place_point_radius`: float, radius of place's circle when place is returned as centre coordinates from osm (default: 7.5)   
(27) `pcr_bool`: bool, True if you want to calculate the physical cyrcadian rhythm (default: False)   
(28) `pcr_window`: int, number of days to look back and forward for calculating the physical cyrcadian rhythm (default: 14)   
(29) `pcr_sample_rate`: int, number of seconds between each sample for calculating the physical cyrcadian rhythm (default: 30)


### Output

(1) summary statistics for all specified participants (.csv)  

(2) imputed trajectories (.csv)\
   Complete trajectories in terms of timestamp, latitude and longitude. By default, it is set to FALSE.

(3) a record (.csv)\
    - Contains start date/time and end date/time for each participant.\
    - Is useful for tracking whose data during which time range have been processed, especially for the online algorithm.

(4) all_bv_set (.pkl)\
    - It is a dictionary, with the key as user ID and the value as a numpy array with size, where each column represents [start_timestamp, start_latitude, start_longitude, end_timestamp, end_latitude, end_longitude]. If it is your first time run the code, it is set to NULL by default. If you want to continue your analysis from here in the future, all_bv_set is expected to be an input in your new analysis and it will be updated in that run. The size of the file should be fixed overtime.

(5) all_memory_dict (.pkl)\
    - It is also a dictionary, with the key as user ID and the value as a numpy array of other parameters for the user. If it is your first time run the code, it is set to NULL by default. If you want to continue your analysis from here in the future, all_memory_dict is expected to be an input in your new analysis and it will be updated in that run. The size of the file should be fixed overtime.

(6) locations_log (.json)\
    - json file created if `save_osm_log` is set to True. It contains information on the places visited by the user, their tags and the time of visit.

## Description of functions in package:

`data2mobmat.py`
This file contains the functions to convert the raw GPS data to a mobility matrix (2d numpy array), where each column represents movement status(flight/pause/undecided), starting latitude, starting longitude, starting timestamp, ending latitude, ending longitude, ending timestamp. This module focuses on summarizing observed data to trajectories but not unobserved period.

- Its main function is `gps_to_mobmat` which calls the required functions in the right order (see [[Link to paper | doi....]] for details on the algorithm
- It contains various functions to calculate distance on the globe: `cartesian`, `shortest_dist_to_great_circle`, `great_circle_dist` and `pairwise_great_circle_dist`
- In addition, it has a few helper functions:
- `collapse_data`: the GPS data is usually sampled at 1 Hz. We collapse the data every 10 seconds and calculate the average to reduce the noise in the raw data.
- `exist_knot`: given a matrix with columns [timestamp, latitude, longitude], return if the trajectories depicted by those coordinates can be approximated as a straight line. The parameter $w$ represents the tolerance of deviation. It return 1 if there exists at least one knot in the trajectory and it returns 0 otherwise.
- `extract_flights`: given a matrix with columns [timestamp, latitude, longitude] in a burst period (when the GPS is on), return a summary of trajectories (2d array) with columns as [movement status, start_timestamp, start_latitude, start_longitude, end_timestamp, end_latitude, end_longitude]. It uses the helper funtions `mark_single_measure`, `mark_complete_pause`, `detect_knots` and `prepare_output_data`.
- `infer_mobmat`: tidy up the trajectory matrix (infer undecided pieces, combine flights/pauses.). It uses the helper functions `compute_flight_positions`, `compute_future_flight_positions`, `infer_status_and_positions`, `merge_pauses_and_bridge_gaps` and `correct_missing_intervals`.

`sogp_gps.py`
This file is the core of sparse online Gaussian Process. It covers the algorithm described in [Csato and Opper (2001)](https://eprints.soton.ac.uk/259182/1/gp2.pdf).

- `calculate_k0`: a kernel function to measure the similarity between x1 and x2.
- `update_similarity`, `update_similarity_all`, `update_e_hat`, `update_gamma`, `update_q`, `update_s_hat`, `update_eta`, `update_alpha_hat`, `update_c_hat`, `update_s`, `update_alpha`, `update_c`, `update_q_mat`, `update_alpha_vec`, `update_c_mat`, `update_q_mat2`, `update_s_mat`: are the updating rules for each parameters in the algorithm.
- `sogp`: A key function of this model. Given an 2d array of latitude and longitude, return a basis vector set of fixed size and relevant parameters for the updates in the future. It uses the helper functions `calculate_sigma_max`, `update_system_given_gamma_tol`, `update_system_otherwise` and `pruning_bv`.
- `bv_select`: The master function. Given the observed trajectory matrix, return representative trajectories of a fixed size and relevant parameters for the updates in the future.

`mobmat2traj.py`
This file imputes the missing trajectories based on the observed trajectory matrix.

- Its main functions are `impute_gps` (for bi-directional imputation) and `imp_to_traj` (for combining pauses, flights shared by both observed and missing intervals, also combining consecutive flight with slightly different directions as one longer flight). It uses the helper functions `calculate_delta`, `adjust_delta_if_needed`, `calculate_position`, `update_table`, `forward_impute` and `backward_impute`.
- It contains two functions that are also used for generating summary statistics: `num_sig_places` (identify number of locations where participant spends x consecutive minutes, and is at least y m away from other locations) and `locate_home` (identify location that a participant spends most time between 9pm and 9 am). They use helper functions `update_existing_place` and `add_new_place`.
- It contains various helper functions:
  - `calculate_k1`: the kernel function returns the similarity between the given triplet and every triplet in the basis vector set.
  - `indicate_flight`: determine if a flight occurs at the current time and location
  - `adjust_direction`: adjust the direction of the sampled flight if it is not likely to happen in the real world.
  - `multiplier`: return a coefficient to accelerate the imputation process based on the duration of the missing interval.
  - `checkbound`: check if the destination will be out of a reasonable range given the sampled flight
  - `create_tables`: initialize three 2d numpy arrays, one to store observed flights, one to store pauses, and one to store missing intervals.

`traj2stats.py`
This file converts the imputed trajectory matrix to summary statistics.
- `Hyperparameters`: dataclass to store the hyperparameters for the imputation and summary statistics.
- `transform_point_to_circle`: transform a transforms a set of cooordinates to a shapely circle with a provided radius.
- `get_nearby_locations`: return a dictionary of nearby locations, a dictionary of nearby locations' names, and a dictionary of nearby locations' coordinates.
- `gps_summaries`: converts the imputed trajectory matrix to summary statistics.
- `gps_quality_check`: checks the data quality of GPS data. If the quality is poor, the imputation will not be executed.
- `gps_stats_main`: this is the main function of the jasmine module and it calls every function defined before. It is the function you should use as an end user. 

## List of summary statistics

The summary statistics that are generated are listed below:

|     Variable                                  |     Type     |     Description of Variable                                                                                                                                                                                                                |     Description of What it Measures |
|-----------------------------------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
|     Observed duration                                 |     Float    |     The total time when the GPS is on                                                                                                                                                                                   |     This variable quantifies the missingness and the uncertainty in all other estimates.| 
|     Observed duration in day                                 |     Float    |     The total time when the GPS is on from 8AM to 8PM                                                                                                                                                                                |     This variable quantifies the missingness in daytime and the majority of uncertainty in all other estimates.| 
|     Observed duration at night                               |     Float    |     The total time when the GPS is on from 8PM to 8AM                                                                                                                                                                                 |     This variable quantifies the missingness at night and the minority of uncertainty in all other estimates since the user is most likely at home.| 
|     Home time                                 |     Float    |     Time spent at home over the course of a day (in hours)                                                                                                                                                                                    |     “Home” is the most frequently visited significant location for a person between the hours of 8pm and 8am each day over the course of follow up.|                                                                                                                                                                                                                                                                             
|     Distance traveled                             |     Float    |     Total distance travelled over the course of a day (in km)                                                                                                                                                                                 |     The sum of lengths of all flights. A flight is defined to be a longest straight-line trip of a particle from one location to another without a directional change or pause. Please find the technical details [here](https://github.com/onnela-lab/forest/wiki/Jasmine-documentation#other-technical-details). |                                                                                                                                                                                                                                                                                                                                                                                                                               
|     Radius of gyration                        |     Float       |     Average radius that a person travels from their center   over the course of a day (in km)                                                                                                                                                  |     Centroid = the average of each ‘place visited’ (see   definition ‘significant location’) over the course of a day, with weights   proportional to the amount of time spent in the location.      The radius of gyration is calculated using a time-weighted   average of the distance between each place and the centroid, where weights   are measured in the same way. |   
|     Maximum diameter                          |   Float           |     Largest distance between any two places that a person visited in a day (in km) |      |                                                                                                                                                                                                                                                                                                                                                                                                                                        
|     Maximum distance from home                |    Float          |     Largest distance between any places that a person visited in a day and their home (in km)|   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
|     Number of significant locations           |      int        |     Number of significant visited at any point over the course of a day                                                                                                                                                            |     Significant locations are distinct pauses which are at least 15 minutes long and 50 meters apart. They are determined using K-means clustering on locations that a patient visits over the course of follow up. Set K=K+1 and repeat clustering until two significant locations are within 100 meters of one another. Then use the results from the previous step (K-1) as the total number of significant locations. |
| Total flight time                             |     Float    |     Total time spent in flight over the course of a day (in hours)                                                                                                                                                                                 |     A flight is defined to be a longest straight-line trip of a particle from one location to another without a directional change or pause.
|     Average flight length                     |     Float         |     Average of the length of all flights (straight line movement) that took place over the course of a day (in km)                                                                                                                           |     GPS is converted into a sequence of flights (straight line movement) and pauses (time spent stationary).  A flight is defined to be a longest straight-line trip of a particle from one location to another without a directional change or pause. Note that a long flight could be composed of several short flights with different directions, but when calculating the average, it is the mean of those short flights. Please find the technical details [here](https://github.com/onnela-lab/forest/wiki/Jasmine-documentation#other-technical-details).|                                                                                                                                                                                                                           
|     Standard deviation of flight length       |    Float          |     Standard deviation of the length of all flights (straight line movement) that took place over the course of a day (in km)                                                                                                                |     GPS is converted into a sequence of flights (straight line movement) and pauses (time spent stationary). The standard deviation of  flights of the day is reported. |
| Average flight duration                      |     Float         |     Average of the duration of all flights (straight line movement) that took place over the course of a day (in hours)                                                                                                                      |     GPS is converted into a sequence of flights (straight line movement) and pauses (time spent stationary). The average of the duration of  flights of the day is reported. |
| Standard deviation of flight duration        |     Float         |     Standard deviation of the duration of all flights (straight line movement) that took place over the course of a day (in hours)                                                                                                           |     GPS is converted into a sequence of flights (straight line movement) and pauses (time spent stationary). The standard deviation of the duration of  flights of the day is reported. |
| Total pause time                             |     Float         |     Total time spent in pause over the course of a day (in hours)                                                                                                                                                                                 |     A pause is defined to be a longest time spent stationary without a directional change or flight. |
|     Average pause duration                   |       Float       |     Average of the duration of all pauses that took place over the course of a day (in hour)   | We consider that a participant has a pause if the distance that he has moved during a 30-s period is less than `r` m. By default, `r`=10.|                                                                                                                                                                                                                                                     
|     Standard deviation of flight duration     |    Float          |     Standard deviation of the duration of all pauses that took place over the course of a day (in hour) |     GPS is converted into a sequence of flights (straight line movement) and pauses (time spent stationary). The standard deviation of duration of pauses over the course of a day is reported. |                                                                                                                                                                                                            
|     Significant location entropy              |      Float        |     Entropy measure based on the proportion of time spent at significant locations over the course of a day |     Letting p_i be the proportion of the day spent at significant location I, significant location entropy is calculated as   -\sum_{i} p_i*log(p_i), where the sum occurs over all non-zero p_i for that day. | 
|     Minutes of GPS data missing               |      Not Available        |     Number of minutes of GPS data missing over the course of a day | |                                                                                                                                                                                                                                                                                                                                                                                                                                 
|     Physical circadian rhythm                 |         Float     |     A continuous measurement of routine in the interval [0,1] that scores a day with 0 if there was a complete break from routine and 1 if the person followed the exact same routine as have in every other day of follow up |     For a detailed description of how this measure is calculated, see Canzian and Musolesi's 2015 paper in the Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing, titled "Trajectories of depression: unobtrusive monitoring of   depressive states by means of smartphone mobility traces analysis."   Their procedure was followed using 30-min increments as a bin size.|
|     Physical circadian rhythm stratified      |     Float         |     A continuous measurement of routine in the interval [0,1] that scores a day with 0 if there was a complete break from routine and 1 if the person followed the exact same routine as have in every other day of follow up    |  Calculated in the same way as Physical circadian rhythm, except the procedure is repeated separately for weekends and weekdays. |


### Other technical details
- Definition of flights and pauses  
A flight is defined to be a longest straight-line trip of a particle from one location to another without a directional change or pause. Technically, we define the straight line between A and B to be a flight if and only if the following conditions are met. (1) The distance between any two consecutively sampled positions between and is larger than `r` meters (i.e., no pause during a flight). (2) When we draw a straight line from A to B, the sampled positions between these two endpoints are at a distance less than `w` meters from the line. The distance between the line and a position is the length of a perpendicular line from that position to the line. (3) For the next sampled position C after B, positions and the straight line between A and C do not satisfy conditions (1) and (2). By default, two consecutive sampled positions are 10 seconds apart, `w` = `r` = 10 meters. We consider that a participant has a pause if the distance that he has moved during a 30 second period is less than `r` meters. By default, `r`=10.  

- Definition of a place  
A place is defined as a location where a person has paused for at least 15 minutes.
