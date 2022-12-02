# Bonsai

## Usage

Bonsai is used to simulate realistic GPS trajectories and call/text log data that resemble the ones returned from the Beiwe API. It is used to test the forest library using generated datasets.

## Installation Instruction

For instructions on how to install forest, please visit [here](https://github.com/onnela-lab/forest).
`from forest import bonsai`

## Data

### Input

Bonsai can be concepually divided into two parts: GPS and call/text log data.

#### GPS

To generate GPS data call function `bonsai.sim_gps_data` with the following parameters:

* n_persons: (int) number of people to simulate
* location: (str) indicating country and city to simulate at,
                format "Country_2_letter_ISO_code/City_Name"
* start_date: (datetime.date) start date of trajectories
* end_date: (datetime.date) end date of trajectories,
            end date is not included in the trajectories
* cycle: (int) the sum of on-cycle and off_cycle,
            unit is minute
* percentage: (float) the missing rate, in other words,
            the proportion of off_cycle, should be within [0,1]
* api_key: (str), api key for open route service
            https://openrouteservice.org/

You can also use the optional argument of `attributes_dict` if you want to simulate people with different attributes. The format of `attributes_dict` is a dictionary with the following keys and values:

* keys: (str) to represent the user id for which we define the attributes.
    Can be of the form: "User 1", "User 2", "User 3", etc. or "Users 2-4", etc.
* values: (dict) to represent the attributes of the users in dictionary format. Format (keys: values):
  * main_employment: (str) has to be in ["office", "university", "none"] to represent the main employment of the user.
        If they are working, they will be simulated to be at work during the work hours.
        If they are studying, they will be simulated to be at school during the school hours.
        If they are not working or studying, they will be simulated at random during the day.
  * vehicle: (str) has to be in ["bus", "car", "bicycle", "foot"] to represent what is their usual means of travel.
        Influences how fast they are moving when heading to more distant destinations.
        For bus they also have to walk to a bus stop and walk off the bus before going to their destination.
  * travelling_status: (int) 0-10, influences how far they should travel.
        0 means they will not travel at all.
        10 means they will travel the furthest.
        Influences how far they travel when going to work or school.
        Influences how far they travel when going to a random destination.
  * active_status: (int) 0-10, influences how active they are
        0 means they will not go outside the house at all.
        10 means they will want to be outside the house as much as possible.
        Influences how often they go to random destinations.
  * preferred_exits: (list) list which can contain any of the following values
        ("cafe", "bar", "restaurant", "park", "cinema", "dance", "fitness_centre").
        Influences what kind of destinations they will go to most often when going to random destinations.
  * travelling_status-{day}, active_status-{day}: (int) change the travelling or active status after day {day}.
        For example, travelling_status-3: 5 means that after day 3, the travelling status will be 5.
        This can be used to simulate people who are on vacation or sick.
        Change their behaviour after a certain day.

#### Call/Text Log

To generate call/text log data call function `bonsai.sim_log_data` with the following parameters:

* output_folder: (str) path to the output file

### Output

The GPS method will return a `pandas.DataFrame` with the GPS trajectories of the simulated users. The dataframe will have the following columns:

* user: (int) the id of the user, starts from 1 and goes up to n_persons
* timestamp: (int) the timestamp of the GPS point
* UTC time: (int) the timestamp of the GPS point in UTC time
* latitude: (float) the latitude of the GPS point
* longitude: (float) the longitude of the GPS point
* altitude: (float) the altitude of the GPS point
* accuracy: (float) the accuracy of the GPS point

The call/text log method will save csv files of call/text log data for 2 users that resemble the data returned from the Beiwe research platform.

##  Description of functions in package: 
`simulate_gps_data.py`
This file contains the classes and functions needed to simulate GPS data.

* `PossibleExits`: (Enum) contains the possible exits for the users
* `Vehicle`: (Enum) contains the possible vehicles for the users
* `Occupation`: (Enum) contains the possible occupations for the users
* `ActionType`: (Enum) contains the possible actions for the users
* `get_path`: (function) Calculates paths between sets of coordinates. This function takes 2 sets of coordinates and
    a mean of transport and using the openroute api
    calculates the set of nodes to traverse
    from location1 to location2 along with the duration
    and distance of the flight.
* `get_basic_path`: (function) Subsets paths depending on transport for optimisation.
    This function takes a path from get_path() function and subsets it
    to a specific number of nodes.
* `bounding_box`: (function) Calculates the bounding box of a set of coordinates.
* `Attributes`: (class) Contains the attributes of a user.
* `Person`: (class) Contains all the information of a person and is used to simulate their GPS data. Has the following methods:
  * `__init__`: This function sets the basic attributes and information
        to be used of the person.
  * `set_travelling_status`: Update preferred locations of exits
        depending on new travelling status.
  * `set_active_status`: Update active status.
  * `update_preferred_places`: This function updates the set of preferred exits for the day,
        after an action has been performed.
  * `choose_preferred_exit`: This function samples through the possible actions for the person,
        depending on his attributes and the time.
  * `end_of_day_reset`: Reset preferred exits of the day. To run when a day ends
  * `calculate_trip`: This function uses the openrouteservice api to produce the path
        from person's house to destination and back.
  * `choose_action`: This function decides action for person to take.
* `gen_basic_traj`: (function) This function generates basic trajectories between 2 points.
* `gen_basic_pause`: (function) This function generates basic trajectories for a pause.
* `gen_route_traj`: (function) This function generates basic trajectories between multiple points.
* `gen_all_traj`: (function) Generates trajectories for a single person.
* `remove_data`: (function) Only keeps observed data from simulated trajectories
    depending on cycle and percentage.
* `prepare_data`: (function) Prepares the data in a dataframe.
* `process_switches`: (function) Preprocesses the attributes of each person in what relates switching active or travelling status.
* `load_attributes`: (function) Loads the attributes of each person.
* `generate_addresses`: (function) Generates multiple addresses.
* `generate_nodes`: (function) Generates multiple amenities coordinates.
* `sim_gps_data`: (function) Generates gps trajectories.
* `gps_to_csv`: (function) Writes gps trajectories to csv files.
