import os
import sys
import numpy as np
import pandas as pd
import re
import datetime
import time
import requests
import json
import overpy
from timezonefinder import TimezoneFinder
from ..poplar.legacy.common_funcs import datetime2stamp,stamp2datetime
from ..jasmine.data2mobmat import great_circle_dist


R = 6.371*10**6

def getPath(lat1: float, lon1: float, lat2: float, lon2: float, transport: str, api_key: str) -> tuple[np.ndarray, float]:
	"""
	This function takes 2 sets of coordinates and a mean of transport and using the openroute api
	calculates the set of nodes to traverse from location1 to location2 along with the duration 
	and distance of the flight. \n
	Args: 
		lat1, lon1: coordinates of start point
		lat2, lon2: coordinates of end point
		transport: means of transportation, can be one of the following: (car, bus, foot, bicycle)
		api_key: api key collected from https://openrouteservice.org/dev/#/home \n
	Return: 
		path_coordinates: 2d numpy array containing [lat,lon] of route
		distance: distance of trip in meters
	"""
	if great_circle_dist(lat1, lon1, lat2, lon2) < 250:
		return np.array([[lon1, lat1], [lon2, lat2]]), great_circle_dist(lat1, lon1, lat2, lon2)
	else: 

		if transport == 'car' or transport == 'bus':
			transport = 'driving-car'
		elif transport == 'foot':
			transport = 'foot-walking'
		elif transport == 'bicycle':
			transport = 'cycling-regular'
		
		headers = {
			'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
		}
		call = requests.get('https://api.openrouteservice.org/v2/directions/{}?api_key={}&start={},{}&end={},{}'.format(transport,api_key,lon1,lat1,lon2,lat2),
							headers=headers, timeout=90)
		
		if call.reason == 'OK' and call.status_code == 200:
			res = json.loads(call.text)['features'][0]
			path_coordinates = res['geometry']['coordinates']
			#distance = res['properties']['segments'][0]['distance'] # meters
			
			if (path_coordinates[0] != [lon1, lat1]):
				path_coordinates[0] = [lon1, lat1]
			if (path_coordinates[-1] != [lon2, lat2]):
				path_coordinates[-1] = [lon2, lat2]

			return np.array(path_coordinates), great_circle_dist(lat1, lon1, lat2, lon2)
		else:
			time.sleep(60)
			call = requests.get('https://api.openrouteservice.org/v2/directions/{}?api_key={}&start={},{}&end={},{}'.format(transport,api_key,lon1,lat1,lon2,lat2),
							headers=headers, timeout=90)

			if call.reason == 'OK' and call.status_code == 200:
				res = json.loads(call.text)['features'][0]
				path_coordinates = res['geometry']['coordinates']
				#distance = res['properties']['segments'][0]['distance'] # meters
				
				if (path_coordinates[0] != [lon1, lat1]):
					path_coordinates[0] = [lon1, lat1]
				if (path_coordinates[-1] != [lon2, lat2]):
					path_coordinates[-1] = [lon2, lat2]

				return np.array(path_coordinates), great_circle_dist(lat1, lon1, lat2, lon2)
			else:
				print("Openroute service failed with code " + str(call.status_code) + ", because of " + call.reason)
				sys.exit()

def basicPath(path: np.ndarray, transport: str) -> np.ndarray:
	"""
	This function takes a path from getPath() function and subsets it
	to the provided number of nodes.\n
	Args: 
		path: 2d numpy array
		length: integer
		transport: str
	Return: 
		subset of original path that represents the flight
	"""

	distance = great_circle_dist(path[0][1], path[0][0],
		path[-1][1], path[-1][0])

	if transport in ['foot', 'bicycle']:
		length = 2 + distance//200
	else:
		length = 2 + distance//400

	if transport == 'bus':
		length += 2 # bus route start and end

	if length >= len(path):
		basic_path = path
	else:
		indexes = list(range(0, len(path), int(len(path)/(length-1))))
		if len(indexes) < length:
			indexes.append(len(path)-1)
		else:
			indexes[-1] = len(path) - 1

		indexes2 = []
		for i in range(len(indexes) - 1):
			if (path[indexes[i]] != path[indexes[i+1]]).any():
				indexes2.append(indexes[i])
		indexes2.append(indexes[-1])
		basic_path = path[indexes2]
	
	return basic_path

def boundingBox(lat: float, lon: float, radius: int) -> tuple[float]:
	"""
	This function outputs an area around a set of coordinates.\n
	Args: 
		lat, lon: set of coordinates
		radius: radius in meters of area around coordinates
	Return: 
		tuple of 4 elements that represents a bounding box around the coordinates provided
	"""
	r_earth = 6371
	lat_const = (radius / (1000*r_earth)) * (180 / np.pi)
	lon_const = (radius / (1000*r_earth)) * (180 / np.pi) / np.cos(lat * np.pi/180)
	return (lat-lat_const, lon-lon_const, lat+lat_const, lon+lon_const)

class Person:
	"""
	This class represents a person whose trajectories we want to simulate. 
	"""

	def __init__(self, house_address: tuple, attributes: list, all_nodes: dict):
		"""
		This function sets the basic attributes and information to be used of the person.\n
		Args: 
			house_address: tuple, coordinates of primary home
			   attributes: list, consists of various information
			   attributes = [vehicle, main_employment, athletic_status, active_status, travelling_status, preferred_exits]
				* vehicle = 0: nothing, 1: own car, 2: own bicycle | used for distances and time of flights
				* main_employment = 0: nothing, 1: worker, 2:student | used for routine action in weekdays
				* active_status = 0-10 | used for probability in free time to take an action or stay home
				* travelling status = 0-10 | used to derive amount of distance travelled
				* preferred_exits = [x1, x2, x3] | used to sample action when free time where x1-x3 are amenities (str)  
				all_nodes: dictionary, contains overpass nodes of amenities near house

		"""

		self.house_address = house_address
		self.vehicle = attributes[0]
		self.main_employment = attributes[1]
		self.active_status = attributes[2]
		self.travelling_status = attributes[3]
		self.preferred_exits = attributes[4]
		
		self.preferred_exits_today = attributes[4]
		self.office_today = False

		self.trips = {}
		
		# define place of employment
		self.house_area = boundingBox(house_address[0], house_address[1], 2000)
		if self.main_employment > 0:
			if self.main_employment == 1:
				employment_str = "office"
			elif self.main_employment == 2:
				employment_str = "university"
			
			if len(all_nodes[employment_str]) != 0:
				i = np.random.choice(range(len(all_nodes[employment_str])), 1)[0]
				
				while all_nodes[employment_str][i] == self.house_address:
					i = np.random.choice(range(len(all_nodes[employment_str])), 1)[0]
					
				self.office_address = all_nodes[employment_str][i]

				no_office_days = np.random.binomial(5, self.active_status/10)
				self.office_days = np.random.choice(range(5), no_office_days, replace=False)
				self.office_days.sort()
			else:
				self.office_address = ""
		else:
			self.office_days = []
			
			
			
					
		# define favorite places
		self.possible_exits = ['cafe', 'bar', 'restaurant', 'park', 'cinema', 'dance', 'fitness']

		for exit in self.possible_exits:
			if len(all_nodes[exit]) > 3:
				random_places = np.random.choice(range(len(all_nodes[exit])), 3, replace=False).tolist()
				places_selected = [(place[0], place[1]) for place in np.array(all_nodes[exit])[random_places]]
				if self.house_address in places_selected:
					places_selected = [pl for pl in places_selected if pl != self.house_address]
					r_id = np.random.choice(range(len(all_nodes[exit])), 1)[0]
					while r_id in random_places:
						r_id = np.random.choice(range(len(all_nodes[exit])), 1)[0]
					place = all_nodes[exit][r_id]
					places_selected.append((place[0], place[1]))

				setattr(self, exit+'_places', places_selected)
			else:
				setattr(self, exit+'_places', [(place[0], place[1]) for place in all_nodes[exit] if (place[0], place[1]) != self.house_address])

			distances = [great_circle_dist(self.house_address[0], self.house_address[1], place[0], place[1]) for place in getattr(self, exit+'_places')]
			order = np.argsort(distances)
			setattr(self, exit+'_places_ordered', np.array(getattr(self, exit+'_places'))[order].tolist())

		# remove all exits which have no places nearby
		possible_exits2 = self.possible_exits.copy()
		for act in possible_exits2:
			if len(getattr(self, act+'_places')) == 0:
				self.possible_exits.remove(act)

		# order preferred places by travelling_status
		travelling_status_norm = (self.travelling_status**2) /  (self.travelling_status**2 + (10-self.travelling_status)**2)
		for act in self.possible_exits:
			act_places = getattr(self, act + '_places_ordered').copy()

			places = []
			for i in range(len(act_places) - 1, -1, -1):
				index = np.random.binomial(i, travelling_status_norm)
				places.append(act_places[index])
				del act_places[index]

			setattr(self, act + '_places', places)

		# find places near office
		if self.main_employment > 0 and len(self.office_address) > 0:
			self.office_exits = []
			self.office_codes = []
			for exit in ['cafe', 'restaurant', 'park', 'fitness']:
				for place in getattr(self, exit+'_places'):
					if great_circle_dist(self.office_address[0], self.office_address[1], place[0], place[1]) < 500:
						self.office_exits.append(place)
						self.office_codes.append(exit)

	def setTravellingStatus(self, travelling_status: int):
		"""
		Update preferred locations of exits depending on new travelling status.\n
		Args:
			travelling_status: 0-10 | int indicating new travelling_status
		"""

		setattr(self, 'travelling_status', travelling_status)

		travelling_status_norm = (travelling_status**2) /  (travelling_status**2 + (10-travelling_status)**2)
		for act in self.possible_exits:
			act_places = getattr(self, act + '_places_ordered').copy()

			places = []
			for i in range(len(act_places) - 1, -1, -1):
				index = np.random.binomial(i, travelling_status_norm)
				places.append(act_places[index])
				del act_places[index]

			setattr(self, act + '_places', places)

	def setActiveStatus(self, active_status: int):
		"""
		Update active status.\n
		Args:
			active_status: 0-10 | int indicating new travelling_status
		"""

		setattr(self, 'active_status', active_status)

		if self.main_employment > 0 and self.office_address != "":
			no_office_days = np.random.binomial(5, active_status/10)
			self.office_days = np.random.choice(range(5), no_office_days, replace=False)
			self.office_days.sort()
		
	def updatePreferredExits(self, exit_code: str):
		"""
		This function updates the set of preferred exits for the day, after an action has been performed.\n
		Args: 
			exit_code: str, representing the action which was performed.
		"""
		
		if exit_code in self.preferred_exits_today:
			index_of_code = self.preferred_exits_today.index(exit_code)
			if index_of_code == (len(self.preferred_exits_today) - 1):
				probs = np.array([0 if c in self.preferred_exits_today else 1 for c in self.possible_exits])
				probs = probs/sum(probs)
				self.preferred_exits_today[-1] = np.random.choice(self.possible_exits, 1, p=probs.tolist())[0]
			else:
				self.preferred_exits_today[index_of_code], self.preferred_exits_today[index_of_code+1] = self.preferred_exits_today[index_of_code+1], self.preferred_exits_today[index_of_code]
		
	def choosePreferredExit(self, t_s: float, update: bool = True) -> tuple[str, tuple]:
		"""
		This function samples through the possible actions for the person, 
		depending on his attributes and the time.\n
		Args: 
			t_s: float, current time in seconds
			update: boolean, to update preferrences
		Return:
			selected_action_decoded: str, selected action to perform
			selected_location: tuple, selected location's coordinates
		"""

		probs_of_staying_home = [1-self.active_status/10, self.active_status/10]
		if np.random.choice([0,1], 1, p=probs_of_staying_home)[0] == 0:
			return 'home', self.house_address


		time_now = t_s % (24 * 60 * 60)
		hr_now = time_now / (60*60)
		
		active_coef = (10 - self.active_status) / 4

		if hr_now < 9+active_coef:
			return 'home', self.house_address
		elif hr_now > 22-active_coef: 
			return 'home_night', self.house_address
			
		possible_exits2 = self.possible_exits.copy()
		
		actions = []
		probabilities = []
		ratios = [6, 3, 1]
		for i in range(len(self.preferred_exits_today)):
			preferred_action = self.preferred_exits_today[i]
			if preferred_action in possible_exits2:
				actions.append(preferred_action)
				probabilities.append(ratios[i])
				possible_exits2.remove(preferred_action)
		
		# remaining ones
		for act in possible_exits2:
			actions.append(act)
			probabilities.append(0.25)

		probabilities = np.array(probabilities)       
		probabilities = probabilities/sum(probabilities)
		
		selected_action = np.random.choice(actions, 1, p=probabilities)[0]
		
		if update:
			self.updatePreferredExits(selected_action)
				
		action_locations = getattr(self, selected_action+'_places')
		ratios2 = [7, 2, 1][:len(action_locations)]
		probabilities2 = np.array(ratios2)       
		probabilities2 = probabilities2/sum(probabilities2)
		
		selected_location_index = np.random.choice(range(len(action_locations)), 1, p=probabilities2)[0]
		selected_location = action_locations[selected_location_index]
		
		return selected_action, selected_location
		
	def choosePreferredOfficeExit(self, api_key: str) -> tuple[np.ndarray, np.ndarray, str]:
		"""
		This function samples through the possible exits when the person is at work or university.\n
		Args: 
			api_key: str, api key of open route service
		Return:
			  np.ndarray, array of nodes to travel from office address to location of exit
		 	  np.ndarray, array of nodes to travel back from location of exit to office address
					 str, exit selected
		"""

		chosen_exit_index = np.random.choice(range(len(self.office_exits)), 1)[0]
		chosen_exit = self.office_exits[chosen_exit_index]
		self.updatePreferredExits(self.office_codes[chosen_exit_index])

		go_path, _ = self.calculateTrip(self.office_address, chosen_exit, api_key) 
		return_path, _ = self.calculateTrip(chosen_exit, self.office_address, api_key) 

		return go_path, return_path, self.office_codes[chosen_exit_index]

	def endOfDayReset(self):
		"""
		Reset preferred exits of the day. To run when a day ends
		"""
		self.preferred_exits_today = self.preferred_exits
		self.office_today = False

	def calculateTrip(self, origin: tuple, destination: tuple, api_key: str) -> tuple[np.ndarray, str]:
		"""
		This function uses the openrouteservice api to produce the path
		from person's house to destination and back.\n
		Args: 
			destination: tuple, coordinates for destination
			origin: tuple, coordinates for origin
			api_key: str, openrouteservice api key
		Return:
			path: 2d numpy array, containing [lat,lon] of route from origin to destination
			transport: str, means of transport
		"""


		distance = great_circle_dist(origin[0], origin[1], destination[0], destination[1])
		transportations = {0: 'bus', 1: 'car', 2: 'bicycle'}

		if distance <= 1000:
			transport = 'foot'
		else:
			transport = transportations[self.vehicle]

		coords_str = str(origin[0])+'_'+str(origin[1])+'_'+str(destination[0])+'_'+str(destination[1])
		if coords_str in self.trips.keys():
			path = self.trips[coords_str]
		else:
			path, _ = getPath(origin[0], origin[1], destination[0], destination[1], transport, api_key)
			path = basicPath(path, transport)

			path = [[x[1], x[0]] for x in path]

			self.trips[coords_str] = path

		return path, transport

	def chooseAction(self, t_s: float, day_now: int, update: bool = True) -> tuple[str, tuple[float, float], list[int, int], str]:
		"""
		This function decides action for person to take.\n
		Args: 
			t_s: int, current time in seconds
			day_now: int, day of the week
		Return:
			str, 'p', 'p_night' or 'fpf' indicating pause, pause for the night or flight-pause-flight
			tuple, destination's coordinates
			list, contains [minimum, maximum] duration of pause in seconds
			str, exit code
		"""
		time_now = t_s % (24 * 60 * 60)

		if time_now == 0: 
			if day_now < 5 and self.main_employment > 0:
				return 'p', self.house_address,[8 * 3600, 9 * 3600], 'home_morning'
			else: 
				return 'p', self.house_address,[8 * 3600, 12 * 3600], 'home_morning'
		
		if not self.office_today:
			if update:
				self.office_today = not self.office_today
			if day_now in self.office_days:
				return 'fpf', self.office_address, [7 * 3600, 9 * 3600], 'office'
			elif day_now < 5:
				return 'p', self.house_address, [7 * 3600, 9 * 3600], 'office_home'

		exit, location = self.choosePreferredExit(t_s, update)

		if exit == 'home':
			if time_now + 2*3600 > 24*3600 - 1:
				return 'p_night', self.house_address,[24 * 3600 - time_now, 24 * 3600 - time_now], "home_night"
			return 'p', self.house_address,[0.5 * 3600, 2 * 3600], exit
		elif exit == 'home_night':
			return 'p_night', self.house_address,[24 * 3600 - time_now, 24 * 3600 - time_now], exit
		else:
			return 'fpf', location, [0.5 * 3600 + 1.5 * 3600 * (self.active_status - 1)/9, 1 * 3600 + 1.5 * 3600 * (self.active_status - 1)/9], exit

def gen_basic_traj(l_s: tuple[float], l_e: tuple[float], vehicle: str, t_s: float) -> tuple[np.ndarray, float]:
	"""
	This function generates basic trajectories between 2 points.\n
	Args: 
		l_s: tuple, coordinates of start point
		l_e: tuple, coordinates of end point
		vehicle: str, means of transportation, can be one of the following: (car, bus, walk, bike)
		t_s: float, starting time
	Return: 
		numpy.ndarray, containing the trajectories
		float, total distance travelled
	"""
	traj_list = []
	[lat_s, lon_s] = l_s
	if vehicle == 'walk':
		spd_range = [1.2, 1.6]
	elif vehicle == 'bike':
		spd_range = [7, 11]
	else:
		spd_range = [10, 14]
	d = great_circle_dist(l_s[0],l_s[1],l_e[0],l_e[1])
	traveled = 0
	t_e = t_s
	while traveled < d:
		r_spd = np.random.uniform(spd_range[0], spd_range[1], 1)[0]
		r_time = int(np.around(np.random.uniform(30, 120, 1), 0))
		mov = r_spd*r_time
		if traveled + mov > d or d - traveled - mov < spd_range[1]:
			mov = d - traveled
			r_time = int(np.around(mov/r_spd,0))
		traveled = traveled + mov
		t_e = t_s + r_time
		ratio = traveled/d
		## temp = ratio*l_e + (1-ratio)*l_s
		[lat_e, lon_e] = [ratio*l_e[0] + (1-ratio)*l_s[0], ratio*l_e[1] + (1-ratio)*l_s[1]]
		for i in range(r_time):
			newline = [t_s+i+1, (i+1)/r_time*lat_e+(r_time-i-1)/r_time*lat_s,
				(i+1)/r_time*lon_e+(r_time-i-1)/r_time*lon_s]
			traj_list.append(newline)
		lat_s = lat_e; lon_s = lon_e; t_s = t_e
		if traveled < d and vehicle == 'bus':
			r_time = int(np.around(np.random.uniform(20, 60, 1),0))
			t_e = t_s + r_time
			for i in range(r_time):
				newline = [t_s+i+1, lat_s, lon_s]
				traj_list.append(newline)
			t_s = t_e
	traj_array = np.array(traj_list)
	err_lat = np.random.normal(loc=0.0, scale= 2*1e-5, size= traj_array.shape[0])
	err_lon = np.random.normal(loc=0.0, scale= 2*1e-5, size= traj_array.shape[0])
	traj_array[:,1] = traj_array[:,1] + err_lat
	traj_array[:,2] = traj_array[:,2] + err_lon
	return traj_array, d

def gen_basic_pause(l_s: tuple[float], t_s: float, t_e_range: list[float], t_diff_range: list[float]) -> np.ndarray:
	"""
	This function generates basic trajectories for a pause.\n
	Args: 
		l_s: tuple, coordinates of pause location
		t_s: float, starting time
		t_e_range: list, limits of ending time (None if t_diff_range used)
		t_diff_range: list, limits of duration (None if t_e_range used)
	Return: 
		numpy.ndarray, containing the trajectories
	"""
	traj_list = []
	if t_e_range is None:
		r_time = int(np.around(np.random.uniform(t_diff_range[0], t_diff_range[1], 1), 0))
	else:
		r_time = int(np.around(np.random.uniform(t_e_range[0], t_e_range[1], 1), 0) - t_s)
	std = 1*1e-5
	for i in range(r_time):
		newline = [t_s+i+1, l_s[0], l_s[1]]
		traj_list.append(newline)
	traj_array = np.array(traj_list)
	err_lat = np.random.normal(loc=0.0, scale= std, size= traj_array.shape[0])
	err_lon = np.random.normal(loc=0.0, scale= std, size= traj_array.shape[0])
	traj_array[:,1] = traj_array[:,1] + err_lat
	traj_array[:,2] = traj_array[:,2] + err_lon
	return traj_array

def gen_route_traj(route: list, vehicle: str, t_s: float) -> tuple[np.ndarray, float]:
	"""
	This function generates basic trajectories between multiple points.\n
	Args: 
		route: list, contains coordinates of multiple locations
		vehicle: str, means of transportation, can be one of the following: (car, bus, walk, bike)
		t_s: float, starting time
	Return: 
		numpy.ndarray, containing the trajectories
		float, total distance travelled
	"""
	total_d = 0
	traj = np.zeros((1,3))
	for i in range(len(route)-1):
		l_s = route[i]
		l_e = route[i+1]
		try:
			trip, d = gen_basic_traj(l_s, l_e, vehicle, t_s)
		except IndexError:
			route[i+1] = l_s
			continue
		total_d = total_d + d
		t_s = trip[-1,0]
		traj = np.vstack((traj,trip))
		if (i+1)!=len(route)-1 and vehicle=='bus':
			trip = gen_basic_pause(l_e, t_s, None, [5,120])
			t_s = trip[-1,0]
			traj = np.vstack((traj,trip))
	traj = traj[1:,:]
	return traj, total_d

def gen_all_traj(house_address: str, attributes: list, switches: dict, all_nodes: dict, start_date: datetime.date, end_date: datetime.date, api_key: str) -> tuple[np.ndarray, list, list]:
	"""
	Generates trajectories for a single person.\n
	Args:
		house_address: (str) indicating the house address of the person to generate
		attributes: (list) contains the attributes required to generate a class Person
		switches: (dictionary) contains changes of attributes in between the simulation
		all_nodes: (dictionary) contains all locations of all amenities around the house address 
		start_date: (datetime.date object) start date of trajectories
		end_date: (datetime.date object) end date of trajectories, end date is not included in the trajectories
		api_key: (str) api key for open route service
	Returns:
		traj: (numpy.ndarray) contains the gps trajectories of a single person, first column is time, second column is lattitude and third column is longitude
		home_time_list: (list) contains the time spent at home each day in seconds
		total_d_list: (list) contains the total distance travelled each day in meters
	"""

	person = Person(house_address, attributes, all_nodes)
	if len(person.possible_exits) < 4 or (person.main_employment > 0 and len(person.office_address) == 0):
		return [], [], []

	val_active_change = -1
	time_active_change = -1
	val_travel_change = -1
	time_travel_change = -1
	if len(switches.keys()) != 0:
		for key in switches.keys():
			key_list = key.split("-")
			if key_list[0] == "active_status":
				time_active_change = int(key_list[1]) - 1
				val_active_change = switches[key]
			elif key_list[0] == "travelling_status":
				time_travel_change = int(key_list[1]) - 1
				val_travel_change = switches[key]
			
	current_date = start_date
	
	t_s = 0
	traj = np.zeros((1,3))
	traj[0,0] = t_s
	traj[0,1] = person.house_address[0]
	traj[0,2] = person.house_address[1]
	
	home_time = 0
	total_d = 0
	
	home_time_list = []
	total_d_list = []

	while current_date < end_date:

		if t_s == time_travel_change*24*3600:
			person.setTravellingStatus(val_travel_change)
		if t_s == time_active_change*24*3600:
			person.setActiveStatus(val_active_change)

		current_weekdate = current_date.weekday()
		action, location, limits, exit = person.chooseAction(t_s, current_weekdate)

		if action == 'p':
			
			res = gen_basic_pause(location, t_s, None, limits)
			
			if location == person.house_address:
				home_time += res[-1, 0] - res[0, 0] + 1
				
			traj = np.vstack((traj, res))
			t_s = res[-1, 0]
				
		elif action == 'fpf':
			t_s_list = []
			d_temp = 0

			go_path, transport = person.calculateTrip(person.house_address, location, api_key)
			return_path, _ = person.calculateTrip(location, person.house_address, api_key)
			transport2 = transport
			if transport2 == 'foot':
				transport2 = 'walk'
			elif transport2 == 'bicycle':
				transport2 = 'bike'
			
			# Flight 1
			res1, d1 = gen_route_traj(go_path, transport2, t_s)
			t_s1 = res1[-1, 0]
			t_s_list.append(t_s1)
			traj2 = res1
			d_temp += d1

			if exit == 'office' and len(person.office_codes) != 0 and np.random.binomial(1, person.travelling_status/10, 1)==1:

				rndm_coefficient = np.random.uniform(0,0.6,1)[0]
				limits2 = [l * rndm_coefficient for l in limits]

				# Pause 1
				res2 = gen_basic_pause(location, t_s1, None, limits2)
				t_s2 = res2[-1, 0]
				t_s_list.append(t_s2)
				traj2 = np.vstack((traj2, res2))

				go_path2, return_path2, _ = person.choosePreferredOfficeExit(api_key)
				# Flight 1.1
				res3, d11 = gen_route_traj(go_path2, 'foot', t_s2)
				t_s3 = res3[-1, 0]
				t_s_list.append(t_s3)
				traj2 = np.vstack((traj2, res3))

				# Pause 2
				res4 = gen_basic_pause(return_path2[0], t_s3, None, [15 * 60, 60 * 60])
				t_s4 = res4[-1, 0]
				t_s_list.append(t_s4)
				traj2 = np.vstack((traj2, res4))

				# Flight 1.2
				res5, d12 = gen_route_traj(return_path2, 'foot', t_s4)
				t_s5 = res5[-1, 0]
				t_s_list.append(t_s5)
				traj2 = np.vstack((traj2, res5))

				# Pause 3
				limits3 = [max(l * (1-rndm_coefficient) - (t_s5-t_s2), 0) for l in limits]
				res6 = gen_basic_pause(location, t_s5, None, limits3)
				t_s6 = res6[-1, 0]
				t_s_list.append(t_s6)
				traj2 = np.vstack((traj2, res6))

				d_temp += d11 + d12

			else:
				# Pause
				res6 = gen_basic_pause(location, t_s1, None, limits)
				t_s6 = res6[-1, 0]
				t_s_list.append(t_s6)
				traj2 = np.vstack((traj2, res6))

			# Flight 2
			res7, d2 = gen_route_traj(return_path, transport2, t_s6)        
			t_s7 = res7[-1, 0]
			traj3 = np.vstack((traj2, res7))

			action3, location3, duration3, exit3  = person.chooseAction(t_s7, current_weekdate, update=False)
			update_exit = False
			if action3 == "fpf":
				if great_circle_dist(location[0], location[1], location3[0], location3[1]) < 5:
					update_exit = True
					# Pause 2
					res8 = gen_basic_pause(location, t_s6, None, duration3)
					t_s8 = res8[-1, 0]
					t_s_list.append(t_s8)
					traj2 = np.vstack((traj2, res8))

					# Flight 2
					res9 ,d21 = gen_route_traj(return_path, transport2, t_s8)
					t_s9 = res9[-1,0]
					t_s_list.append(t_s9)
					traj2 = np.vstack((traj2, res9))
					d_temp += d21
 
				elif great_circle_dist(location[0], location[1], location3[0], location3[1]) < great_circle_dist(person.house_address[0], person.house_address[1], location3[0], location3[1]):
					update_exit = True
					go_path3, _ = person.calculateTrip(location, location3, api_key)
					return_path3, _ = person.calculateTrip(location3, person.house_address, api_key)

					# Flight 2
					res8, d21 = gen_route_traj(go_path3, transport2, t_s6) 
					t_s8 = res8[-1, 0] 
					t_s_list.append(t_s8)
					traj2 = np.vstack((traj2, res8))

					# Pause 2.1
					res9 = gen_basic_pause(location3, t_s8, None, duration3)
					t_s9 = res9[-1, 0]
					t_s_list.append(t_s9)
					traj2 = np.vstack((traj2, res9))

					# Flight 3
					res10, d22 = gen_route_traj(return_path3, transport2, t_s9) 
					t_s10 = res10[-1, 0] 
					t_s_list.append(t_s10)
					traj2 = np.vstack((traj2, res10))

					d_temp += d21 + d22


			if t_s_list[-1] - (current_date - start_date).days * 24 * 3600 < 24 * 3600 and update_exit:
				person.updatePreferredExits(exit3)
				
				t_s = t_s_list[-1]
				traj = np.vstack((traj, traj2))
				total_d += d_temp

			elif t_s7 - (current_date - start_date).days * 24 * 3600 < 24 * 3600: 
				t_s = t_s7
				traj = np.vstack((traj, traj3))
				total_d += d_temp + d2

			else:
				# pause
				res = gen_basic_pause(person.house_address, t_s, None, [15*60, 30*60])
				t_s = res[-1, 0]
				traj = np.vstack((traj, res))
			
		elif action == 'p_night':
			if limits[0]+limits[1] != 0:
				res = gen_basic_pause(location, t_s, None, limits)
				
				if location == person.house_address:
					home_time += res[-1, 0] - res[0, 0] + 1
					
				traj = np.vstack((traj, res))
				t_s = res[-1, 0]
			
			current_date += datetime.timedelta(days=1)
			person.endOfDayReset()
			
			home_time_list.append(home_time)
			total_d_list.append(total_d)
			
			home_time = 0
			total_d = 0
		
	
	traj = traj[:-1,:] 	
		
	return traj, home_time_list, total_d_list

def remove_data(full_data: np.ndarray, cycle: int, p: float, day: int) -> np.ndarray:
	"""
	Only keeps observed data from simulated trajectories dpending on cycle and p.\n
	Args:
		full_data: (numpy.ndarray) contains the complete trajectories
		cycle: (int) on_period + off_period of observations, in minutes
		p: (float) off_period/cycle, in between 0 and 1
		day: (int) number of days in full_data
	Returns:
		obs_data: (numpy.ndarray) contains the trajectories of the on period.
	"""
	sample_dur = int(np.around(60*cycle*(1-p),0))
	for i in range(day):
		start = int(np.around(np.random.uniform(0, 60*cycle, 1),0))+86400*i
		index_cycle = np.arange(start, start + sample_dur)
		if i == 0:
			index_all = index_cycle
		while index_all[-1]< 86400*(i+1):
			index_cycle = index_cycle + cycle*60
			index_all = np.concatenate((index_all, index_cycle))
		index_all = index_all[index_all<86400*(i+1)]
	index_all = np.concatenate((np.arange(600),index_all, np.arange(86400*day-600,86400*day)))
	index_all = np.unique(index_all)
	obs_data = full_data[index_all,:]
	return obs_data

def prepare_data(obs: np.ndarray, s: int, tz_str: str) -> pd.DataFrame:
	"""
	Perpares the data in a dataframe.\n
	Args:
		obs: (numpy.ndarray) observed trajectories.
		s: (int) timestamp of starting day
		tz_str: (str) timezone
	Returns:
		new: (pandas.DataFrame) final dataframe of simulated gps data.
	"""
	utc_start = stamp2datetime(s, tz_str)
	utc_start_stamp = datetime2stamp(utc_start, "UTC")

	new = np.zeros((obs.shape[0],6))
	new[:,0] = (obs[:,0] + s)*1000
	new[:,1] = (obs[:,0] + utc_start_stamp)*1000
	new[:,2] = obs[:,1]
	new[:,3] = obs[:,2]
	new[:,4] = 0
	new[:,5] = 20
	new = pd.DataFrame(new,columns=['timestamp','UTC time','latitude',
			'longitude','altitude','accuracy'])
	return(new)

def impute2second(traj: np.ndarray) -> np.ndarray:
	secondwise = []
	for i in range(traj.shape[0]):
		for j in np.arange(int(traj[i,3]),int(traj[i,6])):
			ratio = (j-traj[i,3])/(traj[i,6]-traj[i,3])
			lat = ratio*traj[i,1]+(1-ratio)*traj[i,4]
			lon = ratio*traj[i,2]+(1-ratio)*traj[i,5]
			newline = [int(j-traj[0,3]), lat, lon]
			secondwise.append(newline)
	secondwise = np.array(secondwise)
	return secondwise

def int2str(h: int) -> str:
	"""
	Converts numbers to 2 digit strings.\n
	Args:
		h: (int) 
	Returns:
		str of h, 2 digit minimum
	"""
	if h<10:
		return str(0)+str(h)
	else:
		return str(h)

vehicle_dict = {'foot': 0, 'car': 1, 'bicycle': 2}
possible_exits_list = ['cafe', 'bar', 'restaurant', 'park', 'cinema', 'dance', 'fitness']
main_employment_dict = {'none': 0, 'work': 1, 'student': 2}

def process_attributes(attributes: dict[int, list], key: str, user: int) -> tuple[list, dict]:
	"""
	Preprocesses the attributes of each person.\n
	Args:
		attributes: (dictionary) contains attributes of each person, loaded from json file.
		key: (str) a key from attributes.keys()
		user: (int) number of user
	Returns:
		attrs: (list) list of attributes for a user
		switches: (dictionary) contains possible changes of attributes in between of simulation  
	"""
	attrs = []
	switches = {}

	if "vehicle" in attributes[key].keys():
		if attributes[key]['vehicle'] in vehicle_dict:
			pass
		else:
			print("For User "+ str(user) + " vehicle was not in ['foot', 'car', 'bicycle']")
			return []
		attrs.append(vehicle_dict[attributes[key]['vehicle']])
	else:
		attrs.append(np.random.choice(range(3), 1)[0])

	if "main_employment" in attributes[key].keys():
		if attributes[key]['main_employment'] in main_employment_dict:
			pass
		else:
			print("For User "+ str(user) + " main_employment was not in ['none', 'work', 'student']")
			return []
		attrs.append(main_employment_dict[attributes[key]['main_employment']])
	else:
		attrs.append(np.random.choice(range(3), 1)[0])

	if "active_status" in attributes[key].keys():
		if attributes[key]['active_status'] in range(11):
			pass
		else:
			print("For User "+ str(user) + " active_status was not in between 0-10")
			return []
		attrs.append(attributes[key]['active_status'])
	else:
		attrs.append(np.random.choice(range(11), 1)[0])

	if "travelling_status" in attributes[key].keys():
		if attributes[key]['travelling_status'] in range(11):
			pass
		else:
			print("For User "+ str(user) + " travelling_status was not in between 0-10")
			return []
		attrs.append(attributes[key]['travelling_status'])
	else:
		attrs.append(np.random.choice(range(11), 1)[0])
	
	
	if "preferred_exits" in attributes[key].keys():

		for exit in attributes[key]['preferred_exits']:
			if exit not in possible_exits_list:
				print("For User "+ str(user) + " exit " + exit +  " is not in ['cafe', 'bar', 'restaurant', 'park', 'cinema', 'dance', 'fitness']")

		preferred_exits = attributes[key]['preferred_exits']
		possible_exits2 = [x for x in possible_exits_list if x not in preferred_exits]
		
		random_exits = np.random.choice(possible_exits2, 3 - len(preferred_exits), replace=False).tolist()
		for choice in random_exits:
			preferred_exits.append(choice)

		attrs.append(preferred_exits)
	else:
		attrs.append(np.random.choice(possible_exits_list, 3, replace=False).tolist())

	for x in attributes[key].keys():
		key_list = x.split("-")
		if len(key_list) == 2:
			switches[x] = attributes[key][x]

	return attrs, switches

def sim_GPS_data(N: int, location: str, start_date: list[int], end_date: list[int], cycle: int, p: float, api_key: str, data_folder: str, attributes_dir: str = None):
	"""
	Generates gps trajectories.\n
	Args:
		N: (int) number of people to simulate
		location: (str) indicating country and city to simulate at, format "Country_2_letter_ISO_code/City_Name"
		start_date: (list) start date of trajectories, format  [year, month, day]
		end_date: (list) end date of trajectories, end date is not included in the trajectories, format [year, month, day]
		cycle: (int) the sum of on-cycle and off_cycle, unit is minute
		p: (float) the missing rate, in other words, the proportion of off_cycle, should be within [0,1]
		api_key: (str), api key for open route service https://openrouteservice.org/
		data_folder: (str) directory to save trajectories
		attributes_dir: (str) directory to json file containing attributes for each user, optional
	"""

	print("Loading Attributes...")
	attributes_dictionary = {}
	switches_dictionary = {}


	if attributes_dir != None:
		attributes = json.load(open(attributes_dir))
		for key in attributes.keys():
			users = re.search(r"[0-9]*-?[0-9]+", key).group(0).split('-')
			if len(users) == 0:
				print("Wrong format in attributes.json on " + key)
				sys.exit()
			elif len(users) == 1:
				user = int(users[0])
				attrs, switches = process_attributes(attributes, key, user)
				if len(attrs) == 0:
					sys.exit()
				attributes_dictionary[user] = attrs
				switches_dictionary[user] = switches
			else:
				for user in range(int(users[0]), int(users[1]) + 1):
					attrs, switches = process_attributes(attributes, key, user)
					if len(attrs) == 0:
						sys.exit()
					attributes_dictionary[user] = attrs
					switches_dictionary[user] = switches
	
	for user in range(1, N + 1):
		if user not in attributes_dictionary.keys():
			# attributes: [vehicle, main_employment, athletic_status, active_status, travelling_status, preferred_exits]
			# look at definition of Person class
			attributes_dictionary[user] = [np.random.choice(range(3), 1)[0], np.random.choice(range(3), 1)[0],
			np.random.choice(range(11), 1)[0], np.random.choice(range(11), 1)[0], np.random.choice(possible_exits_list, 3, replace=False).tolist()]
		if user not in switches_dictionary.keys():
			switches_dictionary[user] = {}
	
	print("Gathering Addresses...")
	try:
		location_ctr = location.split("/")[0]
		location_city = location.split("/")[1]
	except IndexError:
		print("Location provided did not have the correct format.")
		sys.exit()

	api = overpy.Overpass()

	overpy_query = """
	[out:json];
	area["ISO3166-1"="{}"][admin_level=2] -> .country;
	area["name"="{}"] -> .city;
	node(area.country)(area.city)["addr:street"];
	out center {};
	""".format(location_ctr, location_city, str(150))
	
	try:
		r = api.query(overpy_query)
	except:
		time.sleep(30)
		try:
			r = api.query(overpy_query)	
		except:
			print("Too many Overpass requests in a short time. Please try again in a minute.")
			sys.exit()

	if len(r.nodes) == 0:
		overpy_query = """
		[out:json];
		area["ISO3166-1"="{}"][admin_level=2] -> .country;
		area["name:en"="{}"] -> .city;
		node(area.country)(area.city)["addr:street"];
		out center {};
		""".format(location_ctr, location_city, str(100))

		try:
			r = api.query(overpy_query)
		except:
			time.sleep(30)
			try:
				r = api.query(overpy_query)	
			except:
				print("Too many Overpass requests in a short time. Please try again in a minute.")
				sys.exit()

	try:
		index = np.random.choice(range(len(r.nodes)), 100, replace=False)
	except ValueError:
		print("Overpass query came back empty. Check that the location argument, ISO code and city name, did not have any misspellings.")
		sys.exit()
	nodes = np.array(r.nodes)[index]

	location_coords = (float(nodes[0].lat), float(nodes[0].lon))
	
	obj = TimezoneFinder()
	tz_str = obj.timezone_at(lng=location_coords[1], lat=location_coords[0])
	
	start_date = datetime.date(start_date[0],start_date[1],start_date[2])
	end_date = datetime.date(end_date[0],end_date[1],end_date[2])
	no_of_days = (end_date - start_date).days

	s = datetime2stamp([start_date.year,start_date.month,start_date.day,0,0,0],tz_str)*1000

	if os.path.exists(data_folder)==False:
		os.mkdir(data_folder)

	user = 0
	ind = 0
	print("Starting to generate trajectories...")
	while user < N:

		house_address = (float(nodes[ind].lat), float(nodes[ind].lon))
		house_area = boundingBox(house_address[0], house_address[1], 2000)

		attrs = attributes_dictionary[user+1]
		if attrs[1] == 1:
			q_employment = 'node' + str(house_area) + '["office"];'
		elif attrs[1] == 2:
			house_area2 = boundingBox(house_address[0], house_address[1], 3000)
			q_employment = 'node' + str(house_area2) + '["amenity"="university"];\n\t\t\tway' + str(house_area2) + '["amenity"="university"];'
		else:
			q_employment = ""
		
		q = """
		[out:json];
		(
			node{0}["amenity"="cafe"];
			node{0}["amenity"="bar"];
			node{0}["amenity"="restaurant"];
			node{0}["amenity"="cinema"];
			node{0}["leisure"="park"];
			node{0}["leisure"="dance"];
			node{0}["leisure"="fitness_centre"];
			way{0}["amenity"="cafe"];
			way{0}["amenity"="bar"];
			way{0}["amenity"="restaurant"];
			way{0}["amenity"="cinema"];
			way{0}["leisure"="park"];
			way{0}["leisure"="dance"];
			way{0}["leisure"="fitness_centre"];
			{1}
		);
		out center;
		""".format(house_area, q_employment)

		overpass_url = "http://overpass-api.de/api/interpreter"
		response = requests.get(overpass_url, params={'data': q}, timeout=5*60)
		if response.status_code == 200:
			pass
		else:
			time.sleep(60)
			response = requests.get(overpass_url, params={'data': q}, timeout=5*60)
		try:
			res = response.json()
		except:
			time.sleep(60)
			response = requests.get(overpass_url, params={'data': q}, timeout=5*60)
			try:
				res = response.json()
			except:
				print("Too many Overpass requests in a short time. Please try again in a minute.")
				sys.exit()
				
		all_nodes = {'cafe': [], 'bar': [], 'restaurant': [],
		'cinema': [], 'park': [], 'dance': [],
		'fitness': [], 'office': [], 'university': []}

		for element in res['elements']:
			if element['type'] == 'node':
				lon = element['lon']
				lat = element['lat']
			elif 'center' in element:
				lon = element['center']['lon']
				lat = element['center']['lat']

			if 'office' in element['tags']:
				all_nodes['office'].append((lat, lon))

			if 'amenity' in element['tags']:
				if element['tags']['amenity'] == 'cafe':
					all_nodes['cafe'].append((lat, lon))
				if element['tags']['amenity'] == 'bar':
					all_nodes['bar'].append((lat, lon))
				if element['tags']['amenity'] == 'restaurant':
					all_nodes['restaurant'].append((lat, lon))
				if element['tags']['amenity'] == 'cinema':
					all_nodes['cinema'].append((lat, lon))
				if element['tags']['amenity'] == 'university':
					all_nodes['university'].append((lat, lon))
			elif 'leisure' in element['tags']:
				if element['tags']['leisure'] == 'park':
					all_nodes['park'].append((lat, lon))
				if element['tags']['leisure'] == 'dance':
					all_nodes['dance'].append((lat, lon))
				if element['tags']['leisure'] == 'fitness_centre':
					all_nodes['fitness'].append((lat, lon))


		if os.path.exists(data_folder+"/user_"+str(user+1))==False:
			os.mkdir(data_folder+"/user_"+str(user+1))
		if os.path.exists(data_folder+"/user_"+str(user+1)+"/gps")==False:
			os.mkdir(data_folder+"/user_"+str(user+1)+"/gps")


		all_traj,all_T,all_D = gen_all_traj(house_address, attributes_dictionary[user+1], switches_dictionary[user+1],all_nodes, start_date, end_date, api_key)
		if len(all_traj) == 0:
			ind += 1
			continue
		all_D = np.array(all_D)/1000
		all_T = np.array(all_T)/3600

		print("User_"+str(user+1))
		print("	distance(km): ", all_D.tolist())
		print("	hometime(hr): ", all_T.tolist())
		obs = remove_data(all_traj,cycle,p,no_of_days)
		obs_pd = prepare_data(obs, s/1000, tz_str)
		for i in range(no_of_days):
			for j in range(24):
				s_lower = s+i*24*60*60*1000+j*60*60*1000
				s_upper = s+i*24*60*60*1000+(j+1)*60*60*1000
				temp = obs_pd[(obs_pd["timestamp"]>=s_lower)&(obs_pd["timestamp"]<s_upper)]
				[y,m,d,h,mins,sec] = stamp2datetime(s_lower/1000,"UTC")
				filename = str(y)+"-"+int2str(m)+"-"+int2str(d)+" "+int2str(h)+"_00_00.csv"
				temp.to_csv(data_folder+"/user_"+str(user+1)+"/gps/"+filename,index = False)

		user += 1
		ind += 1
