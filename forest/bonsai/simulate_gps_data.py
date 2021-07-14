import os
import numpy as np
import pandas as pd
import re
import datetime
import time
import requests
import json
import overpy
from timezonefinder import TimezoneFinder
from forest.poplar.legacy.common_funcs import datetime2stamp,stamp2datetime
from forest.jasmine.data2mobmat import great_circle_dist


api_key = "5b3ce3597851110001cf6248551c505f7c61488a887356ff5ea924d5"
R = 6.371*10**6

def getPath(lat1, lon1, lat2, lon2, transport, api_key):
	"""
	This function takes 2 sets of coordinates and a mean of transport and using the openroute api
	calculates the set of nodes to traverse from location1 to location2 along with the duration 
	and distance of the flight
	Args: 
	lat1, lon1: coordinates of start point
	lat2, lon2: coordinates of end point
	 transport: means of transportation, can be one of the following: (car, bus, foot, bicycle)
	   api_key: api key collected from https://openrouteservice.org/dev/#/home
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
							headers=headers)
		
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
			print(call.status_code, call.reason)
			return np.array([[lon1, lat1], [lon2, lat2]]), great_circle_dist(lat1, lon1, lat2, lon2)
	
	
	
def basicPath(path, transport):
	"""
	This function takes a path from getPath() function and subsets it
	to the provided number of nodes
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



def createAddressQuery(ctr_iso, city, results):
	"""
	This function outputs a query to parse in the overpass api to
	output a certain number of random addressess
	Args: 
		ctr_iso: str, country 2 letter ISO code
		   city: str, city name
		results: int, number of addresses to output
	Return: 
		oberpass query string
	"""


	q = """
	[out:json];
	area["ISO3166-1"="{}"][admin_level=2] -> .country;
	area[name="{}"] -> .city;
	node(area.country)(area.city)["addr:street"];
	out center {};
	""".format(ctr_iso, city, str(results))
	
	return q
	
def boundingBox(lat, lon, radius):
	"""
	This function outputs an area around a set of coordinates
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

def createAmenityQuery(area, amenity):
	"""
	This function outputs a query to parse in the overpass api to
	output a certain number of random addressess of certain amenities
	Args: 
		area: tuple with 4 elements, bounding box area to look at, output from boundingBox() function
	 amenity: str, amenity to look for can be one of the following
		 ['cafe', 'bar', 'restaurant', 'university', 'marketplace','station', 'bank','hospital', 
		  'pharmacy','cinema', 'theatre', 'fitness_centre', 'park', 'dance', 'grocery', 'supermarket', 'clothes', 'office']
	Return: 
		overpass query string
	"""
	# amenity in ['cafe', 'bar', 'restaurant', 'university', 'marketplace','station', 'bank','hospital', 
	# 'pharmacy','cinema', 'theatre', 'fitness_centre', 'park', 'dance', 'grocery', 'supermarket', 'clothes', 'office']
	if amenity == 'station':
		code = 'public_transport' 
	elif amenity in ['fitness_centre', 'sports_centre', 'dance', 'park']:
		code = 'leisure'
	elif amenity in ['grocery', 'supermarket', 'clothes']:
		code = 'shop'
	else:
		code = 'amenity' 

	q = """
	[out:json];
	node{}["{}"="{}"];
	out 100;
	""".format(area,code,amenity)
	
	if amenity == 'office':
		q = """
			[out:json];
			node{}["office"];
			out 100;
			""".format(area)
	
	return q

def getQueryResults(api, query_str, n):
	"""
	This function runs an overpass query and randomly selects nodes from the results
	Args: 
		   api: overpass api object
	 query_str: str, representing an overpass query
			 n: int, number of nodes to sample
	Return: 
		list of coordinates
	"""

	
	res = api.query(query_str)
	if len(res.nodes) == 0:
		return []
	elif len(res.nodes) < n:
		return [(float(node.lat), float(node.lon)) for node in res.nodes]
	else:
		index = np.random.choice(range(len(res.nodes)), n, replace=False)
	
		if n == 1:
			index = index[0]
			return (float(res.nodes[index].lat), float(res.nodes[index].lon))
		else:
			return [(float(res.nodes[i].lat), float(res.nodes[i].lon)) for i in index]
	
	


class Person:
	"""
	This class represents a person whose life we want to simulate. 
	"""

	
	def __init__(self, house_address, attributes, all_nodes):
		"""
		This function sets the basic attributes and information to be used of the person.
		Args: 
			house_address: tuple, coordinates of primary home
			   attributes: list, consists of various information
			   attributes = [vehicle, main_employment, athletic_status, active_status, travelling_status, preferred_exits]
				* vehicle = 0: nothing, 1: own car, 2: own_bicycle | used for distances and time of flights
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
				query_str = createAmenityQuery(self.house_area, "office")
			elif self.main_employment == 2:
				query_str = createAmenityQuery(self.house_area, "university")
				
			try:
				self.office_address = getQueryResults(overpy.Overpass(), query_str, 1)
			except:
				time.sleep(20)
				self.office_address = getQueryResults(overpy.Overpass(), query_str, 1)
			no_office_days = np.random.randint(3,6,1)[0]
			self.office_days = np.random.choice(range(5), no_office_days, replace=False)
			self.office_days.sort()
		else:
			self.office_days = []
			
			
			
					
		# define favorite places
		self.possible_exits = ['cafe', 'bar', 'restaurant', 'park', 'cinema', 'dance', 'fitness']

		for exit in self.possible_exits:
			if len(all_nodes[exit]) > 3:
				random_places = np.random.choice(range(len(all_nodes[exit])), 3, replace=False).tolist()
				setattr(self, exit+'_places', [(float(place.lat), float(place.lon)) for place in np.array(all_nodes[exit])[random_places]])
			elif len(all_nodes[exit]) == 0:
				setattr(self, exit+'_places', [])
			else:
				setattr(self, exit+'_places', [(float(place.lat), float(place.lon)) for place in np.array(all_nodes[exit])])

			distances = [great_circle_dist(self.house_address[0], self.house_address[1], place[0], place[1]) for place in getattr(self, exit+'_places')]
			order = np.argsort(distances)
			setattr(self, exit+'_places_ordered', np.array(getattr(self, exit+'_places'))[order].tolist())


		possible_exits2 = self.possible_exits.copy()
		for act in possible_exits2:
			if len(getattr(self, act+'_places')) == 0:
				self.possible_exits.remove(act)


		travelling_status_norm = self.travelling_status / 10
		for act in self.possible_exits:
			act_places = getattr(self, act + '_places_ordered')

			places = []
			for i in range(len(act_places) - 1, -1, -1):
				index = np.random.binomial(i, travelling_status_norm)
				places.append(act_places[index])
				del act_places[index]

			setattr(self, act + '_places', places)


	def updatePreferredLocations(self, travelling_status):
		"""
		Update preferred locations of exits depending on new travelling status.
		Args:
			travelling_status: 0-10 | int indicating new travelling_status
		"""

		setattr(self, 'travelling_status', travelling_status)

		travelling_status_norm = self.travelling_status / 10
		for act in self.possible_exits:
			act_places = getattr(self, act + '_places')

			places = []
			for i in range(len(act_places) - 1, -1, -1):
				index = np.random.binomial(i, travelling_status_norm)
				places.append(act_places[index])
				del act_places[index]

			setattr(self, act + '_places', places)

		
		
	def updatePreferredExits(self, exit_code):
		"""
		This function updates the set of preferred exits for the day, after an action has been performed.
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
		
	def choosePreferredExit(self, t_s):
		"""
		This function samples through the possible actions for the person, 
		depending on his attributes and the time.
		Args: 
			t_s: float, current time in seconds
		Return:
			  selected_action_decoded: str, selected action to perform
					selected_location: tuple, selected location's coordinates
		"""

		probs_of_staying_home = [1-0.8*self.active_status/10, 0.8*self.active_status/10]
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
		
		self.updatePreferredExits(selected_action)
				
		action_locations = getattr(self, selected_action+'_places')
		ratios2 = [7, 2, 1][:len(action_locations)]
		probabilities2 = np.array(ratios2)       
		probabilities2 = probabilities2/sum(probabilities2)
		
		selected_location_index = np.random.choice(range(len(action_locations)), 1, p=probabilities2)[0]
		selected_location = action_locations[selected_location_index]
		
		return selected_action, selected_location

	
		
	def endOfDayReset(self):
		"""
		Reset preferred exits of the day. To run when a day ends
		"""
		self.preferred_exits_today = self.preferred_exits
		self.office_today = False

	
	def calculateTrip(self, destination, api_key):
		"""
		This function uses the openrouteservice api to produce the path
		from person's house to destination and back.
		Args: 
			destination: tuple, coordinates for destination
				api_key: str, openrouteservice api key
		Return:
			  go_path: 2d numpy array, containing [lat,lon] of route from house to destination
		  return_path: 2d numpy array, containing [lat,lon] of route from destination to house
			transport: str, means of transport
		"""


		home_distance = great_circle_dist(self.house_address[0], self.house_address[1], destination[0], destination[1])
		transportations = {0: 'bus', 1: 'car', 2: 'bicycle'}

		if home_distance <= 1000:
			transport = 'foot'
		else:
			transport = transportations[self.vehicle]

		if str(destination[0])+'_'+str(destination[1])+'_go' in self.trips.keys():
			go_path = self.trips[str(destination[0])+'_'+str(destination[1])+'_go']
			return_path = self.trips[str(destination[0])+'_'+str(destination[1])+'_return']
		else:
			path, _ = getPath(self.house_address[0], self.house_address[1], destination[0], destination[1], transport, api_key)
			go_path = basicPath(path, transport)

			path2, _ = getPath(destination[0], destination[1], self.house_address[0], self.house_address[1], transport, api_key)
			return_path = basicPath(path2, transport)

			go_path = [[x[1], x[0]] for x in go_path]
			return_path = [[x[1], x[0]] for x in return_path]

			self.trips[str(destination[0])+'_'+str(destination[1])+'_go'] = go_path
			self.trips[str(destination[0])+'_'+str(destination[1])+'_return'] = return_path

		return go_path, return_path, transport

	def chooseAction(self, t_s, day_now):
		"""
		This function decides action for person to take.
		Args: 
				  t_s: int, current time in seconds
			  day_now: int, day of the week
		Return:
			  str, 'p', 'p_night' or 'fpf' indicating pause, pause for the night or flight-pause-flight
			tuple, destination's coordinates
			 list, contains [minimum, maximum] duration of pause in seconds
		"""
		time_now = t_s % (24 * 60 * 60)

		if time_now == 0: 
			if day_now < 5 and self.main_employment > 0:
				return 'p', self.house_address,[8 * 3600, 9 * 3600], 'home_morning'
			else: 
				return 'p', self.house_address,[8 * 3600, 12 * 3600], 'home_morning'
		
		if not self.office_today:
			self.office_today = not self.office_today
			if day_now in self.office_days:
				return 'fpf', self.office_address, [7 * 3600, 9 * 3600], 'office'
			elif day_now < 5:
				return 'p', self.house_address, [7 * 3600, 9 * 3600], 'office_home'

		exit, location = self.choosePreferredExit(t_s)

		if exit == 'home':
			if time_now + 2*3600 > 24*3600 - 1:
				return 'p_night', self.house_address,[24 * 3600 - time_now, 24 * 3600 - time_now], "home_night"
			return 'p', self.house_address,[0.5 * 3600, 2 * 3600], exit
		elif exit == 'home_night':
			return 'p_night', self.house_address,[24 * 3600 - time_now, 24 * 3600 - time_now], exit
		else:
			return 'fpf', location, [0.5 * 3600, 2 * 3600], exit



def gen_basic_traj(l_s, l_e, vehicle, t_s):
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

def gen_basic_pause(l_s, t_s, t_e_range, t_diff_range):
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

def gen_route_traj(route, vehicle, t_s):
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

def gen_all_traj(house_address, attributes, all_nodes, start_date, end_date):
	"""
	Generates trajectories for a single person.
	Args:
		house_address: (str) indicating the house address of the person to generate
		attributes: (list) contains the attributes required to generate a class Person
		all_nodes: (dictionary) contains all locations of all amenities around the house address 
		start_date: (datetime.date object) start date of trajectories
		end_date: (datetime.date object) end date of trajectories, end date is not included in the trajectories
	Returns:
		traj: (numpy.ndarray) contains the gps trajectories of a single person, first column is time, second column is lattitude and third column is longitude
		home_time_list: (list) contains the time spent at home each day in seconds
		total_d_list: (list) contains the total distance travelled each day in meters
	"""

	person = Person(house_address, attributes, all_nodes)
		
	if len(person.possible_exits) < 4 or (person.main_employment > 0 and len(person.office_address) == 0):
		return [], [], []
			
	current_date = start_date
	
	t_s = 0
	traj = np.zeros((1,3))
	traj[0,0] = t_s
	traj[0,1] = person.house_address[0]
	traj[0,2] = person.house_address[1]
	
	home_time = 0
	total_d = 0
	daily_actions = []
	
	home_time_list = []
	total_d_list = []
	daily_actions_list = []

	while current_date < end_date:

		#if t_s == 4*24*3600:
		#	person.updatePreferredLocations(3)

		current_weekdate = current_date.weekday()
		action, location, limits, exit = person.chooseAction(t_s, current_weekdate)
		if action == 'p':
			
			res = gen_basic_pause(location, t_s, None, limits)
			
			if location == person.house_address:
				home_time += res[-1, 0] - res[0, 0]
				
			daily_actions.append(exit)

			traj = np.vstack((traj, res))
			t_s = res[-1, 0]
				
		elif action == 'fpf':
			go_path, return_path, transport = person.calculateTrip(location, api_key)
			transport2 = transport
			if transport2 == 'foot':
				transport2 = 'walk'
			elif transport2 == 'bicycle':
				transport2 = 'bike'
			
			# Flight 1
			res1, d1 = gen_route_traj(go_path, transport2, t_s)
			
			t_s1 = res1[-1, 0]
			# Pause
			res2 = gen_basic_pause(location, t_s1, None, limits)
			
			t_s3 = res2[-1, 0]
			# Flight 2
			res3, d3 = gen_route_traj(return_path, transport2, t_s3)        
			
			t_s4 = res3[-1, 0]
			
			if (t_s4%(24*3600)) < 24*60*60:
				daily_actions.append(exit)
				t_s = t_s4
				traj = np.vstack((traj, res1, res2, res3))
				total_d += (d1 + d3)
				if location == person.house_address:
					home_time += res2[-1, 0] - res2[0, 0]
			
		elif action == 'p_night':
			daily_actions.append(exit)
			if limits[0]+limits[1] != 0:
				res = gen_basic_pause(location, t_s, None, limits)
				
				if location == person.house_address:
					home_time += res[-1, 0] - res[0, 0]
					
				traj = np.vstack((traj, res))
				t_s = res[-1, 0]
			
			current_date += datetime.timedelta(days=1)
			person.endOfDayReset()
			
			home_time_list.append(home_time)
			total_d_list.append(total_d)
			daily_actions_list.append(daily_actions)
			
			home_time = 0
			total_d = 0
			daily_actions = []
	
	traj = traj[:-1,:] 	
		
	return traj, home_time_list, total_d_list
		
		   

## cycle is minute
def remove_data(full_data,cycle,p,day):
	## keep the first and last 10 minutes,on-off-on-off,cycle=on+off,p=off/cycle
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

def prepare_data(obs, s):
	new = np.zeros((obs.shape[0],6))
	new[:,0] = (obs[:,0] + s)*1000
	new[:,1] = 0
	new[:,2] = obs[:,1]
	new[:,3] = obs[:,2]
	new[:,4] = 0
	new[:,5] = 20
	new = pd.DataFrame(new,columns=['timestamp','UTC time','latitude',
			'longitude','altitude','accuracy'])
	return(new)

def impute2second(traj):
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

def int2str(h):
	if h<10:
		return str(0)+str(h)
	else:
		return str(h)

vehicle_dictionary = {'foot':0, 'car':1, 'bicycle':2}
possible_exits = ['cafe', 'bar', 'restaurant', 'park', 'cinema', 'dance', 'fitness']
main_employment_dictionary = {'none':0, 'work':1, 'student':2}

def process_attributes(attributes, key, user):

	attrs = []

	if "vehicle" in attributes[key].keys():
		if attributes[key]['vehicle'] in ['foot', 'car', 'bicycle']:
			pass
		else:
			print("For User "+ str(user) + " vehicle was not in ['foot', 'car', 'bicycle']")
			return []
		attrs.append(vehicle_dictionary[attributes[key]['vehicle']])
	else:
		attrs.append(np.random.choice(range(3), 1)[0])

	if "main_employment" in attributes[key].keys():
		if attributes[key]['main_employment'] in ['none', 'work', 'student']:
			pass
		else:
			print("For User "+ str(user) + " main_employment was not in ['none', 'work', 'student']")
			return []
		attrs.append(main_employment_dictionary[attributes[key]['main_employment']])
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
			if exit not in possible_exits:
				print("For User "+ str(user) + " exit " + exit +  " is not in ['cafe', 'bar', 'restaurant', 'park', 'cinema', 'dance', 'fitness']")

		preferred_exits = attributes[key]['preferred_exits']
		possible_exits2 = [x for x in possible_exits if x not in preferred_exits]
		
		random_exits = np.random.choice(possible_exits2, 3 - len(preferred_exits), replace=False).tolist()
		for choice in random_exits:
			preferred_exits.append(choice)

		attrs.append(preferred_exits)
	else:
		attrs.append(np.random.choice(possible_exits, 3, replace=False).tolist())

	return attrs

def sim_GPS_data(N, location, start_date, end_date, cycle, p, data_folder, attributes_dir = None):
	"""
	Generates gps trajectories.
	Args:
		N: (int) number of people to simulate
		location: (str) indicating country and city to simulate at, format "Country_2_letter_ISO_code/City_Name"
		start_date: (str) start date of trajectories, format "day/month/year"
		end_date: (str) end date of trajectories, end date is not included in the trajectories, format "day/month/year"
		cycle: (int) the sum of on-cycle and off_cycle, unit is minute
		p: (float) the missing rate, in other words, the proportion of off_cycle, should be within [0,1]
		data_folder: (str) directory to save trajectories
		attributes_dir: (str) directory to json file containing attributes for each user, optional
	Returns:
		a pandas dataframe with only observations from on_cycles, which mimics the real data file
	"""

	attributes_dictionary = {}
	attributes = json.load(open(attributes_dir))


	if attributes_dir != None:
		for key in attributes.keys():
			users = re.search(r"[0-9]*-?[0-9]+", key).group(0).split('-')
			if len(users) == 0:
				print("Wrong format in attributes.json on " + key)
				return None
			elif len(users) == 1:
				user = int(users[0])
				attrs = process_attributes(attributes, key, user)
				if len(attrs) == 0:
					return None
				attributes_dictionary[user] = attrs
			else:
				for user in range(int(users[0]), int(users[1]) + 1):
					attrs = process_attributes(attributes, key, user)
					if len(attrs) == 0:
						return None
					attributes_dictionary[user] = attrs

	location_ctr = location.split("/")[0]
	location_city = location.split("/")[1]
	q = """
		area["ISO3166-1"="{}"][admin_level=2];
		(
		node["place"="city"]["name"="{}"](area);
		);
		out;
		""".format(location_ctr, location_city)

	api = overpy.Overpass()
	res = api.query(q)
	location_coords = (float(res.nodes[0].lat), float(res.nodes[0].lon))
	
	obj = TimezoneFinder()
	tz_str = obj.timezone_at(lng=location_coords[0], lat=location_coords[1])

	start_date = np.array(start_date.split("/")).astype(int)
	end_date = np.array(end_date.split("/")).astype(int)
    
	start_date = datetime.date(start_date[2],start_date[1],start_date[0])
	end_date = datetime.date(end_date[2],end_date[1],end_date[0])
	no_of_days = (end_date - start_date).days

	s = datetime2stamp([start_date.year,start_date.month,start_date.day,0,0,0],tz_str)*1000
	if os.path.exists(data_folder)==False:
		os.mkdir(data_folder)

	overpy_query = createAddressQuery(location_ctr, location_city, 100)
	try:
		r = api.query(overpy_query)
	except:
		time.sleep(20)
		r = api.query(overpy_query)	
	index = np.random.choice(range(len(r.nodes)), 100, replace=False)
	nodes = np.array(r.nodes)[index]
		
	user = 0
	i = 0
	while user < N:

		house_address = (float(nodes[i].lat), float(nodes[i].lon))
		house_area = boundingBox(house_address[0], house_address[1], 2000)

		
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
		);
		out;
		""".format(house_area)
		try:
			res = api.query(q)
		except:
			time.sleep(20)
			res = api.query(q)

		all_nodes = {'cafe': [], 'bar': [], 'restaurant': [],
		'cinema': [], 'park': [], 'dance': [],
		'fitness': []}

		for node in res.nodes:
			if 'amenity' in node.tags.keys():
				if node.tags['amenity'] == 'cafe':
					all_nodes['cafe'].append(node)
				if node.tags['amenity'] == 'bar':
					all_nodes['bar'].append(node)
				if node.tags['amenity'] == 'restaurant':
					all_nodes['restaurant'].append(node)
				if node.tags['amenity'] == 'cinema':
					all_nodes['cinema'].append(node)
			elif 'leisure' in node.tags.keys():
				if node.tags['leisure'] == 'park':
					all_nodes['park'].append(node)
				if node.tags['leisure'] == 'dance':
					all_nodes['dance'].append(node)
				if node.tags['leisure'] == 'fitness_centre':
					all_nodes['fitness'].append(node)


		if os.path.exists(data_folder+"/user_"+str(user+1))==False:
			os.mkdir(data_folder+"/user_"+str(user+1))
		if os.path.exists(data_folder+"/user_"+str(user+1)+"/gps")==False:
			os.mkdir(data_folder+"/user_"+str(user+1)+"/gps")


		all_traj,all_T,all_D = gen_all_traj(house_address, attributes_dictionary[user+1], all_nodes, start_date, end_date)
		if len(all_traj) == 0:
			i += 1
			continue
		all_D = np.array(all_D)/1000
		all_T = np.array(all_T)/3600

		print("User_"+str(user+1))
		print("distance(km): ", all_D.tolist())
		print("hometime(hr): ", all_T.tolist())
		obs = remove_data(all_traj,cycle,p,no_of_days)
		obs_pd = prepare_data(obs, s/1000)
		for i in range(no_of_days):
			for j in range(24):
				s_lower = s+i*24*60*60*1000+j*60*60*1000
				s_upper = s+i*24*60*60*1000+(j+1)*60*60*1000
				temp = obs_pd[(obs_pd["timestamp"]>=s_lower)&(obs_pd["timestamp"]<s_upper)]
				[y,m,d,h,mins,sec] = stamp2datetime(s_lower/1000,"UTC")
				filename = str(y)+"-"+int2str(m)+"-"+int2str(d)+" "+int2str(h)+"_00_00.csv"
				temp.to_csv(data_folder+"/user_"+str(user+1)+"/gps/"+filename,index = False)

		user += 1
		i += 1
