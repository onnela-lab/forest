"""
Module to simulate realistic GPS trajectories
of a number of people anywhere in the world.
"""

import os
import time
from typing import Tuple

import numpy as np
import openrouteservice
import pandas as pd

from forest.jasmine.data2mobmat import great_circle_dist
from forest.poplar.legacy.common_funcs import datetime2stamp, stamp2datetime

b_start = [42.3696, -71.1131]
home_g = [42.3678, -71.1138]
home_e = [42.3646, -71.1128]
hmart = [42.3651, -71.1027]
gym = [42.3712, -71.1197]
mit = [42.3589, -71.0992]
turn1 = [42.3473, -71.0877]
turn2 = [42.3433, -71.1025]
turn3 = [42.3389, -71.1073]
turn4 = [42.3641, -71.1181]
turn5 = [42.3526, -71.1106]
turn6 = [42.3609, -71.0708]
b_end = [42.3370, -71.1073]
hsph = [42.3356, -71.1038]
work = [42.3444, -71.1135]
movie = [42.3524, -71.0645]
restaurant = [42.3679,-71.1089]
R = 6.371*10**6


def get_path(start: Tuple[float, float], end: Tuple[float, float], transport: str,
             api_key: str) -> Tuple[np.ndarray, float]:
    """Calculates paths between sets of coordinates

    This function takes 2 sets of coordinates and
    a mean of transport and using the openroute api
    calculates the set of nodes to traverse
    from location1 to location2 along with the duration
    and distance of the flight.
    Args:
        start: coordinates of start point (lat, lon)
        end: coordinates of end point (lat, lon)
        transport: means of transportation,
            can be one of the following:
            (car, bus, foot, bicycle)
        api_key: api key collected from
            https://openrouteservice.org/dev/#/home
    Returns:
        path_coordinates: 2d numpy array containing [lat,lon] of route
        distance: distance of trip in meters
    Raises:
        RuntimeError: An error when openrouteservice does not 
            return coordinates of route as expected
    """

    lat1, lon1 = start
    lat2, lon2 = end
    distance = great_circle_dist(lat1, lon1, lat2, lon2)

    if distance < 250:
        return (np.array([[lat1, lon1], [lat2, lon2]]),
                distance)

    if transport in ("car", "bus"):
        transport = "driving-car"
    elif transport == "foot":
        transport = "foot-walking"
    elif transport == "bicycle":
        transport = "cycling-regular"
    client = openrouteservice.Client(key=api_key)
    coords = ((lon1, lat1), (lon2, lat2))

    try:
        routes = client.directions(coords, profile=transport, format="geojson")
    except Exception:
        raise RuntimeError(
            "Openrouteservice did not return proper trajectories."
        )

    coordinates = routes["features"][0]["geometry"]["coordinates"]
    path_coordinates = [[coord[1], coord[0]] for coord in coordinates]

    # sometimes if exact coordinates of location are not in a road
    # the starting or ending coordinates of route will be returned
    # in the nearer road which can be slightly different than
    # the ones provided
    if path_coordinates[0] != [lat1, lon1]:
        path_coordinates[0] = [lat1, lon1]
    if path_coordinates[-1] != [lat2, lon2]:
        path_coordinates[-1] = [lat2, lon2]

    return np.array(path_coordinates), distance


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
        trip, d = gen_basic_traj(l_s, l_e, vehicle, t_s)
        total_d = total_d + d
        t_s = trip[-1,0]
        traj = np.vstack((traj,trip))
        if (i+1)!=len(route)-1 and vehicle=='bus':
            trip = gen_basic_pause(l_e, t_s, None, [5,120])
            t_s = trip[-1,0]
            traj = np.vstack((traj,trip))
    traj = traj[1:,:]
    return traj, total_d

def gtraj_with_regular_visits(day):
    total_d = 0
    dur = np.random.uniform(15,40,1)[0]
    t_s = (day-1) * 24 * 60 * 60
    traj = np.zeros((1,3))
    traj[0,0] = t_s
    traj[0,1] = home_g[0]
    traj[0,2] = home_g[1]
    home_morning = gen_basic_pause(home_g, t_s, None, [7.5*3600, 8.5*3600])
    t0 = home_morning[-1,0]-home_morning[0,0]
    traj = np.vstack((traj,home_morning))
    t_s = home_morning[-1,0]
    home2gym,d = gen_route_traj([home_g, b_start, gym], 'walk', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,home2gym))
    t_s = home2gym[-1,0]
    workout = gen_basic_pause(gym, t_s, None, [dur*60, dur*60])
    traj = np.vstack((traj,workout))
    t_s = workout[-1,0]
    gym2bus,d = gen_basic_traj(gym, b_start, 'walk', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,gym2bus))
    t_s = gym2bus[-1,0]
    waitbus = gen_basic_pause(b_start, t_s, None, [3*60, 6*60])
    traj = np.vstack((traj,waitbus))
    t_s = waitbus[-1,0]
    bus_time = t_s
    onbus,d = gen_route_traj([b_start,hmart,mit,turn1,turn2,turn3,b_end], 'bus', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,onbus))
    t_s = onbus[-1,0]
    bus2hsph,d = gen_basic_traj(b_end, hsph, 'walk', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,bus2hsph))
    t_s = bus2hsph[-1,0]
    athsph = gen_basic_pause(hsph, t_s, None, [5*3600, 6*3600])
    traj = np.vstack((traj,athsph))
    t_s = athsph[-1,0]
    hsph2bus,d = gen_basic_traj(hsph, b_end, 'walk', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,hsph2bus))
    t_s = hsph2bus[-1,0]
    waitbus = gen_basic_pause(b_end, t_s, None, [3*60, 6*60])
    traj = np.vstack((traj,waitbus))
    t_s = waitbus[-1,0]
    onbus,d = gen_route_traj([b_end,turn3,turn2,turn1,mit,hmart,b_start], 'bus', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,onbus))
    t_s = onbus[-1,0]
    bus2home,d = gen_basic_traj(b_start, home_g, 'walk', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,bus2home))
    t_s = bus2home[-1,0]
    home_night = gen_basic_pause(home_g, t_s, [day*24*60*60-1, day*24*60*60-1], None)
    t1 = home_night[-1,0]-home_night[0,0]
    total_d = total_d + d
    traj = np.vstack((traj,home_night))
    return traj, total_d/1000, (t0+t1)/3600

def gtraj_with_one_visit(day):
    total_d = 0
    dur = np.random.uniform(15,40,1)[0]
    t_s = (day-1) * 24 * 60 * 60
    traj = np.zeros((1,3))
    traj[0,0] = t_s
    traj[0,1] = home_g[0]
    traj[0,2] = home_g[1]
    home_morning = gen_basic_pause(home_g, t_s, None, [9*3600, 9.5*3600])
    t0 = home_morning[-1,0]-home_morning[0,0]
    traj = np.vstack((traj,home_morning))
    t_s = home_morning[-1,0]
    home2bus,d = gen_basic_traj(home_g, b_start, 'walk', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,home2bus))
    t_s = home2bus[-1,0]
    waitbus = gen_basic_pause(b_start, t_s, None, [3*60, 6*60])
    traj = np.vstack((traj,waitbus))
    t_s = waitbus[-1,0]
    onbus,d = gen_route_traj([b_start,hmart,mit], 'bus', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,onbus))
    t_s = onbus[-1,0]
    atmit = gen_basic_pause(mit, t_s, None, [2.5*3600, 3*3600])
    traj = np.vstack((traj,atmit))
    t_s = atmit[-1,0]
    mit2hmart,d = gen_basic_traj(mit, hmart, 'walk', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,mit2hmart))
    t_s = mit2hmart[-1,0]
    hmart_time = t_s
    athmart = gen_basic_pause(hmart, t_s, None, [15*60, 25*60])
    traj = np.vstack((traj,athmart))
    t_s = athmart[-1,0]
    hmart2home,d = gen_route_traj([hmart,b_start,home_g], 'walk', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,hmart2home))
    t_s = hmart2home[-1,0]
    home_afternoon = gen_basic_pause(home_g, t_s, None, [4*3600, 5*3600])
    t1 = home_afternoon[-1,0]-home_afternoon[0,0]
    traj = np.vstack((traj,home_afternoon))
    t_s = home_afternoon[-1,0]
    home2movie,d = gen_route_traj([home_g,b_start,hmart,mit,turn1,movie], 'bus', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,home2movie))
    t_s = home2movie[-1,0]
    atmovie = gen_basic_pause(movie, t_s, None, [dur*60, dur*60])
    traj = np.vstack((traj,atmovie))
    t_s = atmovie[-1,0]
    movie2home,d = gen_route_traj([movie,turn1,mit,hmart,b_start,home_g], 'bus', t_s)
    total_d = total_d + d
    traj = np.vstack((traj,movie2home))
    t_s = movie2home[-1,0]
    home_night = gen_basic_pause(home_g, t_s, [day*24*60*60-1, day*24*60*60-1], None)
    t2 = home_night[-1,0]-home_night[0,0]
    traj = np.vstack((traj,home_night))
    return traj, total_d/1000, (t0+t1+t2)/3600

def gtraj_random(day):
    total_d = 0
    t_s = (day-1) * 24 * 60 * 60
    traj = np.zeros((1,3))
    traj[0,0] = t_s
    traj[0,1] = home_g[0]
    traj[0,2] = home_g[1]
    home_morning = gen_basic_pause(home_g, t_s, None, [9*3600, 9.5*3600])
    t0 = home_morning[-1,0]-home_morning[0,0]
    traj = np.vstack((traj,home_morning))
    t_s = home_morning[-1,0]
    randnum = np.random.randint(0,3,1)
    if randnum == 0:
        home2bus,d = gen_basic_traj(home_g, b_start, 'walk', t_s)
        total_d = total_d + d
        traj = np.vstack((traj,home2bus))
        t_s = home2bus[-1,0]
        waitbus = gen_basic_pause(b_start, t_s, None, [3*60, 6*60])
        traj = np.vstack((traj,waitbus))
        t_s = waitbus[-1,0]
        onbus, d = gen_route_traj([b_start,hmart,mit], 'bus', t_s)
        total_d = total_d + d
        traj = np.vstack((traj,onbus))
        t_s = onbus[-1,0]
        atmit = gen_basic_pause(mit, t_s, None, [2.5*3600, 3*3600])
        traj = np.vstack((traj,atmit))
        t_s = atmit[-1,0]
        mit2hmart,d = gen_basic_traj(mit, hmart, 'walk', t_s)
        total_d = total_d + d
        traj = np.vstack((traj,mit2hmart))
        t_s = mit2hmart[-1,0]
        hmart_time = t_s
        athmart = gen_basic_pause(hmart, t_s, None, [15*60, 25*60])
        traj = np.vstack((traj,athmart))
        t_s = athmart[-1,0]
        hmart2home,d = gen_route_traj([hmart,b_start,home_g], 'walk', t_s)
        total_d = total_d + d
        traj = np.vstack((traj,hmart2home))
        t_s = hmart2home[-1,0]
    if randnum == 1:
        home2hmart,d = gen_route_traj([home_g,b_start,hmart], 'walk', t_s)
        total_d = total_d + d
        traj = np.vstack((traj,home2hmart))
        t_s = home2hmart[-1,0]
        athmart = gen_basic_pause(hmart, t_s, None, [15*60, 25*60])
        traj = np.vstack((traj,athmart))
        t_s = athmart[-1,0]
        hmart2res,d = gen_basic_traj(hmart, restaurant, 'walk', t_s)
        total_d = total_d + d
        traj = np.vstack((traj,hmart2res))
        t_s = hmart2res[-1,0]
        atres = gen_basic_pause(restaurant, t_s, None, [45*60, 60*60])
        traj = np.vstack((traj,atres))
        t_s = atres[-1,0]
        res2home,d = gen_route_traj([restaurant,b_start,home_g], 'walk', t_s)
        total_d = total_d + d
        traj = np.vstack((traj,res2home))
        t_s = res2home[-1,0]
    if randnum == 2:
        home2home,d = gen_route_traj([home_g,b_start,gym,turn4,home_g], 'walk', t_s)
        total_d = total_d + d
        traj = np.vstack((traj,home2home))
        t_s = home2home[-1,0]
    home_night = gen_basic_pause(home_g, t_s, [day*24*60*60-1, day*24*60*60-1], None)
    t1 = home_night[-1,0]-home_night[0,0]
    traj = np.vstack((traj,home_night))
    return traj, total_d/1000, (t0+t1)/3600

def gen_all_traj():
    all_D = []
    all_T = []
    gtraj,D,T = gtraj_with_regular_visits(1)
    all_D.append(D)
    all_T.append(T)
    all_gtraj = gtraj
    for i in np.arange(2,15):
        if sum(np.array([3,5,8,10,12]==i))==1:
            gtraj,D,T = gtraj_with_regular_visits(i)
        elif i == 2:
            gtraj,D,T = gtraj_with_one_visit(i)
        else:
            gtraj,D,T = gtraj_random(i)
        all_gtraj = np.vstack((all_gtraj,gtraj))
        all_D.append(D)
        all_T.append(T)
    return all_gtraj,all_D,all_T

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

def prepare_data(obs):
    s = datetime2stamp([2020,8,24,0,0,0],'America/New_York')
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

def sim_GPS_data(cycle,p,data_folder):
    ## only two parameters
    ## cycle is the sum of on-cycle and off_cycle, unit is minute
    ## p is the missing rate, in other words, the proportion of off_cycle, should be within [0,1]
    ## it returns a pandas dataframe with only observations from on_cycles, which mimics the real data file
    ## the data are the trajectories over two weeks, sampled at 1Hz, with an obvious activity pattern
    s = datetime2stamp([2020,8,24,0,0,0],'America/New_York')*1000
    if os.path.exists(data_folder)==False:
        os.mkdir(data_folder)
    for user in range(2):
        if os.path.exists(data_folder+"/user_"+str(user+1))==False:
            os.mkdir(data_folder+"/user_"+str(user+1))
        if os.path.exists(data_folder+"/user_"+str(user+1)+"/gps")==False:
            os.mkdir(data_folder+"/user_"+str(user+1)+"/gps")
        all_traj,all_D,all_T = gen_all_traj()
        print("User_"+str(user+1))
        print("distance(km): ", all_D)
        print("hometime(hr): ", all_T)
        obs = remove_data(all_traj,cycle,p,14)
        obs_pd = prepare_data(obs)
        for i in range(14):
            for j in range(24):
                s_lower = s+i*24*60*60*1000+j*60*60*1000
                s_upper = s+i*24*60*60*1000+(j+1)*60*60*1000
                temp = obs_pd[(obs_pd["timestamp"]>=s_lower)&(obs_pd["timestamp"]<s_upper)]
                [y,m,d,h,mins,sec] = stamp2datetime(s_lower/1000,"UTC")
                filename = str(y)+"-"+int2str(m)+"-"+int2str(d)+" "+int2str(h)+"_00_00.csv"
                temp.to_csv(data_folder+"/user_"+str(user+1)+"/gps/"+filename,index = False)
