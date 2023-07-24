import sys
import math
import numpy as np
import scipy.stats as stat
from ..poplar.legacy.common_funcs import stamp2datetime
from .data2mobmat import great_circle_dist, great_circle_dist_vec, exist_knot

## the details of the functions are in paper [Liu and Onnela (2020)]
def num_sig_places(data,dist):
    """
    Args: data, 2d array like mobmat (k*7)
          dist, a scalar, hyperparameters, radius of a significant place
    Return: loc_x: a list of latitudes of significant places
            loc_y: a list of longitudes of significant places
            num_xy: a list of frequency/counts (appear in the dataset) of a significant places
            t_xy: a list of duration at those significant places
    """
    loc_x = []; loc_y = []; num_xy=[]; t_xy = []
    for i in range(data.shape[0]):
        if len(loc_x)==0:
            loc_x.append(data[i,1])
            loc_y.append(data[i,2])
            num_xy.append(1)
            t_xy.append(data[i,6]-data[i,3])
        else:
            d = []
            for j in range(len(loc_x)):
                d.append(great_circle_dist(data[i,1],data[i,2],loc_x[j],loc_y[j]))
            index = d.index(min(d))
            if min(d)>dist:
                loc_x.append(data[i,1])
                loc_y.append(data[i,2])
                num_xy.append(1)
                t_xy.append(data[i,6]-data[i,3])
            else:
                loc_x[index] = (loc_x[index]*num_xy[index]+data[i,1])/(num_xy[index]+1)
                loc_y[index] = (loc_y[index]*num_xy[index]+data[i,2])/(num_xy[index]+1)
                num_xy[index] = num_xy[index] + 1
                t_xy[index] = t_xy[index]+data[i,6]-data[i,3]
    return loc_x,loc_y,num_xy,t_xy

def locate_home(MobMat,tz_str):
    """
    Args: MobMat, a k*7 2d array, output from InferMobMat()
          tz_str, timezone, string
    Return: home_x, home_y, two scalar, represent the latitude and longtitude of user's home
    Raises:
        RuntimeError: if not enough data to infer home location

    """
    ObsTraj = MobMat[MobMat[:,0]==2,:]
    hours = []
    for i in range(ObsTraj.shape[0]):
        time_list = stamp2datetime((ObsTraj[i,3]+ObsTraj[i,6])/2,tz_str)
        hours.append(time_list[3])
    hours = np.array(hours)
    if ((hours >= 19) + (hours <= 9)).sum() <= 0: 
        raise RuntimeError(
            "No home location found: Too few observations at night"
        )
    home_pauses = ObsTraj[((hours>=19)+(hours<=9))*ObsTraj[:,0]==2,:]
    loc_x,loc_y,num_xy,t_xy = num_sig_places(home_pauses,20)
    home_index = num_xy.index(max(num_xy))
    home_x, home_y = loc_x[home_index],loc_y[home_index]
    return home_x,home_y

def K1(method,current_t,current_x,current_y,BV_set,pars):
    """
    Args: method, string, should be 'TL', or 'GL' or 'GLC'
          current_t, current_x, current_y are scalars
          BV_set, 2d array, the (subset of) output from BV_select()
          pars, a list of parameters
    Return: 1d array of similarity measures between this triplet and each in BV_set
    """
    [l1,l2,a1,a2,b1,b2,b3,g] = pars
    mean_x = ((BV_set[:,1] + BV_set[:,4])/2).astype(float)
    mean_y = ((BV_set[:,2] + BV_set[:,5])/2).astype(float)
    mean_t = ((BV_set[:,3] + BV_set[:,6])/2).astype(float)
    if method=="TL":
        k1 = np.exp(-abs(current_t-mean_t)/l1)*np.exp(-(np.sin(abs(current_t-mean_t)/86400*math.pi))**2/a1)
        k2 = np.exp(-abs(current_t-mean_t)/l2)*np.exp(-(np.sin(abs(current_t-mean_t)/604800*math.pi))**2/a2)
        return b1/(b1+b2)*k1+b2/(b1+b2)*k2
    if method=="GL":
        d = great_circle_dist_vec(current_x,current_y,mean_x,mean_y)
        return np.exp(-d/g)
    if method=="GLC":
        k1 = np.exp(-abs(current_t-mean_t)/l1)*np.exp(-(np.sin(abs(current_t-mean_t)/86400*math.pi))**2/a1)
        k2 = np.exp(-abs(current_t-mean_t)/l2)*np.exp(-(np.sin(abs(current_t-mean_t)/604800*math.pi))**2/a2)
        d = great_circle_dist_vec(current_x,current_y,mean_x,mean_y)
        k3 = np.exp(-d/g)
        return b1*k1+b2*k2+b3*k3

def I_flight(method,current_t,current_x,current_y,dest_t,dest_x,dest_y,BV_set,switch,num,pars):
    """
    Args: method, string, should be 'TL', or 'GL' or 'GLC'
          current_t, current_x, current_y, dest_t,dest_x,dest_y are scalars
          BV_set, 2d array, the (subset of) output from BV_select()
          switch: the number of binary variables we want to generate, this controls the difficulty to change
             the status from flight to pause or from pause to flight
          num: check top k similarities (avoid the cumulative effect of many low prob trajs)
    Return: 1d array of 0 and 1, of length switch, indicator of a incoming flight
    """
    K = K1(method,current_t,current_x,current_y,BV_set,pars)
    flight_K = K[BV_set[:,0]==1]
    pause_K = K[BV_set[:,0]==2]
    sorted_flight = np.sort(flight_K)[::-1]
    sorted_pause = np.sort(pause_K)[::-1]
    p0 = np.mean(sorted_flight[0:num])/(np.mean(sorted_flight[0:num])+np.mean(sorted_pause[0:num])+1e-8)
    d_dest = great_circle_dist(current_x,current_y,dest_x,dest_y)
    v_dest = d_dest/(dest_t-current_t+0.0001)
    ## design an exponential function here to adjust the probability based on the speed needed
    ## p = p0*exp(|v-2|+/s)  v=2--p=p0   v=14--p=1
    if p0 < 1e-5:
        p0 = 1e-5
    if p0 > 1-1e-5:
        p0 = 1-1e-5
    s = -12/np.log(p0)
    p1 = min(1,p0*np.exp(min(max(0,v_dest-2)/s,1e2)))
    out = stat.bernoulli.rvs(p1,size=switch)
    return out

def adjust_direction(linearity,delta_x,delta_y,start_x,start_y,end_x,end_y,origin_x,origin_y,dest_x,dest_y):
    """
    Args: linearity, a scalar that controls the smoothness of a trajectory
          a large linearity tends to have a more linear traj from starting point toward destination
          a small one tends to have more random directions

          delta_x,delta_y,start_x,start_y,end_x,end_y,origin_x,origin_y,dest_x,dest_y are scalars
    Return: 2 scalars, represent the adjusted dispacement in two axises
    """
    norm1 = np.sqrt((dest_x-origin_x)**2+(dest_y-origin_y)**2)
    k = np.random.uniform(low=0, high=linearity) ## this is another parameter which controls the smoothness
    new_x = delta_x + k*(dest_x-origin_x)/norm1
    new_y = delta_y + k*(dest_y-origin_y)/norm1
    norm2 = np.sqrt(delta_x**2 + delta_y**2)
    norm3 = np.sqrt(new_x**2 + new_y**2)
    norm_x = new_x*norm2/norm3
    norm_y = new_y*norm2/norm3
    inner = np.inner(np.array([end_x-start_x,end_y-start_y]),np.array([norm_x,norm_y]))
    if inner < 0:
        return -norm_x, -norm_y
    else:
        return norm_x, norm_y

def multiplier(t_diff):
    """
    Args: a scalar, difference in time (unit in second)
    Return: a scalar, a multiplication coefficient
    """
    if t_diff<=30*60:
        return 1
    elif t_diff<=180*60:
        return 5
    elif t_diff<=1080*60:
        return 10
    else:
        return 50

def checkbound(current_x,current_y,start_x,start_y,end_x,end_y):
    """
    Args: all scalars
    Return: 1/0, indicates whether (current_x, current_y) is out of the boundary determiend by
            starting and ending points
    """
    max_x = max(start_x,end_x)
    min_x = min(start_x,end_x)
    max_y = max(start_y,end_y)
    min_y = min(start_y,end_y)
    if current_x<max_x+0.01 and current_x>min_x-0.01 and current_y<max_y+0.01 and current_y>min_y-0.01:
        return 1
    else:
        return 0

def create_tables(MobMat, BV_set):
    """
    Args: MobMat, 2d array, output from InferMobMat()
          BV_set, 2d array, output from BV_select()
    Return: 3 2d arrays, one for observed flights, one for observed pauses, one for missing interval
            (where the last two cols are the status of previous obs traj and next obs traj)
    """
    n = np.shape(MobMat)[0]
    m = np.shape(BV_set)[0]
    index = [BV_set[i,0]==1 for i in range(m)]
    flight_table = BV_set[index,:]
    index = [BV_set[i,0]==2 for i in range(m)]
    pause_table = BV_set[index,:]
    mis_table = np.zeros((1, 8))
    for i in range(n-1):
        if MobMat[i+1,3]!=MobMat[i,6]:
            ## also record if it's flight/pause before and after the missing interval
            mov = np.array([MobMat[i,4],MobMat[i,5],MobMat[i,6],MobMat[i+1,1],MobMat[i+1,2],MobMat[i+1,3],MobMat[i,0],MobMat[i+1,0]])
            mis_table = np.vstack((mis_table,mov))
    mis_table = np.delete(mis_table,0,0)
    return flight_table, pause_table, mis_table

def ImputeGPS(MobMat,BV_set,method,switch,num,linearity,tz_str,pars):
    """
    This is the algorithm for the bi-directional imputation in the paper
    Args: MobMat, 2d array, output from InferMobMat()
          BV_set, 2d array, output from BV_select()
          method, string, should be 'TL', or 'GL' or 'GLC'
          switch, the number of binary variables we want to generate, this controls the difficulty to change
             the status from flight to pause or from pause to flight
          linearity, a scalar that controls the smoothness of a trajectory
               a large linearity tends to have a more linear traj from starting point toward destination
               a small one tends to have more random directions
          tz_str, timezone
    Return: 2d array simialr to MobMat, but it is a complete imputed traj (first-step result)
            with headers [imp_s,imp_x0,imp_y0,imp_t0,imp_x1,imp_y1,imp_t1]
    """
    home_x,home_y = locate_home(MobMat,tz_str)
    sys.stdout.write("Imputing missing trajectories ..." + '\n')
    flight_table, pause_table, mis_table = create_tables(MobMat, BV_set)
    imp_x0 = np.array([]); imp_x1 = np.array([])
    imp_y0 = np.array([]); imp_y1 = np.array([])
    imp_t0 = np.array([]); imp_t1 = np.array([])
    imp_s = np.array([])
    for i in range(mis_table.shape[0]):
        mis_t0 = mis_table[i,2]; mis_t1 = mis_table[i,5]
        nearby_flight = sum((flight_table[:,6]>mis_t0-12*60*60)*(flight_table[:,3]<mis_t1+12*60*60))
        d_diff = great_circle_dist(mis_table[i,0],mis_table[i,1],mis_table[i,3],mis_table[i,4])
        t_diff = mis_table[i,5] - mis_table[i,2]
        D1 = great_circle_dist(mis_table[i,0],mis_table[i,1],home_x,home_y)
        D2 = great_circle_dist(mis_table[i,3],mis_table[i,4],home_x,home_y)
        ## if a person remains at the same place at the begining and end of missing, just assume he satys there all the time
        if mis_table[i,0]==mis_table[i,3] and mis_table[i,1]==mis_table[i,4]:
            imp_s = np.append(imp_s,2)
            imp_x0 = np.append(imp_x0, mis_table[i,0])
            imp_x1 = np.append(imp_x1, mis_table[i,3])
            imp_y0 = np.append(imp_y0, mis_table[i,1])
            imp_y1 = np.append(imp_y1, mis_table[i,4])
            imp_t0 = np.append(imp_t0, mis_table[i,2])
            imp_t1 = np.append(imp_t1, mis_table[i,5])
        elif d_diff>300000:
            v_diff = d_diff/t_diff
            if v_diff>210:
                imp_s = np.append(imp_s,1)
                imp_x0 = np.append(imp_x0, mis_table[i,0])
                imp_x1 = np.append(imp_x1, mis_table[i,3])
                imp_y0 = np.append(imp_y0, mis_table[i,1])
                imp_y1 = np.append(imp_y1, mis_table[i,4])
                imp_t0 = np.append(imp_t0, mis_table[i,2])
                imp_t1 = np.append(imp_t1, mis_table[i,5])
            else:
                v_random = np.random.uniform(low=244, high=258)
                t_need = d_diff/v_random
                t_s = np.random.uniform(low = mis_table[i,2], high = mis_table[i,5]-t_need)
                t_e = t_s + t_need
                imp_s = np.append(imp_s,[2,1,2])
                imp_x0 = np.append(imp_x0, [mis_table[i,0],mis_table[i,0],mis_table[i,3]])
                imp_x1 = np.append(imp_x1, [mis_table[i,0],mis_table[i,3],mis_table[i,3]])
                imp_y0 = np.append(imp_y0, [mis_table[i,1],mis_table[i,1],mis_table[i,4]])
                imp_y1 = np.append(imp_y1, [mis_table[i,1],mis_table[i,4],mis_table[i,4]])
                imp_t0 = np.append(imp_t0, [mis_table[i,2],t_s,t_e])
                imp_t1 = np.append(imp_t1, [t_s,t_e,mis_table[i,5]])
        ## add one more check about how many flights observed in the nearby 24 hours
        elif nearby_flight<=5 and t_diff>6*60*60 and min(D1,D2)>50:
            if d_diff<3000:
                v_random = np.random.uniform(low=1, high=1.8)
                t_need = min(d_diff/v_random,t_diff)
            else:
                v_random = np.random.uniform(low=13, high=32)
                t_need = min(d_diff/v_random,t_diff)
            if t_need == t_diff:
                imp_s = np.append(imp_s,1)
                imp_x0 = np.append(imp_x0, mis_table[i,0])
                imp_x1 = np.append(imp_x1, mis_table[i,3])
                imp_y0 = np.append(imp_y0, mis_table[i,1])
                imp_y1 = np.append(imp_y1, mis_table[i,4])
                imp_t0 = np.append(imp_t0, mis_table[i,2])
                imp_t1 = np.append(imp_t1, mis_table[i,5])
            else:
                t_s = np.random.uniform(low = mis_table[i,2], high = mis_table[i,5]-t_need)
                t_e = t_s + t_need
                imp_s = np.append(imp_s,[2,1,2])
                imp_x0 = np.append(imp_x0, [mis_table[i,0],mis_table[i,0],mis_table[i,3]])
                imp_x1 = np.append(imp_x1, [mis_table[i,0],mis_table[i,3],mis_table[i,3]])
                imp_y0 = np.append(imp_y0, [mis_table[i,1],mis_table[i,1],mis_table[i,4]])
                imp_y1 = np.append(imp_y1, [mis_table[i,1],mis_table[i,4],mis_table[i,4]])
                imp_t0 = np.append(imp_t0, [mis_table[i,2],t_s,t_e])
                imp_t1 = np.append(imp_t1, [t_s,t_e,mis_table[i,5]])
        else:
            ## solve the problem that a person has a trajectory like flight/pause/flight/pause/flight...
            ## we want it more like flght/flight/flight/pause/pause/pause/flight/flight...
            ## start from two ends, we make it harder to change the current pause/flight status by drawing multiple random
            ## variables form bin(p0) and require them to be all 0/1
            ## "switch" is the number of random variables
            start_t = mis_table[i,2]; end_t = mis_table[i,5]
            start_x = mis_table[i,0]; end_x = mis_table[i,3]
            start_y = mis_table[i,1]; end_y = mis_table[i,4]
            start_s = mis_table[i,6]; end_s = mis_table[i,7]
            if t_diff>4*60*60 and min(D1,D2)<=50:
                t_need = min(d_diff/0.6,t_diff)
                if D1<=50:
                    imp_s = np.append(imp_s,2)
                    imp_t0 = np.append(imp_t0,start_t)
                    imp_t1 = np.append(imp_t1,end_t-t_need)
                    imp_x0 = np.append(imp_x0,start_x)
                    imp_x1 = np.append(imp_x1,start_x)
                    imp_y0 = np.append(imp_y0,start_y)
                    imp_y1 = np.append(imp_y1,start_y)
                    start_t = end_t-t_need
                else:
                    imp_s = np.append(imp_s,2)
                    imp_t0 = np.append(imp_t0,start_t+t_need)
                    imp_t1 = np.append(imp_t1,end_t)
                    imp_x0 = np.append(imp_x0,end_x)
                    imp_x1 = np.append(imp_x1,end_x)
                    imp_y0 = np.append(imp_y0,end_y)
                    imp_y1 = np.append(imp_y1,end_y)
                    end_t = start_t + t_need
            counter = 0
            while start_t < end_t:
                if abs(start_x-end_x)+abs(start_y-end_y)>0 and end_t-start_t<30: ## avoid extreme high speed
                    imp_s = np.append(imp_s,1)
                    imp_t0 = np.append(imp_t0,start_t)
                    imp_t1 = np.append(imp_t1,end_t)
                    imp_x0 = np.append(imp_x0,start_x)
                    imp_x1 = np.append(imp_x1,end_x)
                    imp_y0 = np.append(imp_y0,start_y)
                    imp_y1 = np.append(imp_y1,end_y)
                    start_t = end_t
                    ## should check the missing legnth first, if it's less than 12 hours, do the following, otherewise,
                    ## insert home location at night most visited places in the interval as known
                elif start_x==end_x and start_y==end_y:
                    imp_s = np.append(imp_s,2)
                    imp_t0 = np.append(imp_t0,start_t)
                    imp_t1 = np.append(imp_t1,end_t)
                    imp_x0 = np.append(imp_x0,start_x)
                    imp_x1 = np.append(imp_x1,end_x)
                    imp_y0 = np.append(imp_y0,start_y)
                    imp_y1 = np.append(imp_y1,end_y)
                    start_t = end_t
                else:
                    if counter % 2 == 0:
                        direction = 'forward'
                    else:
                        direction = 'backward'

                    if direction == 'forward':
                        direction =''
                        I0 = I_flight(method,start_t,start_x,start_y,end_t,end_x,end_y,BV_set,switch,num,pars)
                        if (sum(I0==1)==switch and start_s==2) or (sum(I0==0)<switch and start_s==1):
                            weight = K1(method,start_t,start_x,start_y,flight_table,pars)
                            normalize_w = (weight+1e-5)/sum(weight+1e-5)
                            flight_index = np.random.choice(flight_table.shape[0], p=normalize_w)
                            delta_x = flight_table[flight_index,4]-flight_table[flight_index,1]
                            delta_y = flight_table[flight_index,5]-flight_table[flight_index,2]
                            delta_t = flight_table[flight_index,6]-flight_table[flight_index,3]
                            if(start_t + delta_t > end_t):
                                temp = delta_t
                                delta_t = end_t-start_t
                                delta_x = delta_x*delta_t/temp
                                delta_y = delta_y*delta_t/temp
                            delta_x,delta_y = adjust_direction(linearity,delta_x,delta_y,start_x,start_y,end_x,end_y,mis_table[i,0],mis_table[i,1],mis_table[i,3],mis_table[i,4])
                            try_t = start_t + delta_t
                            try_x = (end_t-try_t)/(end_t-start_t+1e-5)*(start_x+delta_x)+(try_t-start_t+1e-5)/(end_t-start_t)*end_x
                            try_y = (end_t-try_t)/(end_t-start_t+1e-5)*(start_y+delta_y)+(try_t-start_t+1e-5)/(end_t-start_t)*end_y
                            mov1 = great_circle_dist(try_x,try_y,start_x,start_y)
                            mov2 =  great_circle_dist(end_x,end_y,start_x,start_y)
                            check1 = checkbound(try_x,try_y,mis_table[i,0],mis_table[i,1],mis_table[i,3],mis_table[i,4])
                            check2 = (mov1<mov2)*1
                            if end_t>start_t and check1==1 and check2==1:
                                imp_s = np.append(imp_s,1)
                                imp_t0 = np.append(imp_t0,start_t)
                                current_t = start_t + delta_t
                                imp_t1 = np.append(imp_t1,current_t)
                                imp_x0 = np.append(imp_x0,start_x)
                                current_x = (end_t-current_t)/(end_t-start_t)*(start_x+delta_x)+(current_t-start_t)/(end_t-start_t)*end_x
                                imp_x1 = np.append(imp_x1,current_x)
                                imp_y0 = np.append(imp_y0,start_y)
                                current_y = (end_t-current_t)/(end_t-start_t)*(start_y+delta_y)+(current_t-start_t)/(end_t-start_t)*end_y
                                imp_y1 = np.append(imp_y1,current_y)
                                start_x = current_x; start_y = current_y; start_t = current_t; start_s=1
                                counter = counter+1
                            if end_t>start_t and check2==0:
                                sp_log = np.log(mov1) - np.log(delta_t) ## this number can be very close to zero, so numpy can error out 
                                ## if this is run as just normal division
                                t_need = np.exp(np.log(mov2) - sp_log)
                                imp_s = np.append(imp_s,1)
                                imp_t0 = np.append(imp_t0,start_t)
                                current_t = start_t + t_need
                                imp_t1 = np.append(imp_t1,current_t)
                                imp_x0 = np.append(imp_x0,start_x)
                                imp_x1 = np.append(imp_x1,end_x)
                                imp_y0 = np.append(imp_y0,start_y)
                                imp_y1 = np.append(imp_y1,end_y)
                                start_x = end_x; start_y = end_y; start_t = current_t; start_s=1
                                counter = counter+1
                            else:
                                weight = K1(method,start_t,start_x,start_y,pause_table,pars)
                                normalize_w = (weight+1e-5)/sum(weight+1e-5)
                                pause_index = np.random.choice(pause_table.shape[0], p=normalize_w)
                                delta_t = (pause_table[pause_index,6]-pause_table[pause_index,3])*multiplier(end_t-start_t)
                                if start_t + delta_t < end_t:
                                    imp_s = np.append(imp_s,2)
                                    imp_t0 = np.append(imp_t0,start_t)
                                    current_t = start_t + delta_t
                                    imp_t1 = np.append(imp_t1,current_t)
                                    imp_x0 = np.append(imp_x0,start_x)
                                    imp_x1 = np.append(imp_x1,start_x)
                                    imp_y0 = np.append(imp_y0,start_y)
                                    imp_y1 = np.append(imp_y1,start_y)
                                    start_t = current_t
                                    start_s = 2
                                    counter = counter+1
                                else:
                                    imp_s = np.append(imp_s,1)
                                    imp_t0 = np.append(imp_t0,start_t)
                                    imp_t1 = np.append(imp_t1,end_t)
                                    imp_x0 = np.append(imp_x0,start_x)
                                    imp_x1 = np.append(imp_x1,end_x)
                                    imp_y0 = np.append(imp_y0,start_y)
                                    imp_y1 = np.append(imp_y1,end_y)
                                    start_t = end_t

                    if direction == 'backward':
                        direction = ''
                        I1 = I_flight(method,end_t,end_x,end_y,start_t,start_x,start_y,BV_set,switch,num,pars)
                        if (sum(I1==1)==switch and end_s==2) or (sum(I1==0)<switch and end_s==1):
                            weight = K1(method,end_t,end_x,end_y,flight_table,pars)
                            normalize_w = (weight+1e-5)/sum(weight+1e-5)
                            flight_index = np.random.choice(flight_table.shape[0], p=normalize_w)
                            delta_x = -(flight_table[flight_index,4]-flight_table[flight_index,1])
                            delta_y = -(flight_table[flight_index,5]-flight_table[flight_index,2])
                            delta_t = flight_table[flight_index,6]-flight_table[flight_index,3]
                            if(start_t + delta_t > end_t):
                                temp = delta_t
                                delta_t = end_t-start_t
                                delta_x = delta_x*delta_t/temp
                                delta_y = delta_y*delta_t/temp
                            delta_x,delta_y = adjust_direction(linearity,delta_x,delta_y,end_x,end_y,start_x,start_y,mis_table[i,3],mis_table[i,4],mis_table[i,0],mis_table[i,1])
                            try_t = end_t - delta_t
                            try_x = (end_t-try_t)/(end_t-start_t+1e-5)*start_x+(try_t-start_t)/(end_t-start_t+1e-5)*(end_x+delta_x)
                            try_y = (end_t-try_t)/(end_t-start_t+1e-5)*start_y+(try_t-start_t)/(end_t-start_t+1e-5)*(end_y+delta_y)
                            mov1 = great_circle_dist(try_x,try_y,end_x,end_y)
                            mov2 =  great_circle_dist(end_x,end_y,start_x,start_y)
                            check1 = checkbound(try_x,try_y,mis_table[i,0],mis_table[i,1],mis_table[i,3],mis_table[i,4])
                            check2 = (mov1<mov2)*1
                            if end_t>start_t and check1==1 and check2==1:
                                imp_s = np.append(imp_s,1)
                                imp_t1 = np.append(imp_t1,end_t)
                                current_t = end_t - delta_t
                                imp_t0 = np.append(imp_t0,current_t)
                                imp_x1 = np.append(imp_x1,end_x)
                                current_x = (end_t-current_t)/(end_t-start_t)*start_x+(current_t-start_t)/(end_t-start_t)*(end_x+delta_x)
                                imp_x0 = np.append(imp_x0,current_x)
                                imp_y1 = np.append(imp_y1,end_y)
                                current_y = (end_t-current_t)/(end_t-start_t)*start_y+(current_t-start_t)/(end_t-start_t)*(end_y+delta_y)
                                imp_y0 = np.append(imp_y0,current_y)
                                end_x = current_x; end_y = current_y; end_t = current_t; end_s = 1
                                counter = counter+1
                            if end_t>start_t and check2==0:
                                sp_log = np.log(mov1) - np.log(delta_t) ## this number can be very close to zero, so numpy can error out 
                                ## if this is run as just normal division
                                t_need = np.exp(np.log(mov2) - sp_log)
                                imp_s = np.append(imp_s,1)
                                imp_t1 = np.append(imp_t1,end_t)
                                current_t = end_t - t_need
                                imp_t0 = np.append(imp_t0,current_t)
                                imp_x1 = np.append(imp_x1,end_x)
                                imp_x0 = np.append(imp_x0,start_x)
                                imp_y1 = np.append(imp_y1,end_y)
                                imp_y0 = np.append(imp_y0,start_y)
                                end_x = start_x; end_y = start_y; end_t = current_t; end_s = 1
                                counter = counter+1
                            else:
                                weight = K1(method,end_t,end_x,end_y,pause_table,pars)
                                normalize_w = (weight+1e-5)/sum(weight+1e-5)
                                pause_index = np.random.choice(pause_table.shape[0], p=normalize_w)
                                delta_t = (pause_table[pause_index,6]-pause_table[pause_index,3])*multiplier(end_t-start_t)
                                if start_t + delta_t < end_t:
                                    imp_s = np.append(imp_s,2)
                                    imp_t1 = np.append(imp_t1,end_t)
                                    current_t = end_t - delta_t
                                    imp_t0 = np.append(imp_t0,current_t)
                                    imp_x0 = np.append(imp_x0,end_x)
                                    imp_x1 = np.append(imp_x1,end_x)
                                    imp_y0 = np.append(imp_y0,end_y)
                                    imp_y1 = np.append(imp_y1,end_y)
                                    end_t = current_t
                                    end_s = 2
                                    counter = counter+1
                                else:
                                    imp_s = np.append(imp_s,1)
                                    imp_t1 = np.append(imp_t1,end_t)
                                    imp_t0 = np.append(imp_t0,start_t)
                                    imp_x0 = np.append(imp_x0,start_x)
                                    imp_x1 = np.append(imp_x1,end_x)
                                    imp_y0 = np.append(imp_y0,start_y)
                                    imp_y1 = np.append(imp_y1,end_y)
                                    end_t = start_t
    imp_table=np.stack([imp_s,imp_x0,imp_y0,imp_t0,imp_x1,imp_y1,imp_t1], axis=1)
    imp_table = imp_table[imp_table[:,3].argsort()].astype(float)
    return imp_table

def Imp2traj(imp_table,MobMat,itrvl,r,w,h):
    """
    This function tidies up the first-step imputed trajectory, such as combining pauses, flights shared by
    both observed and missing intervals, also combine consecutive flight with slightly different directions
    as one longer flight
    Args: imp_table, 2d array, output from ImputeGPS()
          MobMat, 2d array, output from InferMobMat()
          itrvl: the window size of moving average,  unit is second
              r: the maximum radius of a pause
              w: a threshold for distance, if the distance to the great circle is greater than
                 this threshold, we consider there is a knot
              h: a threshold of distance, if the movemoent between two timestamps is less than h,
                 consider it as a pause and a knot
    Return: 2d array, the final imputed trajectory, with one more columm compared to imp_table
            which is an indicator showing if the peice of traj is imputed (0) or observed (1)
    """
    sys.stdout.write("Tidying up the trajectories..." + '\n')
    mis_table = np.zeros((1, 8))
    for i in range(np.shape(MobMat)[0]-1):
        if MobMat[i+1,3]!=MobMat[i,6]:
            ## also record if it's flight/pause before and after the missing interval
            mov = np.array([MobMat[i,4],MobMat[i,5],MobMat[i,6],MobMat[i+1,1],MobMat[i+1,2],MobMat[i+1,3],MobMat[i,0],MobMat[i+1,0]])
            mis_table = np.vstack((mis_table,mov))
    mis_table = np.delete(mis_table,0,0)

    traj = []
    for k in range(mis_table.shape[0]):
        index = (imp_table[:,3]>=mis_table[k,2])*(imp_table[:,6]<=mis_table[k,5])
        temp = imp_table[index,:]
        a = 0
        b = 1
        while a < temp.shape[0]:
            if b < temp.shape[0]:
                if temp[b,0] == temp[a,0]:
                    b = b + 1
            if b==temp.shape[0] or temp[min(b,temp.shape[0]-1),0]!=temp[a,0]:
                start = a
                end = b-1
                a = b
                b = b+1
                if temp[start,0]==2:
                    traj.append([2,temp[start,1],temp[start,2],temp[start,3],temp[end,4],temp[end,5],temp[end,6]])
                elif end == start:
                    traj.append([1,temp[start,1],temp[start,2],temp[start,3],temp[end,4],temp[end,5],temp[end,6]])
                else:
                    mat = np.vstack((temp[start,1:4],temp[np.arange(start,end+1),4:7]))
                    mat = np.append(mat,np.arange(0,mat.shape[0]).reshape(mat.shape[0],1),1)
                    complete = 0
                    knots = [0,mat.shape[0]-1]
                    while complete == 0:
                        mat_list = []
                        for i in range(len(knots)-1):
                            mat_list.append(mat[knots[i]:min(knots[i+1]+1,mat.shape[0]-1),:])
                        knot_yes = np.empty(len(mat_list))
                        knot_pos = np.empty(len(mat_list))
                        for i in range(len(mat_list)):
                            knot_yes[i] , knot_pos[i] = exist_knot(mat_list[i],w)
                        if sum(knot_yes)==0:
                            complete = 1
                        else:
                            for i in range(len(mat_list)):
                                if knot_yes[i]==1:
                                    knots.append(int((mat_list[i])[int(knot_pos[i]),3]))
                            knots.sort()
                    out = []
                    for j in range(len(knots)-1):
                        traj.append([1,mat[knots[j],0],mat[knots[j],1],mat[knots[j],2],mat[knots[j+1],0],mat[knots[j+1],1],mat[knots[j+1],2]])
    traj = np.array(traj)
    if traj.shape[0]!=0:
        traj = np.hstack((traj,np.zeros((traj.shape[0],1))))
        full_traj = np.vstack((traj,MobMat))
    else:
        full_traj = MobMat
    float_traj = full_traj[full_traj[:,3].argsort()].astype(float)
    final_traj = float_traj[float_traj[:,6]-float_traj[:,3]>0,:]
    return final_traj
