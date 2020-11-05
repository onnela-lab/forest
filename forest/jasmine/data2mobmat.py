from logging import getLogger
import sys
import math
import numpy as np
from itertools import groupby


logger = getLogger(__name__)

## the radius of the earth
R = 6.371*10**6

def unique(anylist):
    """
    Args: anylist, an arbitrary list
    Return: a list of unique items in anylist
    """
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in anylist:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def cartesian(lat,lon):
    """
    Args: latitude and longitude of a location, range[-180, 180],
          each could be a scalar or a vector (1d np.array), but should be of same length(data type)
    Return: the corresponding cartesian coordiantes ((0,0,0) as geocenter)
    """
    lat = lat/180*math.pi
    lon = lon/180*math.pi
    z = R*np.sin(lat)
    u = R*np.cos(lat)
    x = u*np.cos(lon)
    y = u*np.sin(lon)
    return x,y,z

def great_circle_dist(lat1,lon1,lat2,lon2):
    """
    Args: latitude and longitude of location1 and location2, range[-180, 180]
          each could be a scalar or a vector (1d np.array), but should be of same length(data type)
    Return: the great circle distance between the two locations (unit is meter)
            data type matches the input
    """
    lat1 = lat1/180*math.pi
    lon1 = lon1/180*math.pi
    lat2 = lat2/180*math.pi
    lon2 = lon2/180*math.pi
    temp = np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)+np.sin(lat1)*np.sin(lat2)

    ## due to measurement errors, temp may be out of the domain of 'arccos'
    ## lat, lon sometimes could be np.array, so there are two version of this correction
    if isinstance(temp,np.ndarray):
        temp[temp>1]=1
        temp[temp<-1]=-1
    else:
        if temp>1:
            temp=1
        if temp<-1:
            temp=-1
    theta = np.arccos(temp)
    d = theta*R
    return d

def shortest_dist_to_great_circle(lat,lon,lat_start,lon_start,lat_end,lon_end):
    """
    Args: latitude and longitude of location1,location2 and location3, range[-180, 180]
          each could be a scalar or a vector (1d np.array), but should be of same length(data type)
    Return: the shortest distance from location 1 to the great circle determined by location 2 and 3
            (the path which is perpendicular to the great circle)
            unit is meter, data type matches the input
    """
    ## if loc2 and 3 are too close to determine a great circle, return 0
    if abs(lat_start-lat_end)<1e-6 and abs(lon_start-lon_end)<1e-6:
        return np.zeros(len(lat))
    else:
        x,y,z = cartesian(lat,lon)
        x_start,y_start,z_start = cartesian(lat_start,lon_start)
        x_end,y_end,z_end = cartesian(lat_end,lon_end)
        cross_product = np.cross(np.array([x_start,y_start,z_start]),np.array([x_end,y_end,z_end]))
        N = cross_product/(np.linalg.norm(cross_product)+1e-6)
        C = np.array([x,y,z])/R
        temp = np.dot(N,C)
        ## make temp fall into the domain of 'arccos'
        if isinstance(temp,np.ndarray):
            temp[temp>1]=1
            temp[temp<-1]=-1
        else:
            if temp>1:
                temp=1
            if temp<-1:
                temp=-1
        NOC = np.arccos(temp)
        d = abs(math.pi/2-NOC)*R
        return d

def pairwise_great_circle_dist(latlon_array):
    """
    Args: latlon_array should be a n by 2 np.array. The first column is latitude and the second is longitude.
          each element should be within [-180, 180]
    Return: a list of distances between any pair
    """
    dist = []
    k = np.shape(latlon_array)[0]
    for i in range(k-1):
        for j in np.arange(i+1,k):
            dist.append(great_circle_dist(latlon_array[i,0],latlon_array[i,1],latlon_array[j,0],latlon_array[j,1]))
    return dist

def collapse_data(data, itrvl, accuracylim):
    """
    Args: data: the pd dataframe from read_data()
          itrvl: the window size of moving average,  unit is second
          accuracylim: a threshold. We filter out GPS record with accuracy higher than this threshold.
    Return: a 2d numpy array, average over the measures every itrvl seconds
            with first col as an indicator
            if it is 1, it is observed, and
                 second col as timestamp
                 third col as latitude
                 fourth col as longitude
            if it is 4, it is an missing interval and
                 second col is starting timestamp
                 third col is ending timestamp
                 fourth is none
    """
    data = data[data.accuracy<accuracylim]
    t_start = np.array(data.timestamp)[0]/1000
    t_end = np.array(data.timestamp)[-1]/1000
    avgmat = np.empty([int(np.ceil((t_end-t_start)/itrvl))+2,4])
    sys.stdout.write("Collapse data within " + str(itrvl)+" second intervals ..."+'\n')
    IDam = 0
    count = 0
    nextline=[1,t_start+itrvl/2,data.iloc[0,2],data.iloc[0,3]]
    numitrvl=1
    for i in np.arange(1,data.shape[0]):
        if data.iloc[i,0]/1000 < t_start+itrvl:
            nextline[2]=nextline[2]+data.iloc[i,2]
            nextline[3]=nextline[3]+data.iloc[i,3]
            numitrvl=numitrvl+1
        else:
            nextline[2]=nextline[2]/numitrvl
            nextline[3]=nextline[3]/numitrvl
            avgmat[IDam,:]=nextline
            count=count+1
            IDam=IDam+1
            nummiss=int(np.floor((data.iloc[i,0]/1000-(t_start+itrvl))/itrvl))
            if nummiss>0:
                avgmat[IDam,:] = [4,t_start+itrvl,t_start+itrvl*(nummiss+1),None]
                count=count+1
                IDam=IDam+1
            t_start=t_start+itrvl*(nummiss+1)
            nextline[0]=1
            nextline[1]=t_start+itrvl/2
            nextline[2]=data.iloc[i,2]
            nextline[3]=data.iloc[i,3]
            numitrvl=1
    avgmat = avgmat[0:count,:]
    return avgmat

def ExistKnot(mat,w):
    """
    Args: mat: avgmat from collapse_data()
            w: a threshold for distance, if the distance to the great circle is greater than
               this threshold, we consider there is a knot
    Return: an binary (1/0) to show if there is a knot, and if yes, where the knot is (loc of the largest distance)
    """
    n = mat.shape[0]
    if n>1:
        lat_start = mat[0,2]
        lon_start = mat[0,3]
        lat_end = mat[n-1,2]
        lon_end = mat[n-1,3]
        lat = mat[:,2]
        lon = mat[:,3]
        d = shortest_dist_to_great_circle(lat,lon,lat_start,lon_start,lat_end,lon_end)
        if max(d)<w:
            return 0, None
        else:
            return 1, np.argmax(d)
    else:
        return 0, None

def ExtractFlights(mat,itrvl,r,w,h):
    """
    This function calls ExistKnot().
    Args:   mat: avgmat from collapse_data(), just one observed chunk without missing intervals
          itrvl: the window size of moving average,  unit is second
              r: the maximam radius of a pause
              w: a threshold for distance, if the distance to the great circle is greater than
                 this threshold, we consider there is a knot
              h: a threshold of distance, if the movemoent between two timestamps is less than h,
                 consider it as a pause and a knot
    Return: a 2d numpy array of trajectories, with headers as
            [status, lat_start, lon_start, stamp_start, lat_end, lon_end, stamp_end]
            status: if there is only one measure in this chunk, mark it as status "3" (unknown)
                    flight status is '1' and pause status is '2'
    """
    ## sometimes mat is a 1d array and sometimes it's 2d array
    ## which correspond to if and elif below
    try:
        if len(mat.shape)==1:
            out = np.array([3,mat[2],mat[3],mat[1]-itrvl/2,None,None,mat[1]+itrvl/2])
        elif len(mat.shape)==2 and mat.shape[0]==1:
            out = np.array([3,mat[0,2],mat[0,3],mat[0,1]-itrvl/2,None,None,mat[0,1]+itrvl/2])
        else:
            n = mat.shape[0]
            mat = np.hstack((mat,np.arange(n).reshape((n,1))))
            ## pause only
            if n>1 and max(pairwise_great_circle_dist(mat[:,2:4]))<r:
                m_lon = (mat[0,2]+mat[n-1,2])/2
                m_lat = (mat[0,3]+mat[n-1,3])/2
                out = np.array([2,m_lon,m_lat,mat[0,1]-itrvl/2,m_lon,m_lat,mat[n-1,1]+itrvl/2])
            ## if it's not pause only, there is at least one flight
            else:
                complete = 0
                knots = [0,n-1]
                mov = np.array([great_circle_dist(mat[i,2],mat[i,3],mat[i+1,2],mat[i+1,3]) for i in range(n-1)])
                pause_index = np.arange(0,n-1)[mov<h]
                temp = []
                for j in range(len(pause_index)-1):
                    if pause_index[j+1]-pause_index[j]==1:
                        temp.append(pause_index[j])
                        temp.append(pause_index[j+1])
                ## all the consequential numbers in between are inserted twice, but start and end are inserted once
                long_pause = np.unique(temp)[np.array([len(list(group)) for key, group in groupby(temp)])==1]
                ## pause 0,1,2, correspond to point [0,1,2,3], so the end number should plus 1
                long_pause[np.arange(1,len(long_pause),2)] = long_pause[np.arange(1,len(long_pause),2)]+1
                ## the key is to update the knot list and sort them
                knots.extend(long_pause.tolist())
                knots.sort()
                knots = unique(knots)
                ## while loop until there is no more knot
                while complete == 0:
                    mat_list = []
                    for i in range(len(knots)-1):
                        mat_list.append(mat[knots[i]:min(knots[i+1]+1,n-1),:])
                    knot_yes = np.empty(len(mat_list))
                    knot_pos = np.empty(len(mat_list))
                    for i in range(len(mat_list)):
                        knot_yes[i] , knot_pos[i] = ExistKnot(mat_list[i],w)
                    if sum(knot_yes)==0:
                        complete = 1
                    else:
                        for i in range(len(mat_list)):
                            if knot_yes[i]==1:
                                knots.append(int((mat_list[i])[int(knot_pos[i]),4]))
                        knots.sort()
                out = []
                for j in range(len(knots)-1):
                    start = knots[j]
                    end = knots[j+1]
                    mov = np.array([great_circle_dist(mat[i,2],mat[i,3],mat[i+1,2],mat[i+1,3]) for i in np.arange(start,end)])
                    if sum(mov>=h)==0:
                        m_lon = (mat[start,2]+mat[end,2])/2
                        m_lat = (mat[start,3]+mat[end,3])/2
                        nextline = [2, m_lon,m_lat,mat[start,1],m_lon,m_lat,mat[end,1]]
                    else:
                        nextline = [1, mat[start,2],mat[start,3],mat[start,1],mat[end,2],mat[end,3],mat[end,1]]
                    out.append(nextline)
                out = np.array(out)
        return out
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))

def GPS2MobMat(data, itrvl, accuracylim, r, w, h):
    """
    This function takes raw input (GPS) as input and return the first-step trajectory mat as output
    It calls collapse_data() and ExtractFlights(). Additionally, it divides the trajectory mat
    into obs - mis - obs - mis, then does ExtractFlights() on each observed chunk and stack them over
    Args: data: pd dataframe from read_data()
         itrvl: the window size of moving average,  unit is second
             r: the maximum radius of a pause
             w: a threshold for distance, if the distance to the great circle is greater than
                this threshold, we consider there is a knot
             h: a threshold of distance, if the movemoent between two timestamps is less than h,
                consider it as a pause and a knot
    Return: a 2d numpy array of all observed trajectories(first-step), with headers as
            [status, lat_start, lon_start, stamp_start, lat_end, lon_end, stamp_end]
    """
    try:
        avgmat = collapse_data(data, itrvl, accuracylim)
        outmat = np.zeros(7)
        curind = 0
        sys.stdout.write("Extract flights and pauses ..."+'\n')
        for i in range(avgmat.shape[0]):
            if avgmat[i,0]==4:
                ## divide the intermitted observeds chunk by the missing intervals (status=4)
                ## extract the flights and pauses from each observed chunk
                temp = ExtractFlights(avgmat[np.arange(curind,i),:],itrvl,r,w,h)
                outmat = np.vstack((outmat,temp))
                curind=i+1
        if curind<avgmat.shape[0]:
            #print(np.arange(curind,avgmat.shape[0]))
            temp = ExtractFlights(avgmat[np.arange(curind,avgmat.shape[0]),:],itrvl,r,w,h)
            outmat = np.vstack((outmat,temp))
        mobmat = np.delete(outmat,0,0)
        return mobmat
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))

def InferMobMat(mobmat,itrvl,r):
    """
    Args: mobmat: a 2d numpy array (output from GPS2MobMat())
           itrvl: the window size of moving average,  unit is second
               r: the maximam radius of a pause
    Return: a 2d numpy array as a final trajectory mat
            compared to the origimal mobmat, this function
            (1) infer the unknown status ('code==3' in the script below)
            (2) combine close pauses
            (3) join all flights (the end of first flight and the start of next flight)
                and pauses to form a continuous trajectory
            it also hstack one more column on the mobmat from GPS2MobMat(), which is a col of 1, indicating
            they are observed intsead of imputed, for future use.
    """
    try:
        sys.stdout.write("Infer unclassified windows ..."+'\n')
        code = mobmat[:,0]
        x0 = mobmat[:,1]; y0 = mobmat[:,2]; t0 = mobmat[:,3]
        x1 = mobmat[:,4]; y1 = mobmat[:,5]; t1 = mobmat[:,6]

        for i in range(len(code)):
            if code[i]==3 and i==0:
                code[i]=2
                x1[i] = x0[i]
                y1[i] = y0[i]
            if code[i]==3 and i>0:
                d = great_circle_dist(x0[i],y0[i],x1[i-1],y1[i-1])
                if t0[i]-t1[i-1]<=itrvl*3:
                    if d<r:
                        code[i]=2
                        x1[i] = x0[i]
                        y1[i] = y0[i]
                    else:
                        code[i]=1
                        s_x = x0[i]-itrvl/2/(t0[i]-t1[i-1])*(x0[i]-x1[i-1])
                        s_y = y0[i]-itrvl/2/(t0[i]-t1[i-1])*(y0[i]-y1[i-1])
                        e_x = x0[i]+itrvl/2/(t0[i]-t1[i-1])*(x0[i]-x1[i-1])
                        e_y = y0[i]+itrvl/2/(t0[i]-t1[i-1])*(y0[i]-y1[i-1])
                        x0[i] = s_x; x1[i]=e_x
                        y0[i] = s_y; y1[i]=e_y
                if t0[i]-t1[i-1]>itrvl*3:
                    if (i+1)<len(code):
                        f = great_circle_dist(x0[i],y0[i],x0[i+1],y0[i+1])
                        if t0[i+1]-t1[i]<=itrvl*3:
                            if f<r:
                                code[i]=2
                                x1[i] = x0[i]
                                y1[i] = y0[i]
                            else:
                                code[i]=1
                                s_x = x0[i]-itrvl/2/(t0[i+1]-t1[i])*(x0[i+1]-x0[i])
                                s_y = y0[i]-itrvl/2/(t0[i+1]-t1[i])*(y0[i+1]-y0[i])
                                e_x = x0[i]+itrvl/2/(t0[i+1]-t1[i])*(x0[i+1]-x0[i])
                                e_y = y0[i]+itrvl/2/(t0[i+1]-t1[i])*(y0[i+1]-y0[i])
                                x0[i] = s_x; x1[i]=e_x
                                y0[i] = s_y; y1[i]=e_y
                        else:
                            code[i]=2
                            x1[i] = x0[i]
                            y1[i] = y0[i]
                    else:
                        code[i]=2
                        x1[i] = x0[i]
                        y1[i] = y0[i]
            mobmat[i,:] = [code[i],x0[i],y0[i],t0[i],x1[i],y1[i],t1[i]]

        ## merge consecutive pauses
        sys.stdout.write("Merge consecutive pauses and bridge gaps ..."+'\n')
        k = []
        for j in np.arange(1,len(code)):
            if code[j]==2 and code[j-1]==2 and t0[j]==t1[j-1]:
                k.append(j-1)
                k.append(j)
        ## all the consequential numbers in between are inserted twice, but start and end are inserted once
        rk = np.unique(k)[np.array([len(list(group)) for key, group in groupby(k)])==1]
        for j in range(int(len(rk)/2)):
            start = rk[2*j]
            end = rk[2*j+1]
            mx = np.mean(x0[np.arange(start,end+1)])
            my = np.mean(y0[np.arange(start,end+1)])
            mobmat[start,:] = [2,mx,my,t0[start],mx,my,t1[end]]
            mobmat[np.arange(start+1,end+1),0]=5
        mobmat = mobmat[mobmat[:,0]!=5,:]

        ## check missing intervals, if starting and ending point are close, make them same
        new_pauses = []
        for j in np.arange(1,mobmat.shape[0]):
            if mobmat[j,3] > mobmat[j-1,6]:
                d = great_circle_dist(mobmat[j,1],mobmat[j,2],mobmat[j-1,4],mobmat[j-1,5])
                if d<10:
                    if mobmat[j,0]==2 and mobmat[j-1,0]==2:
                        initial_x = mobmat[j-1,4]
                        initial_y = mobmat[j-1,5]
                        mobmat[j,1] = mobmat[j,4] = mobmat[j-1,1] = mobmat[j-1,4] = initial_x
                        mobmat[j,2] = mobmat[j,5] = mobmat[j-1,2] = mobmat[j-1,5] = initial_y
                    if mobmat[j,0]==1 and mobmat[j-1,0]==2:
                        mobmat[j,1] = mobmat[j-1,4]
                        mobmat[j,2] = mobmat[j-1,5]
                    if mobmat[j,0]==2 and mobmat[j-1,0]==1:
                        mobmat[j-1,4] = mobmat[j,1]
                        mobmat[j-1,5] = mobmat[j,2]
                    if mobmat[j,0]==1 and mobmat[j-1,0]==1:
                        mean_x = (mobmat[j,1] + mobmat[j-1,4])/2
                        mean_y = (mobmat[j,2] + mobmat[j-1,5])/2
                        mobmat[j-1,4] = mobmat[j,1] = mean_x
                        mobmat[j-1,5] = mobmat[j,2] = mean_y
                    new_pauses.append([2,mobmat[j,1],mobmat[j,2],mobmat[j-1,6],mobmat[j,1],mobmat[j,2],mobmat[j,3],0])
        new_pauses = np.array(new_pauses)

        ## connect flights and pauses
        for j in np.arange(1,mobmat.shape[0]):
            if mobmat[j,0]*mobmat[j-1,0]==2 and mobmat[j,3]==mobmat[j-1,6]:
                if mobmat[j,0]==1:
                    mobmat[j,1] = mobmat[j-1,4]
                    mobmat[j,2] = mobmat[j-1,5]
                if mobmat[j-1,0]==1:
                    mobmat[j-1,4] = mobmat[j,1]
                    mobmat[j-1,5] = mobmat[j,2]

        mobmat = np.hstack((mobmat,np.ones(mobmat.shape[0]).reshape(mobmat.shape[0],1)))
        mobmat = np.vstack((mobmat,new_pauses))
        mobmat = mobmat[mobmat[:,3].argsort()].astype(float)
        return mobmat
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))
