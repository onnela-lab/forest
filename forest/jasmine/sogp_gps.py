import logging
import math
import numpy as np


logger = logging.getLogger(__name__)

## the radius of the earth
R = 6.371*10**6

def K0(x1,x2,pars):
    """
    Args: x1,x2 are 1d numpy arrays with length 2
          x1 = [a,b], with a as timestamp and b as latitude or longitude
          pars, a list of parameters
    Return: a scalar, a smi
    """
    [l1,l2,l3,a1,a2,b1,b2,b3] = pars
    k1 = np.exp(-abs(x1[0]-x2[0])/l1)*np.exp(-(np.sin(abs(x1[0]-x2[0])/86400*math.pi))**2/a1)
    k2 = np.exp(-abs(x1[0]-x2[0])/l2)*np.exp(-(np.sin(abs(x1[0]-x2[0])/604800*math.pi))**2/a2)
    k3 = np.exp(-abs(x1[1]-x2[1])/l3)
    return b1*k1+b2*k2+b3*k3

## all the functions below are based on the equations in [Csato and Opper (2002)]
## ref: http://www.cs.ubbcluj.ro/~csatol/SOGP/thesis/Gaussian_Process.html#SECTION00521000000000000000
## ref: http://www.cs.ubbcluj.ro/~csatol/SOGP/thesis/Sparsity_in.html#cha:sparse

def update_K(bv,K,pars):
    """
    similarity matrix between bv's
    Args: bv, a list of basis vectors
          K, 2d numpy array
    Return: 2d numpy array
    """
    if len(bv)==0:
        mat = np.array([1])
    else:
        d = np.shape(K)[0]
        row = np.ones(d)
        column = np.ones([d+1,1])
        for i in range(d):
            row[i] = column[i,0] = K0(bv[-1][:-1],bv[i][:-1],pars)
        mat = np.hstack([np.vstack([K,row]),column])
    return mat

def update_k(bv,x_current,pars):
    """
    similarity vector between the current input with all bv's, t starts from 0
    Args: bv, a list of basis vectors
          x_current, current input, X[i,:], 1d array
    Return: 1d numpy array
    """
    d = len(bv)
    if d==0:
        out = np.array([0])
    if d>=1:
        out = np.zeros(d)
        for i in range(d):
            out[i] = K0(x_current,bv[i][:-1],pars)
    return out

def update_e_hat(Q,k):
    """
    Args: Q, 2d numpy array
          k, 1d numpy array
    Return: 1d numpy array
    """
    if np.shape(Q)[0]==0:
        out = np.array([0])
    else:
        out = np.dot(Q,k)
    return out

def update_gamma(k,e_hat):
    """
    Args: k, 1d numpy array
          e_hat, 1d numpy array
    Return: scalar
    """
    return 1-np.dot(k,e_hat)

def update_q(k,alpha,sigmax,y_current):
    """
    Args: k, alpha, e_hat: 1d numpy array
          sigmax, y_current: scalar
    Return: scalar
    """
    if len(alpha)==0:
        out = y_current/sigmax
    else:
        out = (y_current-np.dot(k,alpha))/sigmax
    return out

def update_s_hat(C,k,e_hat):
    """
    Args: C: 2d numpy array
          k, e_hat: 1d numpy array
    Return: 1d numpy array
    """
    return np.dot(C,k)+e_hat

def update_eta(gamma,sigmax):
    """
    Args: gamma and sigmax: scalar
    Return: scalar
    """
    r = -1/sigmax
    return 1/(1+gamma*r)

def update_alpha_hat(alpha,q,eta,s_hat):
    """
    Args: q, eta: scalar
          alpha, s_hat: 1d numpy array
    Return: 1d numpy array
    """
    return alpha+q*eta*s_hat

def update_c_hat(C,sigmax,eta,s_hat):
    """
    Args: sigmax, eta: scalar
          C: 2d array
          s_hat: 1d array
    Return: 2d array
    """
    r = -1/sigmax
    return C+r*eta*np.outer(s_hat,s_hat)

def update_s(C,k):
    """
    Args: C: 2d array
          k: 1d array
    Return: 1d array
    """
    if np.shape(C)[0]==0:
        s = np.array([1])
    else:
        temp = np.dot(C,k)
        s = np.append(temp,1)
    return s

def update_alpha(alpha,q,s):
    """
    Args: alpha, s: 1d array
          q: scalar
    Return: 1d array
    """
    T_alpha = np.append(alpha,0)
    new_alpha = T_alpha + q*s
    return new_alpha

def update_c(C,sigmax,s):
    """
    Args: C: 2d array
          sigmax: scalar
          s: 1d array
    Return: 1d array
    """
    d = np.shape(C)[0]
    if d==0:
        U_c = np.array([0])
    else:
        U_c = np.hstack([np.vstack([C,np.zeros(d)]),np.zeros([d+1,1])])
    r = -1/sigmax
    new_c = U_c+r*np.outer(s,s)
    return new_c

def update_Q(Q,gamma,e_hat):
    """
    Args: Q: 2d array
          gamma: scalar
          e_hat: 1d array
    Return: 2d array
    """
    d = np.shape(Q)[0]
    if d==0:
        out = np.array([1])
    else:
        temp = np.append(e_hat,-1)
        new_Q = np.hstack([np.vstack([Q,np.zeros(d)]),np.zeros([d+1,1])])
        out = new_Q + 1/gamma*np.outer(temp,temp)
    return out

def update_alpha_vec(alpha,Q,C):
    """
    Args: alpha: 1d array
          Q, C: 2d array
    Return: 1d array
    """
    t = len(alpha)-1
    return alpha[:t]-alpha[t]/(C[t,t]+Q[t,t])*(Q[t,:t]+C[t,:t])

def update_c_mat(C,Q):
    """
    Args: Q, C: 2d array
    Return: 2d array
    """
    t = np.shape(C)[0]-1
    return C[:t,:t]+np.outer(Q[t,:t],Q[t,:t])/Q[t,t]-np.outer(Q[t,:t]+C[t,:t],Q[t,:t]+C[t,:t])/(Q[t,t]+C[t,t])

def update_q_mat(Q):
    """
    Args: Q, 2d array
    Return: 2d array
    """
    t = np.shape(Q)[0]-1
    return Q[:t,:t]-np.outer(Q[t,:t],Q[t,:t])/Q[t,t]

def update_s_mat(k_mat,s_mat,index,Q):
    """
    Args: k_mat, s_mat, Q: 2d array
          index: 1d array of intergers
    Return: 2d array
    """
    k_mat =  (k_mat[index,:])[:,index]
    s_mat =  (s_mat[index,:])[:,index]
    step1 = k_mat-k_mat.dot(s_mat).dot(k_mat)
    step2 = (step1[:-1,:])[:,:-1]
    step3 = Q - Q.dot(step2).dot(Q)
    return step3

def SOGP(X,Y,sigma2,tol,d,pars,Q,C,alpha,bv):
    """
    (1) If it is the first time to process the data, Q,C,alpha,bv should be empty,
        this function takes X, Y, (sigma2, tol, d) [parameters] as input and returns
        a list of basis vectors [bv] of length d and other summarized knownledge of
        processed (X,Y) as Q, C, alpha in order to use next time in a online manner
    (2) If we already have (Q,C,alpha,bv) from previous update, then (X,Y) should be
        new data, and this function will update Q, C, alpha and bv. In this scenario,
        d should be greater or equal to len(bv)
    ## This is the key function of sparse online gaussian process
    Args: X: 2d array (n*p)  Y: 1d array (n)
          sigma2, tol, d: scalar, hyperparameters
          pars, a list of parameters
    Return: a dictionary with bv,alpha: 1d array (d) Q,C: 2d array (d*d)
    """
    n = len(Y)
    I = 0 ## an indicator shows if it is the first time that the number of bvs hits d
    for i in range(n):
        if X.ndim==1:
            x_current = X[i]
        else:
            x_current = X[i,:]
        y_current = Y[i]
        k = update_k(bv,x_current,pars)
        if np.shape(C)[0]==0:
            sigmax = 1+sigma2
        else:
            sigmax = 1+sigma2+k.dot(C).dot(k)
        q = update_q(k,alpha,sigmax,y_current)
        r = -1/sigmax
        e_hat = update_e_hat(Q,k)
        gamma = update_gamma(k,e_hat)
        if gamma<tol:
            s = update_s_hat(C,k,e_hat)
            eta = update_eta(gamma,sigmax)
            alpha = update_alpha_hat(alpha,q,eta,s)
            C = update_c_hat(C,sigmax,eta,s)
        else:
            s = update_s(C,k)
            alpha = update_alpha(alpha,q,s)
            C = update_c(C,sigmax,s)
            Q = update_Q(Q,gamma,e_hat)
            if X.ndim==1:
                new_point = np.array([x_current,y_current])
            else:
                new_point = np.concatenate((x_current,[y_current]))
            bv.append(new_point)
            if len(bv)>=d:
                I = I + 1
            if I==1:
                ## the sample size hits d first time, calculate K once and then update it in another way
                K = np.zeros([d,d])
                for i in range(d):
                    for j in range(d):
                        K[i,j] = K0(bv[i][:-1],bv[j][:-1],pars)
                S = np.linalg.inv(np.linalg.inv(C)+K)

            if len(bv)>d:
                alpha_vec = update_alpha_vec(alpha,Q,C)
                c_mat = update_c_mat(C,Q)
                q_mat = update_q_mat(Q)
                s_mat = np.hstack([np.vstack([S,np.zeros(d)]),np.zeros([d+1,1])])
                s_mat[d,d] = 1/sigma2
                k_mat = update_K(bv,K,pars)
                eps = np.zeros(d)
                for j in range(d):
                    eps[j] = alpha_vec[j]/(q_mat[j,j]+c_mat[j,j])-s_mat[j,j]/q_mat[j,j]+np.log(1+c_mat[j,j]/q_mat[j,j])
                loc = np.where(eps == np.min(eps))[0][0]
                del bv[loc]
                if loc==0:
                    #index = np.append(np.arange(1,d+1),0)
                    index = np.concatenate((np.arange(1,d+1),[0]))
                else:
                    #index = np.append(np.append(np.arange(0,loc),np.arange(loc+1,d+1)),loc)
                    index = np.concatenate((np.arange(0,loc),np.arange(loc+1,d+1),[loc]))
                alpha = update_alpha_vec(alpha[index],(Q[index,:])[:,index],(C[index,:])[:,index])

                C = update_c_mat((C[index,:])[:,index],(Q[index,:])[:,index])
                Q = update_q_mat((Q[index,:])[:,index])
                S = update_s_mat(k_mat,s_mat,index,Q)
                K = (k_mat[index[:d],:])[:,index[:d]]
    output = {'bv':bv,'alpha':alpha,'Q':Q,'C':C}
    return output

def BV_select(MobMat,sigma2,tol,d,pars,memory_dict,BV_set):
    """
    This function is an application of SOGP() on GPS data. We first treat latitude as Y,
    [longitude,timestamp] as X, then we treat longitude as Y and [latitude, timestamp] as X.
    Furthurmore, we select basis vectors from flights and pauses separately. This means there
    are 4 scenarios, and we combine the basis vectors from all scenarios as the final bv set.
    Args: MobMat: 2d array, output from InferMobMat() in data2mobmat.py
          sigma2, tol, d: scalar, hyperparameters
          memory_dict: a dictionary of dictionary from SOGP()
    Return: a dictionary with bv [trajectory], bv_index, and an updated memory_dict
    """
    logger.info("Selecting basis vectors ...")
    flight_index = MobMat[:,0]==1
    pause_index = MobMat[:,0]==2
    mean_x = (MobMat[:,1]+MobMat[:,4])/2
    mean_y = (MobMat[:,2]+MobMat[:,5])/2
    mean_t = (MobMat[:,3]+MobMat[:,6])/2
    ## use t as the unique key to match bv and mobmat

    if memory_dict == None:
        memory_dict = {}
        memory_dict['1'] = {'bv':[],'alpha':[],'Q':[],'C':[]}
        memory_dict['2'] = {'bv':[],'alpha':[],'Q':[],'C':[]}
        memory_dict['3'] = {'bv':[],'alpha':[],'Q':[],'C':[]}
        memory_dict['4'] = {'bv':[],'alpha':[],'Q':[],'C':[]}

    X = np.transpose(np.vstack((mean_t,mean_x)))[flight_index]
    Y = mean_y[flight_index]
    result1 = SOGP(X,Y,sigma2,tol,d,pars,memory_dict['1']['Q'],memory_dict['1']['C'],memory_dict['1']['alpha'],memory_dict['1']['bv'])
    bv1 = result1['bv']
    t1 = np.array([bv1[j][0] for j in range(len(bv1))])

    X = np.transpose(np.vstack((mean_t,mean_x)))[pause_index]
    Y = mean_y[pause_index]
    result2 = SOGP(X,Y,sigma2,tol,d,pars,memory_dict['2']['Q'],memory_dict['2']['C'],memory_dict['2']['alpha'],memory_dict['2']['bv'])
    bv2 = result2['bv']
    t2 = np.array([bv2[j][0] for j in range(len(bv2))])

    X = np.transpose(np.vstack((mean_t,mean_y)))[flight_index]
    Y = mean_x[flight_index]
    result3 = SOGP(X,Y,sigma2,tol,d,pars,memory_dict['3']['Q'],memory_dict['3']['C'],memory_dict['3']['alpha'],memory_dict['3']['bv'])
    bv3 = result3['bv']
    t3 = np.array([bv3[j][0] for j in range(len(bv3))])

    X = np.transpose(np.vstack((mean_t,mean_y)))[pause_index]
    Y = mean_x[pause_index]
    result4 = SOGP(X,Y,sigma2,tol,d,pars,memory_dict['4']['Q'],memory_dict['4']['C'],memory_dict['4']['alpha'],memory_dict['4']['bv'])
    bv4 = result4['bv']
    t4 = np.array([bv4[j][0] for j in range(len(bv4))])

    unique_t = np.unique(np.concatenate((np.concatenate((t1,t2)),np.concatenate((t3,t4)))))
    if BV_set != None:
        all_candidates = np.vstack((BV_set,MobMat))
        all_t = (all_candidates[:,3]+all_candidates[:,6])/2
    else:
        all_candidates = MobMat
        all_t = mean_t
    index = []
    for j in range(len(all_t)):
        if np.any(unique_t == all_t[j]):
            index.append(j)
    index = np.array(index)
    BV_set = all_candidates[index,:]
    memory_dict['1'] = result1
    memory_dict['2'] = result2
    memory_dict['3'] = result3
    memory_dict['4'] = result4
    return {'BV_set':BV_set,'memory_dict':memory_dict}
