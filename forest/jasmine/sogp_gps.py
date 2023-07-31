"""This module implements the sparse online gaussian process
    algorithm for GPS data.
    The algorithm is based on the paper [Csato and Opper (2002)]
    and the code is based on the matlab code from the author of the paper.
    ref: http://www.cs.ubbcluj.ro/~csatol/SOGP/thesis/Gaussian_Process.html
    ref: http://www.cs.ubbcluj.ro/~csatol/SOGP/thesis/Sparsity_in.html
    The algorithm is used to select basis vectors from GPS data,
    and the selected basis vectors are used to construct the mobility matrix.
"""
import math
import sys
from typing import Dict, Tuple

import numpy as np


def calculate_K0(x1: np.ndarray, x2: np.ndarray, pars: list) -> float:
    """This function calculates the similarity between two points

    Args:
        x1, x2: np.ndarrays with length 2
          x1 = [a,b], with a as timestamp and b as latitude or longitude
        pars: list, a list of parameters used in the similarity function
    Returns:
        float, the similarity between x1 and x2
    """
    [l1, l2, l3, a1, a2, b1, b2, b3] = pars
    k1 = np.exp(-abs(x1[0] - x2[0]) / l1) * np.exp(
        -((np.sin(abs(x1[0] - x2[0]) / 86400 * math.pi)) ** 2) / a1
    )
    k2 = np.exp(-abs(x1[0] - x2[0]) / l2) * np.exp(
        -((np.sin(abs(x1[0] - x2[0]) / 604800 * math.pi)) ** 2) / a2
    )
    k3 = np.exp(-abs(x1[1] - x2[1]) / l3)
    return b1 * k1 + b2 * k2 + b3 * k3


def update_K(bv: list, K: np.ndarray, pars: list) -> np.ndarray:
    """Update the similarity matrix between basis vectors.

    Args:
        bv: list, a list of basis vectors.
        K: np.ndarray, a 2D array.
        pars: list, a list of parameters used in the similarity function.

    Returns:
        np.ndarray, updated 2D array.
    """
    if len(bv) == 0:
        return np.array([1])

    d = np.shape(K)[0]
    row = np.ones(d)
    column = np.ones([d + 1, 1])

    for i in range(d):
        x1, x2 = bv[-1][:-1], bv[i][:-1]
        row[i] = column[i, 0] = calculate_K0(x1, x2, pars)

    return np.hstack([np.vstack([K, row]), column])


def update_k(bv: list, x1: np.ndarray, pars: list) -> np.ndarray:
    """
    Compute the similarity vector between
        the current input and all basis vectors.

    Args:
        bv: list, a list of basis vectors.
        x1: np.ndarray, the current input, which is a 1D array.
        pars: list, a list of parameters used in the similarity function.

    Returns:
        np.ndarray, 1D array indicating similarity.
    """
    d = len(bv)

    if d == 0:
        return np.array([0])

    out = np.zeros(d)

    for i in range(d):
        x2 = bv[i][:-1]
        out[i] = calculate_K0(x1, x2, pars)

    return out


def update_e_hat(Q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Update the estimated vector e_hat.

    Args:
        Q: np.ndarray, a 2D array.
        k: np.ndarray, a 1D array.

    Returns:
        np.ndarray, updated 1D array.
    """
    if np.shape(Q)[0] == 0:
        return np.array([0])

    return np.dot(Q, k)


def update_gamma(k: np.ndarray, e_hat: np.ndarray) -> float:
    """
    Update the scalar gamma.

    Args:
        k: np.ndarray, a 1D array.
        e_hat: np.ndarray, a 1D array.

    Returns:
        float, updated scalar gamma.
    """
    return 1 - np.dot(k, e_hat)


def update_q(
    k: np.ndarray, alpha: np.ndarray, sigmax: float, y_current: float
) -> float:
    """
    Update the scalar q.

    Args:
        k: np.ndarray, a 1D array.
        alpha: np.ndarray, a 1D array.
        sigmax: float, a scalar.
        y_current: float, a scalar.

    Returns:
        float, updated scalar q.
    """
    if len(alpha) == 0:
        return y_current / sigmax

    return (y_current - np.dot(k, alpha)) / sigmax


def update_s_hat(
    C: np.ndarray, k: np.ndarray, e_hat: np.ndarray
) -> np.ndarray:
    """
    Update the s_hat vector.

    Args:
        C: np.ndarray, a 2D array.
        k: np.ndarray, a 1D array.
        e_hat: np.ndarray, a 1D array.

    Returns:
        np.ndarray, updated 1D array s_hat.
    """
    return np.dot(C, k) + e_hat


def update_eta(gamma: float, sigmax: float) -> float:
    """
    Update the scalar eta.

    Args:
        gamma: float, a scalar.
        sigmax: float, a scalar.

    Returns:
        float, updated scalar eta.
    """
    r = -1 / sigmax
    return 1 / (1 + gamma * r)


def update_alpha_hat(
    alpha: np.ndarray, q: float, eta: float, s_hat: np.ndarray
) -> np.ndarray:
    """
    Update the alpha_hat vector.

    Args:
        alpha: np.ndarray, a 1D array.
        q: float, a scalar.
        eta: float, a scalar.
        s_hat: np.ndarray, a 1D array.

    Returns:
        np.ndarray, updated 1D array alpha_hat.
    """
    return alpha + q * eta * s_hat


def update_c_hat(
    C: np.ndarray, sigmax: float, eta: float, s_hat: np.ndarray
) -> np.ndarray:
    """
    Update the c_hat matrix.

    Args:
        C: np.ndarray, a 2D array.
        sigmax: float, a scalar.
        eta: float, a scalar.
        s_hat: np.ndarray, a 1D array.

    Returns:
        np.ndarray, updated 2D array c_hat.
    """
    r = -1 / sigmax
    return C + r * eta * np.outer(s_hat, s_hat)


def update_s(C: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Update the s vector.

    Args:
        C: np.ndarray, a 2D array.
        k: np.ndarray, a 1D array.

    Returns:
        np.ndarray, updated 1D array s.
    """
    if np.shape(C)[0] == 0:
        return np.array([1])

    temp = np.dot(C, k)
    return np.append(temp, 1)


def update_alpha(alpha: np.ndarray, q: float, s: np.ndarray) -> np.ndarray:
    """
    Update the alpha vector.

    Args:
        alpha: np.ndarray, a 1D array.
        q: float, a scalar.
        s: np.ndarray, a 1D array.

    Returns:
        np.ndarray, updated 1D array alpha.
    """
    T_alpha = np.append(alpha, 0)
    new_alpha = T_alpha + q * s
    return new_alpha


def update_c(C: np.ndarray, sigmax: float, s: np.ndarray) -> np.ndarray:
    """
    Update the C matrix.

    Args:
        C: np.ndarray, a 2D array.
        sigmax: float, a scalar.
        s: np.ndarray, a 1D array.

    Returns:
        np.ndarray, updated 2D array C.
    """
    d = np.shape(C)[0]
    if d == 0:
        U_c = np.array([0])
    else:
        U_c = np.hstack([np.vstack([C, np.zeros(d)]), np.zeros([d + 1, 1])])
    r = -1 / sigmax
    new_c = U_c + r * np.outer(s, s)
    return new_c


def update_Q(Q: np.ndarray, gamma: float, e_hat: np.ndarray) -> np.ndarray:
    """
    Update the Q matrix.

    Args:
        Q: np.ndarray, a 2D array.
        gamma: float, a scalar.
        e_hat: np.ndarray, a 1D array.

    Returns:
        np.ndarray, updated 2D array Q.
    """
    d = np.shape(Q)[0]
    if d == 0:
        return np.array([1])

    temp = np.append(e_hat, -1)
    new_Q = np.hstack([np.vstack([Q, np.zeros(d)]), np.zeros([d + 1, 1])])
    return new_Q + 1 / gamma * np.outer(temp, temp)


def update_alpha_vec(
    alpha: np.ndarray, Q: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """
    Update the alpha vector.

    Args:
        alpha: np.ndarray, a 1D array.
        Q: np.ndarray, a 2D array.
        C: np.ndarray, a 2D array.

    Returns:
        np.ndarray, updated 1D array alpha.
    """
    t = len(alpha) - 1
    return alpha[:t] - alpha[t] / (C[t, t] + Q[t, t]) * (Q[t, :t] + C[t, :t])


def update_c_mat(C: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Update the C matrix.

    Args:
        C: np.ndarray, a 2D array.
        Q: np.ndarray, a 2D array.

    Returns:
        np.ndarray, updated 2D array C.
    """
    t = np.shape(C)[0] - 1
    return (
        C[:t, :t]
        + np.outer(Q[t, :t], Q[t, :t]) / Q[t, t]
        - np.outer(
            Q[t, :t] + C[t, :t], Q[t, :t] + C[t, :t]
        ) / (Q[t, t] + C[t, t])
    )


def update_q_mat(Q: np.ndarray) -> np.ndarray:
    """
    Update the Q matrix.

    Args:
        Q: np.ndarray, a 2D array.

    Returns:
        np.ndarray, updated 2D array Q.
    """
    t = np.shape(Q)[0] - 1
    return Q[:t, :t] - np.outer(Q[t, :t], Q[t, :t]) / Q[t, t]


def update_s_mat(
    k_mat: np.ndarray, s_mat: np.ndarray, index: np.ndarray, Q: np.ndarray
) -> np.ndarray:
    """
    Update the s matrix.

    Args:
        k_mat: np.ndarray, a 2D array.
        s_mat: np.ndarray, a 2D array.
        index: np.ndarray, a 1D array of integers.
        Q: np.ndarray, a 2D array.

    Returns:
        np.ndarray, updated 2D array s_mat.
    """
    k_mat = (k_mat[index, :])[:, index]
    s_mat = (s_mat[index, :])[:, index]
    step1 = k_mat - k_mat.dot(s_mat).dot(k_mat)
    step2 = (step1[:-1, :])[:, :-1]
    step3 = Q - Q.dot(step2).dot(Q)
    return step3


def calculate_sigma_max(C: np.ndarray, k: np.ndarray, sigma2: float) -> float:
    """Calculate sigma max.

    Args:
        C: np.ndarray, a 2D array.
        k: np.ndarray, a 1D array.
        sigma2: float, a scalar.

    Returns:
        float, sigma max.
    """
    if np.shape(C)[0] == 0:
        return 1 + sigma2
    else:
        return 1 + sigma2 + k.dot(C).dot(k)


def update_system_given_gamma_tol(
    C: np.ndarray, Q: np.ndarray, alpha: np.ndarray,
    k: np.ndarray, q: float, gamma: float, sigmax: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Update system when gamma is less than tolerance.

    Args:
        C: np.ndarray, a 2D array.
        Q: np.ndarray, a 2D array.
        alpha: np.ndarray, a 1D array.
        k: np.ndarray, a 1D array.
        q: float, a scalar.
        gamma: float, a scalar.
        sigmax: float, a scalar.

    Returns:
        alpha: np.ndarray, a 1D array.
        C: np.ndarray, a 2D array.
    """
    e_hat = update_e_hat(Q, k)
    s = update_s_hat(C, k, e_hat)
    eta = update_eta(gamma, sigmax)
    alpha = update_alpha_hat(alpha, q, eta, s)
    C = update_c_hat(C, sigmax, eta, s)
    return alpha, C


def update_system_otherwise(
    C: np.ndarray, Q: np.ndarray, alpha: np.ndarray, k: np.ndarray,
    q: float, sigmax: float, gamma: float, e_hat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update system when gamma is greater or equal to tolerance.

    Args:
        C: np.ndarray, a 2D array.
        Q: np.ndarray, a 2D array.
        alpha: np.ndarray, a 1D array.
        k: np.ndarray, a 1D array.
        q: float, a scalar.
        sigmax: float, a scalar.
        gamma: float, a scalar.
        e_hat: np.ndarray, a 1D array.

    Returns:
        alpha: np.ndarray, a 1D array.
        C: np.ndarray, a 2D array.
        Q: np.ndarray, a 2D array.
    """
    s = update_s(C, k)
    alpha = update_alpha(alpha, q, s)
    C = update_c(C, sigmax, s)
    Q = update_Q(Q, gamma, e_hat)
    return alpha, C, Q


def pruning_bv(
    bv: list, alpha: np.ndarray, Q: np.ndarray, C: np.ndarray,
    S: np.ndarray, K: np.ndarray, sigma2: float, d: int, pars: list
) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prune the basis vectors.

    Args:
        bv: list, a list of basis vectors.
        alpha: np.ndarray, a 1D array.
        Q: np.ndarray, a 2D array.
        C: np.ndarray, a 2D array.
        S: np.ndarray, a 2D array.
        K: np.ndarray, a 2D array.
        sigma2: float, a scalar.
        d: int, a scalar.
        pars: list, a list of parameters used in the similarity function.

    Returns:
        bv: list, a list of basis vectors.
        alpha: np.ndarray, a 1D array.
        Q: np.ndarray, a 2D array.
        C: np.ndarray, a 2D array.
        S: np.ndarray, a 2D array.
        K: np.ndarray, a 2D array.
    """
    alpha_vec = update_alpha_vec(alpha, Q, C)
    c_mat = update_c_mat(C, Q)
    q_mat = update_q_mat(Q)
    s_mat = np.hstack([np.vstack([S, np.zeros(d)]), np.zeros([d + 1, 1])])
    s_mat[d, d] = 1 / sigma2
    k_mat = update_K(bv, K, pars)
    eps = np.zeros(d)
    for j in range(d):
        eps[j] = (
            alpha_vec[j] / (q_mat[j, j] + c_mat[j, j])
            - s_mat[j, j] / q_mat[j, j]
            + np.log(1 + c_mat[j, j] / q_mat[j, j])
        )
    loc = np.where(eps == np.min(eps))[0][0]
    del bv[loc]
    index: np.ndarray
    if loc == 0:
        index = np.concatenate((np.arange(1, d + 1), np.array([0])))
    else:
        index = np.concatenate(
            (np.arange(0, loc), np.arange(loc + 1, d + 1), np.array([loc]))
        )
    alpha = update_alpha_vec(
        alpha[index], (Q[index, :])[:, index], (C[index, :])[:, index]
    )

    C = update_c_mat((C[index, :])[:, index], (Q[index, :])[:, index])
    Q = update_q_mat((Q[index, :])[:, index])
    S = update_s_mat(k_mat, s_mat, index, Q)
    K = (k_mat[index[:d], :])[:, index[:d]]

    return bv, alpha, Q, C, S, K


def SOGP(
    X: np.ndarray,
    Y: np.ndarray,
    sigma2: float,
    tol: float,
    d: int,
    pars: list,
    Q: np.ndarray,
    C: np.ndarray,
    alpha: np.ndarray,
    bv: list,
) -> Dict:
    """This is the key function of sparse online gaussian process
    (1) If it is the first time to process the data,
         Q,C,alpha,bv should be empty,
         this function takes X, Y, (sigma2, tol, d)
         [parameters] as input and returns
         a list of basis vectors [bv] of length d
         and other summarized knownledge of
         processed (X,Y) as Q, C, alpha in order
         to use next time in a online manner
    (2) If we already have (Q,C,alpha,bv)
         from previous update, then (X,Y) should be
         new data, and this function will
         update Q, C, alpha and bv. In this scenario,
         d should be greater or equal to len(bv)

    Args:
        X: 2d array (n*d),
         n is the number of samples, d is the dimension of X
        Y: 1d array (n), n is the number of samples
        sigma2: scalar, hyperparameter
        tol: scalar, hyperparameter
        d: scalar, hyperparameter
        pars: list, a list of parameters used in the similarity function
        Q: 2d array (d*d), a summary of previous processed data
        C: 2d array (d*d), a summary of previous processed data
        alpha: 1d array (d), a summary of previous processed data
        bv: list, a list of basis vectors

    Returns:
        a dictionary with:
            bv, alpha: 1d array (d)
            Q, C: 2d array (d*d)
    """
    n = len(Y)
    # an indicator shows if it is the first time that the number of bvs hits d
    indicator = 0

    for i in range(n):
        if X.ndim == 1:
            x_current = X[i]
        else:
            x_current = X[i, :]
        y_current = Y[i]

        k = update_k(bv, x_current, pars)
        sigmax = calculate_sigma_max(C, k, sigma2)
        q = update_q(k, alpha, sigmax, y_current)
        e_hat = update_e_hat(Q, k)
        gamma = update_gamma(k, e_hat)

        if gamma < tol:
            alpha, C = update_system_given_gamma_tol(
                C, Q, alpha, k, q, gamma, sigmax
            )
        else:
            alpha, C, Q = update_system_otherwise(
                C, Q, alpha, k, q, sigmax, gamma, e_hat
            )

            if X.ndim == 1:
                new_point = np.array([x_current, y_current])
            else:
                new_point = np.concatenate((x_current, [y_current]))
            bv.append(new_point)

            if len(bv) >= d:
                indicator += 1

            if indicator == 1:
                # the sample size hits d first time,
                # calculate K once and then update it in another way
                K = np.zeros([d, d])
                for i in range(d):
                    for j in range(d):
                        x1, x2 = bv[i][:-1], bv[j][:-1]
                        K[i, j] = calculate_K0(x1, x2, pars)
                S = np.linalg.inv(np.linalg.inv(C) + K)

            if len(bv) > d:
                bv, alpha, Q, C, S, K = pruning_bv(
                    bv, alpha, Q, C, S, K, sigma2, d, pars
                )

    return {"bv": bv, "alpha": alpha, "Q": Q, "C": C}


def bv_select(
    MobMat: np.ndarray,
    sigma2: float,
    tol: float,
    d: int,
    pars: list,
    memory_dict: dict,
    bv_set: np.ndarray,
) -> Dict:
    """This function is an application of SOGP() on GPS data.
     We first treat latitude as Y,
     [longitude,timestamp] as X, then we treat longitude as Y
     and [latitude, timestamp] as X.
     Furthurmore, we select basis vectors from flights and pauses separately.
     This means there are 4 scenarios, and we combine the basis vectors
     from all scenarios as the final bv set.

    Args:
        MobMat: 2d array, output from InferMobMat() in data2mobmat.py
        sigma2: scalar, hyperparameter
        tol: scalar, hyperparameter
        d: scalar, hyperparameter
        pars: list, a list of parameters used in the similarity function
        memory_dict: dict, a dictionary of dictionaries from SOGP()
        bv_set: 2d array, a set of basis vectors from previous processed data

    Returns:
        a dictionary with bv [trajectory],
        bv_index, and an updated memory_dict
    """
    sys.stdout.write("Selecting basis vectors ...\n")
    flight_index = MobMat[:, 0] == 1
    pause_index = MobMat[:, 0] == 2

    mean_x = (MobMat[:, 1] + MobMat[:, 4]) / 2
    mean_y = (MobMat[:, 2] + MobMat[:, 5]) / 2
    mean_t = (MobMat[:, 3] + MobMat[:, 6]) / 2
    # use t as the unique key to match bv and mobmat

    if memory_dict is None:
        memory_dict = {}
        memory_dict["1"] = {"bv": [], "alpha": [], "Q": [], "C": []}
        memory_dict["2"] = {"bv": [], "alpha": [], "Q": [], "C": []}
        memory_dict["3"] = {"bv": [], "alpha": [], "Q": [], "C": []}
        memory_dict["4"] = {"bv": [], "alpha": [], "Q": [], "C": []}

    X = np.transpose(np.vstack((mean_t, mean_x)))[flight_index]
    Y = mean_y[flight_index]
    result1 = SOGP(
        X,
        Y,
        sigma2,
        tol,
        d,
        pars,
        memory_dict["1"]["Q"],
        memory_dict["1"]["C"],
        memory_dict["1"]["alpha"],
        memory_dict["1"]["bv"],
    )
    bv1 = result1["bv"]
    t1 = np.array([bv1[j][0] for j in range(len(bv1))])

    X = np.transpose(np.vstack((mean_t, mean_x)))[pause_index]
    Y = mean_y[pause_index]
    result2 = SOGP(
        X,
        Y,
        sigma2,
        tol,
        d,
        pars,
        memory_dict["2"]["Q"],
        memory_dict["2"]["C"],
        memory_dict["2"]["alpha"],
        memory_dict["2"]["bv"],
    )
    bv2 = result2["bv"]
    t2 = np.array([bv2[j][0] for j in range(len(bv2))])

    X = np.transpose(np.vstack((mean_t, mean_y)))[flight_index]
    Y = mean_x[flight_index]
    result3 = SOGP(
        X,
        Y,
        sigma2,
        tol,
        d,
        pars,
        memory_dict["3"]["Q"],
        memory_dict["3"]["C"],
        memory_dict["3"]["alpha"],
        memory_dict["3"]["bv"],
    )
    bv3 = result3["bv"]
    t3 = np.array([bv3[j][0] for j in range(len(bv3))])

    X = np.transpose(np.vstack((mean_t, mean_y)))[pause_index]
    Y = mean_x[pause_index]
    result4 = SOGP(
        X,
        Y,
        sigma2,
        tol,
        d,
        pars,
        memory_dict["4"]["Q"],
        memory_dict["4"]["C"],
        memory_dict["4"]["alpha"],
        memory_dict["4"]["bv"],
    )
    bv4 = result4["bv"]
    t4 = np.array([bv4[j][0] for j in range(len(bv4))])

    unique_t = np.unique(
        np.concatenate((np.concatenate((t1, t2)), np.concatenate((t3, t4))))
    )

    if bv_set is not None:
        all_candidates = np.vstack((bv_set, MobMat))
        all_t = (all_candidates[:, 3] + all_candidates[:, 6]) / 2
    else:
        all_candidates = MobMat
        all_t = mean_t

    index = []
    for j, time in enumerate(all_t):
        if np.any(unique_t == time):
            index.append(j)
    index_arr = np.array(index)

    bv_set = all_candidates[index_arr, :]
    memory_dict["1"] = result1
    memory_dict["2"] = result2
    memory_dict["3"] = result3
    memory_dict["4"] = result4

    return {"BV_set": bv_set, "memory_dict": memory_dict}
