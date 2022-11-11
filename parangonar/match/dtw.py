#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains dynamic time warping methods.
"""

import numpy as np

def element_of_metric(vec1, vec2):
    """
    metric that evaluates occurence of vec2 (scalar) in vec1 (vector n-dim)
    """
    return 1 - np.sum(vec2 == vec1)


class DynamicTimeWarping(object):
    """
    pure python vanilla Dynamic Time Warping
    """
    
    def __init__(self, 
                 metric=element_of_metric):
        self.metric = metric

    def __call__(self, X, Y, return_path=True,
                 return_cost_matrix=False):

        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)
        # Compute pairwise distance
        D = cdist(X, Y, self.metric)
        # Compute accumulated cost matrix
        dtwd_matrix = dtw_dmatrix_from_pairwise_dmatrix(D)
        dtwd_distance = dtwd_matrix[-1, -1]

        # Output
        out = (dtwd_distance, )

        if return_path:
            # Compute alignment path
            path = dtw_backtracking(dtwd_matrix)
            out += (path,)
        if return_cost_matrix:
            out += (dtwd_matrix, )
        return out
    
    
def dtw_backtracking(dtwd):
    """
    Decode path from the accumulated dtw cost matrix.

    Parameters
    ----------
    dtwd : np.ndarray
        Accumulated cost matrix (computed with 
        `dtw_dmatrix_from_pairwise_dmatrix`)
    
    Returns
    -------
    path : np.ndarray
       A 2D array of size (n_steps, 2), where i-th row has elements 
       (i_m, i_n) where i_m represents the index in the input array
       and i_n represents the corresponding index in the reference array.
    """
    
    N = dtwd.shape[0]
    M = dtwd.shape[1]

    n = N - 1
    m = M - 1

    step = [n, m]

    path = [step]

    # Initialize step choices
    choices = np.zeros((3, 2), dtype=int)
    # Initialize a vector for candidate distances
    dtwd_candidates = np.zeros(3, dtype=float)
    # initialize boolean variables for stopping decoding
    crit = True

    while crit:

        if n == 0:
            # next point in the path
            m = m - 1

        elif m == 0:
            # next point in the path
            n = n - 1

        else:
            # step sizes
            choices[0, 0] = n - 1
            choices[0, 1] = m - 1
            choices[1, 0] = n - 1
            choices[1, 1] = m
            choices[2, 0] = n
            choices[2, 1] = m - 1

            # accumulated distance from the previous step
            # to the next
            dtwd_candidates[0] = dtwd[n - 1, m - 1]
            dtwd_candidates[1] = dtwd[n - 1, m]
            dtwd_candidates[2] = dtwd[n, m - 1]

            # select the best candidate
            p_l_i = np.argmin(dtwd_candidates)

            # update next indices
            n = choices[p_l_i, 0]
            m = choices[p_l_i, 1]

        step = [n, m]
        # append next step to the path
        path.append(step)

        if n == 0 and m == 0:
            crit = False

    return np.array(path[::-1], dtype=int)


def cdist(arr1, arr2, metric):
    """
    compute array of pairwise distances between 
    the elements of two arrays given a metric
    
    Parameters
    ----------
    arr1: numpy nd array
    
    arr2: numpy nd array
    
    metric> callable
        a metric function
    
    Returns
    -------
    
    pdist_array: numpy 2d array
        array of pairwise distances
    """
    pdist_array = np.ones((arr1.shape[0],arr2.shape[0]))*np.inf
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[0]):
            pdist_array[i, j] = metric(arr1[i], arr2[j])
    return pdist_array


def dtw_dmatrix_from_pairwise_dmatrix(D):
    """
    compute dynamic time warping cost matrix 
    from a pairwise distance matrix

    Parameters
    ----------
    D : double array
        Pairwise distance matrix (computed e.g., with `cdist`).

    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    """
    # Initialize arrays and helper variables
    M = D.shape[0]
    N = D.shape[1]
    # the dtwd distance matrix is initialized with INFINITY
    dtwd = np.ones((M + 1, N + 1),dtype=float) * np.inf
    
    # Compute the distance iteratively
    dtwd[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            c = D[i - 1, j - 1]
            insertion = dtwd[i - 1, j]
            deletion = dtwd[i, j - 1]
            match = dtwd[i - 1, j - 1]
            dtwd[i, j] = c + min((insertion, deletion, match))

    return (dtwd[1:, 1:])
