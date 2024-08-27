#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of Needleman Wunsch and derived algorithms
"""
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import numba
from numba import jit




class NWDistanceMatrix(object):
    """
    An object to hold the accumulated cost matrix for the
    Needleman Wunsch algorithm

    Parameters
    ----------
    gamma : float
        Gap parameter (for initializing the matrix)
    """
    def __init__(self, gamma = 0.1):
        self.gamma = float(gamma)
        self.val_dict = defaultdict(lambda: np.inf)
        self.indices_dict = dict()
        self.xdim = 0
        self.ydim = 0

    @property
    def nw_distance(self):
        """The total accumulated cost of alignment
        """
        return self[self.xdim, self.ydim]

    def __getitem__(self, indices):
        i, j = indices
        if i == 0 and j != 0:
            return self.gamma * j
        elif j == 0 and i != 0:
            return self.gamma * i
        elif i == j == 0:
            return 0
        else:
            return self.val_dict[i, j]

    def __setitem__(self, indices, values):
        if indices[0] > self.xdim:
            self.xdim = indices[0]
        if indices[1] > self.ydim:
            self.ydim = indices[1]
        self.val_dict[indices] = values[0]
        self.indices_dict[indices] = (values[1], values[2])

    def path_step(self, i, j):
        """
        Get the indices and type of message for backtracking
        """
        if i == j == 0:
            return (0, 0), 0
        elif i == 0 and j > 0:
            return (0, j - 1), 1
        elif j == 0 and i > 0:
            return (i - 1, 0), 2
        else:
            return self.indices_dict[i, j]

    @property
    def cost_matrix(self):
        """
        The alignment cost matrix
        """
        cost_matrix = np.ones((self.xdim + 1, self.ydim + 1)) * np.inf
        for i in range(self.xdim + 1):
            for j in range(self.ydim + 1):
                cost_matrix[i, j] = self[i, j]
        return cost_matrix


class NeedlemanWunsch(object):
    """
    Needleman-Wunsch algorithm for aligning sequences.
    """
    def __init__(self, 
                 metric = euclidean, 
                 gamma = 0.1):
        self.metric = metric
        self.gamma = gamma

    def __call__(self, X, Y, return_path=True,
                 window=None,
                 return_cost_matrix=False):
        X = X.astype(float)
        Y = Y.astype(float)
        len_X, len_Y = len(X), len(Y)
        if window is None:
            window = [(i, j) for i in range(len_X) for j in range(len_Y)]
        window = ((i + 1, j + 1) for i, j in window)

        nw_matrix = NWDistanceMatrix(self.gamma)

        for i, j in window:
            dt = self.metric(X[i-1], Y[j-1])

            nw_matrix[i, j] = min(
                (nw_matrix[i-1, j] + self.gamma, (i-1, j), 2),
                (nw_matrix[i, j-1] + self.gamma, (i, j-1), 1),
                (nw_matrix[i-1, j-1] + dt, (i-1, j-1), 0),
                key=lambda a: a[0]
            )

        nw_distance = nw_matrix[len_X, len_Y]

        out = (nw_distance, )

        if return_path:
            path = self.backtracking(nw_matrix)
            out += (path, )

        if return_cost_matrix:
            out += (nw_matrix, )

        return out

    def backtracking(self, nw_matrix):
        path = []
        i, j = nw_matrix.xdim, nw_matrix.ydim
        while not (i == j == 0):
            _, align = nw_matrix.path_step(i, j)
            if align == 0:
                path.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif align == 1:
                path.append((-1, j - 1))
                j -= 1
            elif align == 2:
                path.append((i - 1, -1))
                i -= 1
        path.reverse()
        return np.array(path, dtype=int)


# alias
NW = NeedlemanWunsch


class NeedlemanWunschDynamicTimeWarping(NeedlemanWunsch):
    """
    Needleman-Wunsch Dynamic Time Warping
    """
    def __init__(self, 
                 metric = euclidean, 
                 gamma = 0.1):
        super().__init__(metric=metric,
                         gamma=gamma)

    def __call__(self, X, Y, return_path=True,
                 window=None,
                 return_cost_matrix=False):

        X = X.astype(float)
        Y = Y.astype(float)
        len_X, len_Y = len(X), len(Y)
        if window is None:
            window = [(i, j) for i in range(len_X) for j in range(len_Y)]
        window = ((i + 1, j + 1) for i, j in window)

        nw_matrix = NWDistanceMatrix(self.gamma)

        pairwise_distance = defaultdict(lambda: float('inf'))

        pairwise_distance[0, 0] = 0

        for i, j in window:
            pairwise_distance[i, j] = self.metric(X[i-1], Y[j-1])

            nw_matrix[i, j] = min(
                (nw_matrix[i-1, j] + self.gamma, (i-1, j), 2),
                (nw_matrix[i, j-1] + self.gamma, (i, j-1), 1),
                (nw_matrix[i-1, j-1] + pairwise_distance[i, j], (i-1, j-1), 0),
                (nw_matrix[i-1, j-2] + pairwise_distance[i, j-1] +
                 pairwise_distance[i, j], (i, j-1), 0),
                (nw_matrix[i-2, j-1] + pairwise_distance[i-1, j] +
                 pairwise_distance[i, j], (i-1, j), 0),
                key=lambda a: a[0]
            )

        nw_distance = nw_matrix[len_X, len_Y]

        out = (nw_distance, )

        if return_path:
            path = self.backtracking(nw_matrix)
            out += (path, )

        if return_cost_matrix:
            out += (nw_matrix, )

        return out

    def backtracking(self, nw_matrix):
        path = []
        i, j = nw_matrix.xdim, nw_matrix.ydim

        while not (i == j == 0):
            ij, align = nw_matrix.path_step(i, j)
            if align == 0:
                path.append((i - 1, j - 1))
                i, j = ij
            elif align == 1:
                path.append((-1, j - 1))
                j -= 1
            elif align == 2:
                path.append((i - 1, -1))
                i -= 1
        path.reverse()
        return np.array(path, dtype=int)

# alias
NW_DTW = NeedlemanWunschDynamicTimeWarping

class WeightedNeedlemanWunschTimeWarping(object):
    """
    General Needleman-Wunsch Time Warping algorithm for aligning sequences.
    
        Parameters
    ----------
    directional_weights: np.ndarray
        weights associated with each of the three possible steps
    directions : np.ndarray
        directions.
    directional_distances: list 
        for each direction, the pairwise distance idx which accumulate 
    directional_penalties: np.ndarray
        for each direction, the sum of penalites which accumulate
    """
    def __init__(self, 
                 directions = np.array([[1, 0],[1, 1],[0, 1]]),
                 directional_penalties = np.array([1, 0, 1]),
                 directional_distances = [np.array([]), 
                                          np.array([[0, 0]]),
                                          np.array([])],
                 directional_weights = np.array([1, 1, 1]),
                 metric = euclidean, 
                 gamma = 1):
        self.metric = metric
        self.gamma = gamma
        self.directional_weights = directional_weights
        self.directions = directions
        self.directional_penalties = directional_penalties
        self.directional_distances = directional_distances

    def __call__(self, X, Y, 
                 return_matrices=True,
                 return_cost=False):
        
        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)
        
        #pairwise distances
        pwD = cdist(X,Y,"euclidean")

        cost, path, B = weighted_nwdtw_forward_and_backward(pwD,
                                            self.directional_weights,
                                            self.directions,
                                            self.directional_penalties,
                                            self.directional_distances,
                                            self.gamma)
        out = (path,)
        if return_matrices:
            out += (cost, B)
        if return_cost:
            out += (cost[-1,-1])
        return out

    
# @jit(nopython=True)
def weighted_nwdtw_forward_and_backward(pwD, 
                                      directional_weights = np.array([1, 1, 1]),
                                      directions = np.array([[1, 0],[1, 1],[0, 1]]),
                                      directional_penalties = np.array([1, 0, 1]),
                                      directional_distances = [np.array([]), 
                                                               np.array([[0, 0]]),
                                                               np.array([])],
                                      gamma = 1):
    """
    compute needleman-wunsch dynamic time warping cost matrix
    and backtracking path
    from weighted directions and
    a pairwise distance matrix

    Parameters
    ----------
    D : np.ndarray
        Pairwise distance matrix (computed e.g., with `cdist`).
    directional_weights: np.ndarray
        weights associated with each of the three possible steps
    directions : np.ndarray
        directions.
    directional_distances: list 
        for each direction, the pairwise distance idx which accumulate 
    directional_penalties: np.ndarray
        for each direction, the sum of penalites which accumulate
    gamma: float
        penalty value

    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    path: np.ndarray
        backtracked path
    """
    # Initialize arrays and helper variables
    M = pwD.shape[0]
    N = pwD.shape[1]

    
    # the NW distance matrix is initialized with INFINITY
    D = np.ones((M + 1, N + 1),dtype=float) * np.inf
    # Compute the borders of D
    D[0,:] = np.arange(0, N + 1) * gamma
    D[:,0] = np.arange(0, M + 1) * gamma
    # Backtracking
    B = np.ones((M, N),dtype=np.int8) * -1
    
    # Compute the distance iteratively
    D[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            mincost = np.inf
            minidx = -1
            bestiprev = -1
            bestjprev = -1
            for directionsidx, direction in enumerate(directions):
                (istep, jstep) = direction
                previ = i - istep
                prevj = j - jstep
                if previ >= 0 and prevj >= 0: 
                    prevCost = D[previ, prevj]
                    penaltyCost = directional_penalties[directionsidx] * gamma           
                    distanceCost = np.sum([pwD[i - dis[0] - 1, j - dis[1] - 1] for dis in directional_distances[directionsidx]])
                    cost = directional_weights[directionsidx] * (prevCost + penaltyCost + distanceCost)                    
                    if cost < mincost:
                        mincost = cost
                        minidx = directionsidx
                        bestiprev = previ
                        bestjprev = prevj
                        
            D[i, j] = mincost
            B[i - 1, j - 1] = minidx

    # return (dtwd[1:, 1:])
    n = N - 1
    m = M - 1
    step = [m, n]
    path = [step]
    # initialize boolean variables for stopping decoding
    while n > 0 or m > 0:
       
        backtracking_pointer = B[m, n]
        bt_vector = directions[backtracking_pointer]
        m -= bt_vector[0]
        n -= bt_vector[1]
        step = [m, n]
        # append next step to the path
        path.append(step)

    output_path = np.array(path, dtype=np.int32)[::-1]
    output_D = D[1:, 1:]
    return  output_D, output_path, B

# alias
WNWTW = WeightedNeedlemanWunschTimeWarping


class OriginalNeedlemanWunsch(object):
    """
    Origianl Needleman-Wunsch (and Smith-Waterman) algorithm for aligning (sub-)sequences.
    
        Parameters
    ----------
    gamma_penalty: float
        penalty value
    gamma_match: float
        matching value
    threshold: float
        threshold distance between match and penalty
    """
    def __init__(self,
                 metric = euclidean, 
                 gamma_penalty = -1.0,
                 gamma_match = 1.0,
                 threshold = 1.0,
                 smith_waterman = False):
        self.metric = metric
        self.gamma_penalty = gamma_penalty
        self.gamma_match = gamma_match
        self.threshold = threshold
        self.smith_waterman = smith_waterman

    def __call__(self, X, Y, 
                 return_matrices=True,
                 return_cost=False):
        
        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)
        
        #pairwise distances
        pwD = cdist(X,Y,"euclidean")

        cost, path, B = onw_forward_and_backward(pwD,
                                            self.gamma_penalty,
                                            self.gamma_match,
                                            self.threshold,
                                            self.smith_waterman)
        out = (path,)
        if return_matrices:
            out += (cost, B)
        if return_cost:
            out += (cost[-1,-1])
        return out

    
# @jit(nopython=True)
def onw_forward_and_backward(pwD, 
                            gamma_penalty = -1.0,
                            gamma_match = 1.0,
                            threshold = 1.0,
                            smith_waterman = False):
    """
    compute needleman-wunsch cost matrix
    and backtracking path
    from weighted directions and
    a pairwise distance matrix

    Parameters
    ----------
    D : np.ndarray
        Pairwise distance matrix (computed e.g., with `cdist`).
    gamma_penalty: float
        penalty value
    gamma_match: float
        matching value
    threshold: float
        threshold distance between match and penalty

    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    path: np.ndarray
        backtracked path
    """
    # Initialize arrays and helper variables
    M = pwD.shape[0]
    N = pwD.shape[1]

    
    # the NW distance matrix is initialized with zero
    D = np.zeros((M + 1, N + 1),dtype=float)
    # Compute the borders of D
    if not smith_waterman:
        D[0,:] = np.arange(0, N + 1) * gamma_penalty
        D[:,0] = np.arange(0, M + 1) * gamma_penalty
    # Backtracking
    B = np.ones((M, N),dtype=np.int8) * -1

    directions = np.array([[1, 0],[1, 1],[0, 1]])
    directional_penalties = np.array([1,0,1])
    directional_distances = np.array([0,1,0])
    # Compute the distance iteratively
    D[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            maxGain = -np.inf
            maxidx = -1
            bestiprev = -1
            bestjprev = -1
            for directionsidx, direction in enumerate(directions):
                (istep, jstep) = direction
                previ = i - istep
                prevj = j - jstep
                if previ >= 0 and prevj >= 0: 
                    distanceGain = directional_distances[directionsidx] * (gamma_match if pwD[i - 1, j - 1] < threshold else gamma_penalty)
                    prevGain = D[previ, prevj] 
                    penaltyGain = directional_penalties[directionsidx] * gamma_penalty           
                    Gain = prevGain + penaltyGain + distanceGain              
                    if Gain > maxGain:
                        maxGain = Gain
                        maxidx = directionsidx
            if smith_waterman:            
                D[i, j] = np.max((maxGain, 0))
            else:
                D[i, j] = maxGain
            B[i - 1, j - 1] = maxidx


    print(D)
    print(B)
    # return (dtwd[1:, 1:])
    n = N - 1
    m = M - 1
    step = [m, n]
    path = [step]
    # initialize boolean variables for stopping decoding
    while n > 0 or m > 0:
       
        backtracking_pointer = B[m, n]
        bt_vector = directions[backtracking_pointer]
        m -= bt_vector[0]
        n -= bt_vector[1]
        step = [m, n]
        # append next step to the path
        path.append(step)

    output_path = np.array(path, dtype=np.int32)[::-1]
    output_D = D[1:, 1:]
    return  output_D, output_path, B

# alias
ONW = OriginalNeedlemanWunsch

if __name__ == "__main__":


    A = np.array([[1,2,3,4,1,2,3,4,5,6]]).T
    B = np.array([[1,2,3,4,5,6]]).T

    # Like NWTW in Grachten 
    # matcher = WNWTW(directions = np.array([[1, 0],[1, 1],[0, 1],[2, 1],[1, 2]]),
    #              directional_penalties = np.array([1, 0, 1, 0, 0]),
    #              directional_distances = [np.array([]), 
    #                                       np.array([[0, 0]]),
    #                                       np.array([]),
    #                                       np.array([[0, 0],[1, 0]]),
    #                                       np.array([[0, 0],[0, 1]])],
    #              directional_weights = np.array([1, 1, 1]),
    #              gamma = 0.5)

    # allowing diagonal warping at any angle
    # matcher = WNWTW(directions = np.array([[1, 0],[1, 1],[0, 1],[1, 0],[0, 1]]),
    #              directional_penalties = np.array([1, 0, 1, 0, 0]),
    #              directional_distances = [np.array([]), 
    #                                       np.array([[0, 0]]),
    #                                       np.array([]),
    #                                       np.array([[0, 0]]),
    #                                       np.array([[0, 0]])],
    #              directional_weights = np.array([1, 2, 1, 1, 1]),
    #              gamma = 0.5)

    matcher = ONW(smith_waterman = True,
                  gamma_penalty = -0.5)

    path, cost, back = matcher(A, B, return_matrices = True)