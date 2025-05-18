#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of Needleman Wunsch and derived algorithms
"""
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean, cdist
from numba import jit
# helpers and metrics
from .metrics import (cdist_local, 
                      element_of_set_metric,
                      bounded_recursion,
                      onset_pitch_duration_metric)

class NWDistanceMatrix(object):
    """
    An object to hold the accumulated cost matrix for the
    Needleman Wunsch algorithm

    Parameters
    ----------
    gamma : float
        Gap parameter (for initializing the matrix)
    """

    def __init__(self, gamma=0.1):
        self.gamma = float(gamma)
        self.val_dict = defaultdict(lambda: np.inf)
        self.indices_dict = dict()
        self.xdim = 0
        self.ydim = 0

    @property
    def nw_distance(self):
        """The total accumulated cost of alignment"""
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

    def __init__(self, metric=euclidean, gamma=0.1):
        self.metric = metric
        self.gamma = gamma

    def __call__(self, X, Y, return_path=True, window=None, return_cost_matrix=False):
        X = X.astype(float)
        Y = Y.astype(float)
        len_X, len_Y = len(X), len(Y)
        if window is None:
            window = [(i, j) for i in range(len_X) for j in range(len_Y)]
        window = ((i + 1, j + 1) for i, j in window)

        nw_matrix = NWDistanceMatrix(self.gamma)

        for i, j in window:
            dt = self.metric(X[i - 1], Y[j - 1])

            nw_matrix[i, j] = min(
                (nw_matrix[i - 1, j] + self.gamma, (i - 1, j), 2),
                (nw_matrix[i, j - 1] + self.gamma, (i, j - 1), 1),
                (nw_matrix[i - 1, j - 1] + dt, (i - 1, j - 1), 0),
                key=lambda a: a[0],
            )

        nw_distance = nw_matrix[len_X, len_Y]

        out = (nw_distance,)

        if return_path:
            path = self.backtracking(nw_matrix)
            out += (path,)

        if return_cost_matrix:
            out += (nw_matrix,)

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
    Needleman-Wunsch Dynamic Time Warping as introduced by Grachten et al.
    """

    def __init__(self, 
                 metric=euclidean, 
                 gamma=0.1):
        super().__init__(metric=metric, gamma=gamma)

    def __call__(self, X, Y, return_path=True, window=None, return_cost_matrix=False):
        X = X.astype(float)
        Y = Y.astype(float)
        len_X, len_Y = len(X), len(Y)
        if window is None:
            window = [(i, j) for i in range(len_X) for j in range(len_Y)]
        window = ((i + 1, j + 1) for i, j in window)

        nw_matrix = NWDistanceMatrix(self.gamma)

        pairwise_distance = defaultdict(lambda: float("inf"))

        pairwise_distance[0, 0] = 0

        for i, j in window:
            pairwise_distance[i, j] = self.metric(X[i - 1], Y[j - 1])

            nw_matrix[i, j] = min(
                (nw_matrix[i - 1, j] + self.gamma, (i - 1, j), 2),
                (nw_matrix[i, j - 1] + self.gamma, (i, j - 1), 1),
                (nw_matrix[i - 1, j - 1] + pairwise_distance[i, j], (i - 1, j - 1), 0),
                (
                    nw_matrix[i - 1, j - 2]
                    + pairwise_distance[i, j - 1]
                    + pairwise_distance[i, j],
                    (i, j - 1),
                    0,
                ),
                (
                    nw_matrix[i - 2, j - 1]
                    + pairwise_distance[i - 1, j]
                    + pairwise_distance[i, j],
                    (i - 1, j),
                    0,
                ),
                key=lambda a: a[0],
            )

        nw_distance = nw_matrix[len_X, len_Y]

        out = (nw_distance,)

        if return_path:
            path = self.backtracking(nw_matrix)
            out += (path,)

        if return_cost_matrix:
            out += (nw_matrix,)

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
    metric: callable
        the pairwise distance metric to be used between the input
    cdist_fun: callable
        the pairwise distance to be used (scipy cdist or local cdist)
    """

    def __init__(
        self,
        directions=np.array([[1, 0], [1, 1], [0, 1]]),
        directional_penalties=np.array([1, 0, 1]),
        directional_distances=[np.array([]), np.array([[0, 0]]), np.array([])],
        directional_weights=np.array([1, 1, 1]),
        metric=euclidean,
        cdist_fun=cdist,
        gamma=1,
    ):
        self.metric = metric
        self.cdist_fun = cdist_fun
        self.gamma = gamma
        self.directional_weights = directional_weights
        self.directions = directions
        self.directional_penalties = directional_penalties
        self.directional_distances = directional_distances
        

    def __call__(self, X, Y, return_matrices=True, return_cost=False):
        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)

        # pairwise distances
        pwD = self.cdist_fun(X, Y, self.metric)

        cost, path, B = weighted_nwdtw_forward_and_backward(
            pwD,
            self.directional_weights,
            self.directions,
            self.directional_penalties,
            self.directional_distances,
            self.gamma,
        )
        out = (path,)
        if return_matrices:
            out += (cost, B)
        if return_cost:
            out += cost[-1, -1]
        return out


def weighted_nwdtw_forward_and_backward(
    pwD,
    directional_weights=np.array([1, 1, 1]),
    directions=np.array([[1, 0], [1, 1], [0, 1]]),
    directional_penalties=np.array([1, 0, 1]),
    directional_distances=[np.array([]), np.array([[0, 0]]), np.array([])],
    gamma=1,
):
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
    D = np.ones((M + 1, N + 1), dtype=float) * np.inf
    # Compute the borders of D
    D[0, :] = np.arange(0, N + 1) * gamma
    D[:, 0] = np.arange(0, M + 1) * gamma
    # Backtracking
    B = np.ones((M, N), dtype=np.int8) * -1

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
                    distanceCost = np.sum(
                        [
                            pwD[i - dis[0] - 1, j - dis[1] - 1]
                            for dis in directional_distances[directionsidx]
                        ]
                    )
                    cost = directional_weights[directionsidx] * (
                        prevCost + penaltyCost + distanceCost
                    )
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
    return output_D, output_path, B


# alias
WNWTW = WeightedNeedlemanWunschTimeWarping


class OriginalNeedlemanWunsch(object):
    """
    Original Needleman-Wunsch (and Smith-Waterman) algorithm for aligning (sub-)sequences.

    Parameters
    ----------
    metric: callable
        the pairwise distance metric to be used between the input
    cdist_fun: callable
        the pairwise distance to be used (scipy cdist or local cdist)
    gamma_penalty: float
        penalty value
    gamma_match: float
        matching value
    gap_penalty: float
        gap penalty value
    threshold: float
        threshold distance between match and penalty
    """

    def __init__(
        self,
        metric=euclidean,
        cdist_fun=cdist,
        gamma_penalty=-1.0,
        gamma_match=1.0,
        gap_penalty=-0.5,
        threshold=1.0,
        smith_waterman=False,
    ):
        self.metric = metric
        self.cdist_fun = cdist_fun
        self.gamma_penalty = gamma_penalty
        self.gamma_match = gamma_match
        self.gap_penalty = gap_penalty
        self.threshold = threshold
        self.smith_waterman = smith_waterman

    def __call__(self, X, Y, return_matrices=True, return_cost=False):
        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)

        # pairwise distances
        pwD = self.cdist_fun(X, Y, self.metric)

        cost, path, B = onw_forward_and_backward(
            pwD,
            self.gamma_penalty,
            self.gamma_match,
            self.threshold,
            self.gap_penalty,
            self.smith_waterman,
        )
        out = (path,)
        if return_matrices:
            out += (cost, B)
        if return_cost:
            out += cost[-1, -1]
        return out


def onw_forward_and_backward(
    pwD,
    gamma_penalty=-1.0,
    gamma_match=1.0,
    threshold=1.0,
    gap_penalty=0.5,
    smith_waterman=False,
):
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
    gap_penalty: float
        gap penalty value
    smith-waterman: bool
        compute smith-waterman local alignment

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
    D = np.zeros((M + 1, N + 1), dtype=float)
    # Compute the borders of D
    if not smith_waterman:
        D[0, :] = np.arange(0, N + 1) * gamma_penalty
        D[:, 0] = np.arange(0, M + 1) * gamma_penalty
    # Backtracking
    B = np.ones((M, N), dtype=np.int8) * -1

    directions = np.array([[1, 0], [1, 1], [0, 1]])
    directional_penalties = np.array([1, 0, 1])
    directional_distances = np.array([0, 1, 0])
    # Compute the distance iteratively
    D[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            maxGain = -np.inf
            maxidx = -1
            for directionsidx, direction in enumerate(directions):
                (istep, jstep) = direction
                previ = i - istep
                prevj = j - jstep
                if previ >= 0 and prevj >= 0:
                    distanceGain = directional_distances[directionsidx] * (
                        gamma_match if pwD[i - 1, j - 1] < threshold else gamma_penalty
                    )
                    prevGain = D[previ, prevj]
                    penaltyGain = directional_penalties[directionsidx] * gap_penalty
                    Gain = prevGain + penaltyGain + distanceGain
                    if Gain > maxGain:
                        maxGain = Gain
                        maxidx = directionsidx
            if smith_waterman:
                D[i, j] = np.max((maxGain, 0))
            else:
                D[i, j] = maxGain
            B[i - 1, j - 1] = maxidx

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
    return output_D, output_path, B


# alias
ONW = OriginalNeedlemanWunsch


class BoundedSmithWaterman(object):
    """
    Bounded Smith-Waterman algorithm for aligning (sub-)sequences.
    
        Parameters
    ----------
    gamma_penalty: float
        penalty value
    gamma_match: float
        matching value
    threshold: float
        threshold distance between match and penalty
    metric: callable
        the pairwise distance metric to be used between the input
    cdist_fun: callable
        the pairwise distance to be used (scipy cdist or local cdist)
    """
    def __init__(self,
                 gamma_penalty = -1.0,
                 gamma_match = 1.0,
                 threshold = 1.0,
                 metric=element_of_set_metric,
                 cdist_fun=cdist_local,
                 directions = np.array([[1, 0],[1, 1],[0, 1]]),
                 directional_penalties = np.array([0,0,0]),
                 directional_distances = np.array([1,1,1]),
                 gain_min_val = 0, 
                 gain_max_val = 10, 
                 gain_slope_at_min = 1
                 ):
        self.metric = metric
        self.cdist_fun = cdist_fun
        self.gamma_penalty = gamma_penalty
        self.gamma_match = gamma_match
        self.threshold = threshold
        self.directions = directions
        self.directional_penalties = directional_penalties
        self.directional_distances = directional_distances
        self.gain_min_val = gain_min_val
        self.gain_max_val = gain_max_val
        self.gain_slope_at_min = gain_slope_at_min

    def __call__(self, X, Y):
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
        
        # pairwise distances
        pwD = self.cdist_fun(X,Y,self.metric)

        cost, B = bsw_forward(pwD,
                            self.gamma_penalty,
                            self.gamma_match,
                            self.threshold,
                            self.directions,
                            self.directional_penalties,
                            self.directional_distances,
                            self.gain_min_val,
                            self.gain_max_val,
                            self.gain_slope_at_min)
        out = (cost, B)
        return out
    
    def from_similarity_matrix(self,pwD):
        cost, B = bsw_forward(pwD,
                            self.gamma_penalty,
                            self.gamma_match,
                            self.threshold,
                            self.directions,
                            self.directional_penalties,
                            self.directional_distances,
                            self.gain_min_val,
                            self.gain_max_val,
                            self.gain_slope_at_min)
        out = (cost, B)
        return out


@jit(nopython=True)
def bsw_forward(pwD, 
                gamma_penalty = -1.0,
                gamma_match = 1.0,
                threshold = 1.0,
                directions = np.array([[1, 0],[1, 1],[0, 1]]),
                directional_penalties = np.array([0,0,0]),
                directional_distances = np.array([1,1,1]),
                gain_min_val = 0, 
                gain_max_val = 10, 
                gain_slope_at_min = 1):
    """
    compute needleman-wunsch cost matrix
    and backtracking path
    from weighted directions and
    a pairwise distance matrix

    Parameters
    ----------
    pwD : np.ndarray
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
    # the SW distance matrix is initialized with zero
    D = np.zeros((M + 1, N + 1),dtype=float)
    # Backtracking
    B = np.ones((M, N),dtype=np.int8) * -1
    lower_bound = 0 
    max_steps_below_1 = 20
    # Compute the distance iteratively
    D[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            maxGain = -np.inf
            maxidx = -1
            maxPrevGain = -np.inf
            gamma = (gamma_match if pwD[i - 1, j - 1] < threshold else gamma_penalty)
            for directionsidx, direction in enumerate(directions):
                (istep, jstep) = direction
                previ = i - istep
                prevj = j - jstep
                if previ >= 0 and prevj >= 0: 
                    # match gain or loss scaled by distance weight accumulated for this direction
                    distanceGain = directional_distances[directionsidx] * gamma
                    prevGain = D[previ, prevj] 
                    # penalties incured by this direction
                    penaltyGain = directional_penalties[directionsidx] * gamma_penalty           
                    Gain = prevGain + distanceGain + penaltyGain              
                    if Gain > maxGain:
                        maxGain = Gain
                        maxidx = directionsidx
                        maxPrevGain = prevGain

            if maxGain - maxPrevGain >= 0:          
                D[i, j] = bounded_recursion(maxPrevGain, 
                                            min_val = gain_min_val, 
                                            max_val = gain_max_val, 
                                            slope_at_min = gain_slope_at_min)
            else:

                if maxPrevGain > lower_bound and maxGain < lower_bound:
                    # don't move vertically more than ten steps without match
                    # if maxPrevGain < 0.5 ** max_steps_below_1 and maxidx in [1,2]:
                    #     D[i, j] = lower_bound
                    # else:
                    D[i, j] = maxPrevGain * 0.5
                else:
                    D[i, j] = max(maxGain, lower_bound)
            B[i - 1, j - 1] = maxidx 


    output_D = D[1:, 1:]
    return  output_D, B

# alias
BSW = BoundedSmithWaterman

class SubPartDynamicProgramming(object):
    """
    Monophonic Subpart Alignment with Dynamic Programming.

    Parameters
    ----------
    weights : np.ndarray
        three weights associated with onset, dur, pitch distances
    tempo_factor : float
        moving average recursion factor for tempo update
    """

    def __init__(
        self,
        weights = np.array([1,0.5,2]),
        tempo_factor = 0.1
    ):
        self.weights = weights
        self.tempo_factor = tempo_factor

    def __call__(self, X, Y):
        """
        Parameters
        ----------
        X : np.ndarray
            performance note array.
        Y : np.ndarray
            score note array.
        """

        out = subpart_DP_forward_and_backward(X, Y, 
                                              self.weights, 
                                              self.tempo_factor)
        return out

def subpart_DP_forward_and_backward(
    pna, sna, 
    weights = np.array([1,0.5,2]),
    tempo_factor = 0.1
):
    """
    Parameters
    ----------
    pna : np.ndarray
        performance note array.
    sna : np.ndarray
        score note array.
    weights : np.ndarray
        three weights associated with onset, dur, pitch distances
    tempo_factor : float
        moving average recursion factor for tempo update

    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    path: np.ndarray
        backtracked path
    """

    # Initialize arrays and helper variables
    M = pna.shape[0]
    N = sna.shape[0]
    
    # the cost vector is initialized with INFINITY
    C = np.ones((N + 1), dtype=float) * np.inf
    C[0] = 0
    # cost matrix just for debugging
    D = np.ones((M, N), dtype=float) * np.inf
    # guess the initial tempo from the whole length
    init_tempo = (max(pna["onset_sec"]) - min(pna["onset_sec"])) / (max(sna["onset_beat"]) - min(sna["onset_beat"]))  
    # Tempo vector in [sec / beat]
    T = np.full((N + 1), init_tempo, dtype=float)
    # performance index of the score 
    perf_P = np.zeros(N+1, dtype=int)
    # backtracking
    B = np.ones((M, N), dtype=int) * -1

    # extend the inputs with a dummy start
    dummy_pna = np.copy(pna[0:1])
    dummy_pna["onset_sec"] -= init_tempo
    pna = np.concatenate((dummy_pna,pna))
    dummy_sna = np.copy(sna[0:1])
    dummy_sna["onset_beat"] -= 1
    sna = np.concatenate((dummy_sna,sna))

    # Compute the cost iteratively
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            
            prevtempo = T[j - 1] # tempo at last score index
            prev_perf_idx = perf_P[j - 1]
            local_dist, new_tempo = onset_pitch_duration_metric(
                pitch_s=sna[j]["pitch"],
                pitch_p=pna[i]["pitch"],
                onset_s=sna[j]["onset_beat"],
                onset_p=pna[i]["onset_sec"],
                prev_onset_s=sna[j-1]["onset_beat"],
                prev_onset_p=pna[prev_perf_idx]["onset_sec"],
                duration_s = sna[j]["duration_beat"],
                duration_p = pna[i]["duration_sec"],
                tempo=prevtempo,  # sec / beat
                weights=weights,
                tempo_factor=tempo_factor
            )
            cost = C[j-1] + local_dist
            if cost < C[j]:
                C[j] = cost
                T[j] = new_tempo
                perf_P[j] = i
            
            # store the index of the performance note to backtrack to
            B[i-1, j-1] = perf_P[j - 1] - 1
            # store a global cost matrix, for debugging
            D[i-1, j-1] = C[j]

    n = N - 1
    m = perf_P[N] - 1
    step = [m, n]
    path = [step]
    for sna_step in range(N-1):
        backtracking_perf_idx = B[m, n]
        m = backtracking_perf_idx
        n -= 1
        step = [m, n]
        path.append(step)

    path = np.array(path, dtype=np.int32)[::-1]
    output_path = path[:,0]
    # ppath = perf_P[1:] - 1
    return C, output_path, D, B


if __name__ == "__main__":
    A = np.array([[1, 2, 3, 4, 1, 2, 3, 4, 5, 6]]).T
    B = np.array([[1, 2, 3, 4, 5, 6]]).T

    # Like NWTW in Grachten
    # matcher = WNWTW(directions = np.array([[1, 0],[1, 1],[0, 1],[2, 1],[1, 2]]),
    #              directional_penalties = np.array([1, 0, 1, 0, 0]),
    #              directional_distances = [np.array([]),
    #                                       np.array([[0, 0]]),
    #                                       np.array([]),
    #                                       np.array([[0, 0],[1, 0]]),
    #                                       np.array([[0, 0],[0, 1]])],
    #              directional_weights = np.array([1, 2, 1, 3, 3]),
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

    matcher = ONW(smith_waterman=True, gamma_penalty=-0.5)

    path, cost, back = matcher(A, B, return_matrices=True)
