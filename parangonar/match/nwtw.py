#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of Needleman Wunsch and derived algorithms
"""
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean


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
