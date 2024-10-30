#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains dynamic time warping methods.
"""

import numpy as np
from scipy.spatial.distance import cdist
import numba
from numba import jit

# helpers and metrics


def element_of_metric(vec1, vec2):
    """
    metric that evaluates occurence of vec2 (scalar) in vec1 (vector n-dim)
    """
    return 1 - np.sum(vec2 == vec1)


def element_of_set_metric(element_, set_):
    """
    metric that evaluates occurence of vec2 (scalar) in vec1 (vector n-dim)
    """
    if element_ in set_:
        return 0.0
    else:
        return 1.0


def l2(vec1, vec2):
    """
    l2 metric between vec1 and vec2
    """
    return np.sqrt(np.sum((vec2 - vec1) ** 2))


def invert_matrix(S, inversion="reciprocal", positive=False):
    """
    simple converter from similarity to distance matrix
    and vice versa
    """
    if inversion == "reciprocal":
        D = 1 / (S + 1e-4)
    else:
        D = -S
    if positive:
        D -= D.min()
    return D


def dnw(vec1, vec2):
    """
    normalized and weighted distance for LNCO/LNSO features.
    https://ieeexplore.ieee.org/abstract/document/6333860/
    """
    vec1_l1 = np.abs(vec1).sum()
    vec2_l1 = np.abs(vec2).sum()
    dn = np.abs(vec1 - vec2).sum() / (vec1_l1 + vec2_l1 + 1e-7)  # L1 okay?
    dampening_factor = ((vec1_l1 + vec2_l1 + 1e-7) / 2) ** (
        1 / 4
    )  # safety eps not necessary
    return dn * dampening_factor


def cdist_local(arr1, arr2, metric):
    """
    compute array of pairwise distances between
    the elements of two arrays given a metric

    Parameters
    ----------
    arr1: numpy nd array

    arr2: numpy nd array

    metric: callable
        a metric function

    Returns
    -------

    pdist_array: numpy 2d array
        array of pairwise distances
    """
    pdist_array = np.ones((arr1.shape[0], arr2.shape[0])) * np.inf
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[0]):
            pdist_array[i, j] = metric(arr1[i], arr2[j])
    return pdist_array


# DTW classes


class WeightedDynamicTimeWarping(object):
    """
    Generalized Weighted Dynamic Time Warping.

    Parameters
    ----------
    directional_weights: np.ndarray
        weights associated with each of the three possible steps
    directions : double array
        directions.

    """

    def __init__(
        self,
        directional_weights=np.array([1, 1, 1]),
        directions=np.array([[1, 0], [1, 1], [0, 1]]),
        metric="euclidean",
        cdist_local=False,
    ):
        self.directional_weights = directional_weights
        self.directions = directions
        self.metric = metric
        self.cdist_local = cdist_local

    def __call__(self, X, Y, return_matrices=False, return_cost=False):
        """
        Parameters
        ----------
        X : np.ndarray
            sequence 1 features, 1 row per step.
        Y : np.ndarray
            sequence 2 features, 1 row per step.
        return_matrices: bool
            return accumulated cost matrix
        return_cost : bool
            return accumulated cost of the minimizing path.

        Returns
        -------
        path : np.ndarray
            Accumulated cost matrix
        """

        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)
        # Compute pairwise distance
        if self.cdist_local:
            pwD = cdist_local(X, Y, self.metric)
        else:
            pwD = cdist(X, Y, self.metric)

        out = self.from_distance_matrix(
            pwD, return_matrices=return_matrices, return_cost=return_cost
        )
        return out

    def from_distance_matrix(self, pwD, return_matrices=False, return_cost=False):
        """
            Parameters
        ----------
        pwD : np.ndarray
            pairwise distance matrix
        return_matrices: bool
            return accumulated costmatrix, backtracking, and
            starting point matrix
        return_cost : bool
            return accumulated cost of the minimizing path.

        Returns
        -------
        path : np.ndarray
            Accumulated cost matrix
        """

        D, path = weighted_dtw_forward_and_backward(
            pwD, self.directional_weights, self.directions
        )
        out = (path,)
        if return_matrices:
            out += (D,)
        if return_cost:
            out += (D[path[-1, 0], path[-1, 1]],)
        return out


# alias
WDTW = WeightedDynamicTimeWarping


class DynamicTimeWarping(object):
    """
    pure python vanilla Dynamic Time Warping
    """

    def __init__(self, metric="euclidean", cdist_local=False):
        self.metric = metric
        self.cdist_local = cdist_local

    def __call__(self, X, Y, return_path=True, return_cost_matrix=False):
        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)
        # Compute pairwise distance
        if self.cdist_local:
            D = cdist_local(X, Y, self.metric)
        else:
            D = cdist(X, Y, self.metric)
        # Compute accumulated cost matrix
        dtwd_matrix = dtw_dmatrix_from_pairwise_dmatrix(D)
        dtwd_distance = dtwd_matrix[-1, -1]

        # Output
        out = (dtwd_distance,)

        if return_path:
            # Compute alignment path
            path = dtw_backtracking(dtwd_matrix)
            out += (path,)
        if return_cost_matrix:
            out += (dtwd_matrix,)
        return out


# alias
DTW = DynamicTimeWarping


class DynamicTimeWarpingSingleLoop(object):
    """
    pure python vanilla Dynamic Time Warping
    """

    def __init__(self, metric=element_of_set_metric):
        self.metric = metric

    def __call__(self, X, Y, return_path=True, return_cost_matrix=False):
        # Compute the pw distances and accumulated cost matrix
        dtwd_matrix = cdist_dtw_single_loop(X, Y, self.metric)
        # dtwd_matrix = dtw_dmatrix_from_pairwise_dmatrix(D)
        dtwd_distance = dtwd_matrix[-1, -1]

        # Output
        out = (dtwd_distance,)

        if return_path:
            # Compute alignment path
            path = dtw_backtracking(dtwd_matrix)
            out += (path,)
        if return_cost_matrix:
            out += (dtwd_matrix,)
        return out


# alias
DTWSL = DynamicTimeWarpingSingleLoop


class FlexDynamicTimeWarping(object):
    """
    FlexDTW: https://ismir2023program.ismir.net/poster_235.html
    from two vectors

    Parameters
    ----------
    directional_weights: np.ndarray
        weights associated with each of the three possible steps
    directions : double array
        directions.
    buffer: int
        buffer zone for flexible path end point

    """

    def __init__(
        self,
        directional_weights=np.array([1, 1, 1]),
        directions=np.array([[1, 0], [1, 1], [0, 1]]),
        buffer=1,
        metric="euclidean",
        cdist_local=False,
    ):
        self.directional_weights = directional_weights
        self.directions = directions
        self.buffer = buffer
        self.metric = metric
        self.cdist_local = cdist_local

    def __call__(self, X, Y, return_matrices=False, return_cost=False):
        """
        Parameters
        ----------
        X : np.ndarray
            sequence 1 features, 1 row per step.
        Y : np.ndarray
            sequence 2 features, 1 row per step.
        return_matrices: bool
            return accumulated costmatrix, backtracking, and
            starting point matrix
        return_cost : bool
            return accumulated cost of the minimizing path.

        Returns
        -------
        path : np.ndarray
            Accumulated cost matrix
        """

        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)
        # Compute pairwise distance
        if self.cdist_local:
            pwD = cdist_local(X, Y, self.metric)
        else:
            pwD = cdist(X, Y, self.metric)

        out = self.from_distance_matrix(
            pwD, return_matrices=return_matrices, return_cost=return_cost
        )
        return out

    def from_distance_matrix(self, pwD, return_matrices=False, return_cost=False):
        """
            Parameters
        ----------
        pwD : np.ndarray
            pairwise distance matrix
        return_matrices: bool
            return accumulated costmatrix, backtracking, and
            starting point matrix
        return_cost : bool
            return accumulated cost of the minimizing path.

        Returns
        -------
        path : np.ndarray
            Accumulated cost matrix
        """
        path, D, B, S = flexdtw_forward_and_backward(
            pwD, self.directional_weights, self.directions, self.buffer
        )
        out = (path,)
        if return_matrices:
            out += (
                D,
                B,
                S,
            )
        if return_cost:
            out += D[path[-1, 0], path[-1, 1]]
        return out


# alias
FDTW = FlexDynamicTimeWarping

# DTW fw + bw


@jit(nopython=True)
def weighted_dtw_forward_and_backward(
    pwD,
    directional_weights=np.array([1, 1, 1]),
    directions=np.array([[1, 0], [1, 1], [0, 1]]),
):
    """
    compute dynamic time warping cost matrix
    and backtracking path
    from weighted directions and
    a pairwise distance matrix

    Parameters
    ----------
    D : np.ndarray
        Pairwise distance matrix (computed e.g., with `cdist`).
    directional_weights: np.ndarray
        weights associated with each of the three possible steps
    directions : double array
        directions.

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
    # the dtwd distance matrix is initialized with INFINITY
    D = np.ones((M + 1, N + 1), dtype=float) * np.inf
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
                    cost = (
                        D[previ, prevj]
                        + pwD[i - 1, j - 1] * directional_weights[directionsidx]
                    )
                    if cost < mincost:
                        mincost = cost
                        minidx = directionsidx
                        bestiprev = previ
                        bestjprev = prevj

            D[i, j] = (
                D[bestiprev, bestjprev]
                + pwD[i - 1, j - 1] * directional_weights[minidx]
            )
            B[i - 1, j - 1] = minidx

    # return (dtwd[1:, 1:])
    n = N - 1
    m = M - 1
    step = [m, n]
    path = [step]
    # initialize boolean variables for stopping decoding
    crit = True
    while crit:
        if n == 0 and m == 0:
            crit = False
        else:
            backtracking_pointer = B[m, n]
            bt_vector = directions[backtracking_pointer]
            m -= bt_vector[0]
            n -= bt_vector[1]
            step = [m, n]
        # append next step to the path
        path.append(step)

    output_path = np.array(path, dtype=np.int32)[::-1]
    output_D = D[1:, 1:]
    return output_D, output_path[1:, :]


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
    dtwd = np.ones((M + 1, N + 1), dtype=float) * np.inf

    # Compute the distance iteratively
    dtwd[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            c = D[i - 1, j - 1]
            insertion = dtwd[i - 1, j]
            match = dtwd[i - 1, j - 1]
            deletion = dtwd[i, j - 1]
            dtwd[i, j] = c + min((insertion, deletion, match))

    return dtwd[1:, 1:]


def cdist_dtw_single_loop(arr1, arr2, metric):
    """

    compute  a pairwise distance matrix
    and its dynamic time warping cost matrix

    Parameters
    ----------

    arr1: numpy nd array or list

    arr2: numpy nd array or list

    metric> callable
        a metric function

    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    """
    # Initialize arrays and helper variables
    M = len(arr1)  # arr1.shape[0]
    N = len(arr2)  # arr2.shape[0]

    # pdist_array = np.ones((M,N))*np.inf
    # the dtwd distance matrix is initialized with INFINITY
    dtwd = np.ones((M + 1, N + 1), dtype=float) * np.inf

    # Compute the distance iteratively
    dtwd[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            # pdist_array[i-1, j-1] = metric(arr1[i-1], arr2[j-1])
            # c = pdist_array[i - 1, j - 1]
            c = metric(arr1[i - 1], arr2[j - 1])
            insertion = dtwd[i - 1, j]
            deletion = dtwd[i, j - 1]
            match = dtwd[i - 1, j - 1]
            dtwd[i, j] = c + min((insertion, deletion, match))

    return dtwd[1:, 1:]  # pdist_array


# FDTW fw + bw


@jit(nopython=True)
def flexdtw_forward_and_backward(
    pwD,
    directional_weights=np.array([1, 1, 1]),
    directions=np.array([[1, 0], [1, 1], [0, 1]]),
    buffer=1,
):
    """
    compute felxDTW cost matrix,
    backtrace matrix,
    and starting point matrix
    from a pairwise distance matrix

    Parameters
    ----------
    pwD : double array
        Pairwise distance matrix (computed e.g., with `cdist`).
    directional_weights : double array
        weights for each direction
    directions : double array
        directions.
    buffer : int
        buffer for candidate end points

    Returns
    -------
    D : np.ndarray
        Accumulated cost matrix
    B : np.ndarray
        backtrace matrix
    S : np.ndarray
        starting point matrix
    """
    # Initialize arrays and helper variables
    M = pwD.shape[0]
    N = pwD.shape[1]
    D = np.zeros((M, N))  # * np.inf
    B = np.zeros((M, N), dtype=np.int8)  # * -1
    S = np.zeros((M, N), dtype=np.int32)  # * -1
    # initialize matrices
    M_idx = np.arange(M)
    N_idx = np.arange(N)
    D[0, :] = pwD[0, :]
    D[:, 0] = pwD[:, 0]
    S[:, 0] = M_idx
    S[0, :] = -N_idx
    if buffer > N - 2 or buffer > M - 2:
        buffer = min((N - 2, M - 2))
        # raise ValueError("buffer size needs to be smaller than matrix dimensions")

    # Compute the distance iteratively
    for i in range(1, M):
        for j in range(1, N):
            mincost = np.inf
            minidx = -1
            bestiprev = -1
            bestjprev = -1
            for directionsidx, direction in enumerate(directions):
                (istep, jstep) = direction
                previ = i - istep
                prevj = j - jstep
                if previ >= 0 and prevj >= 0:
                    cost = (
                        D[previ, prevj] + pwD[i, j] * directional_weights[directionsidx]
                    )
                    if S[previ, previ] >= 0:
                        dist = i + (j - S[previ, prevj])
                    else:
                        dist = i + (j + S[previ, prevj])
                    cost_per_mb = cost / dist

                    if cost_per_mb < mincost:
                        mincost = cost_per_mb
                        minidx = directionsidx
                        bestiprev = previ
                        bestjprev = prevj

            D[i, j] = D[bestiprev, bestjprev] + pwD[i, j] * directional_weights[minidx]
            B[i, j] = minidx
            S[i, j] = S[bestiprev, bestjprev]

    # get end point
    endpoint_candidates_m = np.column_stack(
        (
            np.full(N - buffer - 1, M - 1, dtype=np.int32),
            np.arange(buffer, N - 1, dtype=np.int32),
        )
    )  # bottom row
    endpoint_candidates_n = np.column_stack(
        (
            np.arange(buffer, M - 1, dtype=np.int32),
            np.full(M - buffer - 1, N - 1, dtype=np.int32),
        )
    )  # right column

    ep_c = np.concatenate(
        (
            endpoint_candidates_m,
            np.array([[M - 1, N - 1]], dtype=np.int32),
            endpoint_candidates_n,
        )
    )
    # endpoints_values = D[ep_c[:,0],ep_c[:,1]] / (np.sum(ep_c, axis = 1) - np.abs(S[ep_c[:,0],ep_c[:,1]]))
    endpoints_values = np.zeros(ep_c.shape[0])
    for idx, ep_c_cand in enumerate(ep_c):
        ep_c1 = ep_c_cand[0]
        ep_c2 = ep_c_cand[1]
        endpoints_values[idx] = D[ep_c1, ep_c2] / (
            ep_c1 + ep_c2 - np.abs(S[ep_c1, ep_c2])
        )

    minimal_ep = np.argmin(endpoints_values)
    m = ep_c[minimal_ep, 0]
    n = ep_c[minimal_ep, 1]
    step = (m, n)
    path = []
    path.append(step)
    # loop over backtracking matrix
    crit = True
    while crit:
        if n == 0 or m == 0:
            crit = False
        else:
            backtracking_pointer = B[m, n]
            bt_vector = directions[backtracking_pointer, :]
            m -= bt_vector[0]
            n -= bt_vector[1]
            step = (m, n)
        # append next step to the path
        path.append(step)
    output_path = np.array(path, dtype=np.int32)[::-1]
    return output_path[1:, :], D, B, S
    # return 1,2,3,4


def flexdtw_dmatrix_from_pairwise_dmatrix(pwD, directional_weights=np.array([1, 1, 1])):
    """
    compute felxDTW cost matrix,
    backtrace matrix,
    and starting point matrix
    from a pairwise distance matrix

    Parameters
    ----------
    pwD : double array
        Pairwise distance matrix (computed e.g., with `cdist`).

    Returns
    -------
    D : np.ndarray
        Accumulated cost matrix
    B : np.ndarray
        backtrace matrix
    S : np.ndarray
        starting point matrix
    """
    # Initialize arrays and helper variables
    M = pwD.shape[0]
    N = pwD.shape[1]
    dw = directional_weights
    D = np.ones((M, N), dtype=float) * np.inf
    B = np.zeros((M, N), dtype=np.int8)  # * -1
    S = np.zeros((M, N), dtype=np.int32)  # * -1
    # initialize matrices
    M_idx = np.arange(M)
    N_idx = np.arange(N)
    D[:, 0] = pwD[:, 0]
    D[0, :] = pwD[0, :]
    S[:, 0] = M_idx
    S[0, :] = -N_idx
    # Compute the distance iteratively
    for i in range(1, M):
        for j in range(1, N):
            c = pwD[i, j]
            S_local = np.array([S[i - 1, j], S[i - 1, j - 1], S[i, j - 1]])
            D_local = np.array(
                [
                    dw[0] * c + D[i - 1, j],
                    dw[1] * c + D[i - 1, j - 1],
                    dw[2] * c + D[i, j - 1],
                ]
            )
            B_local = D_local / (i + j - np.abs(S_local))
            B_dir = np.argmin(B_local)
            B[i, j] = B_dir
            D[i, j] = D_local[B_dir]
            S[i, j] = S_local[B_dir]
    return D, B, S


def flexdtw_backtracking(D, B, S, buffer=1):
    """
    Decode path from the accumulated dtw cost matrix,
    backtrace matrix, and starting point matrix

    Parameters
    ----------
    D : np.ndarray
        Accumulated cost matrix
    B : np.ndarray
        backtrace matrix
    S : np.ndarray
        starting point matrix

    Returns
    -------
    path : np.ndarray
       A 2D array of size (n_steps, 2), where i-th row has elements
       (i_m, i_n) where i_m represents the index in the input array
       and i_n represents the corresponding index in the reference array.
    """

    N = D.shape[0]
    M = D.shape[1]
    if buffer > N - 2 or buffer > M - 2:
        raise ValueError("buffer size needs to be smaller than matrix dimensions")

    endpoint_candidates_n = np.column_stack(
        (np.arange(buffer, N - 1), np.full(N - buffer - 1, M - 1))
    )  # right column
    endpoint_candidates_m = np.column_stack(
        (np.full(M - buffer - 1, N - 1), np.arange(buffer, M - 1))
    )  # bottom row
    ep_c = np.concatenate(
        (endpoint_candidates_n, np.array([[N - 1, M - 1]]), endpoint_candidates_m)
    )

    endpoints_values = D[ep_c[:, 0], ep_c[:, 1]] / (
        np.sum(ep_c, axis=1) - np.abs(S[ep_c[:, 0], ep_c[:, 1]])
    )
    minimal_ep = np.argmin(endpoints_values)
    n = ep_c[minimal_ep, 0]
    m = ep_c[minimal_ep, 1]
    step = np.array([n, m])
    path = [step]

    # initialize boolean variables for stopping decoding
    crit = True
    backtracking_vectors = np.array([[-1, 0], [-1, -1], [0, -1]])

    while crit:
        if step[0] == 0 or step[1] == 0:
            crit = False

        else:
            backtracking_pointer = B[step[0], step[1]]
            bt_vector = backtracking_vectors[backtracking_pointer, :]
            step = np.copy(step) + bt_vector
            path.append(step)

    return np.array(path[::-1], dtype=int)


if __name__ == "__main__":
    A = np.array([[1, 2, 3, 4, 1, 2, 3, 4, 5, 6]]).T
    B = np.array([[1, 2, 3, 4, 5, 6]]).T
    dtwmatcher = WDTW()
    p = dtwmatcher(A, B)
