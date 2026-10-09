#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains dynamic time warping methods.
"""

from typing import Tuple, Callable
import numpy as np
from scipy.spatial.distance import euclidean, cdist

# helpers and metrics
from .metrics import cdist_local, element_of_set_metric
from ..decorators import numba_jit as jit

ReturnType02 = (
    tuple[float] | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]
)
ReturnType04 = (
    tuple[np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, float]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
)


# DTW / DP classes


class WeightedDynamicTimeWarping(object):
    """
    Generalized Weighted Dynamic Time Warping.

    Parameters
    ----------
    directional_weights: np.ndarray
        weights associated with each of the three possible steps
    directions : double array
        directions.
    metric: callable
        the pairwise distance metric to be used between the input
    cdist_fun: callable
        the pairwise distance to be used (scipy cdist or local cdist)

    """

    def __init__(
        self,
        directional_weights: np.ndarray = np.array([1, 1, 1]),
        directions: np.ndarray = np.array([[1, 0], [1, 1], [0, 1]]),
        metric: Callable = euclidean,
        cdist_fun: Callable = cdist,
    ) -> None:
        self.directional_weights = directional_weights
        self.directions = directions
        self.metric = metric
        self.cdist_fun = cdist_fun

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        return_matrices: bool = False,
        return_cost: bool = False,
    ):
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
        pwD = self.cdist_fun(X, Y, self.metric)

        out = self.from_distance_matrix(
            pwD, 
            return_matrices=return_matrices, 
            return_cost=return_cost
        )
        return out

    def from_distance_matrix(
        self, pwD: np.ndarray, return_matrices: bool = False, return_cost: bool = False
    ):
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

    Parameters
    ----------
    metric: callable
        the pairwise distance metric to be used between the input
    cdist_fun: callable
        the pairwise distance to be used (scipy cdist or local cdist)
    """

    def __init__(
        self, metric: Callable = euclidean, cdist_fun: Callable = cdist
    ) -> None:
        self.metric = metric
        self.cdist_fun = cdist_fun

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        return_path: bool = True,
        return_cost_matrix: bool = False,
    ) -> ReturnType02:
        X = np.asanyarray(X, dtype=float)
        Y = np.asanyarray(Y, dtype=float)
        # Compute pairwise distance
        pwD = self.cdist_fun(X, Y, self.metric)
        # Compute accumulated cost matrix
        dtwd_matrix = dtw_dmatrix_from_pairwise_dmatrix(pwD)
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

    Parameters
    ----------
    metric: callable
        the pairwise distance metric to be used between the input
    """

    def __init__(self, metric: Callable = element_of_set_metric) -> None:
        self.metric = metric

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        return_path: bool = True,
        return_cost_matrix: bool = False,
    ) -> ReturnType02:
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
    metric: callable
        the pairwise distance metric to be used between the input
    cdist_fun: callable
        the pairwise distance to be used (scipy cdist or local cdist)
    """

    def __init__(
        self,
        directional_weights: np.ndarray = np.array([1, 1, 1]),
        directions: np.ndarray = np.array([[1, 0], [1, 1], [0, 1]]),
        buffer: int = 1,
        metric: Callable = euclidean,
        cdist_fun: Callable = cdist,
    ) -> None:
        self.directional_weights = directional_weights
        self.directions = directions
        self.buffer = buffer
        self.metric = metric
        self.cdist_fun = cdist_fun

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        return_matrices: bool = False,
        return_cost: bool = False,
    ) -> ReturnType04:
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
        pwD = self.cdist_fun(X, Y, self.metric)

        out = self.from_distance_matrix(
            pwD, return_matrices=return_matrices, return_cost=return_cost
        )
        return out

    def from_distance_matrix(
        self,
        pwD: np.ndarray,
        return_matrices: bool = False,
        return_cost: bool = False,
    ) -> ReturnType04:
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


class JumpDynamicTimeWarping(object):
    """
    Jump Weighted Dynamic Time Warping (JumpDTW).

    Parameters
    ----------
    directional_weights: np.ndarray
        weights associated with each of the three possible steps
    directions : double array
        directions.
    metric: callable
        the pairwise distance metric to be used between the input
    cdist_fun: callable
        the pairwise distance to be used (scipy cdist or local cdist)

    """

    def __init__(
        self,
        directional_weights: np.ndarray = np.array([1, 1, 1]),
        directions: np.ndarray = np.array([[1, 0], [1, 1], [0, 1]]),
        metric: Callable = euclidean,
        cdist_fun: Callable = cdist,
    ) -> None:
        self.directional_weights = directional_weights
        self.directions = directions
        self.metric = metric
        self.cdist_fun = cdist_fun

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        jumps: np.ndarray = np.empty((0, 2), dtype=np.int64),
        return_matrices: bool = False,
        return_cost: bool = False,
        seg_assign_start_id_map: dict = None,
        seg_assign_end_id_map: dict = None,
        seg_assign_from_to_map: dict = None,

    ):
        """
        Parameters
        ----------
        X : np.ndarray
            sequence 1 features, 1 row per step.
        Y : np.ndarray
            sequence 2 features, 1 row per step.
        jumps : np.ndarray
            array of jump indices: (n,2)
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
        pwD = self.cdist_fun(X, Y, self.metric)

        out = self.from_distance_matrix(
            pwD, 
            jumps=jumps,
            return_matrices=return_matrices, 
            return_cost=return_cost,
            seg_assign_start_id_map = seg_assign_start_id_map,
            seg_assign_end_id_map = seg_assign_end_id_map,
            seg_assign_from_to_map = seg_assign_from_to_map
        )
        return out

    def from_distance_matrix(
        self, 
        pwD: np.ndarray, 
        jumps: np.ndarray = np.empty((0, 2), dtype=np.int64),
        return_matrices: bool = False, 
        return_cost: bool = False,
        seg_assign_start_id_map: dict = None,
        seg_assign_end_id_map: dict = None,
        seg_assign_from_to_map: dict = None,


    ):
        """
            Parameters
        ----------
        pwD : np.ndarray
            pairwise distance matrix
        jumps : np.ndarray
            array of jump indices: (n,2)
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

        D, path, path_list = weighted_jump_dtw_forward_and_backward(
            pwD, 
            self.directional_weights, 
            self.directions,
            jumps,
            seg_assign_start_id_map,
            seg_assign_end_id_map,
            seg_assign_from_to_map

        )
        out = (path,)
        if return_matrices:
            out += (D, path_list)
        if return_cost:
            out += (D[path[-1, 0], path[-1, 1]],)
        return out


# alias
JDTW = JumpDynamicTimeWarping






# DTW fw + bw


@jit(nopython=True)
def weighted_dtw_forward_and_backward(
    pwD: np.ndarray,
    directional_weights: np.ndarray = np.array([1, 1, 1]),
    directions: np.ndarray = np.array([[1, 0], [1, 1], [0, 1]]),
) -> Tuple[np.ndarray, np.ndarray]:
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
                istep, jstep = direction
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


def dtw_backtracking(dtwd: np.ndarray) -> np.ndarray:
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


def dtw_dmatrix_from_pairwise_dmatrix(D: np.ndarray) -> np.ndarray:
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


def cdist_dtw_single_loop(
    arr1: np.ndarray, arr2: np.ndarray, metric: Callable
) -> np.ndarray:
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
    pwD: np.ndarray,
    directional_weights: np.ndarray = np.array([1, 1, 1]),
    directions: np.ndarray = np.array([[1, 0], [1, 1], [0, 1]]),
    buffer: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                istep, jstep = direction
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


def flexdtw_dmatrix_from_pairwise_dmatrix(
    pwD: np.ndarray, directional_weights: np.ndarray = np.array([1, 1, 1])
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def flexdtw_backtracking(
    D: np.ndarray, B: np.ndarray, S: np.ndarray, buffer: int = 1
) -> np.ndarray:
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


# JDTW

import numpy as np
from numba import jit
from typing import Tuple


@jit(nopython=True)
def weighted_jump_dtw_forward_and_backward(
    pwD: np.ndarray,
    directional_weights: np.ndarray = np.array([1, 1, 1]),
    directions: np.ndarray = np.array([[1, 0], [1, 1], [0, 1]]),
    jumps: np.ndarray = np.empty((0, 2), dtype=np.int64),
    seg_assign_start_id_map: dict = None,
    seg_assign_end_id_map: dict = None,
    seg_assign_from_to_map: dict = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    compute JumpDTW cost matrix and backtracking path.

    In addition to the normal DTW directions, 
    the forward pass allows jumps between 
    specified columns. A jump is represented as:

        (jump_from_index, jump_to_index)

    and operates in adjacent rows:

        (i, jump_from_index) -> (i-1, jump_to_index)

    Parameters
    ----------
    pwD : np.ndarray
        Pairwise distance matrix of shape (M, N).

    directional_weights : np.ndarray
        Weights associated with each normal DTW direction.

    directions : np.ndarray
        Array of normal DTW directions, e.g.

            [[1, 0],
             [1, 1],
             [0, 1]]

    jumps : np.ndarray
        Array of shape (K, 2), where each row contains:

            [jump_from_index, jump_to_index]

        The indices refer to columns of pwD, i.e. the range [0, N-1].

    Returns
    -------
    output_D : np.ndarray
        Accumulated DTW cost matrix of shape (M, N).

    output_path : np.ndarray
        Backtracked path through the cost matrix.
    """

    M = pwD.shape[0]
    N = pwD.shape[1]

    # Accumulated cost matrix.
    D = np.ones((M + 1, N + 1), dtype=np.float64) * np.inf

    # Backtracking information.
    #
    # >= 0  -> normal DTW direction index
    # < -1  -> jump
    #
    # For jumps, we store -(jump_index + 2), so that:
    #   -2 -> jumps[0]
    #   -3 -> jumps[1]
    #   ...

    B = np.ones((M, N), dtype=np.int64) * -1

    D[0, 0] = 0.0
    # center the pairwise distances
    pwD = (pwD - pwD.min()) / (pwD.max()-pwD.min())

    jump_tos = set(jumps[:,1])
    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    for i in range(1, M + 1):
        for j in range(1, N + 1):

            mincost = D[i, j]#np.inf
            minidx = -1
            best_is_jump = False
            best_jump_idx = -1

            # ----------------------------------------------------------
            # Normal DTW directions
            # ----------------------------------------------------------
            for directionsidx, direction in enumerate(directions):

                istep = direction[0]
                jstep = direction[1]

                previ = i - istep
                prevj = j - jstep

                if previ >= 0 and prevj >= 0:

                    cost = (
                        D[previ, prevj]
                        + pwD[i - 1, j - 1]
                        * directional_weights[directionsidx]
                    )

                    if cost < mincost:
                        mincost = cost
                        minidx = directionsidx
                        best_is_jump = False

            # ----------------------------------------------------------
            # Jump transitions
            # ----------------------------------------------------------
            
            if j - 1 in jump_tos:
                for jump_idx in range(jumps.shape[0]):

                    jump_from = jumps[jump_idx, 0]
                    jump_to = jumps[jump_idx, 1]
                    if j - 1 == jump_to:
                        # print(jump_to, jump_from)
                        prevj = jump_from + 1
                        previ = i - 1
                        dist_jump = abs(jump_to - jump_from)

                        if prevj >= 0 and prevj < N + 1 and previ >= 0:
                            cost = D[previ, prevj] + pwD[i - 1, j - 1] - 0.16 * dist_jump

                            if cost < mincost:
                                mincost = cost
                                best_is_jump = True
                                best_jump_idx = jump_idx


            # ----------------------------------------------------------
            # Store best predecessor
            # ----------------------------------------------------------
            D[i, j] = mincost

            if best_is_jump:
                B[i - 1, j - 1] = -(best_jump_idx + 2)
            else:
                B[i - 1, j - 1] = minidx

    # ------------------------------------------------------------------
    # Backtracking
    # ------------------------------------------------------------------




    n = N - 1
    m = M - 1

    step = [m, n]
    path = [step]

    crit = True
    path_list = ["END"]

    if (
        (seg_assign_from_to_map is not None) and 
        (seg_assign_start_id_map is not None) and
        (seg_assign_end_id_map is not None)
        ):

        last_seen_end_id = ""
        while crit:

            staged_id = seg_assign_end_id_map.get(n, None)
            if staged_id is not None:
                last_seen_end_id = staged_id
                # print(last_seen_end_id)

            seg_id = seg_assign_start_id_map.get(n, None)
            if seg_id is not None:
                # print("start id", seg_id)
                if seg_id == last_seen_end_id:
                    tos = seg_assign_from_to_map.get(seg_id, None)

                    if path_list[-1] in tos:
                        path_list.append(seg_id)
                        last_seen_end_id = ""

            if n == 0 and m == 0:
                crit = False

            else:

                backtracking_pointer = B[m, n]

                if backtracking_pointer >= 0:
                    # Normal DTW step
                    bt_vector = directions[backtracking_pointer]
                    m -= bt_vector[0]
                    n -= bt_vector[1]
                   


                elif backtracking_pointer < -1:
                    # print("jump_backtracking")
                    # Jump
                    #
                    # Decode:
                    #   -2 -> jump 0
                    #   -3 -> jump 1
                    #   ...
                    jump_idx = -backtracking_pointer - 2
                    jump_from = jumps[jump_idx, 0]
                    # print("jump_backtracking", jump_from, n)
                    n = jump_from
                    m -= 1

                else:
                    print("invalid pointer")
                    crit = False


                step = [m, n]
            path.append(step)
        




    else:
        while crit:
            # print(m, n)

            if n == 0 and m == 0:
                crit = False

            elif m < 0:
                crit = False

            else:

                backtracking_pointer = B[m, n]
                # print(backtracking_pointer)

                if backtracking_pointer >= 0:
                    # Normal DTW step
                    bt_vector = directions[backtracking_pointer]
                    m -= bt_vector[0]
                    n -= bt_vector[1]

                elif backtracking_pointer < -1:
                    # print("jump_backtracking")
                    # Jump
                    #
                    # Decode:
                    #   -2 -> jump 0
                    #   -3 -> jump 1
                    #   ...
                    jump_idx = -backtracking_pointer - 2
                    jump_from = jumps[jump_idx, 0]
                    # print("jump_backtracking", jump_from, n)
                    n = jump_from
                    m -= 1

                else:
                    print("invalid pointer")
                    crit = False


                step = [m, n]
            path.append(step)



    output_path = np.array(path, dtype=np.int32)[::-1]
    output_D = D[1:, 1:]

    return output_D, output_path[1:, :], path_list[::-1]
    


if __name__ == "__main__":
    A = np.array([[1, 2, 3, 4, 1, 2, 3, 4, 5, 6]]).T
    B = np.array([[1, 2, 3, 4, 5, 6]]).T
    dtwmatcher = WDTW()
    p = dtwmatcher(A, B)
