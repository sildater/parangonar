#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains pairwise distance metrics and other DP helpers.
"""
import numpy as np
from numba import jit



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


@jit(nopython=True)
def bounded_recursion(prev_val, 
              min_val = 0, 
              max_val = 10, 
              slope_at_min = 1):
    """
    a recursive function which when starting at min_val, 
    grows to slope_at_min after one step,
    then continues to grow and asymptotically reaches max_val
    """
    dv = max_val - min_val
    exponential_decay_factor = (slope_at_min) / dv
    normal_prev_val = (prev_val - min_val - dv) / (-dv)
    normal_next_val = (1 - exponential_decay_factor) * normal_prev_val
    next_val = -dv * normal_next_val + dv + min_val
    return next_val