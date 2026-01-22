#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains pairwise distance metrics and other DP helpers.
"""
from typing import Union, Set, List, Callable, Tuple
import numpy as np
from numba import jit


def element_of_metric(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    metric that evaluates occurence of vec2 (scalar) in vec1 (vector n-dim)
    """
    return 1 - np.sum(vec2 == vec1)


def element_of_set_metric(element_: Union[int, float], set_: Set[Union[int, float]]) -> float:
    """
    metric that evaluates occurence of an element in a set
    """
    if element_ in set_:
        return 0.0
    else:
        return 1.0


def element_of_set_metric_se(set_: Set[Union[int, float]], element_: Union[int, float]) -> float:
    """
    metric that evaluates occurence of an element in a set
    """
    if element_ in set_:
        return 0.0
    else:
        return 1.0


def l2(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    l2 metric between vec1 and vec2
    """
    return np.sqrt(np.sum((vec2 - vec1) ** 2))


def invert_matrix(
    S: np.ndarray, inversion: str = "reciprocal", positive: bool = False
) -> np.ndarray:
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


def dnw(vec1: np.ndarray, vec2: np.ndarray) -> float:
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


def cdist_local(
    arr1: np.ndarray, arr2: np.ndarray, metric: Callable
) -> np.ndarray:
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
    l1 = len(arr1)
    l2 = len(arr2)
    pdist_array = np.ones((l1, l2)) * np.inf
    for i in range(l1):
        for j in range(l2):
            pdist_array[i, j] = metric(arr1[i], arr2[j])
    return pdist_array


@jit(nopython=True)
def bounded_recursion(
    prev_val: float, min_val: float = 0, max_val: float = 10, slope_at_min: float = 1
) -> float:
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


def tempo_and_pitch_metric(
    pitch_set_s: Set[int],
    pitch_p: int,
    onset_s: float,
    onset_p: float,
    prev_onset_s: float,
    prev_onset_p: float,
    tempo: float,  # sec / beat
    time_weight: float = 0.5,
    tempo_factor: float = 0.5,
) -> Tuple[float, float]:
    """
    metric that combines
    1) pitch set metric
    2) onset deviation based on tempo

    returns the distance and the tempo associated with it
    """
    # pitch stuff
    if pitch_p in pitch_set_s:
        pitch_dist = 0
    else:
        pitch_dist = 1

    # onset stuff
    estimated_onset = prev_onset_p + (onset_s - prev_onset_s) * tempo
    onset_dist = abs(onset_p - estimated_onset) / tempo  # normalize offset by tempo4

    # tempo stuff
    if onset_s - prev_onset_s > 0:
        current_tempo = (onset_p - prev_onset_p) / (onset_s - prev_onset_s)
    else:
        current_tempo = tempo + (onset_p - prev_onset_p)
    exponential_average_tempo = (
        tempo_factor * current_tempo + (1 - tempo_factor) * tempo
    )

    dist = pitch_dist + time_weight * onset_dist
    return dist, exponential_average_tempo


def onset_pitch_duration_metric(
    pitch_s: int,
    pitch_p: int,
    onset_s: float,
    onset_p: float,
    prev_onset_s: float,
    prev_onset_p: float,
    duration_s: float,
    duration_p: float,
    tempo: float,  # sec / beat
    weights: np.ndarray = np.array([1, 1, 1]),  # onset, dur, pitch
    tempo_factor: float = 0.5,
) -> Tuple[float, float]:
    """
    metric that combines
    1) pitch set metric
    2) onset deviation based on tempo
    3) duration

    returns the distance and the tempo associated with it
    """
    # pitch stuff
    if pitch_p == pitch_s:
        pitch_dist = 0
    else:
        pitch_dist = 1

    # onset stuff
    estimated_onset = prev_onset_p + (onset_s - prev_onset_s) * tempo
    onset_dist = abs(onset_p - estimated_onset) / tempo  # normalize offset by tempo4

    # duration stuff
    estimated_duration = duration_s * tempo
    duration_dist = abs(
        np.log(duration_p / estimated_duration)
    )  # abs log of duration ratio

    # tempo stuff
    if onset_s - prev_onset_s > 0:
        current_tempo = (onset_p - prev_onset_p) / (onset_s - prev_onset_s)
    else:
        current_tempo = tempo + (onset_p - prev_onset_p)
    exponential_average_tempo = (
        tempo_factor * current_tempo + (1 - tempo_factor) * tempo
    )

    dist = weights.dot(np.array([onset_dist, duration_dist, pitch_dist]))
    return dist, exponential_average_tempo
