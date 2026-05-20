#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains dynamic programming algorithms for spectrogram-based
audio-to-score alignment.

Both algorithms operate on pitch-onset features extracted from an IIR
log-frequency spectrogram.  They track a beat-period variable through the
DP matrix and use it to constrain the stretch window around each candidate
jump.

The two flavours differ in how they prune the candidate set at each step:

* :class:`ElasticSpecDP` — evaluates *all* candidates within the stretch
  window and then prunes high-cost columns via an absolute cost threshold.
* :class:`ElasticSpecDPLimited` — retains only the top-N candidates per step
  using a soft-min (log-sum-exp) and discards the rest, trading accuracy for 
  speed and memory on long sequences.
"""

from typing import Optional

import numpy as np

from ..decorators import numba_jit as jit


# ---------------------------------------------------------------------------
# Internal JIT-compiled forward-backward functions
# (verbatim from the original test code; no algorithmic changes)
# ---------------------------------------------------------------------------


@jit(nopython=True)
def _elastic_forward_and_backward_pitch_onset(
    onsets,
    spec,
    spikes,
    pitch_sets,
    bp_init,
    max_stretch_longer,
    max_stretch_shorter,
    stretch_cost,
    spec_cost,
    spike_cost,
    max_bp,
    min_bp,
    alpha,
    cost_threshold,
    spec_slice_len,
):
    """
    Full elastic DP forward-backward pass over pitch-onset features.

    Parameters
    ----------
    onsets : np.ndarray, shape (n_bins, n_frames)
        Onset-activation features (already inverted, i.e. 1 - activation).
    spec : np.ndarray, shape (n_bins, n_frames)
        Spectrogram features (already inverted, i.e. 1 - normalised).
    spikes : np.ndarray, shape (n_spikes,)
        Score-onset positions in spectrogram-frame indices.
    pitch_sets : list of np.ndarray
        Per-spike arrays of pitch-bin indices active at that score onset.
    bp_init : float
        Initial beat-period estimate (frames per score beat).
    max_stretch_longer : float
        Maximum relative stretch above the beat period.
    max_stretch_shorter : float
        Maximum relative compression below the beat period.
    stretch_cost : float
        Weight for the stretch penalty.
    spec_cost : float
        Weight for the spectrogram minimum-fit term.
    spike_cost : float
        Weight for the onset-activation term.
    max_bp : float
        Upper bound on the tracked beat period.
    min_bp : float
        Lower bound on the tracked beat period.
    alpha : float
        Beat-period adaptation rate.
    cost_threshold : int
        Number of lowest-cost columns to retain per row; the rest are
        set to ``inf``.
    spec_slice_len : int
        Length of the spectrogram slice taken around each candidate for
        the spec-fit cost (minimum over the slice).

    Returns
    -------
    D : np.ndarray, shape (n_spikes, n_frames)
        Accumulated cost matrix.
    B : np.ndarray, shape (n_spikes, n_frames), dtype int64
        Backtracking matrix (stores previous column index).
    BP : np.ndarray, shape (n_spikes, n_frames)
        Beat-period matrix.
    path : list of int
        Backtracked path of spectrogram-frame indices, one per spike.
    """
    # Initialize arrays and helper variables
    M = spikes.shape[0]
    N = onsets.shape[1]
    spike_period = np.diff(spikes)

    # accumulated cost matrix is initialized with INFINITY
    D = np.ones((M, N), dtype=float) * np.inf
    # Backtracking
    B = np.ones((M, N), dtype=np.int64) * -1
    # keep track of previous beat period
    BP = np.ones((M, N), dtype=float) * bp_init
    #
    D[0, 0] = 0
    max_bp = float(max_bp)
    min_bp = float(min_bp)
    spec_slice_len = int(spec_slice_len)

    for i in np.arange(0, M - 1, dtype=np.int64):  # loop over spikes
        pitches = pitch_sets[i + 1]

        for j in np.arange(0, N - 1, dtype=np.int64):  # loop over activations
            if D[i, j] < np.inf:
                beat_period = BP[i, j]
                lower_bound = max(
                    min(j + np.floor(beat_period * (1 - max_stretch_shorter) * spike_period[i]), N - 1),
                    j + 1,
                )
                upper_bound = max(
                    min(j + np.ceil(beat_period * (1 + max_stretch_longer)) * spike_period[i] + 1, N),
                    lower_bound + 1,
                )
                candidate_slice = np.arange(lower_bound, upper_bound, dtype=np.int64)

                # prepare the slice of the spectrogram and flux to be checked
                slice_arr = D[i, candidate_slice]
                D_vals = np.empty((len(pitches), len(slice_arr)), dtype=slice_arr.dtype)
                for pitch_rows in range(len(pitches)):
                    D_vals[pitch_rows] = slice_arr

                for pitch_idx, pitch in enumerate(pitches):
                    for candidate_idx, candidate_j in enumerate(candidate_slice):

                        # compute the stretch
                        stretch = max(
                            (candidate_j - j) / (beat_period * spike_period[i]),
                            (beat_period * spike_period[i]) / (candidate_j - j),
                        )
                        stretch_c = min((stretch ** 2 - 1), 1)

                        # onset activation
                        activation = onsets[pitch, candidate_j]
                        # spec slice
                        spec_slice = spec[pitch, candidate_j: candidate_j + spec_slice_len]
                        spec_fit = np.min(spec_slice)

                        # total cost for this candidate
                        candidate_j_cost = (
                            D[i, j]
                            + spike_cost * activation
                            + stretch_cost * stretch_c
                            + spec_fit * spec_cost
                        )
                        D_vals[pitch_idx, candidate_idx] = candidate_j_cost

                for candidate_j_idx, candidate_j in enumerate(candidate_slice):
                    candidate_j_cost = np.min(D_vals[:, candidate_j_idx])
                    if D[i + 1, candidate_j] > candidate_j_cost:
                        D[i + 1, candidate_j] = candidate_j_cost
                        BP[i + 1, candidate_j] = (
                            (1 - alpha) * beat_period
                            + alpha * min(max(float(candidate_j - j) / spike_period[i], min_bp), max_bp)
                        )
                        B[i + 1, candidate_j] = j

        min_cost_at_idx = np.min(D[i + 1, :])
        mask_large_cost = D[i + 1, :] > min_cost_at_idx + cost_threshold
        D[i + 1, mask_large_cost] = np.inf

    # simple backtracking
    path = [N - 1]
    for backwards_i in range(M - 1, 0, -1):
        prev_spike = B[backwards_i, path[-1]]
        path.append(prev_spike)
    return D, B, BP, path[::-1]


@jit(nopython=True)
def _elastic_forward_and_backward_pitch_onset_limit(
    onsets,
    spec,
    spikes,
    pitch_sets,
    bp_init,
    max_stretch_longer,
    max_stretch_shorter,
    stretch_cost,
    spec_cost,
    spike_cost,
    max_bp,
    min_bp,
    alpha,
    spec_slice_len,
    cost_threshold,
    candidate_onset_number,
):
    """
    Limited elastic DP forward-backward pass over pitch-onset features.

    Identical to :func:`_elastic_forward_and_backward_pitch_onset` except
    that only the top-``candidate_onset_number`` candidates are retained
    per step (soft-min via log-sum-exp), and ``cost_threshold`` is
    interpreted as a *rank* cutoff rather than an absolute cost delta.

    Parameters
    ----------
    onsets : np.ndarray, shape (n_bins, n_frames)
        Onset-activation features (already inverted).
    spec : np.ndarray, shape (n_bins, n_frames)
        Spectrogram features (already inverted).
    spikes : np.ndarray, shape (n_spikes,)
        Score-onset positions in spectrogram-frame indices.
    pitch_sets : list of np.ndarray
        Per-spike arrays of pitch-bin indices active at that score onset.
    bp_init : float
        Initial beat-period estimate (frames per score beat).
    max_stretch_longer : float
        Maximum relative stretch above the beat period.
    max_stretch_shorter : float
        Maximum relative compression below the beat period.
    stretch_cost : float
        Weight for the stretch penalty.
    spec_cost : float
        Weight for the spectrogram minimum-fit term.
    spike_cost : float
        Weight for the onset-activation term.
    max_bp : float
        Upper bound on the tracked beat period.
    min_bp : float
        Lower bound on the tracked beat period.
    alpha : float
        Beat-period adaptation rate.
    spec_slice_len : int
        Length of the spectrogram slice taken around each candidate.
    cost_threshold : int
        Rank cutoff: columns whose rank exceeds this value are set to
        ``inf`` (i.e. only the ``cost_threshold`` lowest-cost columns
        are kept).
    candidate_onset_number : int
        Number of top candidates to write into the DP matrix per step
        (soft-min via log-sum-exp).

    Returns
    -------
    D : np.ndarray, shape (n_spikes, n_frames)
        Accumulated cost matrix.
    B : np.ndarray, shape (n_spikes, n_frames), dtype int64
        Backtracking matrix.
    BP : np.ndarray, shape (n_spikes, n_frames)
        Beat-period matrix.
    path : list of int
        Backtracked path of spectrogram-frame indices, one per spike.
    """
    # Initialize arrays and helper variables
    M = spikes.shape[0]
    N = onsets.shape[1]
    spike_period = np.diff(spikes)

    # accumulated cost matrix is initialized with INFINITY
    D = np.ones((M, N), dtype=float) * np.inf
    # Backtracking
    B = np.ones((M, N), dtype=np.int64) * -1
    # keep track of previous beat period
    BP = np.ones((M, N), dtype=float) * bp_init
    #
    D[0, 0] = 0
    max_bp = float(max_bp)
    min_bp = float(min_bp)

    for i in np.arange(0, M - 1, dtype=np.int64):  # loop over spikes
        pitches = pitch_sets[i + 1]

        for j in np.arange(0, N - 1, dtype=np.int64):  # loop over activations
            if D[i, j] < np.inf:
                beat_period = BP[i, j]
                lower_bound = max(
                    min(j + np.floor(beat_period * (1 - max_stretch_shorter) * spike_period[i]), N - 1),
                    j + 1,
                )
                upper_bound = max(
                    min(j + np.ceil(beat_period * (1 + max_stretch_longer)) * spike_period[i] + 1, N),
                    lower_bound + 1,
                )
                candidate_slice = np.arange(lower_bound, upper_bound, dtype=np.int64)

                slice_arr = D[i, candidate_slice]
                D_vals = np.empty((len(pitches), len(slice_arr)), dtype=slice_arr.dtype)
                for pitch_rows in range(len(pitches)):
                    D_vals[pitch_rows] = slice_arr

                for pitch_idx, pitch in enumerate(pitches):
                    for candidate_idx, candidate_j in enumerate(candidate_slice):
                        # compute the stretch
                        stretch = max(
                            (candidate_j - j) / (beat_period * spike_period[i]),
                            (beat_period * spike_period[i]) / (candidate_j - j),
                        )
                        stretch_c = min((stretch ** 2 - 1), 1)

                        # onset activation
                        activation = onsets[pitch, candidate_j]
                        # spec slice
                        spec_slice = spec[pitch, candidate_j: candidate_j + spec_slice_len]
                        spec_fit = np.min(spec_slice)

                        # total cost for this candidate
                        candidate_j_cost = (
                            D[i, j]
                            + spike_cost * activation
                            + stretch_cost * stretch_c
                            + spec_fit * spec_cost
                        )
                        D_vals[pitch_idx, candidate_idx] = candidate_j_cost

                    # try to extract just the top candidates and fill it in the global matrix
                    # softmin-like function
                    D_vals_min = -np.log(np.sum(np.exp(-D_vals), axis=0))
                    sorted_idx = np.argsort(D_vals_min)
                    if i + 1 != M - 1:
                        for min_idx in sorted_idx[:candidate_onset_number]:
                            min_candidate_j_cost = D_vals_min[min_idx]
                            min_candidate_j = candidate_slice[min_idx]
                            if D[i + 1, min_candidate_j] > min_candidate_j_cost:
                                D[i + 1, min_candidate_j] = min_candidate_j_cost
                                BP[i + 1, min_candidate_j] = (
                                    (1 - alpha) * beat_period
                                    + alpha * min(
                                        max(float(min_candidate_j - j) / spike_period[i], min_bp), max_bp
                                    )
                                )
                                B[i + 1, min_candidate_j] = j
                    else:
                        min_idx = sorted_idx[0]
                        min_candidate_j_cost = D_vals_min[min_idx]
                        min_candidate_j = N - 1
                        if D[i + 1, min_candidate_j] > min_candidate_j_cost:
                            D[i + 1, min_candidate_j] = min_candidate_j_cost
                            BP[i + 1, min_candidate_j] = (
                                (1 - alpha) * beat_period
                                + alpha * min(
                                    max(float(min_candidate_j - j) / spike_period[i], min_bp), max_bp
                                )
                            )
                            B[i + 1, min_candidate_j] = j

            # rank-based pruning: keep only the cost_threshold lowest-cost columns
            min_cost_at_idx = np.argsort(D[i + 1, :])
            D[i + 1, min_cost_at_idx[cost_threshold:]] = np.inf

    # simple backtracking
    path = [N - 1]
    for backwards_i in range(M - 1, 0, -1):
        prev_spike = B[backwards_i, path[-1]]
        path.append(prev_spike)
    return D, B, BP, path[::-1]


# ---------------------------------------------------------------------------
# Public callable classes
# ---------------------------------------------------------------------------


class ElasticSpecDP:
    """
    Full elastic DP for audio-to-score alignment using pitch-onset features.

    Evaluates all candidate spectrogram frames within a beat-period stretch
    window at each score onset and prunes high-cost rows via an absolute
    cost threshold after each spike step.

    This class wraps :func:`_elastic_forward_and_backward_pitch_onset`,
    which is JIT-compiled via the shared
    :func:`~parangonar.decorators.numba_jit` decorator (accelerated when
    numba is installed, falls back to pure Python with a warning otherwise).

    Parameters
    ----------
    max_stretch_longer : float
        Maximum relative stretch above the beat period (e.g. ``0.5`` = +50%).
    max_stretch_shorter : float
        Maximum relative compression below the beat period (e.g. ``0.3`` = -30%).
    stretch_cost : float
        Weight for the stretch penalty term.
    spec_cost : float
        Weight for the spectrogram minimum-fit term.
    spike_cost : float
        Weight for the onset-activation term.
    max_bp : float
        Upper bound on the tracked beat period (in spectrogram frames).
    min_bp : float
        Lower bound on the tracked beat period (in spectrogram frames).
    alpha : float
        Beat-period adaptation rate (0 = no adaptation, 1 = full).
    cost_threshold : float
        Absolute cost delta for row pruning: columns whose cost exceeds
        ``min_cost_in_row + cost_threshold`` are set to ``inf``.
    spec_slice_len : int
        Length of the spectrogram slice taken around each candidate for the
        spec-fit cost (minimum over the slice).
    """

    def __init__(
        self,
        max_stretch_longer: float = 0.5,
        max_stretch_shorter: float = 0.3,
        stretch_cost: float = 0.1,
        spec_cost: float = 0.2,
        spike_cost: float = 0.7,
        max_bp: float = 64.0,
        min_bp: float = 1.0,
        alpha: float = 0.5,
        cost_threshold: float = 20.0,
        spec_slice_len: int = 7,
    ):
        self.max_stretch_longer = max_stretch_longer
        self.max_stretch_shorter = max_stretch_shorter
        self.stretch_cost = stretch_cost
        self.spec_cost = spec_cost
        self.spike_cost = spike_cost
        self.max_bp = max_bp
        self.min_bp = min_bp
        self.alpha = alpha
        self.cost_threshold = cost_threshold
        self.spec_slice_len = spec_slice_len

    def __call__(
        self,
        onsets: np.ndarray,
        spec: np.ndarray,
        spikes: np.ndarray,
        pitch_sets: list,
        bp_init: float,
    ) -> tuple:
        """
        Run the elastic DP forward-backward pass.

        Parameters
        ----------
        onsets : np.ndarray, shape (n_bins, n_frames)
            Inverted onset-activation features (1 - activation).
        spec : np.ndarray, shape (n_bins, n_frames)
            Inverted spectrogram features (1 - normalised).
        spikes : np.ndarray, shape (n_spikes,)
            Score-onset positions in spectrogram-frame indices.
        pitch_sets : list of np.ndarray
            Per-spike arrays of pitch-bin indices (0-indexed, 0 = A0).
        bp_init : float
            Initial beat-period estimate (frames per score beat).

        Returns
        -------
        D : np.ndarray, shape (n_spikes, n_frames)
            Accumulated cost matrix.
        B : np.ndarray, shape (n_spikes, n_frames), dtype int64
            Backtracking matrix.
        BP : np.ndarray, shape (n_spikes, n_frames)
            Beat-period matrix.
        path : list of int
            Backtracked path of spectrogram-frame indices, one per spike.
        """
        return _elastic_forward_and_backward_pitch_onset(
            onsets=onsets,
            spec=spec,
            spikes=spikes,
            pitch_sets=pitch_sets,
            bp_init=float(bp_init),
            max_stretch_longer=float(self.max_stretch_longer),
            max_stretch_shorter=float(self.max_stretch_shorter),
            stretch_cost=float(self.stretch_cost),
            spec_cost=float(self.spec_cost),
            spike_cost=float(self.spike_cost),
            max_bp=float(self.max_bp),
            min_bp=float(self.min_bp),
            alpha=float(self.alpha),
            cost_threshold=float(self.cost_threshold),
            spec_slice_len=int(self.spec_slice_len),
        )


class ElasticSpecDPLimited:
    """
    Limited elastic DP for audio-to-score alignment using pitch-onset features.

    Like :class:`ElasticSpecDP` but retains only the top-``N`` candidates per
    DP step using a soft-min (log-sum-exp), reducing memory and compute for
    long sequences.  ``cost_threshold`` is interpreted as a *rank* cutoff
    (number of lowest-cost columns to keep) rather than an absolute cost delta.

    This class wraps :func:`_elastic_forward_and_backward_pitch_onset_limit`,
    which is JIT-compiled via the shared
    :func:`~parangonar.decorators.numba_jit` decorator.

    Parameters
    ----------
    max_stretch_longer : float
        Maximum relative stretch above the beat period.
    max_stretch_shorter : float
        Maximum relative compression below the beat period.
    stretch_cost : float
        Weight for the stretch penalty term.
    spec_cost : float
        Weight for the spectrogram minimum-fit term.
    spike_cost : float
        Weight for the onset-activation term.
    max_bp : float
        Upper bound on the tracked beat period (in spectrogram frames).
    min_bp : float
        Lower bound on the tracked beat period (in spectrogram frames).
    alpha : float
        Beat-period adaptation rate.
    spec_slice_len : int
        Length of the spectrogram slice taken around each candidate for the
        spec-fit cost (minimum over the slice).
    cost_threshold : int
        Rank cutoff: only the ``cost_threshold`` lowest-cost columns are
        kept per row; the rest are set to ``inf``.
    candidate_onset_number : int
        Number of top candidates to write into the DP matrix per step
        (soft-min via log-sum-exp).

    Attributes
    ----------
    spec_slice_len : int
    cost_threshold : int
    candidate_onset_number : int
    """

    def __init__(
        self,
        max_stretch_longer: float = 0.5,
        max_stretch_shorter: float = 0.5,
        stretch_cost: float = 0.1,
        spec_cost: float = 0.2,
        spike_cost: float = 0.7,
        max_bp: float = 64.0,
        min_bp: float = 1.0,
        alpha: float = 0.5,
        spec_slice_len: int = 7,
        cost_threshold: int = 15,
        candidate_onset_number: int = 3,
    ):
        self.max_stretch_longer = max_stretch_longer
        self.max_stretch_shorter = max_stretch_shorter
        self.stretch_cost = stretch_cost
        self.spec_cost = spec_cost
        self.spike_cost = spike_cost
        self.max_bp = max_bp
        self.min_bp = min_bp
        self.alpha = alpha
        self.spec_slice_len = spec_slice_len
        self.cost_threshold = cost_threshold
        self.candidate_onset_number = candidate_onset_number

    def __call__(
        self,
        onsets: np.ndarray,
        spec: np.ndarray,
        spikes: np.ndarray,
        pitch_sets: list,
        bp_init: float,
    ) -> tuple:
        """
        Run the limited elastic DP forward-backward pass.

        Parameters
        ----------
        onsets : np.ndarray, shape (n_bins, n_frames)
            Inverted onset-activation features (1 - activation).
        spec : np.ndarray, shape (n_bins, n_frames)
            Inverted spectrogram features (1 - normalised).
        spikes : np.ndarray, shape (n_spikes,)
            Score-onset positions in spectrogram-frame indices.
        pitch_sets : list of np.ndarray
            Per-spike arrays of pitch-bin indices (0-indexed, 0 = A0).
        bp_init : float
            Initial beat-period estimate (frames per score beat).

        Returns
        -------
        D : np.ndarray, shape (n_spikes, n_frames)
            Accumulated cost matrix.
        B : np.ndarray, shape (n_spikes, n_frames), dtype int64
            Backtracking matrix.
        BP : np.ndarray, shape (n_spikes, n_frames)
            Beat-period matrix.
        path : list of int
            Backtracked path of spectrogram-frame indices, one per spike.
        """
        return _elastic_forward_and_backward_pitch_onset_limit(
            onsets=onsets,
            spec=spec,
            spikes=spikes,
            pitch_sets=pitch_sets,
            bp_init=float(bp_init),
            max_stretch_longer=float(self.max_stretch_longer),
            max_stretch_shorter=float(self.max_stretch_shorter),
            stretch_cost=float(self.stretch_cost),
            spec_cost=float(self.spec_cost),
            spike_cost=float(self.spike_cost),
            max_bp=float(self.max_bp),
            min_bp=float(self.min_bp),
            alpha=float(self.alpha),
            spec_slice_len=int(self.spec_slice_len),
            cost_threshold=int(self.cost_threshold),
            candidate_onset_number=int(self.candidate_onset_number),
        )
