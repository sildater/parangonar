#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains audio-to-score matcher classes.

Each matcher assembles the full alignment pipeline:

1. Convert raw audio to an IIR log-frequency spectrogram.
2. Compute onset-activation and normalised spectrogram features.
3. Prepare the score note array into onset-grouped pitch-set representation.
4. Run the elastic DP algorithm.
5. Convert the DP path into a parangonar-format alignment list.

The public interface mirrors the symbolic matchers in
:mod:`parangonar.match.matchers`: call the matcher with a numpy audio array
and a score note array, and receive a list of alignment event dicts.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import warnings

import numpy as np
from scipy.ndimage import maximum_filter
from partitura.utils.generic import interp1d

from ..audio.spectrogram import IIRSpect
from ..dp.spec_dp import ElasticSpecDP, ElasticSpecDPLimited

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_score(s_array: np.ndarray) -> tuple:
    """
    Convert a structured score note array into onset-grouped pitch-set
    representation and a sorted array of unique onset beats.

    Parameters
    ----------
    s_array : np.ndarray
        Structured score note array with at minimum fields ``onset_beat``
        and ``pitch``.

    Returns
    -------
    features : list of [onset_beat, set_of_pitches]
        One entry per unique score onset.
    unique_onsets : np.ndarray
        Sorted unique ``onset_beat`` values.
    """
    features = []
    unique_onsets = np.unique(s_array["onset_beat"])
    for onset in unique_onsets:
        features.append([onset, set(s_array[s_array["onset_beat"] == onset]["pitch"])])
    return features, unique_onsets


def _pitch_sets_from_features(features: list, pitch_offset: int = 21) -> list:
    """
    Convert the output of :func:`_prepare_score` into the 0-indexed
    pitch-bin arrays expected by the DP algorithm.

    Parameters
    ----------
    features : list
        Output of :func:`_prepare_score`.
    pitch_offset : int
        MIDI note number of the lowest bin (A0 = 21).  Pitches are
        subtracted by this value to produce 0-indexed bin indices.

    Returns
    -------
    pitch_sets : list of np.ndarray
    """
    return [np.array(list(b)) - pitch_offset for a, b in features]


def _compute_audio_features(
    audio_np: np.ndarray,
    sample_rate: int,
    frame_rate: int,
    f_min: float,
    f_max: float,
    n_bins: int,
    log_multiplier: float,
) -> tuple:
    """
    Compute onset-activation and normalised spectrogram features from raw
    audio.

    Parameters
    ----------
    audio_np : np.ndarray
        Raw audio samples (mono or stereo).  Stereo is averaged to mono.
    sample_rate : int
        Sample rate of ``audio_np``.
    frame_rate : int
        Desired spectrogram frame rate in Hz.
    f_min : float
        Lowest filter frequency in Hz.
    f_max : float
        Highest filter frequency in Hz.
    n_bins : int
        Number of IIR filter bands.
    log_multiplier : float
        Log-scaling factor for spectrogram post-processing.

    Returns
    -------
    onsets : np.ndarray, shape (n_bins, n_frames)
        Inverted onset-activation features (1 - normalised).
    coeff : np.ndarray, shape (n_bins, n_frames)
        Inverted normalised spectrogram (1 - normalised).
    """
    hop_length = int(sample_rate / frame_rate)
    n_fft = hop_length  # matches the test code convention

    processor = IIRSpect(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_bins=n_bins,
        log_multiplier=log_multiplier,
    )
    iirspec = processor(audio_np)
    iirlspec = np.log1p(log_multiplier * iirspec)
    iirlspec3 = maximum_filter(iirlspec, size=(3, 1))
    iirlspec4 = np.maximum(0, iirlspec[:, 1:] - iirlspec3[:, :-1])

    # Normalise onset features
    iirlspec4_row_max = iirlspec4.max(axis=1, keepdims=True)
    iirlspec4_norm = iirlspec4 / np.where(
        iirlspec4_row_max == 0, 1.0, iirlspec4_row_max
    )

    # Normalise spectrogram features
    iirspec_row_max = iirspec.max(axis=1, keepdims=True)
    coeff = iirspec / np.where(iirspec_row_max == 0, 1.0, iirspec_row_max)

    # Invert so that "low cost" = "high activation" in the DP
    onsets = 1.0 - iirlspec4_norm
    spec = 1.0 - coeff

    return onsets, spec


def _estimate_audio_window(
    onsets: np.ndarray,
    frame_rate: int,
    onset_threshold_ratio: float = 0.05,
    onset_margin_sec: float = 0.05,
    min_window_sec: float = 1.0,
) -> Optional[tuple]:
    """
    Estimate the musically relevant audio window from onset-activation features.

    Assumes the audio contains silence or background noise before the first
    real onset and after the last real onset.  Finds the first and last
    spectrogram frames whose activation exceeds a relative threshold and
    returns the corresponding time interval, with a safety margin added on
    each side.

    Parameters
    ----------
    onsets : np.ndarray, shape (n_bins, n_frames)
        Inverted onset-activation features as returned by
        :func:`_compute_audio_features` (i.e. 1 - normalised_activation).
    frame_rate : int
        Spectrogram frame rate in Hz.
    onset_threshold_ratio : float
        Fraction of the peak frame energy used as the detection threshold.
        Frames whose (non-inverted) activation exceeds this fraction of the
        global peak are considered "active".
    onset_margin_sec : float
        Safety margin added on each side of the detected window (seconds).
    min_window_sec : float
        Minimum window length enforced even for very short pieces (seconds).

    Returns
    -------
    (start_sec, end_sec) : tuple of float, or None
        Estimated audio window in seconds, or ``None`` if no active frames
        were found (caller should fall back to the full spectrogram).
    """
    # Un-invert to get normalised activation: 0 = silence, 1 = strong onset
    activation = 1.0 - onsets
    # Frame-level energy: max activation across all pitch bins
    frame_energy = activation.max(axis=0)  # shape (n_frames,)

    peak = frame_energy.max()
    if peak == 0.0:
        return None  # all-zero audio; caller handles this

    threshold = onset_threshold_ratio * peak
    active_frames = np.where(frame_energy > threshold)[0]

    if len(active_frames) == 0:
        return None

    first_frame = int(active_frames[0])
    last_frame = int(active_frames[-1])
    n_frames = onsets.shape[1]

    # Add safety margin
    margin_frames = int(onset_margin_sec * frame_rate)
    first_frame = max(0, first_frame - margin_frames)
    last_frame = min(n_frames - 1, last_frame + margin_frames)

    # Enforce minimum window length
    min_window_frames = int(min_window_sec * frame_rate)
    if last_frame - first_frame < min_window_frames:
        centre = (first_frame + last_frame) // 2
        half = min_window_frames // 2
        first_frame = max(0, centre - half)
        last_frame = min(n_frames - 1, first_frame + min_window_frames)
        if last_frame == n_frames - 1:
            first_frame = max(0, last_frame - min_window_frames)

    start_sec = first_frame / frame_rate
    end_sec = (last_frame + 1) / frame_rate  # +1 because slice_end is exclusive

    return start_sec, end_sec, first_frame, last_frame, n_frames


def _path_to_alignment(
    path: list,
    unique_score_onsets: np.ndarray,
    frame_rate: int,
    slice_start: int,
    score_note_array: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Convert a DP backtracked path into a parangonar-format alignment list.

    Parameters
    ----------
    path : list of int
        Backtracked path of spectrogram-frame indices, one per spike.
    unique_score_onsets : np.ndarray
        Sorted unique score onset beats.
    frame_rate : int
        Spectrogram frame rate in Hz.
    slice_start : int
        First spectrogram frame index used in the DP (for time conversion).
    score_note_array : np.ndarray
        Full score note array (used to assign ``score_id`` values).

    Returns
    -------
    alignment : list of dict
        Alignment events with ``label``, ``score_id``, and
        ``performance_time``.
    """
    # Convert frame indices to audio seconds
    predicted_perf_times = np.array(path) / frame_rate + slice_start / frame_rate

    # Build a map from score onset beat -> audio time
    predicted_stime_to_ptime_map = interp1d(
        y=predicted_perf_times,
        x=unique_score_onsets,
        fill_value="extrapolate",
    )

    # Build alignment by matching each score note to its audio time
    alignment = []

    for note in score_note_array:
        sid = note["id"]
        s_onset = note["onset_beat"]
        p_time = float(predicted_stime_to_ptime_map(s_onset))
        alignment.append(
            {"label": "match", "score_id": sid, "performance_time": p_time}
        )

    return alignment


# ---------------------------------------------------------------------------
# Public matcher classes
# ---------------------------------------------------------------------------


class AudioToScoreMatcher:
    """
    Audio-to-score alignment via elastic DP with full candidate evaluation.

    Accepts a raw audio numpy array and a score note array, computes
    IIR-based log-frequency spectrogram features, and runs
    :class:`~parangonar.dp.spec_dp.ElasticSpecDP` to produce a
    score-onset-to-audio-time alignment.

    Parameters
    ----------
    frame_rate : int
        Spectrogram frame rate in Hz (default ``50``).
    f_min : float
        Lowest filter frequency in Hz (default ``27.5`` = A0).
    f_max : float
        Highest filter frequency in Hz (default ``4186.0`` = C8).
    n_bins : int
        Number of IIR filter bands (default ``88``, one per piano key).
    log_multiplier : float
        Log-scaling factor for spectrogram post-processing (default ``1000``).
    pitch_offset : int
        MIDI note number of the lowest frequency bin (default ``21`` = A0).
        Score pitches are subtracted by this value to produce 0-indexed
        bin indices for the DP.
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
    cost_threshold : float
        Absolute cost delta for row pruning in the DP.
    spec_slice_len : int
        Spectrogram slice length (informational; hard-coded to 7 in the
        underlying JIT function).

    Attributes
    ----------
    spec_processor : IIRSpect
        The IIR spectrogram extractor (constructed at call time).
    dp_algo : ElasticSpecDP
        The DP algorithm instance.
    """

    def __init__(
        self,
        frame_rate: int = 50,
        f_min: float = 27.5,
        f_max: float = 4186.0,
        n_bins: int = 88,
        log_multiplier: float = 1000.0,
        pitch_offset: int = 21,
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
        self.frame_rate = frame_rate
        self.f_min = f_min
        self.f_max = f_max
        self.n_bins = n_bins
        self.log_multiplier = log_multiplier
        self.pitch_offset = pitch_offset
        self.dp_algo = ElasticSpecDP(
            max_stretch_longer=max_stretch_longer,
            max_stretch_shorter=max_stretch_shorter,
            stretch_cost=stretch_cost,
            spec_cost=spec_cost,
            spike_cost=spike_cost,
            max_bp=max_bp,
            min_bp=min_bp,
            alpha=alpha,
            cost_threshold=cost_threshold,
            spec_slice_len=spec_slice_len,
        )

    def __call__(
        self,
        audio_np: np.ndarray,
        score_note_array: np.ndarray,
        sample_rate: Optional[int] = None,
        audio_window: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Align ``audio_np`` to ``score_note_array``.

        Parameters
        ----------
        audio_np : np.ndarray
            Raw audio samples (mono or stereo).  Stereo is averaged to mono
            automatically.
        score_note_array : np.ndarray
            Structured score note array with at minimum fields ``onset_beat``
            and ``pitch`` (standard MIDI note numbers, 21–108 for piano).
        sample_rate : int, optional
            Sample rate of ``audio_np`` in Hz.  Required when the audio
            array does not carry sample-rate metadata.
        audio_window : tuple of (start_sec, end_sec), optional
            If given, restrict the spectrogram to this time window before
            running the DP.  When ``None`` (default) the full spectrogram
            is used.

        Returns
        -------
        alignment : list of dict
            Alignment events.  Each dict has ``label`` (``"match"``),
            ``score_id``, and ``performance_time`` (seconds).

        Raises
        ------
        ValueError
            If ``sample_rate`` is not provided and cannot be inferred, or if
            the score note array has fewer than 2 unique onsets.
        """
        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided explicitly or embedded in the "
                "audio array metadata."
            )

        # --- audio preprocessing ---
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)  # stereo -> mono

        onsets, spec = _compute_audio_features(
            audio_np=audio_np,
            sample_rate=sample_rate,
            frame_rate=self.frame_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            n_bins=self.n_bins,
            log_multiplier=self.log_multiplier,
        )

        # --- score preprocessing ---
        features, unique_onsets = _prepare_score(score_note_array)
        pitch_sets = _pitch_sets_from_features(features, pitch_offset=self.pitch_offset)

        if len(unique_onsets) < 2:
            raise ValueError(
                "Score note array must contain at least 2 unique onset beats "
                f"to compute an initial beat-period estimate, got {len(unique_onsets)}."
            )

        # --- determine spectrogram slice ---
        if audio_window is not None:
            start_sec, end_sec = audio_window
            slice_start = max(0, int(start_sec * self.frame_rate))
            slice_end = min(onsets.shape[1], int(end_sec * self.frame_rate) + 1)
        else:
            estimated = _estimate_audio_window(
                onsets=onsets,
                frame_rate=self.frame_rate,
            )
            if estimated is not None:
                (
                    est_start_sec,
                    est_end_sec,
                    est_first_frame,
                    est_last_frame,
                    n_frames,
                ) = estimated
                slice_start = est_first_frame
                slice_end = est_last_frame + 1
                logger.info(
                    "AudioToScoreMatcher: estimated audio window %.3f s – %.3f s "
                    "(frames %d – %d of %d, frame_rate=%d Hz).",
                    est_start_sec,
                    est_end_sec,
                    est_first_frame,
                    est_last_frame,
                    n_frames,
                    self.frame_rate,
                )
            else:
                slice_start = 0
                slice_end = onsets.shape[1]
                logger.info(
                    "AudioToScoreMatcher: could not estimate audio window from onset "
                    "features; using full spectrogram (%.3f s – %.3f s).",
                    slice_start / self.frame_rate,
                    slice_end / self.frame_rate,
                )

        onsets_slice = onsets[:, slice_start:slice_end]
        spec_slice = spec[:, slice_start:slice_end]

        # --- initial beat-period estimate ---
        # frames per score beat  (audio_frames / score_beats)
        bp_average = (slice_end - slice_start) / (unique_onsets[-1] - unique_onsets[0])

        # --- run DP ---
        _D, _B, _BP, path = self.dp_algo(
            onsets=onsets_slice,
            spec=spec_slice,
            spikes=unique_onsets,
            pitch_sets=pitch_sets,
            bp_init=bp_average,
        )

        # --- convert path to alignment ---
        alignment = _path_to_alignment(
            path=path,
            unique_score_onsets=unique_onsets,
            frame_rate=self.frame_rate,
            slice_start=slice_start,
            score_note_array=score_note_array,
        )

        return alignment


class AudioToScoreMatcherLimited:
    """
    Audio-to-score alignment via elastic DP with limited candidate evaluation.

    Identical to :class:`AudioToScoreMatcher` but uses
    :class:`~parangonar.dp.spec_dp.ElasticSpecDPLimited`, which retains only
    the top-N candidates per DP step (soft-min via log-sum-exp), trading a
    small amount of accuracy for speed and memory on long sequences.

    Parameters
    ----------
    frame_rate : int
        Spectrogram frame rate in Hz (default ``50``).
    f_min : float
        Lowest filter frequency in Hz (default ``27.5``).
    f_max : float
        Highest filter frequency in Hz (default ``4186.0``).
    n_bins : int
        Number of IIR filter bands (default ``88``).
    log_multiplier : float
        Log-scaling factor (default ``1000``).
    pitch_offset : int
        MIDI note number of the lowest bin (default ``21``).
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
        Upper bound on the tracked beat period.
    min_bp : float
        Lower bound on the tracked beat period.
    alpha : float
        Beat-period adaptation rate.
    spec_slice_len : int
        Spectrogram slice length.
    cost_threshold : int
        Rank cutoff: only the ``cost_threshold`` lowest-cost columns are
        kept per row.
    candidate_onset_number : int
        Number of top candidates to write into the DP matrix per step.

    Attributes
    ----------
    spec_processor : IIRSpect
        The IIR spectrogram extractor (constructed at call time).
    dp_algo : ElasticSpecDPLimited
        The DP algorithm instance.
    """

    def __init__(
        self,
        frame_rate: int = 50,
        f_min: float = 27.5,
        f_max: float = 4186.0,
        n_bins: int = 88,
        log_multiplier: float = 1000.0,
        pitch_offset: int = 21,
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
        self.frame_rate = frame_rate
        self.f_min = f_min
        self.f_max = f_max
        self.n_bins = n_bins
        self.log_multiplier = log_multiplier
        self.pitch_offset = pitch_offset
        self.dp_algo = ElasticSpecDPLimited(
            max_stretch_longer=max_stretch_longer,
            max_stretch_shorter=max_stretch_shorter,
            stretch_cost=stretch_cost,
            spec_cost=spec_cost,
            spike_cost=spike_cost,
            max_bp=max_bp,
            min_bp=min_bp,
            alpha=alpha,
            spec_slice_len=spec_slice_len,
            cost_threshold=cost_threshold,
            candidate_onset_number=candidate_onset_number,
        )

    def __call__(
        self,
        audio_np: np.ndarray,
        score_note_array: np.ndarray,
        sample_rate: Optional[int] = None,
        audio_window: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Align ``audio_np`` to ``score_note_array`` using the limited DP.

        Parameters
        ----------
        audio_np : np.ndarray
            Raw audio samples (mono or stereo).
        score_note_array : np.ndarray
            Structured score note array with fields ``onset_beat`` and
            ``pitch``.
        sample_rate : int, optional
            Sample rate of ``audio_np`` in Hz.
        audio_window : tuple of (start_sec, end_sec), optional
            Restrict the spectrogram to this time window.

        Returns
        -------
        alignment : list of dict
            Alignment events with ``label``, ``score_id``, and
            ``performance_time``.

        Raises
        ------
        ValueError
            If ``sample_rate`` is not provided, or if the score has fewer
            than 2 unique onsets.
        """
        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided explicitly or embedded in the "
                "audio array metadata."
            )

        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)

        onsets, spec = _compute_audio_features(
            audio_np=audio_np,
            sample_rate=sample_rate,
            frame_rate=self.frame_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            n_bins=self.n_bins,
            log_multiplier=self.log_multiplier,
        )

        features, unique_onsets = _prepare_score(score_note_array)
        pitch_sets = _pitch_sets_from_features(features, pitch_offset=self.pitch_offset)

        if len(unique_onsets) < 2:
            raise ValueError(
                "Score note array must contain at least 2 unique onset beats, "
                f"got {len(unique_onsets)}."
            )

        if audio_window is not None:
            start_sec, end_sec = audio_window
            slice_start = max(0, int(start_sec * self.frame_rate))
            slice_end = min(onsets.shape[1], int(end_sec * self.frame_rate) + 1)
        else:
            estimated = _estimate_audio_window(
                onsets=onsets,
                frame_rate=self.frame_rate,
            )
            if estimated is not None:
                (
                    est_start_sec,
                    est_end_sec,
                    est_first_frame,
                    est_last_frame,
                    n_frames,
                ) = estimated
                slice_start = est_first_frame
                slice_end = est_last_frame + 1
                logger.info(
                    "AudioToScoreMatcherLimited: estimated audio window %.3f s – %.3f s "
                    "(frames %d – %d of %d, frame_rate=%d Hz).",
                    est_start_sec,
                    est_end_sec,
                    est_first_frame,
                    est_last_frame,
                    n_frames,
                    self.frame_rate,
                )
            else:
                slice_start = 0
                slice_end = onsets.shape[1]
                logger.info(
                    "AudioToScoreMatcherLimited: could not estimate audio window from "
                    "onset features; using full spectrogram (%.3f s – %.3f s).",
                    slice_start / self.frame_rate,
                    slice_end / self.frame_rate,
                )

        onsets_slice = onsets[:, slice_start:slice_end]
        spec_slice = spec[:, slice_start:slice_end]

        bp_average = (slice_end - slice_start) / (unique_onsets[-1] - unique_onsets[0])

        _D, _B, _BP, path = self.dp_algo(
            onsets=onsets_slice,
            spec=spec_slice,
            spikes=unique_onsets,
            pitch_sets=pitch_sets,
            bp_init=bp_average,
        )

        alignment = _path_to_alignment(
            path=path,
            unique_score_onsets=unique_onsets,
            frame_rate=self.frame_rate,
            slice_start=slice_start,
            score_note_array=score_note_array,
        )

        return alignment
