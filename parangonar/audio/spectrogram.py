#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains audio spectrogram extraction utilities.

The IIRSpect class implements an IIR-based log-frequency spectrogram
using a bank of 2nd-order Butterworth bandpass filters.
"""

from typing import Optional
import numpy as np
from scipy import signal
from joblib import Parallel, delayed
from itertools import repeat


class IIRSpect:
    """
    IIR-based log-frequency spectrogram.

    Applies a bank of 2nd-order Butterworth bandpass filters (one per
    log-spaced frequency bin) and takes the max-absolute value per hop
    frame, producing a ``(n_bins, n_frames)`` spectrogram.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the input signal.
    n_fft : int
        Window length for the FFT (in samples).  Only used to compute
        ``view_to_the_past = hop_length - n_fft``.
    hop_length : int
        Hop size in samples; also determines the output frame rate.
    f_min : float
        Lower bound of the first filter (Hz).
    f_max : float
        Upper bound of the last filter (Hz).
    n_bins : int
        Number of frequency bins (filters).
    power : int
        Whether to compute magnitudes (1) or energy (2) of the complex
        spectrogram (currently unused in the forward pass).
    log_multiplier : float
        Factor that the magnitudes are multiplied with before adding 1.0
        and taking the logarithm (used externally, not inside this class).
    device : str
        Device string (currently unused; kept for API compatibility).
    rir_prob : float
        With what probability to apply a room impulse response
        (currently unused).
    shift_prob : float
        With what probability to apply pitch shifting (currently unused).
    shift_max : float
        How many semitones (or fractions of semitones) to shift at most
        (currently unused).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 160,
        f_min: float = 27.5,
        f_max: float = 4186.009,
        n_bins: int = 88,
        power: int = 1,
        log_multiplier: float = 1000,
        device: str = "cpu",
        rir_prob: float = 0.0,
        shift_prob: float = 0.0,
        shift_max: float = 0.1,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.log_multiplier = log_multiplier
        self.power = power
        self.device = device
        self.view_to_the_past = hop_length - self.n_fft
        self.boundary_freqs = np.logspace(
            np.log2(0.5 * f_min * 2 ** (23 / 24)),
            np.log2(f_max * 2 ** (1 / 24)),
            n_bins + 1,
            base=2,
        )
        self.center_freqs = np.logspace(np.log2(f_min), np.log2(f_max), n_bins, base=2)
        self.filter_order = 2
        self.nyq = 0.5 * self.sample_rate
        self.filters = []
        for i, f in enumerate(self.boundary_freqs[:-1]):
            low = f / self.nyq
            high = self.boundary_freqs[i + 1] / self.nyq
            coeff_array = signal.butter(
                N=self.filter_order, Wn=[low, high], btype="band", output="sos"
            )
            self.filters.append(coeff_array)

    def apply_sos_and_max_filter(
        self, x: np.ndarray, coeff_array: np.ndarray, num_windows: int
    ) -> np.ndarray:
        """
        Apply a single SOS bandpass filter and take the max-absolute value
        within each hop frame.

        Parameters
        ----------
        x : np.ndarray
            Input audio signal.
        coeff_array : np.ndarray
            Second-order sections coefficients for one bandpass filter.
        num_windows : int
            Number of hop frames to extract.

        Returns
        -------
        np.ndarray
            Max-absolute value per frame, shape ``(num_windows,)``.
        """
        filtered_signal = signal.sosfilt(coeff_array, x)
        output_max_filt_signal = []
        for j in range(num_windows):
            start = j * self.hop_length
            start_past = max(0, start + self.view_to_the_past)
            segment = filtered_signal[start_past : start + self.hop_length]
            output_max_filt_signal.append(np.max(np.abs(segment)))
        return np.array(output_max_filt_signal)

    def __call__(self, x_np: np.ndarray) -> np.ndarray:
        """
        Compute the IIR log-frequency spectrogram.

        Parameters
        ----------
        x_np : np.ndarray
            Input audio signal (mono, 1-D numpy array).

        Returns
        -------
        np.ndarray
            Spectrogram of shape ``(n_bins, n_frames)``.
        """
        num_windows = len(x_np) // self.hop_length
        results = Parallel(n_jobs=-1)(
            delayed(self.apply_sos_and_max_filter)(sig, car, num_windows)
            for sig, car, num_windows in zip(
                repeat(x_np), self.filters, repeat(num_windows)
            )
        )
        spectrogram = np.array(results)
        return spectrogram
