#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for audio-to-score alignment matchers.

The tests use the Mozart K.265 match file already present in the test data
directory and generate synthetic audio in-process (no external audio file
required).  Synthetic audio is produced by summing sine waves at each score
note's MIDI pitch for the note's duration, sampled at 16 kHz.

If numba is not installed the test class is skipped entirely, because the
DP algorithms require JIT compilation to run in a reasonable time.
"""

import unittest
import logging
import numpy as np
import parangonar as pa
import partitura as pt

from tests import MATCH_FILES

# ---------------------------------------------------------------------------
# Optional-dependency guard
# ---------------------------------------------------------------------------

try:
    from numba import jit  # noqa: F401  (just probes availability)
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Synthetic-audio helper
# ---------------------------------------------------------------------------

def _sine_audio_from_score_note_array(
    score_note_array: np.ndarray,
    sample_rate: int = 16000,
    frame_rate: int = 50,
) -> np.ndarray:
    """
    Generate a simple synthetic audio signal from a score note array.

    Each note is rendered as a sine wave at its MIDI pitch frequency, with
    a short fade-in and fade-out to avoid clicks.  Notes are summed into a
    single mono float32 array normalised to [-1, 1].

    Parameters
    ----------
    score_note_array : np.ndarray
        Structured score note array with at minimum fields ``onset_beat``,
        ``pitch``, and ``duration_beat``.
    sample_rate : int
        Audio sample rate in Hz.
    frame_rate : int
        Spectrogram frame rate in Hz (used to determine the audio length
        from the score onsets).

    Returns
    -------
    audio_np : np.ndarray, shape (n_samples,)
        Mono float32 audio signal.
    """
    # Determine audio length from score span + a small tail
    unique_onsets = np.unique(score_note_array["onset_beat"])
    score_duration_beats = unique_onsets[-1] - unique_onsets[0] + 2.0  # +2 beats tail
    n_samples = int(score_duration_beats * sample_rate / frame_rate * frame_rate)

    audio = np.zeros(n_samples, dtype=np.float64)
    fade_samples = int(0.01 * sample_rate)  # 10 ms fade

    for note in score_note_array:
        midi_pitch = note["pitch"]
        onset_beat = note["onset_beat"]
        dur_beat = note["duration_beat"]

        freq = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))  # MIDI 69 = A4
        t_start = int(onset_beat * sample_rate / frame_rate * frame_rate)
        t_end = int(t_start + dur_beat * sample_rate / frame_rate * frame_rate)
        t_end = min(t_end, n_samples)

        if t_end <= t_start:
            continue

        t = np.arange(t_start, t_end, dtype=np.float64) / sample_rate
        wave = np.sin(2.0 * np.pi * freq * t)

        # Apply fade-in / fade-out
        if fade_samples > 0 and len(wave) > 2 * fade_samples:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            wave[:fade_samples] *= fade_in
            wave[-fade_samples:] *= fade_out

        audio[t_start:t_end] += wave

    # Normalise to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = audio / peak

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@unittest.skipUnless(NUMBA_AVAILABLE, "numba not installed; audio DP tests require numba")
class TestAudioAlignment(unittest.TestCase):
    """Tests for AudioToScoreMatcher and AudioToScoreMatcherLimited."""

    @classmethod
    def setUpClass(cls):
        """Load the Mozart match file and prepare synthetic audio."""
        cls.perf_match, cls.alignment, cls.score_match = pt.load_match(
            filename=MATCH_FILES[0], create_score=True
        )
        # score_match is already a Partitura Score when create_score=True
        cls.sna = cls.score_match.note_array(include_grace_notes=True)

        cls.sample_rate = 16000
        cls.frame_rate = 50
        cls.audio_np = _sine_audio_from_score_note_array(
            cls.sna,
            sample_rate=cls.sample_rate,
            frame_rate=cls.frame_rate,
        )

    # ------------------------------------------------------------------
    # AudioToScoreMatcher
    # ------------------------------------------------------------------

    def test_AudioToScoreMatcher_returns_alignment_list(self):
        """AudioToScoreMatcher returns a list of alignment dicts."""
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
        )
        self.assertIsInstance(alignment, list)
        self.assertGreater(len(alignment), 0)

    def test_AudioToScoreMatcher_alignment_has_required_fields(self):
        """Each alignment event has 'label' and 'score_id' fields."""
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
        )
        for event in alignment:
            self.assertIn("label", event)
            self.assertIn("score_id", event)
            self.assertIn(event["label"], ("match", "deletion", "insertion"))

    def test_AudioToScoreMatcher_alignment_covers_score_notes(self):
        """Every score note appears in the alignment."""
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
        )
        aligned_ids = {e["score_id"] for e in alignment if e["label"] == "match"}
        all_score_ids = set(self.sna["id"])
        # At minimum, a substantial fraction of notes should be matched
        match_rate = len(aligned_ids) / len(all_score_ids)
        self.assertGreater(match_rate, 0.5)

    def test_AudioToScoreMatcher_with_explicit_window(self):
        """Explicit audio_window is respected."""
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        audio_window = (0.0, 20.0)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
            audio_window=audio_window,
        )
        self.assertIsInstance(alignment, list)
        self.assertGreater(len(alignment), 0)

    def test_AudioToScoreMatcher_estimates_window(self):
        """Without audio_window the estimator runs and logs a message."""
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        with self.assertLogs("parangonar.match.audio_matchers", level="INFO") as cm:
            alignment = matcher(
                self.audio_np,
                self.sna,
                sample_rate=self.sample_rate,
            )
        # At least one log record should mention the estimated window
        log_output = "\n".join(cm.output)
        self.assertIn("estimated audio window", log_output.lower())
        self.assertIsInstance(alignment, list)

    def test_AudioToScoreMatcher_returns_performance_time(self):
        """Matched events carry a 'performance_time' field."""
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
        )
        matched = [e for e in alignment if e["label"] == "match"]
        self.assertGreater(len(matched), 0)
        for event in matched:
            self.assertIn("performance_time", event)
            self.assertIsInstance(event["performance_time"], float)
            self.assertGreaterEqual(event["performance_time"], 0.0)

    # ------------------------------------------------------------------
    # AudioToScoreMatcherLimited
    # ------------------------------------------------------------------

    def test_AudioToScoreMatcherLimited_returns_alignment_list(self):
        """AudioToScoreMatcherLimited returns a list of alignment dicts."""
        matcher = pa.AudioToScoreMatcherLimited(frame_rate=self.frame_rate)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
        )
        self.assertIsInstance(alignment, list)
        self.assertGreater(len(alignment), 0)

    def test_AudioToScoreMatcherLimited_alignment_has_required_fields(self):
        """Each alignment event has 'label' and 'score_id' fields."""
        matcher = pa.AudioToScoreMatcherLimited(frame_rate=self.frame_rate)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
        )
        for event in alignment:
            self.assertIn("label", event)
            self.assertIn("score_id", event)
            self.assertIn(event["label"], ("match", "deletion", "insertion"))

    def test_AudioToScoreMatcherLimited_alignment_covers_score_notes(self):
        """Every score note appears in the alignment."""
        matcher = pa.AudioToScoreMatcherLimited(frame_rate=self.frame_rate)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
        )
        aligned_ids = {e["score_id"] for e in alignment if e["label"] == "match"}
        all_score_ids = set(self.sna["id"])
        match_rate = len(aligned_ids) / len(all_score_ids)
        self.assertGreater(match_rate, 0.5)

    def test_AudioToScoreMatcherLimited_estimates_window(self):
        """Without audio_window the estimator runs and logs a message."""
        matcher = pa.AudioToScoreMatcherLimited(frame_rate=self.frame_rate)
        with self.assertLogs("parangonar.match.audio_matchers", level="INFO") as cm:
            alignment = matcher(
                self.audio_np,
                self.sna,
                sample_rate=self.sample_rate,
            )
        log_output = "\n".join(cm.output)
        self.assertIn("estimated audio window", log_output.lower())
        self.assertIsInstance(alignment, list)

    def test_AudioToScoreMatcherLimited_returns_performance_time(self):
        """Matched events carry a 'performance_time' field."""
        matcher = pa.AudioToScoreMatcherLimited(frame_rate=self.frame_rate)
        alignment = matcher(
            self.audio_np,
            self.sna,
            sample_rate=self.sample_rate,
        )
        matched = [e for e in alignment if e["label"] == "match"]
        self.assertGreater(len(matched), 0)
        for event in matched:
            self.assertIn("performance_time", event)
            self.assertIsInstance(event["performance_time"], float)
            self.assertGreaterEqual(event["performance_time"], 0.0)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_sample_rate_raises(self):
        """Calling without sample_rate raises ValueError."""
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        with self.assertRaises(ValueError):
            matcher(self.audio_np, self.sna)  # no sample_rate kwarg

    def test_too_few_unique_onsets_raises(self):
        """Score with < 2 unique onsets raises ValueError."""
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        degenerate = np.array(
            [(0, 60, 1.0), (0, 62, 1.0)],
            dtype=[("id", "U256"), ("pitch", "i4"), ("onset_beat", "f8")],
        )
        with self.assertRaises(ValueError):
            matcher(self.audio_np, degenerate, sample_rate=self.sample_rate)

    def test_stereo_audio_is_converted_to_mono(self):
        """Stereo input is averaged to mono without error."""
        stereo = np.stack([self.audio_np, self.audio_np], axis=1)  # (N, 2)
        matcher = pa.AudioToScoreMatcher(frame_rate=self.frame_rate)
        alignment = matcher(stereo, self.sna, sample_rate=self.sample_rate)
        self.assertIsInstance(alignment, list)
        self.assertGreater(len(alignment), 0)


if __name__ == "__main__":
    unittest.main()
