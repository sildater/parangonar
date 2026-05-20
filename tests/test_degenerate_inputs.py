#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for matcher behavior on degenerate inputs.

DualDTWNoteMatcher previously crashed with an IndexError when the
internal `unique_time_tuples_by_onset` dict ended up with fewer than 2
anchor pairs (e.g., a tiny performance against a much larger score, or
pitch-disjoint inputs). It should produce a (possibly poor-quality)
alignment instead, so downstream consumers can decide what to do.
"""

import unittest

import numpy as np
import partitura as pt
import partitura.score as pscore

from parangonar import DualDTWNoteMatcher


def _build_short_score():
    """Build a 4-measure C-major scale Part programmatically."""
    part = pscore.Part("P0", "scale", quarter_duration=4)
    part.add(pscore.TimeSignature(4, 4), start=0)
    pitches = [("C", 4), ("D", 4), ("E", 4), ("F", 4),
               ("G", 4), ("A", 4), ("B", 4), ("C", 5)]
    for i, (s, o) in enumerate(pitches):
        part.add(pscore.Note(step=s, octave=o, alter=None), start=i * 4, end=(i + 1) * 4)
    pscore.add_measures(part)
    return part


def _perf_note_array(pitches, onsets, durations):
    """Build a performance note_array directly without round-tripping MIDI."""
    n = len(pitches)
    dtype = [
        ("onset_sec", "f4"),
        ("duration_sec", "f4"),
        ("onset_tick", "i4"),
        ("duration_tick", "i4"),
        ("pitch", "i4"),
        ("velocity", "i4"),
        ("track", "i4"),
        ("channel", "i4"),
        ("id", "<U16"),
    ]
    arr = np.zeros(n, dtype=dtype)
    arr["onset_sec"] = onsets
    arr["duration_sec"] = durations
    arr["pitch"] = pitches
    arr["velocity"] = 80
    arr["id"] = [f"n{i}" for i in range(n)]
    return arr


class TestDegenerateInputs(unittest.TestCase):
    def setUp(self):
        self.part = _build_short_score()
        self.sna = self.part.note_array(include_grace_notes=True)

    def test_one_note_performance_does_not_crash(self):
        """Aligning a 1-note performance against an 8-note score used to
        raise IndexError in get_score_to_perf_map; now it returns a list."""
        pna = _perf_note_array([60], [0.0], [0.5])
        matcher = DualDTWNoteMatcher()
        result = matcher(self.sna, pna, process_ornaments=True, score_part=self.part)
        self.assertIsInstance(result, list)

    def test_pitch_disjoint_performance_does_not_crash(self):
        """Performance whose pitches are disjoint from the score must not crash."""
        pna = _perf_note_array([34, 37, 42], [0.0, 0.4, 0.8], [0.3, 0.3, 0.3])
        matcher = DualDTWNoteMatcher()
        result = matcher(self.sna, pna, process_ornaments=True, score_part=self.part)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
