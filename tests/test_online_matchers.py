#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for the online matchers.

TOLTWMatcher read its starting tempo off the first score note, which is
0/0 when that note is a grace note, and a NaN starting tempo never tracks.

The transformer matchers could step outside the score when the model
predicts a performance or padding token, e.g. near the last score onset.
"""

import unittest

import numpy as np

from parangonar import (
    OnlinePureTransformerMatcher,
    OnlineTransformerMatcher,
    TOLTWMatcher,
)

NOTE_FIELDS = [
    ("onset_beat", "f4"),
    ("duration_beat", "f4"),
    ("onset_quarter", "f4"),
    ("duration_quarter", "f4"),
    ("pitch", "i4"),
    ("is_grace", "?"),
    ("id", "U256"),
]

PERF_FIELDS = [
    ("onset_sec", "f4"),
    ("duration_sec", "f4"),
    ("pitch", "i4"),
    ("velocity", "i4"),
    ("id", "U256"),
]


def _score(grace_first: bool = False, n_notes: int = 4) -> np.ndarray:
    """Eighth notes in 6/8 (one beat each), optionally after a grace note."""
    notes = [
        (float(i), 1.0, i / 2, 0.5, 60 + i, False, f"n{i}") for i in range(n_notes)
    ]
    if grace_first:
        notes.insert(0, (0.0, 0.0, 0.0, 0.0, 59, True, "g0"))
    return np.array(notes, dtype=NOTE_FIELDS)


def _perf_note(i: int, pitch: int) -> np.ndarray:
    return np.array([(0.5 * i, 0.4, pitch, 64, f"p{i}")], dtype=PERF_FIELDS)[0]


class _LastTokenModel(object):
    """Stand-in model that always predicts the last (padding) token."""

    def __init__(self, torch):
        self.torch = torch

    def __call__(self, tokens):
        out = self.torch.zeros((tokens.shape[1], 1, 2))
        out[-1, 0, 1] = 10.0
        return out


class TestTOLTWInitialTempo(unittest.TestCase):
    def test_reads_the_first_note(self):
        # 90 quarters a minute, two beats to the quarter.
        matcher = TOLTWMatcher(_score(), tracker_type="SLT_OLTW")
        self.assertAlmostEqual(matcher.init_tempo, 60 / 90 / 2)

    def test_skips_a_leading_grace_note(self):
        matcher = TOLTWMatcher(_score(grace_first=True), tracker_type="SLT_OLTW")
        self.assertTrue(np.isfinite(matcher.init_tempo))
        self.assertAlmostEqual(matcher.init_tempo, 60 / 90 / 2)


class TestTransformerMatchersStayInScore(unittest.TestCase):
    def _check(self, matcher):
        matcher.model = _LastTokenModel(matcher.torch)
        for i in range(30):
            position = matcher(_perf_note(i, 60 + i % 4))
            self.assertGreaterEqual(position, 0)
            self.assertLess(position, matcher.N_ref)

    def test_online_transformer_matcher(self):
        self._check(OnlineTransformerMatcher(_score()))

    def test_online_pure_transformer_matcher(self):
        self._check(OnlinePureTransformerMatcher(_score()))


if __name__ == "__main__":
    unittest.main()
