#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for alignment utilities.
"""
import unittest
import numpy as np
from parangonar import (
    AutomaticNoteMatcher,
    DualDTWNoteMatcher,
    TheGlueNoteMatcher,
    AnchorPointNoteMatcher,
    OnlinePureTransformerMatcher,
    OnlineTransformerMatcher,
    fscore_alignments,
)
from parangonar.match import node_array
import partitura as pt

RNG = np.random.RandomState(1984)
from tests import MATCH_FILES


class TestNoteAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perf_match, cls.alignment, cls.score_match = pt.load_match(
            filename=MATCH_FILES[0], create_score=True
        )

    def test_AutomaticNoteMatcher_align(self, **kwargs):
        pna_match = self.perf_match.note_array()
        sna_match = self.score_match.note_array()
        matcher = AutomaticNoteMatcher()
        pred_alignment = matcher(sna_match, pna_match)
        _, _, f_score = fscore_alignments(pred_alignment, self.alignment, "match")
        self.assertTrue(f_score == 1.0)

    def test_DualDTWNoteMatcher_align(self, **kwargs):
        pna_match = self.perf_match.note_array()
        sna_match = self.score_match.note_array(include_grace_notes=True)
        matcher = DualDTWNoteMatcher()
        pred_alignment = matcher(
            sna_match, pna_match, process_ornaments=True, score_part=self.score_match[0]
        )
        _, _, f_score = fscore_alignments(pred_alignment, self.alignment, "match")
        self.assertTrue(f_score == 1.0)

    def test_TheGlueNote_align(self, **kwargs):
        pna_match = self.perf_match.note_array()
        sna_match = self.score_match.note_array()
        matcher = TheGlueNoteMatcher()
        pred_alignment = matcher(sna_match, pna_match)
        _, _, f_score = fscore_alignments(pred_alignment, self.alignment, "match")
        self.assertTrue(abs(f_score - 1.0) < 0.01)

    def testAnchorPointNoteMatcher_align(self, **kwargs):
        pna_match = self.perf_match.note_array()
        sna_match = self.score_match.note_array(include_grace_notes=True)
        nodes = node_array(
            self.score_match[0], self.perf_match[0], self.alignment, node_interval=4
        )
        matcher = AnchorPointNoteMatcher()
        pred_alignment = matcher(sna_match, pna_match, nodes)
        _, _, f_score = fscore_alignments(pred_alignment, self.alignment, "match")
        self.assertTrue(f_score == 1.0)

    def test_OnlineTransformerMatcher_align(self, **kwargs):
        pna_match = self.perf_match.note_array()
        sna_match = self.score_match.note_array(include_grace_notes=True)
        matcher = OnlineTransformerMatcher(sna_match)
        pred_alignment = matcher.offline(pna_match)
        _, _, f_score = fscore_alignments(pred_alignment, self.alignment, "match")
        self.assertTrue(abs(f_score - 1.0) < 0.02)

    def test_OnlinePureTransformerMatcher_align(self, **kwargs):
        pna_match = self.perf_match.note_array()
        sna_match = self.score_match.note_array(include_grace_notes=True)
        matcher = OnlinePureTransformerMatcher(sna_match)
        pred_alignment = matcher.offline(pna_match)
        _, _, f_score = fscore_alignments(pred_alignment, self.alignment, "match")
        self.assertTrue(f_score == 1.0)


if __name__ == "__main__":
    unittest.main()
