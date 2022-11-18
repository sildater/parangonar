#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for alignment utilities.
"""
import unittest
import numpy as np
from parangonar import AutomaticNoteMatcher, fscore_alignments
import partitura as pt

RNG = np.random.RandomState(1984)
from tests import MATCH_FILES


class TestNoteAlignment(unittest.TestCase):
    def test_auto_align(self, **kwargs):
        
        perf_match, alignment, score_match = pt.load_match(
            filename=MATCH_FILES[0],
            create_score=True,
        ) 
        pna_match = perf_match.note_array()
        sna_match = score_match.note_array()
        sdm = AutomaticNoteMatcher()
        pred_alignment = sdm(sna_match, pna_match)
        _, _, f_score = fscore_alignments(pred_alignment, 
                                        alignment, 
                                        "deletion")
        self.assertTrue(f_score == 1.0)
        

        
if __name__ == "__main__":
    unittest.main()
        