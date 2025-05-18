#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for alignment utilities.
"""
import unittest
import numpy as np
from parangonar import (
    fscore_alignments,
)
import partitura as pt

RNG = np.random.RandomState(1984)
from tests import MATCH_FILES


class TestMismatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perf_match, cls.alignment, cls.score_match = pt.load_match(
            filename=MATCH_FILES[0], create_score=True
        )

    def test_reapeat_identification(self, **kwargs):
        
        _, _, f_score = fscore_alignments(pred_alignment, self.alignment, "match")
        self.assertTrue(f_score == 1.0)

    

if __name__ == "__main__":
    unittest.main()
