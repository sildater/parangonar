#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for alignment utilities.
"""
import unittest
import numpy as np
from parangonar import (
    RepeatIdentifier,
    SubPartMatcher,
    fscore_alignments
)
import partitura as pt
from copy import copy

RNG = np.random.RandomState(1984)
from tests import MATCH_FILES, P_FILES, S_FILES


class TestMismatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perf_match, cls.alignment, cls.score_match = pt.load_match(
            filename=MATCH_FILES[0], create_score=True
        )
        cls.perf_mid = pt.load_performance_midi(P_FILES[0])
        cls.score_xml = pt.load_musicxml(S_FILES[0])

    def test_repeat_identification(self, **kwargs):

        ri = RepeatIdentifier()
        found_path, _ = ri(self.score_xml,self.perf_mid)
        true_path = "AABBCCDEDFAB"
        self.assertTrue(true_path == found_path)

    def test_subpart_alignment(self, **kwargs):
        # create a subpart
        sna = self.score_match.note_array()  
        sna_mask = np.arange(0,len(sna),12)
        sna_subpart = sna[sna_mask]
        sort_idx = np.argsort(sna_subpart["onset_beat"])
        sna_subpart = sna_subpart[sort_idx]
        new_alignment = list()
        for m in self.alignment:
            if m["label"] == "match":
                if m["score_id"] in sna_subpart["id"]:
                    new_alignment.append(m)
                else:
                    new_alignment.append({"label": "insertion",
                                "performance_id": m["performance_id"]})
                    new_alignment.append({"label": "deletion", 
                                "score_id": m["score_id"]})
            else:
                new_alignment.append(m)

        pna = self.perf_match.note_array()      
        matcher = SubPartMatcher()
        pred_alignment = matcher.from_note_arrays(sna_subpart,pna, True)
        _, _, f_score = fscore_alignments(pred_alignment, new_alignment, "match")
        
        self.assertTrue(f_score > 0.94)

    

if __name__ == "__main__":
    unittest.main()
