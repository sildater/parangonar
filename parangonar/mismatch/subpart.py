#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods to align subparts of a performance.
"""
import numpy as np
from ..dp.nwtw import SubPartDynamicProgramming
import partitura as pt


class SubPartMatcher(object):
    """
    Subpart Alignment 
    """

    def __init__(
        self
    ):
        self.DP = SubPartDynamicProgramming()

    def preprocess_score(self, score):
        bass_line = score.parts[-1]
        return pt.score.Score(bass_line) 

    def preprocess_performance(self,pna, sna):
        max_score_pitch = np.max(sna["pitch"])
        pitch_mask = pna["pitch"] <= max_score_pitch
        return pna[pitch_mask], pna[~pitch_mask]
    
    def align_from_path(self,
                        path, 
                        sna, pna, 
                        pna_top_voice = None):
        alignment = list()
        pidx = -1
        used_pidx = list()
        for snote_idx in range(len(sna)):
            pnote_idx = path[snote_idx]
            pnote = pna[pnote_idx]    
            snote = sna[snote_idx]
            if np.max(pnote["pitch"] == snote["pitch"] ):
                alignment.append({"label": "match", 
                                "score_id": str(snote["id"]), 
                                "performance_id": str(pnote["id"])})
                used_pidx.append(pidx)
            else:
                alignment.append({"label": "deletion", 
                                "score_id": str(snote["id"])})

        for pidx, pnote in enumerate(pna):
            if pidx not in used_pidx:
                alignment.append({"label": "insertion",
                                "performance_id": str(pnote["id"])})
                
        if pna_top_voice is not None:
            for pnote in pna_top_voice:
                alignment.append({"label": "insertion",
                                "performance_id": str(pnote["id"])})
            
        return alignment

    def __call__(self, score_path, performance_path, preprocess_pna = False):
        """
        Parameters
        ----------
        score_path: str
            path to a score
        performance_path : str
            path to a performance
        """
        score = pt.load_musicxml(score_path,
                         force_note_ids = True)
        bscore = self.preprocess_score(score)
        sna = bscore.note_array()

        performance = pt.load_performance_midi(performance_path)
        pna_original = performance.note_array()
        if preprocess_pna:
            pna_bottom_voice, pna_top_voice = self.preprocess_performance(pna_original, sna)
            _, path, _, _ = self.DP(pna_bottom_voice, sna)
            alignment = self.note_aligner(path, sna, pna_bottom_voice, pna_top_voice)

        else:
            _, path, _, _ = self.DP(pna_original, sna)
            alignment = self.note_aligner(path, sna, pna_original)

        return alignment

