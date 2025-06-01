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

    def __init__(self):
        self.DP = SubPartDynamicProgramming()

    def preprocess_score(self, score):
        bass_line = score.parts[-1]
        return pt.score.Score(bass_line)

    def preprocess_performance(self, pna, sna):
        max_score_pitch = np.max(sna["pitch"])
        min_score_pitch = np.min(sna["pitch"])
        pitch_mask = np.all(
            (pna["pitch"] <= max_score_pitch, pna["pitch"] >= min_score_pitch), axis=0
        )
        return pna[pitch_mask], pna[~pitch_mask]

    def align_from_path(self, path, sna, pna, pna_top_voice=None):
        alignment = list()
        used_pidx = set()
        for snote_idx in range(len(sna)):
            pnote_idx = path[snote_idx]
            pnote = pna[pnote_idx]
            snote = sna[snote_idx]
            if np.max(pnote["pitch"] == snote["pitch"]):
                alignment.append(
                    {
                        "label": "match",
                        "score_id": str(snote["id"]),
                        "performance_id": str(pnote["id"]),
                    }
                )
                used_pidx.add(pnote["id"])
            else:
                alignment.append({"label": "deletion", "score_id": str(snote["id"])})

        for pnote in pna:
            if pnote["id"] not in used_pidx:
                alignment.append(
                    {"label": "insertion", "performance_id": str(pnote["id"])}
                )

        if pna_top_voice is not None:
            for pnote in pna_top_voice:
                alignment.append(
                    {"label": "insertion", "performance_id": str(pnote["id"])}
                )

        return alignment

    def from_note_arrays(self, sna, pna_original, preprocess_pna=True):
        """
        compute subpart alignment from note arrays.

        Parameters
        ----------
        sna: ndarray
            score note array of the subpart
        pna : ndarray
            performance note array of the full performance
        preprocess_pna: bool
            trim the performance pitch range to the subpart
            pitch range before alignment, defaults to True.
        """
        if preprocess_pna:
            pna_bottom_voice, pna_top_voice = self.preprocess_performance(
                pna_original, sna
            )
            _, path, _, _ = self.DP(pna_bottom_voice, sna)
            alignment = self.align_from_path(path, sna, pna_bottom_voice, pna_top_voice)
        else:
            _, path, _, _ = self.DP(pna_original, sna)
            alignment = self.align_from_path(path, sna, pna_original)
        return alignment

    def __call__(self, score_path, performance_path, preprocess_pna=True):
        """
        compute subpart alignment from score
        and performance paths.

        Parameters
        ----------
        score_path: str
            path to a score
        performance_path : str
            path to a performance
        preprocess_pna: bool
            trim the performance pitch range to the subpart
            pitch range before alignment, defaults to True.
        """
        score = pt.load_musicxml(score_path, force_note_ids=True)
        # re
        bscore = self.preprocess_score(score)
        sna = bscore.note_array()

        performance = pt.load_performance_midi(performance_path)
        pna_original = performance.note_array()

        alignment = self.from_note_arrays(sna, pna_original, preprocess_pna)
        return alignment
