#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains condensed, 
single-command methods to align music files
and save the result as csv files.
"""


import parangonar as pa
import partitura as pt
from numpy.lib import recfunctions as rfn
import numpy as np
import pandas as pd
from ..match import (
    DualDTWNoteMatcher
)

def match_midis(ref_midi, # this is usually a score (deadpan) midi file and will be processed assuming that it is
                performance_midi, # this can be any recorded or sequenced midi file that will be matched to the reference
                output_file = "matched_notes.csv", #  a path to a csv file where we store the resulting match info
                shift_onsets_to_zero = False): # whether to shift both note arrays such that they start at zero
    
    ref_midi_p = pt.load_performance_midi(ref_midi)
    performance_midi_p = pt.load_performance_midi(performance_midi)
    ref_midi_na = ref_midi_p.note_array()
    performance_midi_na = performance_midi_p.note_array()
    # ppq = ref_midi_p.performedparts[0].ppq

    # add dummy score fields for the matcher
    ref_midi_na_extended = rfn.append_fields(
        ref_midi_na,
        names = ['onset_beat','duration_beat','is_grace'],
        data =[ref_midi_na['onset_tick'],
               ref_midi_na['duration_tick'], 
               np.zeros_like(ref_midi_na, dtype=np.bool_)],
        dtypes = ["f4", "f4", "b"],
        usemask=False
    )

    # do the matching
    matcher = DualDTWNoteMatcher()
    pred_alignment = matcher(ref_midi_na_extended, performance_midi_na) 

    # shift to zero
    if shift_onsets_to_zero:
        performance_midi_na["onset_tick"] -= np.min(performance_midi_na["onset_tick"])
        performance_midi_na["onset_sec"] -= np.min(performance_midi_na["onset_sec"])
        ref_midi_na["onset_tick"] -= np.min(ref_midi_na["onset_tick"])
        ref_midi_na["onset_sec"] -= np.min(ref_midi_na["onset_sec"])

    # rename fields
    performance_midi_na = rfn.rename_fields(performance_midi_na, {field: f"performance_{field}" for field in performance_midi_na.dtype.names})
    ref_midi_na = rfn.rename_fields(ref_midi_na, {field: f"ref_{field}" for field in ref_midi_na.dtype.names})
    
    # keep only relevant fields
    rel_fields = ["onset_sec", "duration_sec", "onset_tick", "duration_tick",
                  "id", "pitch", "velocity", "channel", "track"]
    performance_na = performance_midi_na[["performance_"+f for f in rel_fields]]
    ref_na = ref_midi_na[["ref_"+f for f in rel_fields]]

    # add alignment column for merging
    map_ref_id_to_idx = {note["ref_id"]:idx for idx,note in enumerate(ref_na)}
    map_performance_id_to_idx = {note["performance_id"]:idx for idx,note in enumerate(performance_na)}
    ref_match_number = np.full_like(ref_na["ref_onset_sec"] , -1)
    performance_match_number = np.full_like(performance_na["performance_onset_sec"] , -2)
    meta_columns = []
    for idx, align_entry in enumerate(pred_alignment):
        if align_entry['label'] == 'match':
            ref_na_idx = map_ref_id_to_idx[align_entry["score_id"]]
            ref_match_number[ref_na_idx] = idx
            performance_na_idx = map_performance_id_to_idx[align_entry["performance_id"]]
            performance_match_number[performance_na_idx] = idx
            meta_columns.append((idx, "match", 
                                 performance_na["performance_onset_sec"][performance_na_idx] - ref_na["ref_onset_sec"][ref_na_idx]))
        elif align_entry['label'] == 'insertion':
            meta_columns.append((idx, "insertion", None))
            performance_na_idx = map_performance_id_to_idx[align_entry["performance_id"]]
            performance_match_number[performance_na_idx] = idx
        elif align_entry['label'] == 'deletion':
            meta_columns.append((idx, "deletion", None))
            ref_na_idx = map_ref_id_to_idx[align_entry["score_id"]]
            ref_match_number[ref_na_idx] = idx

    ref_na_full = rfn.append_fields(
        ref_na,
        names = "match_number",
        data = ref_match_number,
        dtypes = "i4",
        usemask=False
    )

    performance_na_full = rfn.append_fields(
        performance_na,
        names = "match_number",
        data = performance_match_number,
        dtypes = "i4",
        usemask=False
    )
    dt = np.dtype([('match_number', 'i4'), ('alignment_type', 'U256'), ('timing_deviation', 'f4')])
    meta_array = np.array(meta_columns, dtype=dt)
    
    # convert to df, merge, sort, and export
    df1 = pd.DataFrame(ref_na_full)
    df2 = pd.DataFrame(performance_na_full)
    df3 = pd.DataFrame(meta_array)

    merged_df = df1.merge(df2, on='match_number', how='outer').merge(df3, on='match_number', how='outer')
    output_df = merged_df.loc[merged_df.ref_onset_sec.fillna(merged_df.performance_onset_sec).argsort()].drop(columns=['match_number'])
    output_df.to_csv(output_file, index=False)

    return 