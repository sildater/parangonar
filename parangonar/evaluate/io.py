#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains functionality to export alignments
for visualization with pianoprecision:
https://github.com/yucongj/piano-precision
"""

import numpy as np
import partitura as pt
from fractions import Fraction
import csv

def save_pianoprecision_csv(
    alignment,
    performance,
    spart,
    out = "scorealignment.csv"):
    pp_list = convert_alignment_to_list(alignment, spart, performance)
    export_pianoprecision_to_csv(pp_list, out)

def export_pianoprecision_to_csv(pp_list, 
                                 out = "scorealignment.csv"):
    with open(out, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["LABEL","TIME"])
        for integer, frac, flt in pp_list:
            mixed = f"{integer}+{frac.numerator}/{frac.denominator}"
            writer.writerow([mixed, f"{flt:.3f}"])

def convert_alignment_to_list(alignment, spart, performance):
    score_note_array = spart.note_array()
    performance_note_array = performance.note_array()
    beat_map = spart.beat_map
    _, stime_to_ptime_map = pt.musicanalysis.performance_codec.get_time_maps_from_alignment(
                    performance_note_array,
                    score_note_array,
                    alignment)

    measures = np.array(list(spart.iter_all(pt.score.Measure)))
    measure_starts_divs = np.array([m.start.t for m in measures])
    # measure_starts_beats = beat_map(measure_starts_divs)
    measure_sorting_idx = measure_starts_divs.argsort()
    measure_starts_divs = measure_starts_divs[measure_sorting_idx]
    measures = measures[measure_sorting_idx]

    # start_measure_num = 0 if measure_starts_beats.min() < 0 else 1
    start_measure_num = 1
    measure_starts = np.column_stack(
        (
            np.arange(start_measure_num, start_measure_num + len(measure_starts_divs)),
            measure_starts_divs,
            measures,
        )
    )
    used_note_onsets_divs = set()
    line_tuples = list()
    for (mnum, msd, m) in measure_starts:
        snotes = spart.iter_all(pt.score.Note, m.start, m.end, include_subclasses=True)
        for snote in snotes:
            onset_divs = snote.start.t
            if onset_divs in used_note_onsets_divs:
                continue
            else:
                onset_beats = beat_map(onset_divs)
                divs_per_quarter = int(spart.quarter_duration_map(onset_divs))
                fraction_offset_from_measure_begin = Fraction(int(onset_divs - msd), (divs_per_quarter * 4))
                performance_onset_sec = stime_to_ptime_map(onset_beats)
                line_tuple = (int(mnum), fraction_offset_from_measure_begin, performance_onset_sec)
                line_tuples.append(line_tuple)
                used_note_onsets_divs.add(onset_divs)
            
    return line_tuples
