#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains functionality to export alignments
for visualization with pianoprecision:
https://github.com/yucongj/piano-precision
and sonic visualizer:
https://www.sonicvisualiser.org/
"""

import numpy as np
import partitura as pt
from fractions import Fraction
import csv
from pathlib import Path
import numpy.lib.recfunctions as rfn
from partitura.utils.generic import interp1d



def save_pianoprecision_csv(
    performance,
    spart,
    alignment,
    out = "scorealignment.csv"):
    """
    save alignment for pianoprecision

    Parameters
    ----------
    
    performance : object
        a performance object
    spart: object
        a score part object
    alignment: list
        note alignment list of dicts
            
    """
    
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

def save_notes_for_sonic_visualiser(notes_array, output_filename):
    """
    Saves a structured numpy array to a Sonic Visualiser-compatible note file.

    Parameters
    ----------
    alignment: list
        note alignment list of dicts
    performance : object
        a performance object
    spart: object
        a score part object
    """
    with open(output_filename, 'w') as f:
        for note in notes_array:
            onset = float(note['onset'])
            duration = float(note['duration'])
            pitch = note['pitch']
            # Combine ID and pitch or use just pitch if you prefer
            label = f"{pitch}"  # Or f"{note['id']}:{pitch}" if ID is meaningful
            f.write(f"{onset:.6f}\t{duration:.6f}\t{label}\n")

def save_notes_for_sonic_visualiser(note_array, 
                                    out = "notes.csv"):
    """
    Saves a performance note array to a Sonic Visualiser note file.
    
    Parameters
    ----------
    note_array: ndarray
        performance note array
    out : str
        csv filename to store the notes
    """
    with open(out, 'w') as f:
        f.write("TIME,VALUE,DURATION,LEVEL,LABEL\n")
        for note in note_array:
            onset = float(note['onset_sec'])
            duration = float(note['duration_sec'])
            pitch = note['pitch']
            label = note["id"]
            f.write(f"{onset:.9f},{pitch},{duration:.9f},0.8,{label}\n")

def save_attribute_for_sonic_visualiser_instants(note_array, 
                                                attribute_name,
                                        out = "instants.csv"):
    """
    Saves a note array attribute array to a Sonic Visualiser instants file.

    Parameters
    ----------
    note_array: ndarray
        note array
    attribute_name: str
        name of the attribute to use as label
    out : str
        csv filename to store the instants
    """
    # # get unique instants
    # _, unique_idx = np.unique(note_array[attribute_name], return_index = True)
    with open(out, 'w') as f:
        f.write("TIME,LABEL\n")
        for note in note_array:
            onset = float(note['onset_sec'])
            label = str(note[attribute_name])
            f.write(f"{onset:.9f},{label}\n")

def save_attribute_for_sonic_visualiser_time_values(note_array, 
                                                attribute_name,
                                                out = "time_values.csv",
                                                set_range = False):
    """
    Saves a note array attribute array to a Sonic Visualiser time values file.

    Parameters
    ----------
    note_array: ndarray
        note array
    attribute_name: str
        name of the attribute to use for values
    out : str
        csv filename to store the instants
    """
    
    with open(out, 'w') as f:
        f.write("TIME,VALUE,LABEL\n")
        for note in note_array:
            onset = float(note['onset_sec'])
            tvalue = float(note[attribute_name])
            f.write(f"{onset:.9f},{tvalue:.9f},{attribute_name}\n")
        if set_range:
            f.write(f"{onset+1:.9f},{set_range[0]:.9f},min value\n")
            f.write(f"{onset+2:.9f},{set_range[1]:.9f},max value\n")


def compute_snote_pnote_array(performance,
                            score_part, 
                            alignment):
    """
    Saves a note array attribute array to a Sonic Visualiser time values file.

    Parameters
    ----------
    performance: partitura.performance.PerformanceLike
        Performance information, can be a ppart, performance
    score_part: partitura.score.ScoreLike
        Score information, can be a part, score
    alignment : list
        list of alignment dicts
    """

    # nf = pt.musicanalysis.compute_note_array(score_part, 
    #                                              include_pitch_spelling=True,
    #                                              include_key_signature=True,
    #                                              include_time_signature=True,
    #                                              include_metrical_position=True,
    #                                              include_grace_notes=True,
    #                                              feature_functions="all",
    #                                              force_fixed_size=True)

    nf = score_part.note_array()

    pf = pt.musicanalysis.make_performance_features(score_part, 
                                                    performance, 
                                                    alignment, 
                                                    feature_functions="all")
    
    pf = rfn.rename_fields(pf, dict(zip(["p_onset", "p_duration"], ["onset_sec", "duration_sec"])))
    score_fields = ["onset_beat", "duration_beat", "pitch", "id", "voice"]
    performance_fields = [ "id", "onset_sec", "duration_sec", "beat_period", "timing", "articulation_log", "velocity", "pedal_feature.onset_value"]
    score_f = nf[score_fields]
    performance_f = pf[performance_fields]
    merged_ = np.lib.recfunctions.join_by("id", score_f, performance_f, jointype='inner').data
    return merged_

def compute_onsetwise_snote_pnote_array(merged_note_array,
                                        attribute_list = ["velocity", "beat_period","articulation_log"]):
    unique_onset_idx = pt.musicanalysis.performance_codec.get_unique_onset_idxs(merged_note_array["onset_beat"])
    onset_val = pt.musicanalysis.performance_codec.notewise_to_onsetwise(merged_note_array["onset_sec"],unique_onset_idx)
    onset_beat_val = pt.musicanalysis.performance_codec.notewise_to_onsetwise(merged_note_array["onset_beat"],unique_onset_idx)
    array_collector = [np.array(list(zip(onset_val)), dtype = np.dtype([("onset_sec", "f4")])),
                       np.array(list(zip(onset_beat_val)), dtype = np.dtype([("onset_beat", "f4")]))]
    for attribute in attribute_list:
        attribute_val = pt.musicanalysis.performance_codec.notewise_to_onsetwise(merged_note_array[attribute],unique_onset_idx)
        array_collector.append(np.array(list(zip(attribute_val)), dtype = np.dtype([(attribute, "f4")])))
    return rfn.merge_arrays(array_collector, flatten=True, usemask=False)


def save_expression_features_for_sonic_visualiser(merged_note_array,
                                                  out_dir = ".",
                                                  notewise = False,
                                                  onsetwise = True,
                                                  beatwise = 0):
    """
    Saves a note array attribute array to a Sonic Visualiser time values file.

    Parameters
    ----------
    merged_note_array: ndarray
        note array with p and s features: use compute_snote_pnote_array

    """
    out_dir = Path(out_dir)

    if notewise:
        save_attribute_for_sonic_visualiser_time_values(merged_note_array, "velocity", out_dir / Path("nw_velocity.csv"))
        save_attribute_for_sonic_visualiser_time_values(merged_note_array, "beat_period", out_dir / Path("nw_beat_period.csv"))
        save_attribute_for_sonic_visualiser_time_values(merged_note_array, "timing", out_dir / Path("nw_timing.csv"))
        save_attribute_for_sonic_visualiser_time_values(merged_note_array, "articulation_log", out_dir / Path("nw_articulation_log.csv"))
        save_attribute_for_sonic_visualiser_time_values(merged_note_array, "pedal_feature.onset_value", out_dir / Path("nw_pedal.csv"))

    if onsetwise:
        onset_merged_array = compute_onsetwise_snote_pnote_array(merged_note_array)
        save_attribute_for_sonic_visualiser_time_values(onset_merged_array, "velocity", out_dir / Path("ow_velocity.csv"))
        save_attribute_for_sonic_visualiser_time_values(onset_merged_array, "beat_period", out_dir / Path("ow_beat_period.csv"))
        save_attribute_for_sonic_visualiser_time_values(onset_merged_array, "articulation_log", out_dir / Path("ow_articulation_log.csv"))

    if beatwise:
        onset_merged_array = compute_onsetwise_snote_pnote_array(merged_note_array)
        stime_to_ptime_map = stime_to_ptime_map = interp1d(
            y=onset_merged_array["onset_sec"],
            x=onset_merged_array["onset_beat"],
            bounds_error=False,
            fill_value="extrapolate",
        )
        max_beat = np.max(onset_merged_array["onset_beat"])
        beats = np.arange(0, max_beat + beatwise, beatwise)
        perf_beats = stime_to_ptime_map(beats)
        beat_period = np.diff(perf_beats) / beatwise

        perf_beats_array = np.array(list(zip(perf_beats, beats)), dtype=np.dtype([("onset_sec", "f4"), ("onset_beat", "f4")]))
        # perf_beats_array = np.array([perf_beats, beats], dtype = [("onset_sec", "f4"), ("onset_beat", "f4")])
        beat_period_array = np.array(list(zip(perf_beats[:-1], beat_period)), dtype=np.dtype([("onset_sec", "f4"), ("bw_beat_period", "f4")]))
        # beat_period_array = np.array([perf_beats[:-1], beat_period], dtype = [("onset_sec", "f4"), ("bw_beat_period", "f4")])
        save_attribute_for_sonic_visualiser_time_values(beat_period_array, "bw_beat_period", out_dir / Path("bw_beat_period.csv"))
        save_attribute_for_sonic_visualiser_instants(perf_beats_array, "onset_beat", out_dir / Path("beats.csv"))


def save_sonic_visualizer_csvs(
    performance,
    score_part,
    alignment,
    out_dir = ".",
    notewise = False,
    onsetwise = True,
    beatwise = 0):
    """
    save expression features for sonic visualizer

    Parameters
    ----------
    performance : object
        a performance object
    score_part: object
        a score part object
    alignment: list
        note alignment list of dicts   
    """
    out_dir = Path(out_dir)
    merged_note_array = compute_snote_pnote_array(performance,
                                                  score_part, 
                                                    alignment)
    
    save_expression_features_for_sonic_visualiser(merged_note_array,
                                                  out_dir = out_dir,
                                                  notewise = notewise,
                                                  onsetwise = onsetwise,
                                                  beatwise = beatwise)
    
    controls_list = [(cc["time"], cc["value"]) for cc in performance.performedparts[0].controls if cc["number"] == 64]
    controls_array = np.array(controls_list, dtype=np.dtype([("onset_sec", "f4"), ("pedal", "f4")]))
    save_attribute_for_sonic_visualiser_time_values(controls_array, "pedal", out_dir / Path("pedal.csv"))

    save_notes_for_sonic_visualiser(merged_note_array, out_dir / Path("notes.csv"))
