#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains preprocessing methods
"""

import numpy as np
from scipy.interpolate import interp1d

from partitura.utils.music import (compute_pianoroll)

from .dtw import DTW
from .nwtw import NW_DTW, NW


################################### HELPERS ###################################


def alignment_times_from_dtw(score_note_array, 
                             performance_note_array,
                             matcher=DTW(),
                             SCORE_FINE_NODE_LENGTH=1.0,
                             s_time_div=16, p_time_div=16):
    """
    
    Coarse time warping to generate anchor points
    Returns an array, each row is tuple of corresponding 
    times in the score array and the perfromance array,
    respectively.

    Args:
        score_note_array (_type_): _description_
        performance_note_array (_type_): _description_
        matcher (_type_, optional): _description_. Defaults to DTW().
        SCORE_FINE_NODE_LENGTH (float, optional): _description_. Defaults to 1.0.
        s_time_div (int, optional): _description_. Defaults to 16.
        p_time_div (int, optional): _description_. Defaults to 16.

    Returns:
        _type_: _description_
    """
    # _____________ fine alignment ____________
    # compute proper piano rolls
    s_pianoroll = compute_pianoroll(score_note_array,
                                    time_div=s_time_div,
                                    remove_drums=False).toarray()
    p_pianoroll = compute_pianoroll(performance_note_array,
                                    time_div=p_time_div,
                                    remove_drums=False).toarray()
    # make piano rolls binary
    p_pianoroll_ones = np.zeros_like(p_pianoroll)
    p_pianoroll_ones[p_pianoroll > 0.0] = 1.0
    # align the piano rolls
    _, path = matcher(s_pianoroll.T, p_pianoroll_ones.T)
    # compute an alignment of times using the DTW path
    path_array = np.array(path)

    # Post-process path for NW-DTW (insertion and deletions
    # are denoted as -1 in the path)
    if isinstance(matcher, (NW_DTW, NW)):
        valid_idxs = np.where(
            np.logical_and(path_array[:, 0] != -1,
                           path_array[:, 1] != -1))[0]
        path_array = path_array[valid_idxs]

        if len(path_array) == 0:
            path_array = np.array([(path[:, 0].min(), path[:, 1].min()),
                                   (path[:, 1].max(), path[:, 1].max())],
                                  dtype=int)

    times_score = path_array[:, 0] / s_time_div
    times_performance = path_array[:, 1] / p_time_div

    u_times_score = np.unique(times_score)
    u_times_score_idxs = [np.where(times_score == u)[0]
                          for u in u_times_score]
    u_times_performance = np.array(
        [np.min(times_performance[ui])
         for ui in u_times_score_idxs])

    try:
        # Use a mapping to deal with missing values (due to
        # insertions and deletions in NW-related methods)
        # It should not affect the behavior of DTW methods
        # CC: I will check this just in case ;)
        stime_to_ptime_map = interp1d(x=u_times_score,
                                      y=u_times_performance,
                                      kind='linear',
                                      bounds_error=False,
                                      fill_value='extrapolate')
    except ValueError:
        # if there is only one element in u_times_score
        stime_to_ptime_map = \
            lambda x: np.ones_like(x) * u_times_performance[0]  # noqa: E731

    min_score = times_score.min()
    max_score = max(score_note_array["onset_beat"].max() -
                    score_note_array["onset_beat"].min(),
                    SCORE_FINE_NODE_LENGTH)
    max_performance = max(performance_note_array["onset_sec"].max() -
                          performance_note_array["onset_sec"].min(),
                          SCORE_FINE_NODE_LENGTH)

    cut_times_score = np.r_[
        np.arange(min_score, max_score,
                  SCORE_FINE_NODE_LENGTH),
        max_score]

    cut_times_performance = np.clip(
        stime_to_ptime_map(cut_times_score),
        a_min=0,
        a_max=max_performance)

    cut_times_performance += performance_note_array["onset_sec"].min()
    cut_times_score += score_note_array["onset_beat"].min()

    dtw_alignment_times = np.column_stack((cut_times_score,
                                           cut_times_performance))
    return dtw_alignment_times


def traverse_the_alignment_graph(start_id, 
                                 score_ids, 
                                 performance_ids, 
                                 performance_alignment, 
                                 score_alignment, counter, 
                                 max_depth=150):
    #score_ids = [input_id]
    #performance_id = []
    if start_id not in score_ids and counter < max_depth:
        #print("appendin", start_id)
        score_ids.append(start_id)
        new_pids = score_alignment[start_id]
        for pid in new_pids:
            #print("loop point", pid)
            if pid not in performance_ids:
                #print("appendin perf", pid)
                performance_ids.append(pid)
                new_sids = performance_alignment[pid]
                for sid in new_sids:
                    counter += 1
                    traverse_the_alignment_graph(sid, score_ids, performance_ids, performance_alignment, score_alignment, counter, max_depth)
                    
            else:
                continue
    elif counter == 150:
        print("max recursion depth in note graph")
        pass
    else:
        pass
        #print("done")

################################### SEGMENT CUTTING ###################################


def cut_note_arrays(performance_note_array, 
                    score_note_array, 
                    alignment,
                    sfuzziness=0.0, 
                    pfuzziness=0.0, 
                    window_size=1,
                    pfuzziness_relative_to_tempo=False):
    """
    cut note arrays into two lists of corresponding
    note array segments based on anchor points given 
    as "alignment": n by 2 array, first column score, 
    second performace times.


    Args:
        performance_note_array (_type_): _description_
        score_note_array (_type_): _description_
        alignment (_type_): _description_
        sfuzziness (float, optional): _description_. Defaults to 0.0.
        pfuzziness (float, optional): _description_. Defaults to 0.0.
        window_size (int, optional): _description_. Defaults to 1.
        pfuzziness_relative_to_tempo (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    score_note_arrays = []
    performance_note_arrays = []

    if not pfuzziness_relative_to_tempo:
        local_pfuzzines = np.ones_like(alignment[:,1])*pfuzziness
    else:
        approximate_tempo = interp1d(alignment[:-1,0],
                                     np.diff(alignment[:,1])/np.diff(alignment[:,0]),
                                     fill_value="extrapolate")
        local_pfuzzines = approximate_tempo(alignment[:,0])*pfuzziness

    for i in range(len(alignment)-window_size):
        # extract all score notes with onsets inside the closed inter beat interval
        score_window_mask = np.all([score_note_array["onset_beat"] >= alignment[i,0]-sfuzziness,
                                    score_note_array["onset_beat"] < alignment[i+window_size,0]+sfuzziness], axis = 0)
        # extract all performance notes with onsets inside the inter beat interval plus some fuzzy relaxation
        performance_window_mask = np.all([performance_note_array["onset_sec"] >= alignment[i,1]-local_pfuzzines[i],
                                    performance_note_array["onset_sec"] < alignment[i+window_size,1]+local_pfuzzines[i]], axis = 0)
        score_note_arrays.append(score_note_array[score_window_mask])
        performance_note_arrays.append(performance_note_array[performance_window_mask])

    return score_note_arrays, performance_note_arrays


################################### SEGMENT MENDING ###################################


def mend_note_alignments(note_alignments, 
                 performance_note_array, 
                 score_note_array, 
                 node_times, 
                 symbolic_note_matcher, 
                 max_traversal_depth = 150):
    """
    mend note alignments in (overlapping) windows.
    creates a global dictionary of MAPS style alignments
    from a list of windowed MAPS style alignments
    """
                 
    score_alignment = {"insertion":[]}
    performance_alignment = {"deletion":[]}
    alignment = []

    # approximate_position = interp1d(node_times[:,0],node_times[:,1],fill_value="extrapolate")
    # approximate_tempo = interp1d(node_times[:-1,0],np.diff(node_times[:,1])/np.diff(node_times[:,0]),fill_value="extrapolate")

    # score_onsets_by_id = { snote["id"]:snote["onset_beat"] for snote in score_note_array}

    # combine all note alignments in two dictionaries
    for window_id in range(len(note_alignments)):
        for alignment_line in note_alignments[window_id]:
            if alignment_line["label"] == "match":
                if alignment_line["score_id"] in score_alignment.keys():
                    score_alignment[alignment_line["score_id"]].append(alignment_line["performance_id"])
                else:
                    score_alignment[alignment_line["score_id"]] = [alignment_line["performance_id"]]

                if alignment_line["performance_id"] in performance_alignment.keys():
                    performance_alignment[alignment_line["performance_id"]].append(alignment_line["score_id"])
                else:
                    performance_alignment[alignment_line["performance_id"]] = [alignment_line["score_id"]]

            if alignment_line["label"] == "deletion":
                if alignment_line["score_id"] in score_alignment.keys():
                    score_alignment[alignment_line["score_id"]].append("deletion")
                else:
                    score_alignment[alignment_line["score_id"]] = ["deletion"]

            if alignment_line["label"] == "insertion":
                if alignment_line["performance_id"] in performance_alignment.keys():
                    performance_alignment[alignment_line["performance_id"]].append("insertion")
                else:
                    performance_alignment[alignment_line["performance_id"]] = ["insertion"]

    # compute as single unique ids alignment
    used_perf_notes = ["deletion"]
    used_score_notes = ["insertion"]
    for snote_id in score_alignment.keys():

        # IF 1 p_note in alignment:
        if snote_id not in used_score_notes:
            if len(score_alignment[snote_id]) == 1:
                # DELETION
                if score_alignment[snote_id][0] == "deletion":
                    alignment.append({'label': 'deletion', 'score_id': snote_id})
                    used_score_notes.append(snote_id)
                else:
                    score_ids = []
                    performance_ids = []     
                    counter = 0
                    traverse_the_alignment_graph(snote_id, score_ids, performance_ids, performance_alignment, score_alignment, counter, max_depth=max_traversal_depth)
                    # deal with the possible alignments
                    no_insertion = [sid for sid in score_ids if sid != "insertion"]
                    no_deletion = [pid for pid in performance_ids if pid != "deletion"]
                    # if two unique notes match
                    if len(no_deletion) == 1 and len(no_insertion) == 1:
                        alignment.append({'label': 'match', 'score_id': snote_id, 'performance_id': str(score_alignment[snote_id][0])})
                        used_score_notes.append(no_insertion[0])
                        used_perf_notes.append(str(no_deletion[0]))
                    # try realigning
                    else:
                        local_score_note_array = score_note_array[np.any([score_note_array["id"] == str(idx) for idx in no_insertion], axis=0)]
                        masks = [performance_note_array["id"] == str(idx) for idx in no_deletion]
                        local_performance_note_array = performance_note_array[np.any(masks, axis=0)]
                        #print( snote_id, "realigning with score and performance array: ",local_score_note_array.shape, local_performance_note_array.shape)
                        
                        local_alignment = symbolic_note_matcher(local_score_note_array, local_performance_note_array, node_times)
                        #import pdb; pdb.set_trace()
                        used_score_notes += no_insertion
                        used_perf_notes += no_deletion
                        alignment += local_alignment

            
            else:
                # multiple perf_ids aligned to snote_id
                score_ids = []
                performance_ids = []     
                counter = 0
                traverse_the_alignment_graph(snote_id, score_ids, performance_ids, performance_alignment, score_alignment, counter, max_depth=max_traversal_depth)

                # deal with the possible alignments
                no_insertion = [sid for sid in score_ids if sid != "insertion"]
                no_deletion = [pid for pid in performance_ids if pid != "deletion"]
                # if two unique notes match
                if len(no_deletion) == 1 and len(no_insertion) == 1:
                    alignment.append({'label': 'match', 'score_id': snote_id, 'performance_id': str(no_deletion[0])})
                    used_score_notes.append(no_insertion[0])
                    used_perf_notes.append(str(no_deletion[0]))
                
                elif len(no_deletion) == 0:
                    # all deletions
                    alignment.append({'label': 'deletion', 'score_id': snote_id})
                    used_score_notes.append(snote_id)
                else:
                    # try realigning
                    
                    
                    local_score_note_array = score_note_array[np.any([score_note_array["id"] == str(idx) for idx in no_insertion], axis=0)]
                    local_performance_note_array = performance_note_array[np.any([performance_note_array["id"] == str(idx) for idx in no_deletion], axis=0)]
                    
                    #print( snote_id,"realigning (conflicting) with score and performance array: ",local_score_note_array.shape, local_performance_note_array.shape)
                    local_alignment = symbolic_note_matcher(local_score_note_array, local_performance_note_array, node_times)

                    #import pdb; pdb.set_trace()
                    used_score_notes += no_insertion
                    used_perf_notes += no_deletion
                    alignment += local_alignment


    for pnote_id in performance_alignment.keys():
        if pnote_id not in used_perf_notes:
            #print("previously unused performance note: ", pnote_id, performance_alignment[pnote_id])
            alignment.append({'label': 'insertion', 'performance_id': pnote_id})

    return alignment, score_alignment, performance_alignment


