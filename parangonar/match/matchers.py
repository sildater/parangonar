#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains full note matcher classes.
"""
import numpy as np
import torch
from scipy.interpolate import interp1d
from collections import defaultdict

import time
from itertools import combinations
from scipy.special import binom

from .dtw import DTW, element_of_metric
from .nwtw import NW_DTW, NW

from .preprocessors import (mend_note_alignments,
                            cut_note_arrays,
                            alignment_times_from_dtw,
                            note_per_ons_encoding)

from .pretrained_models import (AlignmentTransformer)

################################### SYMBOLIC MATCHERS ###################################


class SimplestGreedyMatcher(object):
    """
    Create alignment in MAPS format (dict) by greedy pitch matching from performance and score note_array
    """
    def __call__(self, score_note_array, performance_note_array):
        alignment = []
        s_aligned = []
        p_aligned = []
        for s_note in score_note_array:
            sid = s_note['id']
            pid = None

            # filter performance notes with matching pitches
            matching_pitches = performance_note_array[s_note['pitch'] == performance_note_array['pitch']]

            for p_note in matching_pitches:
                # take first matching performance note that was not yet aligned
                if p_note not in p_aligned:
                    pid = str(p_note['id'])
                    p_aligned.append(p_note)
                    s_aligned.append(s_note)
                    break

            if pid is not None:
                alignment.append({'label': 'match', 'score_id': sid, 'performance_id': str(pid)})
            else:
                # if score note could not be aligned it counts as a deletion
                alignment.append({'label': 'deletion', 'score_id': sid})

        # check for unaligned performance notes (ie insertions)
        for p_note in performance_note_array:
            if p_note not in p_aligned:
                alignment.append({'label': 'insertion', 'performance_id': str(p_note['id'])})

        return alignment


class SequenceAugmentedGreedyMatcher(object):
    """
    Create alignment in MAPS format (dict) by sequence augmented pitch matching from performance and score note_array
    """
    def __init__(self):
        self.overlap = False

    def __call__(self, 
                 score_note_array, 
                 performance_note_array, 
                 alignment_times, 
                 shift=False, 
                 cap_combinations = 10000):
        alignment = []
        # s_aligned = []
        p_aligned = []

        # DTW gives non-unique times, sometimes...
        # TODO: safety net
        try:
            onset_time_conversion = interp1d(alignment_times[:, 0],
                                             alignment_times[:, 1],
                                             fill_value="extrapolate")
        except ValueError:
            if len(alignment_times) < 2:
                onset_time_conversion = \
                    lambda x: np.ones_like(x) * alignment_times[0, 1]  # noqa: E731

        score_pitches = np.unique(score_note_array["pitch"])
        # loop over pitches and align full sequences of matching pitches in correct order
        # if sequences mismatch in length, classify extra notes as insertions or deletions respectively
        for pitch in score_pitches:
         


            score_notes = score_note_array[pitch == score_note_array['pitch']]
            performance_notes = performance_note_array[pitch == performance_note_array['pitch']]
       
            score_notes_onsets = onset_time_conversion(score_notes["onset_beat"])
            score_notes_onsets_idx = np.argsort(score_notes_onsets)
            score_notes_onsets = score_notes_onsets[score_notes_onsets_idx]
    
            performance_notes_onsets = performance_notes["onset_sec"]
            performance_notes_onsets_idx = np.argsort(performance_notes_onsets)
            performance_notes_onsets = performance_notes_onsets[performance_notes_onsets_idx]

            score_no = score_notes_onsets.shape[0]
            performance_no = performance_notes_onsets.shape[0]
    
            common_no = min(score_no, performance_no)
            extra_no = max(score_no, performance_no)-common_no
            score_longer = np.argmax([performance_no, score_no])

            if score_longer:
                longt = score_notes_onsets
                shortt = performance_notes_onsets
                longid = score_notes_onsets_idx
                shortid = performance_notes_onsets_idx
            else:
                longt = performance_notes_onsets
                shortt = score_notes_onsets
                longid = performance_notes_onsets_idx
                shortid = score_notes_onsets_idx

            diffs = dict()
 
            if shift:
                if cap_combinations is not None:
                    combination_number = binom(max(score_no, performance_no), extra_no)
                    if combination_number > cap_combinations:
                        combs = [np.random.choice(max(score_no, performance_no), extra_no, replace=False) for n in range(cap_combinations)]  

                        print("high number of combinations: ", combination_number, "low number sampled ", len(combs))
                    else:
                        combs = combinations(range(max(score_no, performance_no)), extra_no)
                else:
                    combs = combinations(range(max(score_no, performance_no)), extra_no)
                for omit_idx in combinations(range(max(score_no, performance_no)), extra_no):
                    shortenedt = np.delete(longt,list(omit_idx))
                    optimal_shift = np.mean(shortenedt-shortt)
                    shift_diff = np.sum(np.abs(shortenedt-shortt-optimal_shift*np.ones_like(shortenedt))**2)
                    diffs[shift_diff] = list(omit_idx)


            else:
                
                if cap_combinations is not None:
                    combination_number = binom(max(score_no, performance_no), extra_no)
                    if combination_number > cap_combinations:
                        combs = [np.random.choice(max(score_no, performance_no), extra_no, replace=False) for n in range(cap_combinations)]  

                        print("high number of combinations: ", combination_number, "low number sampled ", len(combs))
                    else:
                        combs = combinations(range(max(score_no, performance_no)), extra_no)
                else:
                    combs = combinations(range(max(score_no, performance_no)), extra_no)
                for omit_idx in combs:

                    shortenedt = np.delete(longt,list(omit_idx))
                    diff = np.sum(np.abs(shortenedt-shortt)**2)
                    diffs[diff] = list(omit_idx)

            best_omit_dist = np.min(list(diffs.keys()))
            best_omit_idx = diffs[best_omit_dist]

            # get the arrays of actual onset times
            aligns = np.delete(longt, best_omit_idx)
            nonaligns = longt[best_omit_idx]
            # get the idx of the original note_arrays
            align_ids = np.delete(longid, best_omit_idx)
            nonalign_ids = longid[best_omit_idx]

            if score_longer:
                for sid, pid in zip(score_notes["id"][align_ids], performance_notes["id"][performance_notes_onsets_idx]):
                    alignment.append({'label': 'match', 'score_id': sid, 'performance_id': str(pid)})
                    p_aligned.append(str(pid))

                for sid in score_notes["id"][nonalign_ids]:
                    alignment.append({'label': 'deletion', 'score_id': sid})

            else:
                for sid, pid in zip(score_notes["id"][score_notes_onsets_idx], performance_notes["id"][align_ids]):
                    alignment.append({'label': 'match', 'score_id': sid, 'performance_id': str(pid)})
                    p_aligned.append(str(pid))

                for pid in performance_notes["id"][nonalign_ids]:
                    alignment.append({'label': 'insertion', 'performance_id': str(pid)})
                    p_aligned.append(str(pid))

        # check for unaligned performance notes (ie insertions)
        for p_note in performance_note_array:
            if str(p_note['id']) not in p_aligned:
                alignment.append({'label': 'insertion', 'performance_id': str(p_note['id'])})

        return alignment
    
    
class OnsetGreedyMatcher(object):
    """
    Create alignment in MAPS format (dict) 
    by pitch matching from an onset-wise
    alignment

    starts from an onset-wise alignment
    and then matches the notes in the
    score and performance that are
    aligned on the same onset
    in a second step, it matches the
    remaining performance notes pitch-wise
    in (-4, 4) score windows.

    """
    def __call__(self, 
                 score_note_array, 
                 performance_note_array,
                 onset_alignment):
        alignment = []
        s_aligned = []
        p_aligned = []
        unique_onsets = np.unique(score_note_array['onset_beat'])
        # for p_no, p_note in enumerate(performance_note_array):
        for p_no, s_onset_no in onset_alignment:
            p_note = performance_note_array[p_no]
            pid = p_note['id']
            sid = None
            
            for k in range(2): # check onsets up to 5 steps in the future
                # s_candidate_mask = onset_alignment[p_no]
                try: 
                    s_onset = unique_onsets[s_onset_no+k]
                    score_note_array_segment = score_note_array[score_note_array['onset_beat'] == s_onset]
                    # filter performance notes with matching pitches
                    matching_pitches = score_note_array_segment[p_note['pitch'] == score_note_array_segment['pitch']]

                    for s_note in matching_pitches:
                        # take first matching performance note that was not yet aligned
                        if s_note not in s_aligned and p_note not in p_aligned:
                            sid = str(s_note['id'])
                            p_aligned.append(p_note)
                            s_aligned.append(s_note)
                            break 
                    
                    if sid is not None or len(matching_pitches) == 0:
                        break
                   
                except:
                    print("next onset trial error in OnsetGreedyMatcher")

            if sid is not None:
                alignment.append({'label': 'match', 'score_id': sid, 'performance_id': str(pid)})
                
        
        # check for unaligned performance notes (ie insertions) 
        for p_no, p_note in enumerate(performance_note_array):
            if p_note not in p_aligned:
                # neighborhood watch
                pid = p_note['id']
                mask  = (onset_alignment[:,0] == p_no)
                sid = None
                s_onset = np.min(unique_onsets[onset_alignment[mask,1]])
                smask  = np.all((score_note_array['onset_beat'] < s_onset + 4, score_note_array['onset_beat'] > s_onset - 4), axis= 0)
                possible_score_notes = score_note_array[smask]
                for s_note in possible_score_notes:
                    if s_note["pitch"] == p_note["pitch"] and s_note not in s_aligned:
                        sid = str(s_note['id'])
                        p_aligned.append(p_note)
                        s_aligned.append(s_note)
                        alignment.append({'label': 'match', 'score_id': sid, 'performance_id': str(pid)})
                        break
                
                # ok enough
                if sid is None:
                    alignment.append({'label': 'insertion', 'performance_id': str(p_note['id'])})
                
        # check for unaligned score notes (ie deletions)
        for s_note in score_note_array:
            if s_note not in s_aligned:           
                alignment.append({'label': 'deletion', 'score_id': str(s_note['id'])})
             
        return alignment


class CleanOnsetMatcher(object):
    """
    Create alignment in MAPS format (dict) 
    by pitch matching from an onset-wise
    alignment.

    1. clean onset alignment,
        don't reuse pitches of previous onsets
    2. create score to perf map
    3. cut the note arrays at every onset tuple!
    4. local mathing
    5. alignment mending

    """
    def __call__(self, 
                 score_note_array, 
                 performance_note_array,
                 onset_alignment):


        # Get time alignments from first unaligned notes
        print("clean dtw alignment")
        s_aligned = []
        p_aligned = []
        time_tuples = defaultdict(list)
        unique_onsets = np.unique(score_note_array['onset_beat'])
        # for p_no, p_note in enumerate(performance_note_array):
        for p_no, s_onset_no in onset_alignment:
            p_note = performance_note_array[p_no]
            pid = str(p_note['id'])
            if pid not in p_aligned:
                s_onset = unique_onsets[s_onset_no]
                if s_onset_no > 0:
                    s_onset_prev = unique_onsets[s_onset_no-1]
                else: 
                    s_onset_prev = -10
                score_note_array_segment = score_note_array[score_note_array['onset_beat'] == s_onset]
                score_note_array_segment_prev = score_note_array[score_note_array['onset_beat'] == s_onset_prev]
                matching_pitches = score_note_array_segment[p_note['pitch'] == score_note_array_segment['pitch']]
            
                for s_note in matching_pitches:
                    sid = str(s_note['id'])
                    # take first matching performance note that was not yet aligned
                    if sid not in s_aligned and s_note["pitch"] not in score_note_array_segment_prev["pitch"]:
                        p_aligned.append(pid)
                        s_aligned.append(sid)
                        time_tuples[s_note["onset_beat"]].append( p_note["onset_sec"])
                        break 
        
        x_score = list()
        y_perf = list()
        for k in time_tuples.keys():
            x_score.append(k)
            y_perf.append(np.min(time_tuples[k]))

        score_to_perf_map = interp1d(x_score,
                                     y_perf,
                                     fill_value="extrapolate")

        x_score_cut_locations = np.array(x_score + [x_score[-1]+10]) - 0.05

        cleaned_alignment = np.column_stack((x_score_cut_locations, score_to_perf_map(x_score_cut_locations)))
        
        print("cut note arrays", cleaned_alignment.shape)

        score_note_arrays, performance_note_arrays = cut_note_arrays(performance_note_array, 
                    score_note_array, 
                    cleaned_alignment,
                    sfuzziness=0.0, 
                    pfuzziness=0.0, 
                    window_size=1,
                    pfuzziness_relative_to_tempo=False)

        symbolic_note_matcher=SequenceAugmentedGreedyMatcher()
        note_alignments = list()

        print("local alignment", len(score_note_arrays))

        for window_id in range(len(score_note_arrays)):
            dtw_alignment_times = cleaned_alignment[window_id:window_id+2, :]
            fine_local_alignment = symbolic_note_matcher(
                    score_note_arrays[window_id],
                    performance_note_arrays[window_id],
                    dtw_alignment_times,
                    shift=False,
                    cap_combinations=100)

            note_alignments.append(fine_local_alignment)

        print("global alignment")

        # MEND windows to global alignment
        global_alignment, score_alignment, \
            performance_alignment = mend_note_alignments(note_alignments, 
                                                    performance_note_array,
                                                    score_note_array, 
                                                    node_times=cleaned_alignment,
                                                    symbolic_note_matcher= symbolic_note_matcher,
                                                    max_traversal_depth=1500)

        return global_alignment


class CleanMatcher(object):
    """
    Create alignment in MAPS format (dict) 
    by pitch matching from an onset-wise
    alignment

    1. get cleaned time tuples from onset alignment
    2. create score to perf map
    3. map each pitch-wise sequence from score to perf
    4. symbolic alignment via onset seq dtw (check threshold) 

    """
    def __call__(self, 
                 score_note_array, 
                 performance_note_array,
                 onset_alignment,
                 onset_threshold=3.0):

        if onset_threshold is None:
            onset_threshold1 = 1000000
        else:
            onset_threshold1 = onset_threshold
        # Get time alignments from first unaligned notes
        time_tuples_by_onset, unique_time_tuples_by_onset, \
            time_tuples_by_pitch, \
                unique_time_tuples = pitch_and_onset_wise_times(performance_note_array, 
                                                                score_note_array, 
                                                                onset_alignment)

        score_to_perf_map = interp1d(unique_time_tuples[:,0],# score onsets
                                     unique_time_tuples[:,1],# perf onsets
                                     fill_value="extrapolate")
        
        note_alignments = list()
        used_score_note_ids = set()
        used_performance_note_ids = set()
        for pitch in np.unique(score_note_array['pitch']):
            score_note_array_pitch = score_note_array[score_note_array['pitch'] == pitch]
            score_note_array_pitch_sort_idx = np.argsort(score_note_array_pitch['onset_beat'])
            score_note_array_pitch = score_note_array_pitch[score_note_array_pitch_sort_idx]
            performance_note_array_pitch = performance_note_array[performance_note_array['pitch'] == pitch]
            estimated_performance_note_onsets = score_to_perf_map(score_note_array_pitch['onset_beat'])

            if performance_note_array_pitch.shape[0] > 1 and score_note_array_pitch.shape[0] > 1:
                s_p_ID_tuples = unique_alignments(estimated_performance_note_onsets, 
                                                performance_note_array_pitch["onset_sec"],
                                                threshold=onset_threshold1)
            
            elif performance_note_array_pitch.shape[0] == 1 and score_note_array_pitch.shape[0] == 1: 
                if np.abs(estimated_performance_note_onsets[0] - performance_note_array_pitch["onset_sec"][0]) < onset_threshold1:
                    s_p_ID_tuples = [(0,0)]
                else:
                    s_p_ID_tuples = []
            else:
                s_p_ID_tuples = []       
            
            for s_ID, p_ID in s_p_ID_tuples:
                note_alignments.append({'label': 'match', 
                                        "score_id": score_note_array_pitch["id"][s_ID], 
                                        "performance_id": performance_note_array_pitch["id"][p_ID]})
                used_score_note_ids.add(score_note_array_pitch["id"][s_ID])
                used_performance_note_ids.add(performance_note_array_pitch["id"][p_ID])

        
        # add unmatched notes
        for score_note in score_note_array:
            if score_note["id"] not in used_score_note_ids:
                note_alignments.append({'label': 'deletion', 'score_id': score_note["id"]})
        
        for performance_note in performance_note_array:
            if performance_note["id"] not in used_performance_note_ids:
                note_alignments.append({'label': 'insertion', 'performance_id': performance_note["id"]})

        return note_alignments


def unique_alignments(xs, ys, threshold = None):
    """
    From two sequences of numbers, return the unique ID
    tuples of aligned values that minimize the sum of
    tupel distances.

    Parameters
    ----------      
    xs : np.array   
        Sequence of numbers
    ys : np.array
        Sequence of numbers
    threshold : float   

    Returns
    -------
    tuples : list
    
    """
    matcher = DTW()
    _, p = matcher(xs.reshape((-1,1)),ys.reshape((-1,1)), return_path=True)

    used_x = set()
    tuples = list()
    if threshold is not None:
        for x in xs:
            if not x in used_x:
                current_x_mask = (xs[p[:,0]] == x)
                if current_x_mask.sum() > 1:
                    current_ids = p[current_x_mask,:]
                    current_y = ys[current_ids[:,1]]
                    current_x = xs[current_ids[:,0]]
                    
                else:
                    y = ys[p[current_x_mask,1]]
                    current_y_mask = (ys[p[:,1]] == y)
                    # print(current_y_mask, current_x_mask, y, ys, xs, x, p[:,:])
                    if current_y_mask.sum() > 1:
                        current_ids = p[current_y_mask,:]
                        current_y = ys[current_ids[:,1]]
                        current_x = xs[current_ids[:,0]]
                
                    else:
                        current_ids = p[current_x_mask,:].reshape(1,2)
                        current_x = x
                        current_y = y

                candidate_dist = np.min(np.abs(current_x - current_y))
                candidate_id = np.argmin(np.abs(current_x - current_y))
                if candidate_dist < threshold:
                    # chosen_y = current_y[candidate_id]
                    # tuples.append((x, chosen_y))
                    tuples.append((current_ids[candidate_id,0], current_ids[candidate_id,1]))
                used_x.update( set(np.unique(current_x)))
    else:
        for x in xs:
            if not x in used_x:
                current_x_mask = (xs[p[:,0]] == x)
                if current_x_mask.sum() > 1:
                    current_ids = p[current_x_mask,:]
                    current_y = ys[current_ids[:,1]]
                    current_x = xs[current_ids[:,0]]
                    
                else:
                    y = ys[p[current_x_mask,1]]
                    current_y_mask = (ys[p[:,1]] == y)

                    if current_y_mask.sum() > 1:
                        current_ids = p[current_y_mask,:]
                        current_y = ys[current_ids[:,1]]
                        current_x = xs[current_ids[:,0]]
                
                    else:
                        current_ids = p[current_x_mask,:].reshape(1,2)
                        current_x = x
                        current_y = y

                candidate_dist = np.min(np.abs(current_x - current_y))
                candidate_id = np.argmin(np.abs(current_x - current_y))
                tuples.append((current_ids[candidate_id,0], current_ids[candidate_id,1]))
                used_x.update( set(np.unique(current_x)))
    return tuples

def pitch_and_onset_wise_times(performance_note_array, 
                               score_note_array, 
                               alignment_ids
                               # return_ids=False
                               ):
    """
    from a performed MIDI note to score onset alignment
    create a list of tuples of (score_onset, performance_onset)
    where the performance_onset is the minimum onset of the performed notes
    at this score onset.
    Also create a list of tuples of (score_pitch, performance_pitch)
    for each pitch in the score.

    Parameters
    ----------
    performance_note_array : np.array
        Array of performed notes
    score_note_array : np.array
        Array of score notes
    alignment_ids : list
        List of tuples of (score_onset_id, performance_note_id)


    Returns
    -------
    time_tuples_by_onset : dict
        Dictionary of list of (performance_onsets)
    unique_time_tuples_by_onset : dict
        Dictionary of (earliest) performance_onsets indexed by score_onsets
    time_tuples_by_pitch : dict
        Dictionary of lists of tuples of (score_onset, performance_onset)
    unique_time_tuples : np.array
        Array of (score_onset, (earliest) performance_onset) tuples

    """
    p_aligned = set()
    time_tuples_by_onset = defaultdict(list)
    time_tuples_by_pitch = defaultdict(list)
    block_by_pitch_by_onset = defaultdict(dict)
    all_pitch_repeat_by_onset = defaultdict(lambda : False)

    # time_tuples_by_onset_id = defaultdict(list)
    # time_tuples_by_pitch_id = defaultdict(list)

    unique_onsets = np.unique(score_note_array['onset_beat'])
    pitches_by_onset = {s_onset: set(score_note_array[score_note_array['onset_beat'] == s_onset]["pitch"]) for s_onset in unique_onsets}
    # keep track of pitches repeated in adjacent score onsets
    for s_o in np.arange(1, len(unique_onsets)):
        complete_block = []
        for p in pitches_by_onset[unique_onsets[s_o]]:
            block_by_pitch_by_onset[unique_onsets[s_o]][p] = p in pitches_by_onset[unique_onsets[s_o-1]]
            complete_block.append(block_by_pitch_by_onset[unique_onsets[s_o]][p])
        # keep track of completely repeated score onsets
        all_pitch_repeat_by_onset[unique_onsets[s_o]] = all(complete_block)
    for p in pitches_by_onset[unique_onsets[0]]:
        block_by_pitch_by_onset[unique_onsets[0]][p] = False
    
    used_pitches_by_onset = defaultdict(set)
    for p_no, s_onset_no in alignment_ids:
        # p_no is the index of the performance note in the performance note array
        # s_onset_no is the index of the score onset in the unique_onsets array
        p_note = performance_note_array[p_no]
        pid = str(p_note['id'])
        ppitch = p_note['pitch']
        if pid not in p_aligned:
            s_onset = unique_onsets[s_onset_no]
            used_pitches = used_pitches_by_onset[s_onset]
            available_pitches = pitches_by_onset[s_onset] 
            s_pitch_used = ppitch in used_pitches
            s_pitch_available = ppitch in available_pitches
            if s_pitch_available and not s_pitch_used:
                if not block_by_pitch_by_onset[s_onset][ppitch]:
                    used_pitches_by_onset[s_onset].add(ppitch)
                    time_tuples_by_pitch[ppitch].append((s_onset, p_note['onset_sec']))
                    time_tuples_by_onset[s_onset].append(p_note['onset_sec'])
                    # time_tuples_by_pitch_id[ppitch].append((s_onset_no, p_no))
                    # time_tuples_by_onset_id[s_onset_no].append(p_no)
                    p_aligned.add(pid)


    # make clean sequences
    onsets_with_performance_times = np.array(list(time_tuples_by_onset.keys()))
    current_s_onset_no = 0
    for s_onset_no in range(len(unique_onsets)):
        if s_onset_no > current_s_onset_no:
            s_onset = unique_onsets[s_onset_no]
            if all_pitch_repeat_by_onset[s_onset]:
                local_s_onset_no = s_onset_no
                s_onset_range = [unique_onsets[s_onset_no - 1]]
                not_last = True
                while(all_pitch_repeat_by_onset[unique_onsets[local_s_onset_no]]):
                    s_onset_range.append(unique_onsets[local_s_onset_no])
                    local_s_onset_no += 1
                    if local_s_onset_no >= len(unique_onsets)-1:
                        not_last = False
                        break
                if not_last:
                    current_s_onset_no = local_s_onset_no - 1
                    s_onset_range = np.array(s_onset_range)
                    first_s_onset_in_range = s_onset_range[0]
                    first_s_onset_out_of_range = unique_onsets[local_s_onset_no]
                    first_s_onset_in_range_aligned = np.max(onsets_with_performance_times[onsets_with_performance_times<=first_s_onset_in_range])
                    first_s_onset_out_of_range_aligned = np.min(onsets_with_performance_times[onsets_with_performance_times>=first_s_onset_out_of_range])
                    first_p_onset_in_range = np.min(time_tuples_by_onset[first_s_onset_in_range_aligned])   
                    first_p_onset_out_of_range = np.min(time_tuples_by_onset[first_s_onset_out_of_range_aligned])   
                    for pitch in pitches_by_onset[s_onset_range[1]]:  
                        pitch_mask = performance_note_array['pitch'] == pitch
                        higher_mask = performance_note_array['onset_sec'] >= first_p_onset_in_range
                        lower_mask = performance_note_array['onset_sec'] < first_p_onset_out_of_range
                        available_pp_notes = performance_note_array[np.all((pitch_mask, higher_mask, lower_mask), axis=0)]
                        if len(available_pp_notes) == len(s_onset_range):
                            for s_onset_local, p_onset_local in zip(s_onset_range, available_pp_notes):
                                time_tuples_by_pitch[pitch].append((s_onset_local, p_onset_local['onset_sec']))
                                time_tuples_by_onset[s_onset_local].append(p_onset_local['onset_sec'])
                        
    # remove outliers
    for s_onset in time_tuples_by_onset.keys():
        sorted_times = np.sort(np.array(time_tuples_by_onset[s_onset]))
        mask = np.abs(sorted_times - np.median(sorted_times)) < 0.1

        if mask.sum() >= 1:
            time_tuples_by_onset[s_onset] = list(sorted_times[mask])
        else:
            time_tuples_by_onset[s_onset] = list(sorted_times)

    unique_time_tuples_by_onset = {s_onset : np.min(time_tuples_by_onset[s_onset]) for s_onset in time_tuples_by_onset.keys()}
    unique_time_tuples = np.array([(tup, unique_time_tuples_by_onset[tup]) for tup in unique_time_tuples_by_onset.keys()])  
    unique_time_tuples = unique_time_tuples[unique_time_tuples[:,0].argsort()]

    # unique_time_tuples_by_onset_id = {s_onset_no : np.min(time_tuples_by_onset_id[s_onset_no]) for s_onset_no in time_tuples_by_onset_id.keys()}
    # unique_time_tuples_id = np.array([(tup, unique_time_tuples_by_onset_id[tup]) for tup in unique_time_tuples_by_onset_id.keys()]) 
    # if not return_ids:
    return time_tuples_by_onset, unique_time_tuples_by_onset, time_tuples_by_pitch, unique_time_tuples
    # else:
    #     return time_tuples_by_onset_id, unique_time_tuples_by_onset_id, time_tuples_by_pitch_id, unique_time_tuples_id

def pitch_and_onset_wise_times_ornament(performance_note_array, 
                               score_note_array, 
                               alignment_ids
                                ):
    """
    from a performed MIDI note to score onset alignment
    create a list of tuples of (score_onset, performance_onset)
    where the performance_onset is the minimum onset of the performed notes
    at this score onset.
    Also create a list of tuples of (score_pitch, performance_pitch)
    for each pitch in the score.

    Parameters
    ----------
    performance_note_array : np.array
        Array of performed notes
    score_note_array : np.array
        Array of score notes
    alignment_ids : list
        List of tuples of (score_onset_id, performance_note_id)


    Returns
    -------
    time_tuples_by_onset : dict
        Dictionary of list of (performance_onsets)
    unique_time_tuples_by_onset : dict
        Dictionary of (earliest) performance_onsets indexed by score_onsets
    time_tuples_by_pitch : dict
        Dictionary of lists of tuples of (score_onset, performance_onset)
    unique_time_tuples : np.array
        Array of (score_onset, (earliest) performance_onset) tuples

    """
    p_aligned = set()
    time_tuples_by_onset = defaultdict(list)
    time_tuples_by_pitch = defaultdict(list)
    block_by_pitch_by_onset = defaultdict(dict)
    all_pitch_repeat_by_onset = defaultdict(lambda : False)

    unique_onsets = np.unique(score_note_array['onset_beat'])
    pitches_by_onset = {s_onset: set(score_note_array[score_note_array['onset_beat'] == s_onset]["pitch"]) for s_onset in unique_onsets}
    
    # # keep track of pitches repeated in adjacent score onsets
    # for s_o in np.arange(1, len(unique_onsets)):
    #     complete_block = []
    #     for p in pitches_by_onset[unique_onsets[s_o]]:
    #         block_by_pitch_by_onset[unique_onsets[s_o]][p] = p in pitches_by_onset[unique_onsets[s_o-1]]
    #         complete_block.append(block_by_pitch_by_onset[unique_onsets[s_o]][p])
    #     # keep track of completely repeated score onsets
    #     all_pitch_repeat_by_onset[unique_onsets[s_o]] = all(complete_block)
    # for p in pitches_by_onset[unique_onsets[0]]:
    #     block_by_pitch_by_onset[unique_onsets[0]][p] = False

    # keep track of pitches repeated in adjacent score onsets
    last_new_pitches = set()
    for s_o in np.arange(0, len(unique_onsets)):
        complete_block = []
        any_out = False
        for p in pitches_by_onset[unique_onsets[s_o]]:
            if p not in last_new_pitches:
                any_out = True
                block_by_pitch_by_onset[unique_onsets[s_o]][p] = False
            else:
                block_by_pitch_by_onset[unique_onsets[s_o]][p] = True
                
        if any_out:
            last_new_pitches = pitches_by_onset[unique_onsets[s_o]]
            all_pitch_repeat_by_onset[unique_onsets[s_o]] = False
        else:
            all_pitch_repeat_by_onset[unique_onsets[s_o]] = True



    # get first onsets of each pitch in each pitch-stable sequence 
    # (dtw alignment messes up the order)
    used_pitches_by_onset = defaultdict(set)
    for p_no, s_onset_no in alignment_ids:
        # p_no is the index of the performance note in the performance note array
        # s_onset_no is the index of the score onset in the unique_onsets array
        p_note = performance_note_array[p_no]
        pid = str(p_note['id'])
        ppitch = p_note['pitch']
        if pid not in p_aligned:
            s_onset = unique_onsets[s_onset_no]
            used_pitches = used_pitches_by_onset[s_onset]
            available_pitches = pitches_by_onset[s_onset]
            s_pitch_used = ppitch in used_pitches
            s_pitch_available = ppitch in available_pitches
            if s_pitch_available and not s_pitch_used:
                if not block_by_pitch_by_onset[s_onset][ppitch]:
                    used_pitches_by_onset[s_onset].add(ppitch)
                    time_tuples_by_pitch[ppitch].append((s_onset, p_note['onset_sec']))
                    time_tuples_by_onset[s_onset].append(p_note['onset_sec'])
                    p_aligned.add(pid)
    # remove outliers
    for s_onset in time_tuples_by_onset.keys():
        sorted_times = np.sort(np.array(time_tuples_by_onset[s_onset]))
        mask = np.abs(sorted_times - np.median(sorted_times)) < 0.1

        if mask.sum() >= 1:
            time_tuples_by_onset[s_onset] = list(sorted_times[mask])
        else:
            time_tuples_by_onset[s_onset] = list(sorted_times)

    # # make clean sequences
    # onsets_with_performance_times = np.array(list(time_tuples_by_onset.keys()))
    # current_s_onset_no = 0
    # for s_onset_no in range(len(unique_onsets)):
    #     if s_onset_no > current_s_onset_no:
    #         s_onset = unique_onsets[s_onset_no]
    #         if all_pitch_repeat_by_onset[s_onset]:
    #             local_s_onset_no = s_onset_no
    #             s_onset_range = [unique_onsets[s_onset_no - 1]]
    #             not_last = True
    #             while(all_pitch_repeat_by_onset[unique_onsets[local_s_onset_no]]):
    #                 s_onset_range.append(unique_onsets[local_s_onset_no])
    #                 local_s_onset_no += 1
    #                 if local_s_onset_no >= len(unique_onsets)-1:
    #                     not_last = False
    #                     break
    #             if not_last:
    #                 current_s_onset_no = local_s_onset_no - 1
    #                 s_onset_range = np.array(s_onset_range)
    #                 first_s_onset_in_range = s_onset_range[0]
    #                 first_s_onset_out_of_range = unique_onsets[local_s_onset_no]
    #                 first_s_onset_in_range_aligned = np.max(onsets_with_performance_times[onsets_with_performance_times<=first_s_onset_in_range])
    #                 first_s_onset_out_of_range_aligned = np.min(onsets_with_performance_times[onsets_with_performance_times>=first_s_onset_out_of_range])
    #                 first_p_onset_in_range = np.min(time_tuples_by_onset[first_s_onset_in_range_aligned])   
    #                 first_p_onset_out_of_range = np.min(time_tuples_by_onset[first_s_onset_out_of_range_aligned])   
    #                 for pitch in pitches_by_onset[s_onset_range[1]]:  
    #                     pitch_mask = performance_note_array['pitch'] == pitch
    #                     higher_mask = performance_note_array['onset_sec'] >= first_p_onset_in_range
    #                     lower_mask = performance_note_array['onset_sec'] < first_p_onset_out_of_range
    #                     available_pp_notes = performance_note_array[np.all((pitch_mask, higher_mask, lower_mask), axis=0)]
    #                     if len(available_pp_notes) == len(s_onset_range):
    #                         for s_onset_local, p_onset_local in zip(s_onset_range, available_pp_notes):
    #                             time_tuples_by_pitch[pitch].append((s_onset_local, p_onset_local['onset_sec']))
    #                             time_tuples_by_onset[s_onset_local].append(p_onset_local['onset_sec'])


    # # remove outliers
    # for s_onset in time_tuples_by_onset.keys():
    #     sorted_times = np.sort(np.array(time_tuples_by_onset[s_onset]))
    #     mask = np.abs(sorted_times - np.median(sorted_times)) < 0.1

    #     if mask.sum() >= 1:
    #         time_tuples_by_onset[s_onset] = list(sorted_times[mask])
    #     else:
    #         time_tuples_by_onset[s_onset] = list(sorted_times)

    unique_time_tuples_by_onset = {s_onset : np.min(time_tuples_by_onset[s_onset]) for s_onset in time_tuples_by_onset.keys()}
    unique_time_tuples = np.array([(tup, unique_time_tuples_by_onset[tup]) for tup in unique_time_tuples_by_onset.keys()])  
    unique_time_tuples = unique_time_tuples[unique_time_tuples[:,0].argsort()]

    return time_tuples_by_onset, unique_time_tuples_by_onset, time_tuples_by_pitch, unique_time_tuples

def pitch_and_onset_wise_times_simple(performance_note_array, 
                               score_note_array, 
                               alignment_ids
                               ):
    """
    from a performed MIDI note to score onset alignment
    create a list of tuples of (score_onset, performance_onset)
    where the performance_onset is the minimum onset of the performed notes
    at this score onset.
    Also create a list of tuples of (score_pitch, performance_pitch)
    for each pitch in the score.

    Parameters
    ----------
    performance_note_array : np.array
        Array of performed notes
    score_note_array : np.array
        Array of score notes
    alignment_ids : list
        List of tuples of (score_onset_id, performance_note_id)


    Returns
    -------
    time_tuples_by_onset : dict
        Dictionary of list of (performance_onsets)
    unique_time_tuples_by_onset : dict
        Dictionary of (earliest) performance_onsets indexed by score_onsets
    time_tuples_by_pitch : dict
        Dictionary of lists of tuples of (score_onset, performance_onset)
    unique_time_tuples : np.array
        Array of (score_onset, (earliest) performance_onset) tuples

    """
    p_aligned = set()
    time_tuples_by_onset = defaultdict(list)
    time_tuples_by_pitch = defaultdict(list)
    block_by_pitch_by_onset = defaultdict(dict)
    all_pitch_repeat_by_onset = defaultdict(lambda : False)

    unique_onsets = np.unique(score_note_array['onset_beat'])
    pitches_by_onset = {s_onset: set(score_note_array[score_note_array['onset_beat'] == s_onset]["pitch"]) for s_onset in unique_onsets}
    
    # keep track of pitches repeated in adjacent score onsets
    for s_o in np.arange(1, len(unique_onsets)):
        complete_block = []
        for p in pitches_by_onset[unique_onsets[s_o]]:
            block_by_pitch_by_onset[unique_onsets[s_o]][p] = p in pitches_by_onset[unique_onsets[s_o-1]]
            complete_block.append(block_by_pitch_by_onset[unique_onsets[s_o]][p])
        # keep track of completely repeated score onsets
        all_pitch_repeat_by_onset[unique_onsets[s_o]] = all(complete_block)
    for p in pitches_by_onset[unique_onsets[0]]:
        block_by_pitch_by_onset[unique_onsets[0]][p] = False

    # get first onsets of each pitch in each pitch-stable sequence 
    # (dtw alignment messes up the order)
    used_pitches_by_onset = defaultdict(set)
    for p_no, s_onset_no in alignment_ids:
        # p_no is the index of the performance note in the performance note array
        # s_onset_no is the index of the score onset in the unique_onsets array
        p_note = performance_note_array[p_no]
        pid = str(p_note['id'])
        ppitch = p_note['pitch']
        if pid not in p_aligned:
            s_onset = unique_onsets[s_onset_no]
            used_pitches = used_pitches_by_onset[s_onset]
            available_pitches = pitches_by_onset[s_onset]
            s_pitch_used = ppitch in used_pitches
            s_pitch_available = ppitch in available_pitches
            if s_pitch_available and not s_pitch_used:
                if not block_by_pitch_by_onset[s_onset][ppitch]:
                    used_pitches_by_onset[s_onset].add(ppitch)
                    time_tuples_by_pitch[ppitch].append((s_onset, p_note['onset_sec']))
                    time_tuples_by_onset[s_onset].append(p_note['onset_sec'])
                    p_aligned.add(pid)

    unique_time_tuples_by_onset = {s_onset : np.min(time_tuples_by_onset[s_onset]) for s_onset in time_tuples_by_onset.keys()}
    unique_time_tuples = np.array([(tup, unique_time_tuples_by_onset[tup]) for tup in unique_time_tuples_by_onset.keys()])  
    unique_time_tuples = unique_time_tuples[unique_time_tuples[:,0].argsort()]

    return time_tuples_by_onset, unique_time_tuples_by_onset, time_tuples_by_pitch, unique_time_tuples

def pitch_and_onset_wise_times_rev(performance_note_array, 
                               score_note_array, 
                               alignment_ids,
                               backwards = True,
                               ):
    """
    from a performed MIDI note to score onset alignment
    create a list of tuples of (score_onset, performance_onset)
    where the performance_onset is the minimum onset of the performed notes
    at this score onset.
    Also create a list of tuples of (score_pitch, performance_pitch)
    for each pitch in the score.

    Parameters
    ----------
    performance_note_array : np.array
        Array of performed notes
    score_note_array : np.array
        Array of score notes
    alignment_ids : list
        List of tuples of (score_onset_id, performance_note_id)


    Returns
    -------
    time_tuples_by_onset : dict
        Dictionary of list of (performance_onsets)
    unique_time_tuples_by_onset : dict
        Dictionary of (earliest) performance_onsets indexed by score_onsets
    time_tuples_by_pitch : dict
        Dictionary of lists of tuples of (score_onset, performance_onset)
    unique_time_tuples : np.array
        Array of (score_onset, (earliest) performance_onset) tuples

    """
    p_aligned = set()
    time_tuples_by_onset = defaultdict(list)
    time_tuples_by_pitch = defaultdict(list)
    block_by_pitch_by_onset = defaultdict(dict)
    all_pitch_repeat_by_onset = defaultdict(lambda : False)

    # --------reverse specials
    unique_onsets = np.unique(score_note_array['onset_beat'])
    if backwards:
        unique_onsets = np.flipud(unique_onsets)
    pitches_by_onset = {s_onset: set(score_note_array[score_note_array['onset_beat'] == s_onset]["pitch"]) for s_onset in unique_onsets}
    # --------reverse specials

    # # keep track of pitches repeated in adjacent score onsets
    # for s_o in np.arange(1, len(unique_onsets)):
    #     complete_block = []
    #     for p in pitches_by_onset[unique_onsets[s_o]]:
    #         block_by_pitch_by_onset[unique_onsets[s_o]][p] = p in pitches_by_onset[unique_onsets[s_o-1]]
    #         complete_block.append(block_by_pitch_by_onset[unique_onsets[s_o]][p])
    #     # keep track of completely repeated score onsets
    #     all_pitch_repeat_by_onset[unique_onsets[s_o]] = all(complete_block)
    # for p in pitches_by_onset[unique_onsets[0]]:
    #     block_by_pitch_by_onset[unique_onsets[0]][p] = False


    last_new_pitches = set()
    for s_o in np.arange(0, len(unique_onsets)):
        complete_block = []
        any_out = False
        for p in pitches_by_onset[unique_onsets[s_o]]:
            if p not in last_new_pitches:
                any_out = True
                block_by_pitch_by_onset[unique_onsets[s_o]][p] = False
            else:
                block_by_pitch_by_onset[unique_onsets[s_o]][p] = True
                
        if any_out:
            last_new_pitches = pitches_by_onset[unique_onsets[s_o]]
            all_pitch_repeat_by_onset[unique_onsets[s_o]] = False
        else:
            all_pitch_repeat_by_onset[unique_onsets[s_o]] = True


    # get first onsets of each pitch in each pitch-stable sequence 
    # (dtw alignment messes up the order)
    used_pitches_by_onset = defaultdict(set)
    for p_no, s_onset_no in alignment_ids:
        # p_no is the index of the performance note in the performance note array
        # s_onset_no is the index of the score onset in the unique_onsets array
        p_note = performance_note_array[p_no]
        pid = str(p_note['id'])
        ppitch = p_note['pitch']
        if pid not in p_aligned:
            s_onset = unique_onsets[s_onset_no]
            used_pitches = used_pitches_by_onset[s_onset]
            available_pitches = pitches_by_onset[s_onset]
            s_pitch_used = ppitch in used_pitches
            s_pitch_available = ppitch in available_pitches
            if s_pitch_available and not s_pitch_used:
                if not block_by_pitch_by_onset[s_onset][ppitch]:
                    used_pitches_by_onset[s_onset].add(ppitch)
                    time_tuples_by_pitch[ppitch].append((s_onset, p_note['onset_sec']))
                    time_tuples_by_onset[s_onset].append(p_note['onset_sec'])
                    p_aligned.add(pid)


    # remove outliers
    for s_onset in time_tuples_by_onset.keys():
        sorted_times = np.sort(np.array(time_tuples_by_onset[s_onset]))
        mask = np.abs(sorted_times - np.median(sorted_times)) < 0.1

        if mask.sum() >= 1:
            time_tuples_by_onset[s_onset] = list(sorted_times[mask])
        else:
            time_tuples_by_onset[s_onset] = list(sorted_times)

    # # make clean sequences
    # onsets_with_performance_times = np.array(list(time_tuples_by_onset.keys()))
    # current_s_onset_no = 0
    # for s_onset_no in range(len(unique_onsets)):
    #     if s_onset_no > current_s_onset_no:
    #         s_onset = unique_onsets[s_onset_no]
    #         if all_pitch_repeat_by_onset[s_onset]:
    #             local_s_onset_no = s_onset_no
    #             s_onset_range = [unique_onsets[s_onset_no - 1]]
    #             not_last = True
    #             while(all_pitch_repeat_by_onset[unique_onsets[local_s_onset_no]]):
    #                 s_onset_range.append(unique_onsets[local_s_onset_no])
    #                 local_s_onset_no += 1
    #                 if local_s_onset_no >= len(unique_onsets)-1:
    #                     not_last = False
    #                     break
    #             if not_last:
    #                 current_s_onset_no = local_s_onset_no - 1
    #                 s_onset_range = np.array(s_onset_range)
    #                 first_s_onset_in_range = s_onset_range[0]
    #                 first_s_onset_out_of_range = unique_onsets[local_s_onset_no]
    #                 # --------reverse specials

    #                 first_s_onset_in_range_aligned = np.min(onsets_with_performance_times[onsets_with_performance_times<=first_s_onset_in_range])
    #                 first_s_onset_out_of_range_aligned = np.max(onsets_with_performance_times[onsets_with_performance_times>=first_s_onset_out_of_range])
    #                 first_p_onset_in_range = np.min(time_tuples_by_onset[first_s_onset_in_range_aligned])   
    #                 first_p_onset_out_of_range = np.min(time_tuples_by_onset[first_s_onset_out_of_range_aligned])   
    #                 for pitch in pitches_by_onset[s_onset_range[1]]:  
    #                     pitch_mask = performance_note_array['pitch'] == pitch
    #                     higher_mask = performance_note_array['onset_sec'] <= first_p_onset_in_range
    #                     lower_mask = performance_note_array['onset_sec'] > first_p_onset_out_of_range
    #                     available_pp_notes = performance_note_array[np.all((pitch_mask, higher_mask, lower_mask), axis=0)]
    #                     if len(available_pp_notes) == len(s_onset_range):
    #                         for s_onset_local, p_onset_local in zip(s_onset_range, available_pp_notes):
    #                             time_tuples_by_pitch[pitch].append((s_onset_local, p_onset_local['onset_sec']))
    #                             time_tuples_by_onset[s_onset_local].append(p_onset_local['onset_sec'])


    # # remove outliers
    # for s_onset in time_tuples_by_onset.keys():
    #     sorted_times = np.sort(np.array(time_tuples_by_onset[s_onset]))
    #     mask = np.abs(sorted_times - np.median(sorted_times)) < 0.1

    #     if mask.sum() >= 1:
    #         time_tuples_by_onset[s_onset] = list(sorted_times[mask])
    #     else:
    #         time_tuples_by_onset[s_onset] = list(sorted_times)

    unique_time_tuples_by_onset = {s_onset : np.min(time_tuples_by_onset[s_onset]) for s_onset in time_tuples_by_onset.keys()}
    unique_time_tuples = np.array([(tup, unique_time_tuples_by_onset[tup]) for tup in unique_time_tuples_by_onset.keys()])  
    unique_time_tuples = unique_time_tuples[unique_time_tuples[:,0].argsort()]

    return time_tuples_by_onset, unique_time_tuples_by_onset, time_tuples_by_pitch, unique_time_tuples

def get_score_to_perf_map(score_note_array, 
                       performance_note_array, 
                       onset_alignment,
                       onset_alignment_reverse):
    
        score_note_array = score_note_array[np.argsort(score_note_array["onset_beat"])]
        # Get time alignments from first unaligned notes
        time_tuples_by_onset_forward, _, _, unique_time_tuples_forward = pitch_and_onset_wise_times_ornament(performance_note_array, 
                                                                score_note_array, 
                                                                onset_alignment)
        # unique_time_tuples = unique_time_tuples_forward
        performance_note_array_rev = np.flipud(performance_note_array)
        score_note_array_no_grace_rev = np.flipud(score_note_array)
        time_tuples_by_onset_reverse,_,_,unique_time_tuples_reverse = pitch_and_onset_wise_times_rev(performance_note_array_rev, 
                                                                score_note_array_no_grace_rev, 
                                                                onset_alignment_reverse)

        # DUAL MATCHER  -----------------------------------------------------------------------------------


        unique_time_tuples_by_onset = defaultdict(float)
        all_keys = np.unique(list(time_tuples_by_onset_forward.keys()) + list(time_tuples_by_onset_reverse.keys()))
        for s_onset in all_keys:
            p_onsets = np.array(list(set(time_tuples_by_onset_forward[s_onset])))
            p_onsets_rev = np.array(list(set(time_tuples_by_onset_reverse[s_onset])))
            if len(p_onsets) > 0 and len(p_onsets_rev) > 0:
                if np.abs(np.median(p_onsets)-np.median(p_onsets_rev)) <= 0.1:
                    unique_time_tuples_by_onset[s_onset] = np.min((np.min(p_onsets), np.min(p_onsets_rev)))
                # else:
                #     unique_time_tuples_by_onset[s_onset] = np.median(list(p_onsets.union(p_onsets_rev)))
            elif len(p_onsets) > 1:
                if np.max(np.abs(np.median(p_onsets)-p_onsets)) <= 0.1:
                    unique_time_tuples_by_onset[s_onset] = np.min(p_onsets) 
            elif len(p_onsets_rev) > 1:
                if np.max(np.abs(np.median(p_onsets_rev)-p_onsets_rev)) <= 0.1:
                    unique_time_tuples_by_onset[s_onset] = np.min(p_onsets_rev)

        # make clean sequences
        unique_onsets = np.unique(score_note_array['onset_beat'])
        additional_time_tuples_by_onset = defaultdict(list)
        onsets_with_performance_times = np.sort(list(unique_time_tuples_by_onset.keys()))
        for id, score_onset in enumerate(onsets_with_performance_times[:-1]):
            # print(id, score_onset, onsets_with_performance_times[id+1])
            first_s_onset_in_range = score_onset
            first_s_onset_out_of_range = onsets_with_performance_times[id+1]
            onset_mask = np.all((unique_onsets >= first_s_onset_in_range, unique_onsets < first_s_onset_out_of_range), axis=0)   
            if onset_mask.sum() > 1:  

                # use time tuples from forward pass and minimum to make sure the boundaries are correct
                first_p_onset_in_range = np.min(unique_time_tuples_by_onset[first_s_onset_in_range])   
                first_p_onset_out_of_range = np.min(unique_time_tuples_by_onset[first_s_onset_out_of_range]) 

                score_mask = np.all((score_note_array['onset_beat'] >= first_s_onset_in_range,
                                    score_note_array['onset_beat'] < first_s_onset_out_of_range), axis=0 )
                pitches = np.unique( score_note_array[score_mask ]["pitch"])
                perf_mask = np.all((performance_note_array['onset_sec'] >= first_p_onset_in_range,
                                    performance_note_array['onset_sec'] < first_p_onset_out_of_range),axis=0)

                for pitch in pitches:  
                    done = False
                    pitch_mask = performance_note_array['pitch'] == pitch
                    s_pitch_mask = score_note_array['pitch'] == pitch
                    
                    available_pp_notes = performance_note_array[np.all((pitch_mask, perf_mask), axis=0)]
                    available_ss_notes = score_note_array[np.all((score_mask, s_pitch_mask), axis=0)]

                    if len(available_pp_notes) == len(available_ss_notes):
                        # import pdb; pdb.set_trace()
                        for s_onset_local, p_onset_local in zip(available_ss_notes, available_pp_notes):
                            additional_time_tuples_by_onset[s_onset_local['onset_beat']].append(p_onset_local['onset_sec'])
                        done = True

                    if done:
                        break

        for s_onset in additional_time_tuples_by_onset.keys():
            if unique_time_tuples_by_onset[s_onset] == 0:
                unique_time_tuples_by_onset[s_onset] = np.min(additional_time_tuples_by_onset[s_onset])

       

        unique_time_tuples = np.array([(tup, unique_time_tuples_by_onset[tup]) for tup in unique_time_tuples_by_onset.keys()])  
        unique_time_tuples = unique_time_tuples[unique_time_tuples[:,0].argsort()]
        
        # # DUAL MATCHER -----------------------------------------------------------------------------------



        # import matplotlib.pyplot as plt
        # plt.plot(unique_time_tuples_forward[:,0], unique_time_tuples_forward[:,1], label="forward", marker ="x", c="r")
        # plt.plot(unique_time_tuples_reverse[:,0], unique_time_tuples_reverse[:,1], label="reverse", marker ='o',linestyle ='-') 
        # plt.plot(unique_time_tuples[:,0], unique_time_tuples[:,1], label="mix", marker ='x', c="g", linestyle = "dashed") 
        # plt.legend()
        # plt.show()


        score_to_perf_map = interp1d(unique_time_tuples[:,0],# score onsets
                                     unique_time_tuples[:,1],# perf onsets
                                     fill_value="extrapolate")
        
        # score_to_perf_map1 = interp1d(unique_time_tuples_forward[:,0],# score onsets
        #                              unique_time_tuples_forward[:,1],# perf onsets
        #                              fill_value="extrapolate")
        
        # from parangonar.evaluate import plot_alignment_mappings
        # plot_alignment_mappings(performance_note_array, score_note_array_no_grace, score_to_perf_map, score_to_perf_map2)
        # import pdb; pdb.set_trace()

        return score_to_perf_map

def na_within(note_array, 
            field="onset_beat", 
            lower_bound = None, 
            upper_bound = None, 
            pitch = None,
            exclusion_ids = None,
            inclusion_ids = None,
            ordered_by_field = True):
    if pitch is None:
        mask_pitch = np.ones_like(note_array["pitch"])
    else:
        mask_pitch = note_array["pitch"] == pitch

    if lower_bound is None:
        mask_lower = np.ones_like(note_array[field])
    else:
        mask_lower = note_array[field] >= lower_bound

    if upper_bound is None:
        mask_upper = np.ones_like(note_array[field])
    else:
        mask_upper = note_array[field] <= upper_bound

    if exclusion_ids is None:
        mask_exclusion = np.ones_like(note_array[field])
    else:
        mask_exclusion = np.array([n not in exclusion_ids for n in note_array["id"]])

    if inclusion_ids is None:
        mask_inclusion = np.ones_like(note_array[field])
    else:
        mask_inclusion = np.array([n in exclusion_ids for n in note_array["id"]])

    mask = np.all((mask_pitch, mask_lower, mask_upper, mask_exclusion, mask_inclusion), axis=0)
    masked_note_array = note_array[mask]

    if ordered_by_field:
        return masked_note_array[masked_note_array[field].argsort()]
    else:
        return masked_note_array 

class CleanOrnamentMatcher(object):
    """
    Create alignment in MAPS format (dict) 
    by pitch matching from an onset-wise
    alignment

    1. get cleaned time tuples from onset alignment
    2. create score to perf map
    3. map each pitch-wise sequence from score to perf
    4. symbolic alignment via onset seq dtw (check threshold)

    difference to CleanMatcher:
    - no grace notes in dtw/score2perf map
    - grace notes are then mixed in with the score notes 
    for symbolic alignment


    """
    def __call__(self, 
                 score_note_array_full, # score notes including grace notes
                 score_note_array_no_grace, # score notes excluding grace notes 
                 score_note_array_grace, # grace notes
                 score_note_array_ornaments, # score notes with ornaments
                 performance_note_array,
                 onset_alignment,
                 onset_alignment_reverse,
                 onset_threshold=None):

        if onset_threshold is None:
            onset_threshold1 = 1000000
        else:
            onset_threshold1 = onset_threshold

        score_to_perf_map = get_score_to_perf_map(score_note_array_no_grace,
                                                    performance_note_array,
                                                    onset_alignment,
                                                    onset_alignment_reverse)
        
        # Mix the grace notes into the score note array
        grace_onsets = np.unique(score_note_array_grace["onset_beat"])
        grace_onsets_perf = score_to_perf_map(grace_onsets)
        for grace_onset, grace_onset_perf in zip(grace_onsets, grace_onsets_perf):
            mask = (score_note_array_grace['onset_beat'] == grace_onset)
            number_of_grace_notes = mask.sum()
            score_note_array_grace["onset_beat"][mask] -= np.linspace(number_of_grace_notes*0.1,0.0, number_of_grace_notes+1)[:-1]
        
        alignment_score_note_array = np.concatenate((score_note_array_no_grace, score_note_array_grace))
        alignment_score_note_array = alignment_score_note_array[alignment_score_note_array['onset_beat'].argsort()]

        # Get symbolic note_alignments
        note_alignments = list()
        used_score_note_ids = set()
        used_performance_note_ids = set()
        perf_id_from_score_id = defaultdict(list)
        for pitch in np.unique(alignment_score_note_array['pitch']):
            score_note_array_pitch =  na_within(alignment_score_note_array, 
                                                field="onset_beat",
                                                pitch = pitch,
                                                ordered_by_field = True)
            performance_note_array_pitch =  na_within(performance_note_array, 
                                                field="onset_sec",
                                                pitch = pitch,
                                                ordered_by_field = True)
            # score_note_array_pitch = alignment_score_note_array[alignment_score_note_array['pitch'] == pitch]
            # score_note_array_pitch_sort_idx = np.argsort(score_note_array_pitch['onset_beat'])
            # score_note_array_pitch = score_note_array_pitch[score_note_array_pitch_sort_idx]
            # performance_note_array_pitch = performance_note_array[performance_note_array['pitch'] == pitch]
            estimated_performance_note_onsets = score_to_perf_map(score_note_array_pitch['onset_beat'])

            if  (performance_note_array_pitch.shape[0] > 1 and score_note_array_pitch.shape[0] > 1) or \
                (performance_note_array_pitch.shape[0] > 1 and score_note_array_pitch.shape[0] == 1) or \
                (performance_note_array_pitch.shape[0] == 1 and score_note_array_pitch.shape[0] > 1):
                s_p_ID_tuples = unique_alignments(estimated_performance_note_onsets, 
                                                  performance_note_array_pitch["onset_sec"],
                                                  threshold=onset_threshold)
            
            elif performance_note_array_pitch.shape[0] == 1 and score_note_array_pitch.shape[0] == 1: 
                if np.abs(estimated_performance_note_onsets[0] - performance_note_array_pitch["onset_sec"][0]) < onset_threshold1:
                    s_p_ID_tuples = [(0,0)]
                else:
                    s_p_ID_tuples = []
            else:
                s_p_ID_tuples = []       
            
            for s_ID, p_ID in s_p_ID_tuples:
                note_alignments.append({'label': 'match', 
                                        "score_id": score_note_array_pitch["id"][s_ID], 
                                        "performance_id": performance_note_array_pitch["id"][p_ID]})
                used_score_note_ids.add(score_note_array_pitch["id"][s_ID])
                used_performance_note_ids.add(performance_note_array_pitch["id"][p_ID])
                perf_id_from_score_id[score_note_array_pitch["id"][s_ID]].append( performance_note_array_pitch["id"][p_ID])

        
        # add unmatched notes
        used_score_mask = list()
        for score_note in score_note_array_full:
            if score_note["id"] not in used_score_note_ids:
                note_alignments.append({'label': 'deletion', 'score_id': score_note["id"]})
                used_score_mask.append(True)
            else:
                used_score_mask.append(False)
        
        used_pid_mask = list()
        for performance_note in performance_note_array:
            if performance_note["id"] not in used_performance_note_ids:
                note_alignments.append({'label': 'insertion', 'performance_id': performance_note["id"]})
                used_pid_mask.append(True)
            else:
                used_pid_mask.append(False)
                


        # add ornaments
        insertions = performance_note_array[np.array(used_pid_mask)]
        deletions = score_note_array_full[np.array(used_score_mask)]

        for ornament in score_note_array_ornaments:
            if len(deletions[deletions["id"] == ornament["id"]]) == 0:
                possible_ornament_notes = list()
                if len(perf_id_from_score_id[ornament["id"]]) > 0:
                    p_id = perf_id_from_score_id[ornament["id"]][0]
                    note_alignments.remove({'label': 'match', 
                                                "score_id": ornament["id"], 
                                                "performance_id":  p_id})
                    note_alignments.append({'label': 'insertion', 'performance_id': p_id})
                    possible_ornament_notes = [performance_note_array[performance_note_array["id"] == p_id]]
                        
                ornament_start = score_to_perf_map(ornament["onset_beat"] )
                ornament_end = score_to_perf_map(ornament["onset_beat"] + ornament["duration_beat"])
                ornament_pitch = ornament["pitch"]
                # find sequence of notes that could belong to the ornament
                
                for p in np.arange(ornament_pitch-2, ornament_pitch+3, 1):
                    possible_ornament_notes.append(na_within(insertions, 
                            field="onset_sec", 
                            pitch=p, 
                            lower_bound=ornament_start-0.25,
                            upper_bound=ornament_end,
                            ordered_by_field=True))
                    
                possible_ornament_notes = np.concatenate(possible_ornament_notes)   
                possible_ornament_notes = possible_ornament_notes[possible_ornament_notes["onset_sec"].argsort()]
                possible_ornament_notes_pitch = possible_ornament_notes[possible_ornament_notes["pitch"] == pitch]
                if len(possible_ornament_notes_pitch) == 0 and len(possible_ornament_notes) > 0:
                    note_alignments.append({'label': 'match', 
                                            "score_id": ornament["id"], 
                                            "performance_id":  possible_ornament_notes[0]["id"]})
                elif len(possible_ornament_notes_pitch) > 0:
                    note_alignments.append({'label': 'match', 
                                            "score_id": ornament["id"], 
                                            "performance_id":  possible_ornament_notes_pitch[0]["id"]})

                # for p in possible_ornament_notes[1:]:
                #     pass
                    # note_alignments.append({'label': 'ornament', 
                    #                         "score_id": ornament["id"], 
                    #                         'performance_id': p["id"]})


            

        return note_alignments

################################### FULL MODEL MATCHERS ###################################


class PianoRollSequentialMatcher(object):
    """
    A matcher that takes a score and a performance 
    as well as a tuples of coarse alignment times
    and returns a list of note alignments.
    """
    def __init__(self,
                 note_matcher=DTW,
                 matcher_kwargs=dict(metric="euclidean"),
                 node_cutter=cut_note_arrays,
                 node_mender=mend_note_alignments,
                 symbolic_note_matcher=SequenceAugmentedGreedyMatcher(),
                 greedy_symbolic_note_matcher=SimplestGreedyMatcher(),
                 alignment_type="dtw",
                 SCORE_FINE_NODE_LENGTH=0.25,
                 s_time_div=16,
                 p_time_div=16,
                 sfuzziness=0.5,
                 pfuzziness=0.5,
                 window_size=1,
                 pfuzziness_relative_to_tempo=True,
                 shift_onsets=False,
                 cap_combinations=None):

        self.note_matcher = note_matcher(**matcher_kwargs)
        self.symbolic_note_matcher = symbolic_note_matcher
        self.node_cutter = node_cutter
        self.node_mender = node_mender
        self.greedy_symbolic_note_matcher = greedy_symbolic_note_matcher
        self.alignment_type = alignment_type
        self.SCORE_FINE_NODE_LENGTH = SCORE_FINE_NODE_LENGTH
        self.s_time_div = s_time_div
        self.p_time_div = p_time_div
        self.sfuzziness = sfuzziness
        self.pfuzziness = pfuzziness
        self.window_size = window_size
        self.pfuzziness_relative_to_tempo = pfuzziness_relative_to_tempo
        self.shift_onsets = shift_onsets
        self.cap_combinations = cap_combinations

    def __call__(self, score_note_array,
                 performance_note_array, 
                 alignment_times):
        
        # cut arrays to windows
        score_note_arrays, performance_note_arrays = self.node_cutter(
            performance_note_array,
            score_note_array,
            np.array(alignment_times),
            sfuzziness=self.sfuzziness, 
            pfuzziness=self.pfuzziness,
            window_size=self.window_size,
            pfuzziness_relative_to_tempo=self.pfuzziness_relative_to_tempo)

        # compute windowed alignments
        note_alignments = []
        dtw_al = []

        for window_id in range(len(score_note_arrays)):  
            if self.alignment_type == "greedy":
                alignment = self.greedy_symbolic_note_matcher(
                    score_note_arrays[window_id],
                    performance_note_arrays[window_id])
                note_alignments.append(alignment)
            else:
                # _____________ fine alignment ____________
                if self.alignment_type == "dtw":
                    if score_note_arrays[window_id].shape[0] == 0 or \
                        performance_note_arrays[window_id].shape[0] == 0:
                        # for empty arrays fall back to linear
                        dtw_alignment_times = np.array(alignment_times)[
                                window_id:window_id+2, :]

                    else:    
                        dtw_alignment_times = alignment_times_from_dtw(
                            score_note_arrays[window_id],
                            performance_note_arrays[window_id],
                            matcher=self.note_matcher,
                            SCORE_FINE_NODE_LENGTH=self.SCORE_FINE_NODE_LENGTH,
                            s_time_div=self.s_time_div,
                            p_time_div=self.p_time_div)
                else:
                    dtw_alignment_times = np.array(alignment_times)[
                        window_id:window_id+2, :]

                dtw_al.append(dtw_alignment_times)
               
                # distance augmented greedy align
                fine_local_alignment = self.symbolic_note_matcher(
                    score_note_arrays[window_id],
                    performance_note_arrays[window_id],
                    dtw_alignment_times,
                    shift=self.shift_onsets,
                    cap_combinations=self.cap_combinations)

                note_alignments.append(fine_local_alignment)

                

        # MEND windows to global alignment
        global_alignment, score_alignment, \
            performance_alignment = self.node_mender(note_alignments, 
                                                    performance_note_array,
                                                    score_note_array, 
                                                    node_times=np.array(alignment_times),
                                                    symbolic_note_matcher= self.symbolic_note_matcher,
                                                    max_traversal_depth=1500)

        return global_alignment


class PianoRollNoNodeMatcher(object):
    def __init__(self,
                 note_matcher=DTW,
                 matcher_kwargs=dict(metric="euclidean"),#"cosine"),
                 node_cutter=cut_note_arrays,
                 node_mender=mend_note_alignments,
                 symbolic_note_matcher=SequenceAugmentedGreedyMatcher(),
                 greedy_symbolic_note_matcher=SimplestGreedyMatcher(),
                 alignment_type="dtw",
                 SCORE_FINE_NODE_LENGTH=0.25,
                 s_time_div=16,
                 p_time_div=16,
                 sfuzziness=4.0,#0.5,
                 pfuzziness=4.0,#0.5,
                 window_size=1,
                 pfuzziness_relative_to_tempo=True,
                 shift_onsets=False,
                 cap_combinations=100):

        self.note_matcher = note_matcher(**matcher_kwargs)
        self.symbolic_note_matcher = symbolic_note_matcher
        self.node_cutter = node_cutter
        self.node_mender = node_mender
        self.greedy_symbolic_note_matcher = greedy_symbolic_note_matcher
        self.alignment_type = alignment_type
        self.SCORE_FINE_NODE_LENGTH = SCORE_FINE_NODE_LENGTH
        self.s_time_div = s_time_div
        self.p_time_div = p_time_div
        self.sfuzziness = sfuzziness
        self.pfuzziness = pfuzziness
        self.window_size = window_size
        self.pfuzziness_relative_to_tempo = pfuzziness_relative_to_tempo
        self.shift_onsets = shift_onsets
        self.cap_combinations = cap_combinations

    def __call__(self, score_note_array,
                 performance_note_array,
                 verbose_time=False):
        
        t1 = time.time()
        # start with DTW
        dtw_alignment_times_init = alignment_times_from_dtw(
                            score_note_array,
                            performance_note_array,
                            matcher=self.note_matcher,
                            SCORE_FINE_NODE_LENGTH=4.0,
                            s_time_div=self.s_time_div,
                            p_time_div=self.p_time_div
                            )
        # cut arrays to windows
        t11 = time.time()
        if verbose_time:
            print(format(t11-t1, ".3f"), "sec : Initial coarse DTW pass")
        score_note_arrays, performance_note_arrays = self.node_cutter(
            performance_note_array,
            score_note_array,
            np.array(dtw_alignment_times_init),
            sfuzziness=self.sfuzziness, 
            pfuzziness=self.pfuzziness,
            window_size=self.window_size,
            pfuzziness_relative_to_tempo=self.pfuzziness_relative_to_tempo)

        # compute windowed alignments
        note_alignments = []
        dtw_al = []

        t2 = time.time()
        if verbose_time:
            print(format(t2-t11, ".3f"), "sec : Cutting")
            
        for window_id in range(len(score_note_arrays)):
            if self.alignment_type == "greedy":
                alignment = self.greedy_symbolic_note_matcher(
                    score_note_arrays[window_id],
                    performance_note_arrays[window_id])
                note_alignments.append(alignment)
            else:
                # _____________ fine alignment ____________
                if self.alignment_type == "dtw":
                    if score_note_arrays[window_id].shape[0] == 0 or \
                        performance_note_arrays[window_id].shape[0] == 0:
                        # for empty arrays fall back to linear
                        dtw_alignment_times = np.array(dtw_alignment_times_init)[
                                window_id:window_id+2, :]

                    else:    
                        dtw_alignment_times = alignment_times_from_dtw(
                            score_note_arrays[window_id],
                            performance_note_arrays[window_id],
                            matcher=self.note_matcher,
                            SCORE_FINE_NODE_LENGTH=self.SCORE_FINE_NODE_LENGTH,
                            s_time_div=self.s_time_div,
                            p_time_div=self.p_time_div)
                else:
                    dtw_alignment_times = np.array(dtw_alignment_times_init)[
                        window_id:window_id+2, :]

                dtw_al.append(dtw_alignment_times)
                
                # distance augmented greedy align
                fine_local_alignment = self.symbolic_note_matcher(
                    score_note_arrays[window_id],
                    performance_note_arrays[window_id],
                    dtw_alignment_times,
                    shift=self.shift_onsets,
                    cap_combinations=self.cap_combinations)

                note_alignments.append(fine_local_alignment)
        t41 = time.time()
        if verbose_time:
            print(format(t41-t2, ".3f"), "sec : Fine-grained DTW passes, symbolic matching")

        
        # MEND windows to global alignment
        global_alignment, score_alignment, \
            performance_alignment = self.node_mender(note_alignments, 
                                                    performance_note_array,
                                                    score_note_array, 
                                                    node_times=np.array(dtw_alignment_times_init),
                                                    symbolic_note_matcher= self.symbolic_note_matcher,
                                                    max_traversal_depth=150)
        t5 = time.time()
        if verbose_time:
            print(format(t5-t41, ".3f"), "sec : Mending")

        return global_alignment

# alias
AutomaticNoteMatcher = PianoRollNoNodeMatcher

# alias
AnchorPointNoteMatcher = PianoRollSequentialMatcher


class ChordEncodingMatcher(object):
    def __init__(self,
                 note_matcher=DTW,
                 matcher_kwargs=dict(metric=element_of_metric,
                                     cdist_local=True),
                 symbolic_note_matcher=OnsetGreedyMatcher(),
                 dtw_window_size=6):

        self.note_matcher = note_matcher(**matcher_kwargs)
        self.symbolic_note_matcher = symbolic_note_matcher
        self.dtw_window_size = dtw_window_size

    def __call__(self, score_note_array,
                 performance_note_array):
        
        # create encodings
        score_note_per_ons_encoding = note_per_ons_encoding(score_note_array)
        
        # match by onset
        matcher = DTW(metric = element_of_metric, cdist_local = True)
        
        _, onset_alignment_path = matcher(performance_note_array["pitch"], 
                                          score_note_per_ons_encoding,  return_path=True)


        # match by note
        global_alignment = self.symbolic_note_matcher(
                                                    score_note_array, 
                                                    performance_note_array,
                                                    onset_alignment_path
                                                    )
                                                
        return global_alignment, onset_alignment_path
    

################################### ONLINE MATCHERS ###################################


class OnlineMatcherOG(object):
    def __init__(self,
                 score_note_array_full,
                 score_note_array_no_grace,
                 score_note_array_grace,
                 score_note_array_ornament
                 ):
        self.score_note_array_full = score_note_array_full
        self.score_note_array_no_grace = score_note_array_no_grace
        self.score_note_array_grace = score_note_array_grace
        self.score_note_array_ornament = score_note_array_ornament
        self.first_p_onset = None
        self.tempo_model = None
        
        self._prev_performance_onset = None
        self._prev_score_onset = None
        self._snote_aligned = set()
        self._pnote_aligned = set()
        self.alignment = []
        self.note_alignments = []
        self.prepare_score()

    def prepare_score(self):
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            self.score_by_pitch[pitch] += list(self.score_note_array_full[
                self.score_note_array_full["pitch"] == pitch])
        
        self.number_of_grace_notes_at_onset = defaultdict(int)
        for s_note in self.score_note_array_grace:
            self.number_of_grace_notes_at_onset[s_note["onset_beat"]] += 1

        self._prev_score_onset = self.score_note_array_full["onset_beat"][0]
        self._unique_score_onsets = np.unique(self.score_note_array_full["onset_beat"])
        self.onset_range_at_onset = dict()
        for s_id, s_onset in enumerate(self._unique_score_onsets[1:-1]):
            self.onset_range_at_onset[s_onset] = [self._unique_score_onsets[s_id], self._unique_score_onsets[s_id+2]]
        self.onset_range_at_onset[self._unique_score_onsets[0]] = [self._unique_score_onsets[0], self._unique_score_onsets[1]]
        self.onset_range_at_onset[self._unique_score_onsets[-1]] = [self._unique_score_onsets[-2], self._unique_score_onsets[-1]]
        # self.onset_range_at_onset[self._unique_score_onsets[-2]] = [self._unique_score_onsets[-3], self._unique_score_onsets[-1]]


        self.aligned_notes_at_onset = defaultdict(list)


    def prepare_performance(self, first_onset, func = None):
        if func is None:
            self.tempo_model = TempoModel(init_beat_period = 0.5,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 1)
        else:
            self.tempo_model = DummyTempoModel(init_beat_period = 0.5,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 1,
                                    func = func)

    def offline(self, performance_note_array, func = None):

        self.prepare_performance(performance_note_array[0]["onset_sec"], func)

        for p_note in performance_note_array[:]:
            self.online(p_note)

        for s_ID, p_ID in self.alignment:
                self.note_alignments.append({'label': 'match', 
                                        "score_id": s_ID, 
                                        "performance_id": p_ID})
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                self.note_alignments.append({'label': 'deletion', 'score_id': score_note["id"]})
        
        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                self.note_alignments.append({'label': 'insertion', 'performance_id': performance_note["id"]})

        return self.note_alignments



    def online(self, performance_note, debug=False):
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        possible_score_onsets = self.onset_range_at_onset[self._prev_score_onset]
        if debug:
            print("--------------------")
            print("Performance note: ", p_id, p_onset, p_pitch)
            print("Current onset: ", self._prev_score_onset)
            print("Possible score onsets: ", possible_score_onsets)

        possible_score_notes = self.score_by_pitch[p_pitch]
        # possible_score_notes = sorted(possible_score_notes, key=lambda x: x["onset_beat"])
        possible_score_notes = [x for x in possible_score_notes \
                                if x["id"] not in self._snote_aligned and \
                                    x["onset_beat"] >= possible_score_onsets[0] and \
                                    x["onset_beat"] <= possible_score_onsets[1] ]


        if len(possible_score_notes) > 0:
            possible_note_onsets_mapped = [self.tempo_model.predict(x["onset_beat"]) for x in possible_score_notes]
            possible_note_onsets_dist = np.abs(np.array(possible_note_onsets_mapped) - p_onset)
            lowest_dist_idx = np.argmin(possible_note_onsets_dist)
            best_note = possible_score_notes[lowest_dist_idx]
            lowest_dist = possible_note_onsets_dist[lowest_dist_idx]

            if debug:
                print("Possible notes onsets: ", [x["onset_beat"] for x in possible_score_notes])
                print("Possible notes mapped: ", possible_note_onsets_mapped)
                print("Best note: ", best_note, lowest_dist)
            if best_note["is_grace"]:
                number_of_local_grace_notes = self.number_of_grace_notes_at_onset[best_note["onset_beat"]]
                if lowest_dist < number_of_local_grace_notes*0.2:
                    self._snote_aligned.add(best_note["id"])
                    self._pnote_aligned.add(p_id)
                    self.alignment.append((best_note["id"], p_id))
            else:
                previous_aligned_p_onsets = self.aligned_notes_at_onset[best_note["onset_beat"]]
                close_enough = True
                if len(previous_aligned_p_onsets) > 0:
                    close_enough = np.abs(p_onset - np.median(previous_aligned_p_onsets)) < 0.25
                if close_enough:
                    self._snote_aligned.add(best_note["id"])
                    self._pnote_aligned.add(p_id)
                    self.alignment.append((best_note["id"], p_id))
                    self.aligned_notes_at_onset[best_note["onset_beat"]].append(p_onset)
                    if best_note["onset_beat"] > self._prev_score_onset:
                        self.tempo_model.update(p_onset, best_note["onset_beat"])
                        self._prev_score_onset = best_note["onset_beat"]
                


    def __call__(self):
                                    
        return None
    

class OnlineMatcher(object):
    def __init__(self,
                 score_note_array_full,
                 score_note_array_no_grace,
                 score_note_array_grace,
                 score_note_array_ornament
                 ):
        self.score_note_array_full = score_note_array_full
        self.score_note_array_no_grace = score_note_array_no_grace
        self.score_note_array_grace = score_note_array_grace
        self.score_note_array_ornament = score_note_array_ornament
        self.first_p_onset = None
        self.tempo_model = None
        
        self._prev_performance_onset = None
        self._prev_score_onset = None
        self._snote_aligned = set()
        self._pnote_aligned = set()
        self.alignment = []
        self.note_alignments = []
        self.prepare_score()

    def prepare_score(self):
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            self.score_by_pitch[pitch] += list(self.score_note_array_full[
                self.score_note_array_full["pitch"] == pitch])
        
        self.number_of_grace_notes_at_onset = defaultdict(int)
        for s_note in self.score_note_array_grace:
            self.number_of_grace_notes_at_onset[s_note["onset_beat"]] += 1

        self._prev_score_onset = self.score_note_array_full["onset_beat"][0]
        self._unique_score_onsets = np.unique(self.score_note_array_full["onset_beat"])
        self.onset_range_at_onset = dict()
        for s_id, s_onset in enumerate(self._unique_score_onsets[1:-1]):
            self.onset_range_at_onset[s_onset] = [self._unique_score_onsets[s_id], self._unique_score_onsets[s_id+2]]
        self.onset_range_at_onset[self._unique_score_onsets[0]] = [self._unique_score_onsets[0], self._unique_score_onsets[1]]
        self.onset_range_at_onset[self._unique_score_onsets[-1]] = [self._unique_score_onsets[-2], self._unique_score_onsets[-1]]
        # self.onset_range_at_onset[self._unique_score_onsets[-2]] = [self._unique_score_onsets[-3], self._unique_score_onsets[-1]]


        self.aligned_notes_at_onset = defaultdict(list)


    def prepare_performance(self, first_onset, func = None):
        if func is None:
            self.tempo_model = TempoModel(init_beat_period = 0.5,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 3)
        else:
            self.tempo_model = DummyTempoModel(init_beat_period = 0.5,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 3,
                                    func = func)

    def offline(self, performance_note_array, func = None):

        self.prepare_performance(performance_note_array[0]["onset_sec"], func)

        for p_note in performance_note_array[:]:
            self.online(p_note)

        for s_ID, p_ID in self.alignment:
                self.note_alignments.append({'label': 'match', 
                                        "score_id": s_ID, 
                                        "performance_id": p_ID})
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                self.note_alignments.append({'label': 'deletion', 'score_id': score_note["id"]})
        
        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                self.note_alignments.append({'label': 'insertion', 'performance_id': performance_note["id"]})

        return self.note_alignments



    def online(self, performance_note, debug=False):
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        possible_score_onsets = self.onset_range_at_onset[self._prev_score_onset]
        if debug:
            print("--------------------")
            print("Performance note: ", p_id, p_onset, p_pitch)
            print("Current onset: ", self._prev_score_onset)
            print("Possible score onsets: ", possible_score_onsets)

        possible_score_notes = self.score_by_pitch[p_pitch]
        # possible_score_notes = sorted(possible_score_notes, key=lambda x: x["onset_beat"])
        possible_score_notes = [x for x in possible_score_notes \
                                if x["id"] not in self._snote_aligned and \
                                    x["onset_beat"] >= possible_score_onsets[0] and \
                                    x["onset_beat"] <= possible_score_onsets[1] ]

        # import pdb; pdb.set_trace()
        if len(possible_score_notes) > 0:
            possible_note_onsets_mapped = [self.tempo_model.predict_ratio(x["onset_beat"], p_onset) for x in possible_score_notes]
            possible_note_onsets_dist = np.abs(np.array(possible_note_onsets_mapped) )#- p_onset)
            lowest_dist_idx = np.argmin(possible_note_onsets_dist)
            best_note = possible_score_notes[lowest_dist_idx]
            lowest_dist = possible_note_onsets_dist[lowest_dist_idx]

            if debug:
                print("Possible notes onsets: ", [x["onset_beat"] for x in possible_score_notes])
                print("Possible notes mapped: ", possible_note_onsets_mapped)
                print("Best note: ", best_note, lowest_dist)
            if best_note["is_grace"]:
                number_of_local_grace_notes = self.number_of_grace_notes_at_onset[best_note["onset_beat"]]
                if lowest_dist < number_of_local_grace_notes*0.2:
                    self._snote_aligned.add(best_note["id"])
                    self._pnote_aligned.add(p_id)
                    self.alignment.append((best_note["id"], p_id))
            else:
                previous_aligned_p_onsets = self.aligned_notes_at_onset[best_note["onset_beat"]]
                close_enough = True
                if len(previous_aligned_p_onsets) > 0:
                    close_enough = np.abs(p_onset - np.median(previous_aligned_p_onsets)) < 0.25
                if close_enough:
                    self._snote_aligned.add(best_note["id"])
                    self._pnote_aligned.add(p_id)
                    self.alignment.append((best_note["id"], p_id))
                    self.aligned_notes_at_onset[best_note["onset_beat"]].append(p_onset)
                    if best_note["onset_beat"] > self._prev_score_onset:
                        self.tempo_model.update(p_onset, best_note["onset_beat"])
                        self._prev_score_onset = best_note["onset_beat"]

            # if p_id == "n306":
            #     import pdb; pdb.set_trace()
            
        else:
            # if the neighboring pitches are different, extend the search
            possible_score_notes = self.score_by_pitch[p_pitch]
            possible_score_notes = [x for x in possible_score_notes \
                                    if x["id"] not in self._snote_aligned and \
                                    x["onset_beat"] > possible_score_onsets[1] and \
                                    x["onset_beat"] <= self._prev_score_onset + 10.0 ]
            
            
            if len(possible_score_notes) > 0:
                possible_note_onsets_mapped = [self.tempo_model.predict_ratio(x["onset_beat"], p_onset) for x in possible_score_notes]
                possible_note_onsets_dist = np.abs(np.array(possible_note_onsets_mapped) )#- p_onset)
                possible_score_notes_id = [x["id"] for x in possible_score_notes]
                lowest_dist_idx = np.argmin(possible_note_onsets_dist)
                best_note = possible_score_notes[lowest_dist_idx]
                lowest_dist = possible_note_onsets_dist[lowest_dist_idx]
                # if p_id == "n42":
                #     import pdb; pdb.set_trace()
                if lowest_dist < 0.2:

                    if debug:
                        print("Second Possible notes onsets: ", [x["onset_beat"] for x in possible_score_notes])
                        print("SecondPossible notes mapped: ", possible_note_onsets_mapped)
                        print("Second  note: ", best_note, lowest_dist)
                    if best_note["is_grace"]:
                        number_of_local_grace_notes = self.number_of_grace_notes_at_onset[best_note["onset_beat"]]
                        if lowest_dist < number_of_local_grace_notes*0.1:
                            self._snote_aligned.add(best_note["id"])
                            self._pnote_aligned.add(p_id)
                            self.alignment.append((best_note["id"], p_id))
                    else:
                        previous_aligned_p_onsets = self.aligned_notes_at_onset[best_note["onset_beat"]]
                        close_enough = True
                        if len(previous_aligned_p_onsets) > 0:
                            close_enough = False
                        if close_enough:
                            self._snote_aligned.add(best_note["id"])
                            self._pnote_aligned.add(p_id)
                            self.alignment.append((best_note["id"], p_id))
                            self.aligned_notes_at_onset[best_note["onset_beat"]].append(p_onset)
                            if best_note["onset_beat"] > self._prev_score_onset:
                                self.tempo_model.update(p_onset, best_note["onset_beat"])
                                self._prev_score_onset = best_note["onset_beat"]
                elif lowest_dist < 1.0:

                    if debug:
                        print("Second Possible notes onsets: ", [x["onset_beat"] for x in possible_score_notes])
                        print("SecondPossible notes mapped: ", possible_note_onsets_mapped)
                        print("Second  note: ", best_note, lowest_dist)
                    if best_note["is_grace"]:
                        number_of_local_grace_notes = self.number_of_grace_notes_at_onset[best_note["onset_beat"]]
                        if lowest_dist < number_of_local_grace_notes*0.1:
                            self._snote_aligned.add(best_note["id"])
                            self._pnote_aligned.add(p_id)
                            self.alignment.append((best_note["id"], p_id))
                    else:
                        previous_aligned_p_onsets = self.aligned_notes_at_onset[best_note["onset_beat"]]
                        close_enough = True
                        if len(previous_aligned_p_onsets) > 0:
                            close_enough = False
                        if close_enough:
                            self._snote_aligned.add(best_note["id"])
                            self._pnote_aligned.add(p_id)
                            self.alignment.append((best_note["id"], p_id))
                            self.aligned_notes_at_onset[best_note["onset_beat"]].append(p_onset)
                            

                # else:
                #     print("No match found for ", p_id, possible_score_notes_id, possible_note_onsets_dist)
                    


    def __call__(self):

        return None
    

class TempoModel(object):
    """
    Base class for synchronization models

    Attributes
    ----------
    """
    def __init__(
        self,
        init_beat_period = 0.5,
        init_score_onset = 0,
        init_perf_onset = 0,
        lookback = 1    
        ):
        self.lookback = lookback
        self.beat_period = init_beat_period
        self.prev_score_onsets = [init_score_onset - 2 * lookback]
        self.prev_perf_onsets = [init_perf_onset - 2 * lookback * self.beat_period]
        self.prev_perf_onsets_at_score_onsets = defaultdict(list)
        self.prev_perf_onsets_at_score_onsets[self.prev_score_onsets[-1]].append(self.prev_perf_onsets[-1])
        self.est_onset = None
        self.score_perf_map = None
        # Count how many times has the tempo model been called
        self.counter = 0
        self.update(init_perf_onset - lookback * self.beat_period, init_score_onset - lookback)

    def predict(
        self,
        score_onset
        ):
        self.est_onset = self.score_perf_map(score_onset - (self.lookback+1)) + \
            (self.lookback+1) * self.beat_period 
        return self.est_onset

    
    def predict_ratio(
        self,
        score_onset,
        perf_onset
        ):
        self.est_onset = self.score_perf_map(score_onset - (self.lookback+1)) + \
            (self.lookback+1) * self.beat_period 
        error = perf_onset - self.est_onset
        offset_score =  score_onset  - self.prev_score_onsets[-1] 
        if offset_score > 0.0:
            return error/(offset_score * self.beat_period)
        else:
            return error

    def update(
        self,
        performed_onset,
        score_onset
        ):

        self.prev_perf_onsets_at_score_onsets[score_onset].append(performed_onset)
        if score_onset == self.prev_score_onsets[-1]:
            #     self.prev_perf_onsets[-1] = 4/5 * self.prev_perf_onsets[-1] + 1/5* performed_onset
            self.prev_perf_onsets[-1] = np.median(self.prev_perf_onsets_at_score_onsets[score_onset])
        else:
            self.prev_score_onsets.append(score_onset)
            self.prev_perf_onsets.append(performed_onset)
            
        self.score_perf_map = interp1d(self.prev_score_onsets[-100:], 
                                       self.prev_perf_onsets[-100:], 
                                       fill_value="extrapolate")
        self.beat_period = np.clip((self.score_perf_map(score_onset) - \
            self.score_perf_map(score_onset - self.lookback))/self.lookback, 0.1, 10.0)
        self.counter += 1


class DummyTempoModel(object):


    """
    Base class for synchronization models

    Attributes
    ----------
    """
    def __init__(
        self,
        init_beat_period = 0.5,
        init_score_onset = 0,
        init_perf_onset = 0,
        lookback = 1,
        func = None
    ):
        
        self.lookback = lookback
        self.beat_period = init_beat_period
        self.score_perf_map = func
        # Count how many times has the tempo model been called
        self.counter = 0

    def predict(
        self,
        score_onset
        ):
        self.est_onset = self.score_perf_map(score_onset)
        return self.est_onset
    
    # def predict_ratio(
    #     self,
    #     score_onset,
    #     perf_onset
    #     ):
    #     self.est_onset = self.score_perf_map(score_onset - self.lookback) + \
    #         self.lookback * self.beat_period 
    #     error = perf_onset - self.est_onset
    #     offset_score =  score_onset  - self.prev_score_onsets[-1] 
    #     if offset_score > 0.0:
    #         return error/(offset_score * self.beat_period)
    #     else:
    #         return error
    def update(
        self,
        performed_onset,
        score_onset
        ):
        self.counter += 1



class OnlineTransformerMatcher(object):
    def __init__(self,
                 score_note_array_full,
                 score_note_array_no_grace,
                 score_note_array_grace,
                 score_note_array_ornament
                 ):
        self.score_note_array_full = np.sort(score_note_array_full, order="onset_beat")
        self.score_note_array_no_grace = np.sort(score_note_array_no_grace, order="onset_beat")
        self.score_note_array_grace = np.sort(score_note_array_grace, order="onset_beat")
        self.score_note_array_ornament = np.sort(score_note_array_ornament, order="onset_beat")
        self.first_p_onset = None
        self.tempo_model = None
        
        self._prev_performance_notes = list()
        self._prev_score_onset = None
        # self._prev_score_onset = self.score_note_array_full[0]["onset_beat"]
        self._snote_aligned = set()
        self._pnote_aligned = set()
        self._pnote_aligned_pitch = list()
        self.alignment = []
        self.note_alignments = []
        self.time_since_nn_update = 0
        self.prepare_score()
        self.prepare_model()

    def prepare_score(self):
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            # self.score_by_pitch[pitch] += list(self.score_note_array_full[
            #     self.score_note_array_full["pitch"] == pitch])
            self.score_by_pitch[pitch] = self.score_note_array_full[self.score_note_array_full["pitch"] == pitch]
        
        self.number_of_grace_notes_at_onset = defaultdict(int)
        for s_note in self.score_note_array_grace:
            self.number_of_grace_notes_at_onset[s_note["onset_beat"]] += 1

        self._prev_score_onset = self.score_note_array_full["onset_beat"][0]
        self._unique_score_onsets = np.unique(self.score_note_array_full["onset_beat"])

        # onset range for forward backward view
        self.onset_range_at_onset = dict()
        for s_id, s_onset in enumerate(self._unique_score_onsets[1:-1]):
            self.onset_range_at_onset[s_onset] = [self._unique_score_onsets[s_id], self._unique_score_onsets[s_id+2]]
        self.onset_range_at_onset[self._unique_score_onsets[0]] = [self._unique_score_onsets[0], self._unique_score_onsets[1]]
        self.onset_range_at_onset[self._unique_score_onsets[-1]] = [self._unique_score_onsets[-2], self._unique_score_onsets[-1]]

        # set of pitches at onset / map from onset to idx in unique onsets
        self.pitches_at_onset_by_id = list()
        self.id_by_onset = dict()

        for i, onset in enumerate(self._unique_score_onsets):
            self.pitches_at_onset_by_id.append(
                set(self.score_note_array_no_grace[
                    self.score_note_array_no_grace["onset_beat"] == onset
                    ]["pitch"])
                )
            self.id_by_onset[onset] = i

        # aligned notes at each onset
        self.aligned_notes_at_onset = defaultdict(list)

    def prepare_performance(self, first_onset, func = None):
        if func is None:
            self.tempo_model = TempoModel(init_beat_period = 0.5,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 3)
        else:
            self.tempo_model = DummyTempoModel(init_beat_period = 0.5,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 3,
                                    func = func)

    def prepare_model(self):
        self.model = AlignmentTransformer(
            token_number = 91,# 21 - 108 + 2 for padding (start_score, end) + 1 for non_pitch
            dim_model = 64,
            dim_class = 2,
            num_heads = 8,
            num_decoder_layers = 6,
            dropout_p = 0.1
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(r"C:\Users\silva\Documents\repos\ZDUDLES\alignment_transformer\alignment_transformer_epoch_positional_70.pt", 
                                map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def offline(self, performance_note_array, func = None):

        self.prepare_performance(performance_note_array[0]["onset_sec"], func)

        for p_note in performance_note_array[:]:
            self.online(p_note)

        for s_ID, p_ID in self.alignment:
                self.note_alignments.append({'label': 'match', 
                                        "score_id": s_ID, 
                                        "performance_id": p_ID})
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                self.note_alignments.append({'label': 'deletion', 'score_id': score_note["id"]})
        
        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                self.note_alignments.append({'label': 'insertion', 'performance_id': performance_note["id"]})

        return self.note_alignments

    def online(self, performance_note, debug=False):
        self.time_since_nn_update += 1
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        self._prev_performance_notes.append(p_pitch)

        possible_score_notes = self.score_by_pitch[p_pitch]

        # align greedily if open note at current oonset
        if p_pitch in self.pitches_at_onset_by_id[self.id_by_onset[self._prev_score_onset]]:
            best_notes = na_within(possible_score_notes, "onset_beat", 
                                    self._prev_score_onset, self._prev_score_onset,
                                    exclusion_ids=self._snote_aligned)
            if len(best_notes) > 0:
                best_note = best_notes[0]
                self.add_note_alignment(p_id, best_note["id"], p_onset, best_note["onset_beat"])
                return
        
        # align with the help of the tempo function
        possible_score_onsets = self.onset_range_at_onset[self._prev_score_onset]
        possible_score_notes = na_within(possible_score_notes, "onset_beat", 
                                         possible_score_onsets[0], possible_score_onsets[1],
                                         exclusion_ids=self._snote_aligned)

        if len(possible_score_notes) > 0:
            possible_note_onsets_mapped = [self.tempo_model.predict_ratio(x["onset_beat"], p_onset) for x in possible_score_notes]
            possible_note_onsets_dist = np.abs(np.array(possible_note_onsets_mapped))
            lowest_dist_idx = np.argmin(possible_note_onsets_dist)
            best_note = possible_score_notes[lowest_dist_idx]
            lowest_dist = possible_note_onsets_dist[lowest_dist_idx]
            # if p_id == "n491":
            #     print("Possible notes onsets: ", [x["onset_beat"] for x in possible_score_notes])
            #     print("Possible notes mapped: ", possible_note_onsets_mapped)
            #     print("Best note: ", best_note, lowest_dist)
            #     import pdb; pdb.set_trace()
            if debug:
                print("Possible notes onsets: ", [x["onset_beat"] for x in possible_score_notes])
                print("Possible notes mapped: ", possible_note_onsets_mapped)
                print("Best note: ", best_note, lowest_dist)
            

                # HEURISTIC: distance based on number of grace notes
                # number_of_local_grace_notes = self.number_of_grace_notes_at_onset[best_note["onset_beat"]]
                # if lowest_dist < number_of_local_grace_notes*0.2:
                #     self._snote_aligned.add(best_note["id"])
                #     self._pnote_aligned.add(p_id)
                #     self.alignment.append((best_note["id"], p_id))

                # HEURISTIC: don't stray too far from previously aligned
                # previous_aligned_p_onsets = self.aligned_notes_at_onset[best_note["onset_beat"]]
                # close_enough = True
                # if len(previous_aligned_p_onsets) > 0:
                #     close_enough = np.abs(p_onset - np.median(previous_aligned_p_onsets)) < 1.5
                # if close_enough:
            if best_note["is_grace"]:
                self.add_note_alignment(p_id, best_note["id"])
            else:
                self.add_note_alignment(p_id, best_note["id"], p_onset, best_note["onset_beat"])

        # only then do we use the neural network
        elif self.time_since_nn_update > 1:
            current_id = self.id_by_onset[self._prev_score_onset]
            s_slice = slice(np.max((current_id-7, 0)), current_id+9 )
            p_slice = slice(-8, None )
            score_seq = self.pitches_at_onset_by_id[s_slice]
            perf_seq = self._prev_performance_notes[p_slice]

            tokenized_score_seq =  tokenize(score_seq, perf_seq, dims = 7)
            out = self.model(torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device))
            pred_id = torch.argmax(torch.softmax(out.squeeze(1),dim=0)[:,1]).cpu().numpy()
            new_pred_id = pred_id - len(perf_seq) - 1 - (current_id - np.max((current_id-7, 0)))

            if debug: #or p_id in ["n"+str(x) for x in range(750, 780)]:
                print("predicted id ", pred_id, new_pred_id, p_id)
                print("predicted score onset, ",      self._unique_score_onsets[current_id + new_pred_id], self._prev_score_onset)
                import pdb; pdb.set_trace()
            if new_pred_id > -2 and new_pred_id < 10:#np.min((6,self.time_since_nn_update+2))	:
                # print("happens", p_id)
                pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
                possible_score_notes = self.score_by_pitch[p_pitch]
                possible_score_notes = sorted(possible_score_notes, key=lambda x: x["onset_beat"])
                possible_score_notes = [x for x in possible_score_notes \
                                        if x["id"] not in self._snote_aligned and \
                                            x["onset_beat"] >=pred_score_onset and \
                                            x["onset_beat"] <=pred_score_onset ]

                if len(possible_score_notes) > 0:
                    # print("alignment transformer: ", possible_score_notes)
                    possible_note_onsets_mapped = [self.tempo_model.predict_ratio(x["onset_beat"], p_onset) for x in possible_score_notes]
                    possible_note_onsets_dist = np.abs(np.array(possible_note_onsets_mapped) )#- p_onset)
                    lowest_dist_idx = np.argmin(possible_note_onsets_dist)
                    best_note = possible_score_notes[lowest_dist_idx]
                    self.time_since_nn_update = 0
                    if debug:
                        lowest_dist = possible_note_onsets_dist[lowest_dist_idx]
                        print("Possible notes onsets: ", [x["onset_beat"] for x in possible_score_notes])
                        print("Possible notes mapped: ", possible_note_onsets_mapped)
                        print("Best note: ", best_note, lowest_dist)
                    if best_note["is_grace"]:
                        self.add_note_alignment(p_id, best_note["id"])
                    else:
                        self.add_note_alignment(p_id, best_note["id"], p_onset, best_note["onset_beat"])
                            

    def add_note_alignment(self,
                           perf_id, score_id, 
                           perf_onset = None, score_onset = None
                           ):
        self.alignment.append((score_id, perf_id))
        self._snote_aligned.add(score_id)
        self._pnote_aligned.add(perf_id)
        if perf_onset is not None and score_onset is not None:
            self.aligned_notes_at_onset[score_onset].append(perf_onset)
            if score_onset >= self._prev_score_onset:
                self.tempo_model.update(perf_onset, score_onset)
                self._prev_score_onset = score_onset

    def __call__(self):

        return None
    


def perf_tokenizer(pitch, dims = 7):
    return np.ones((1,dims), dtype = int) * (pitch - 20)

def score_tokenizer(pitch_set, dims = 7):
    token = np.zeros((1,dims), dtype = int)
    for no, pitch in enumerate(list(pitch_set)):
        if pitch >= 21 and pitch <= 108 and no < dims:
            token[0,no] = pitch - 20
    return token

def perf_to_score_tokenizer(dims = 7):
    return np.ones((1,dims), dtype = int) *89

def end_tokenizer(dims = 7, end_dims=1):
    return np.ones((end_dims,dims), dtype = int) *90


def tokenize(score_segment, perf_segment, dims = 7):
    tokens = list()
    for perf_note in perf_segment:
        perf_token = perf_tokenizer(perf_note, dims)
        tokens.append(perf_token)
    tokens.append(perf_to_score_tokenizer(dims))
    for score_set in score_segment:
        score_token = score_tokenizer(score_set, dims)
        tokens.append(score_token)
    
    end_token = end_tokenizer(dims, 26 - len(tokens))
    tokens.append(end_token)

    return np.row_stack(tokens)
