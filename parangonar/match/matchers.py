#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains full note matcher classes.
"""
import numpy as np
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
    alignment
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
            

                #
                # print(s_onset, matching_pitches.shape)
            
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
            # if pitch == 73:
            #     print("local alignment", pitch, score_note_array_pitch.shape, performance_note_array_pitch.shape)

            #     print(score_note_array[20:35])
            #     print(score_note_array[20:35]["onset_beat"], score_to_perf_map(score_note_array[20:35]['onset_beat']))
            #     print(performance_note_array[20:35])
            #     print(unique_time_tuples)


            if performance_note_array_pitch.shape[0] > 1 and score_note_array_pitch.shape[0] > 1:
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

    # for p_no, p_note in enumerate(performance_note_array):
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
            # if pid == "n2957":
            #     print("here", s_pitch_used, s_pitch_available, s_onset)
            if s_pitch_available and not s_pitch_used:
                if not block_by_pitch_by_onset[s_onset][ppitch]:
                    # if s_onset == 631.0:
                    #     print("here", s_onset, p_note)
                    used_pitches_by_onset[s_onset].add(ppitch)
                    time_tuples_by_pitch[ppitch].append((s_onset, p_note['onset_sec']))
                    time_tuples_by_onset[s_onset].append(p_note['onset_sec'])
                    # time_tuples_by_pitch_id[ppitch].append((s_onset_no, p_no))
                    # time_tuples_by_onset_id[s_onset_no].append(p_no)
                    p_aligned.add(pid)


    # remove outliers
    for s_onset in time_tuples_by_onset.keys():
        sorted_times = np.sort(np.array(time_tuples_by_onset[s_onset]))
        
        if np.median(sorted_times) - sorted_times[0] > 0.1:
            # print("low ",sorted_times)
            sorted_times = np.delete(sorted_times, 0)
            
        elif np.median(sorted_times) - sorted_times[-1] < -0.1:
            # print("high ",sorted_times)
            sorted_times = np.delete(sorted_times, -1)
        time_tuples_by_onset[s_onset] = list(sorted_times)
    

    # make clean sequences
    onsets_with_performance_times = np.array(list(time_tuples_by_onset.keys()))
    current_s_onset_no = 0
    for s_onset_no in range(len(unique_onsets)):
        if s_onset_no > current_s_onset_no:
            s_onset = unique_onsets[s_onset_no]
            if all_pitch_repeat_by_onset[s_onset]:
                local_s_onset_no = s_onset_no
                s_onset_range = [unique_onsets[s_onset_no - 1]]
                while(all_pitch_repeat_by_onset[unique_onsets[local_s_onset_no]]):
                    s_onset_range.append(unique_onsets[local_s_onset_no])
                    local_s_onset_no += 1
                    if local_s_onset_no >= len(unique_onsets)-1:
                        break
                current_s_onset_no = local_s_onset_no - 1
                s_onset_range = np.array(s_onset_range)
                first_s_onset_in_range = s_onset_range[0]
                first_s_onset_out_of_range = unique_onsets[local_s_onset_no]
                first_s_onset_in_range_aligned = np.max(onsets_with_performance_times[onsets_with_performance_times<=first_s_onset_in_range])
                first_s_onset_out_of_range_aligned = np.min(onsets_with_performance_times[onsets_with_performance_times>=first_s_onset_out_of_range])

                # print(s_onset_range, s_onset_no, first_s_onset_out_of_range, time_tuples_by_onset[first_s_onset_in_range],time_tuples_by_onset[first_s_onset_out_of_range])
                try:
                    first_p_onset_in_range = np.min(time_tuples_by_onset[first_s_onset_in_range_aligned])   
                    
                    first_p_onset_out_of_range = np.min(time_tuples_by_onset[first_s_onset_out_of_range_aligned])   
                except:
                    print("error", s_onset_range, s_onset_no, first_s_onset_out_of_range, time_tuples_by_onset[first_s_onset_in_range],time_tuples_by_onset[first_s_onset_out_of_range])
                    break
                for pitch in pitches_by_onset[s_onset_range[0]]:  
                    pitch_mask = performance_note_array['pitch'] == pitch
                    higher_mask = performance_note_array['onset_sec'] >= first_p_onset_in_range
                    lower_mask = performance_note_array['onset_sec'] < first_p_onset_out_of_range
                    available_pp_notes = performance_note_array[np.all((pitch_mask, higher_mask, lower_mask), axis=0)]
                    if len(available_pp_notes) == len(s_onset_range):
                        for s_onset_local, p_onset_local in zip(s_onset_range, available_pp_notes):
                            time_tuples_by_pitch[pitch].append((s_onset_local, p_onset_local['onset_sec']))
                            time_tuples_by_onset[s_onset_local].append(p_onset_local['onset_sec'])
                        

       


    unique_time_tuples_by_onset = {s_onset : np.min(time_tuples_by_onset[s_onset]) for s_onset in time_tuples_by_onset.keys()}
    unique_time_tuples = np.array([(tup, unique_time_tuples_by_onset[tup]) for tup in unique_time_tuples_by_onset.keys()])  

    # unique_time_tuples_by_onset_id = {s_onset_no : np.min(time_tuples_by_onset_id[s_onset_no]) for s_onset_no in time_tuples_by_onset_id.keys()}
    # unique_time_tuples_id = np.array([(tup, unique_time_tuples_by_onset_id[tup]) for tup in unique_time_tuples_by_onset_id.keys()])  

    # if not return_ids:
    return time_tuples_by_onset, unique_time_tuples_by_onset, time_tuples_by_pitch, unique_time_tuples
    # else:
    #     return time_tuples_by_onset_id, unique_time_tuples_by_onset_id, time_tuples_by_pitch_id, unique_time_tuples_id




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