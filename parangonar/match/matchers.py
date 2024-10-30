#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains full note matcher classes.
"""
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
import miditok
import symusic

import time
from itertools import combinations
from scipy.special import binom

from ..dp.dtw import DTW, DTWSL
from ..dp.nwtw import NW_DTW, NW
from .. import THEGLUENOTE_CHECKPOINT

from .gluenote_utils import (
    get_shifted_and_stretched_and_agnostic_midis,
    minimal_note_array_from_symusic,
    note_array_to_symusic_score,
    format_score_performance_alignment,
    format_note_array_alignment,
    DEFAULT_NOTE,
    get_local_path_from_confidence_matrix,
    get_input_to_ref_map
)
from .preprocessors import (mend_note_alignments,
                            cut_note_arrays,
                            alignment_times_from_dtw,
                            note_per_ons_encoding)

from .pretrained_models import (AlignmentTransformer,
                                TheGlueNote)
import torch


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
    
    if len(note_array) == 0:
        return list()
    else:
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
                 onset_threshold=None,
                 process_ornaments=False):

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
        #grace_onsets_perf = score_to_perf_map(grace_onsets)
        #for grace_onset, grace_onset_perf in zip(grace_onsets, grace_onsets_perf):
        for grace_onset in grace_onsets:
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
                
        if process_ornaments:
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
        return note_alignments


def get_note_matches_with_updating_map(
                    note_array,# pitch, onset
                     note_array_ref,# pitch, onset
                     matched_onset_seqs,
                     onset_threshold,
                     unmatched_idx = 100000000):
     
    note_array_idx_range = np.arange(len(note_array))
    note_array_ref_idx_range = np.arange(len(note_array_ref))

    # Get symbolic note_alignments
    note_alignments = list()
    used_note_ids = set()
    used_ref_note_ids = set()
    unique_pitches, pitch_counts = np.unique(np.concatenate((note_array[:,0],note_array_ref[:,0]), axis= 0), return_counts = True)
    pitch_by_quantity = np.argsort(pitch_counts)

    for pitch in unique_pitches[pitch_by_quantity]:

        input_to_ref_map = interp1d(matched_onset_seqs[:,0],
                                    matched_onset_seqs[:,1],
                                    kind = "linear",
                                    fill_value = "extrapolate")

        note_array_pitch_mask = note_array[:,0] == pitch
        note_array_ref_pitch_mask = note_array_ref[:,0] == pitch
        
        note_array_onsets = note_array[note_array_pitch_mask,1]
        note_array_ref_onsets = note_array_ref[note_array_ref_pitch_mask,1]

        note_array_ids = note_array_idx_range[note_array_pitch_mask]
        note_array_ref_ids = note_array_ref_idx_range[note_array_ref_pitch_mask]
        
        estimated_note_array_ref_onsets = input_to_ref_map(note_array_onsets)

        

        if  (note_array_ref_onsets.shape[0] > 1 and note_array_onsets.shape[0] > 1) or \
            (note_array_ref_onsets.shape[0] > 1 and note_array_onsets.shape[0] == 1) or \
            (note_array_ref_onsets.shape[0] == 1 and note_array_onsets.shape[0] > 1):
            try:
                ID_tuples = unique_alignments(estimated_note_array_ref_onsets, 
                                              note_array_ref_onsets,
                                              threshold=onset_threshold)
            except:
                import pdb; pdb.set_trace()
        
        elif note_array_ref_onsets.shape[0] == 1 and note_array_onsets.shape[0] == 1: 
            if np.abs(estimated_note_array_ref_onsets[0] - note_array_ref_onsets[0]) < onset_threshold:
                ID_tuples = [(0,0)]
            else:
                ID_tuples = []
        else:
            ID_tuples = []       
        
        for input_idx, ref_idx in ID_tuples:
            note_alignments.append(
                [note_array_ids[input_idx], note_array_ref_ids[ref_idx]]

            )
            used_note_ids.add(note_array_ids[input_idx])
            used_ref_note_ids.add(note_array_ref_ids[ref_idx])
        
        if len(ID_tuples) > 0:
            ID_tuples_numpy = np.array(ID_tuples)
            new_matches = np.column_stack((note_array_onsets[ID_tuples_numpy[:,0]],note_array_ref_onsets[ID_tuples_numpy[:,1]]))
            matched_onset_seqs = insert_matches_into_matched_seqs(matched_onset_seqs,
                                                                new_matches)


    # add unmatched notes
    for note_idx in note_array_idx_range:
        if note_idx not in used_note_ids:
            note_alignments.append([note_idx, unmatched_idx])

    for ref_idx in note_array_ref_idx_range:
        if ref_idx not in used_ref_note_ids:
            note_alignments.append([unmatched_idx, ref_idx])
   

    note_alignments = np.array(note_alignments)
    note_alignments = note_alignments[np.argsort(note_alignments[:,0]),:]
    return note_alignments


def insert_matches_into_matched_seqs(matched_onset_seqs,
                                     new_matches):
    new_matched_onset_seqs = np.copy(matched_onset_seqs)

    new_lines = list()
    idx_to_delete = list()
    for match in new_matches:
        id_seq1 = np.where(matched_onset_seqs[:,0] == match[0])[0]
        if len(id_seq1) > 0:
           idx_to_delete.append(id_seq1[0])
           new_lines.append(match)

    if len(new_lines) > 0:
        new_lines_numpy = np.array(new_lines)
        deletion_mask = np.array(idx_to_delete)
        new_matched_onset_seqs = np.delete(new_matched_onset_seqs, deletion_mask, axis=0)
        new_matched_onset_seqs = np.concatenate((new_matched_onset_seqs, new_lines_numpy))
        new_matched_onset_seqs = new_matched_onset_seqs[np.argsort(new_matched_onset_seqs[:,0])]

    return new_matched_onset_seqs


################################### ONSET MATCHERS ###################################


class OnsetMatcherDTW(object):
    """
    Create an onset matching using pitch-based DTW from note_arrays
    """
    def __init__(self, 
                 dtw = DTWSL()):
        self.dtw = dtw

    def __call__(self,
                 score_note_array_no_grace, 
                 performance_note_array, 
                 flip = False):

        
        unique_onsets = np.unique(score_note_array_no_grace["onset_beat"])
        score_pitch_at_onsets = list()
        for onset in unique_onsets:
            score_pitch_at_onsets.append(set(score_note_array_no_grace[score_note_array_no_grace["onset_beat"] == onset]["pitch"]))

        if flip:
            score_pitch_at_onsets.reverse()
            _, onset_alignment_path = self.dtw(np.flipud(performance_note_array["pitch"]), 
                                    score_pitch_at_onsets,  
                                    return_path=True)
        else:
            _, onset_alignment_path = self.dtw(performance_note_array["pitch"], 
                                    score_pitch_at_onsets,  
                                    return_path=True)
        
        return onset_alignment_path, unique_onsets

    
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
    

class DualDTWNoteMatcher(object):
    def __init__(self,
                 onset_matcher=OnsetMatcherDTW(),
                 note_matcher=CleanOrnamentMatcher(),
                 ):

        self.onset_matcher = onset_matcher
        self.note_matcher = note_matcher


    def __call__(self, 
                 score_note_array,
                 performance_note_array,
                 process_ornaments = False,
                 score_part = None):
        
        if process_ornaments:
            if score_part is None:
                print("score part is required for ornament extraction")
                score_note_array_ornament = score_note_array
            else:
                # add ornament tags to score_note_array
                notes = score_part.notes_tied
                fields = [("id", "U256"),("ornament", "b")]
                ornament_tags = list()
                for n in notes:
                    ornament = False
                    if n.ornaments:
                        ornament = True
                    ornament_tags.append((n.id, ornament))
                ornament_tags_array = np.array(ornament_tags, dtype=fields)
                note_array_joined = np.lib.recfunctions.join_by("id", score_note_array, ornament_tags_array)
                score_note_array = note_array_joined.data 
                sort_idx = np.lexsort(
                    (score_note_array["duration_div"], score_note_array["pitch"], score_note_array["onset_beat"])
                    )
                score_note_array = score_note_array[sort_idx]
                score_note_array_ornament = score_note_array[score_note_array["ornament"] == True]    
        else:
            score_note_array_ornament = score_note_array
        
        score_note_array_no_grace = score_note_array[score_note_array["is_grace"] == False]    
        score_note_array_grace = score_note_array[score_note_array["is_grace"] == True]

        onset_alignment_path, _ = self.onset_matcher(score_note_array_no_grace, 
                                                     performance_note_array)

        onset_alignment_path_reverse, _ = self.onset_matcher(score_note_array_no_grace, 
                                                             performance_note_array,
                                                             flip = True)
        

        
            

        global_alignment = self.note_matcher(score_note_array, # score notes including grace notes
                                            score_note_array_no_grace, # score notes excluding grace notes 
                                            score_note_array_grace, # grace notes
                                            score_note_array_ornament,
                                            performance_note_array,
                                            onset_alignment_path,
                                            onset_alignment_path_reverse,
                                            onset_threshold=1.5,
                                            process_ornaments=process_ornaments) # TODO: document
        
        return global_alignment


################################### PRETRAINED MATCHERS ###################################


class TheGlueNoteMatcher(object):
    def __init__(self):

        self.prepare_model()
        self.tokenizer = miditok.Structured()
        self.unmatched_idx = 100000000
        self.matching_threshold = 0.5,

    def prepare_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(THEGLUENOTE_CHECKPOINT, 
                                map_location=torch.device(self.device))
        self.model = TheGlueNote(device = self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, 
                 score_note_array, 
                 performance_note_array):
        midi_score = note_array_to_symusic_score(score_note_array)
        midi_performance = note_array_to_symusic_score(performance_note_array)
        alignment = self.run_model_inference(
                        midi_performance,
                        midi_score,
                        performance_note_array = performance_note_array,
                        score_note_array_full = score_note_array,
                        return_formatted_alignment = True
                        )
        return alignment
    
    def run_model_inference(
        self,
        performance_midi,
        score_midi,
        performance_note_array = None,
        score_note_array_full = None,
        return_formatted_alignment = False
        ): 
        
        alignment = self.get_dtw_alignment_from_model(
                                encoder_model = self.model, 
                                tokenizer = self.tokenizer,
                                input_midi1 = score_midi, 
                                input_midi2 = performance_midi,
                                unmatched_idx = self.unmatched_idx)
        
        if return_formatted_alignment and score_note_array_full is not None and performance_note_array is not None:
            alignment = format_score_performance_alignment(score_note_array_full,
                                                                    performance_note_array,
                                                                    alignment,
                                                                    unmatched_idx = self.unmatched_idx)
        return alignment

    def get_dtw_alignment_from_model(
                                self,
                                encoder_model, 
                                tokenizer,
                                input_midi1, 
                                input_midi2,
                                unmatched_idx = 100000000):
        # setup and preprocessing of files
        sequence_length = encoder_model.position_number - 1
        input_midi2, input_midi1 = get_shifted_and_stretched_and_agnostic_midis(input_midi2, input_midi1)
        note_array = minimal_note_array_from_symusic(input_midi1)
        note_array_ref = minimal_note_array_from_symusic(input_midi2)
        tokens1 = tokenizer(input_midi1)
        tokens2 = tokenizer(input_midi2)
        sample = {"s1":np.array(tokens1[0].ids, dtype = int),
                "s2":np.array(tokens2[0].ids, dtype = int)}
        
        no_notes_s1 = len(note_array)
        no_notes_s2 = len(note_array_ref)
        index_shift = int(sequence_length/2)
        no_slices_s1 = no_notes_s1 // (index_shift) + 1
        no_slices_s2 = no_notes_s2 // (index_shift) + 1
        full_similarity_matrix = np.zeros((no_notes_s1, no_notes_s2))

        # loop over windows
        for i in range(no_slices_s1):
            for j in range(no_slices_s2):
                current_idx_1 = [i*index_shift, (i+2)*index_shift]
                current_idx_2 = [j*index_shift, (j+2)*index_shift]

                # pad and prepare the note sequences
                s1 = sample["s1"][current_idx_1[0]*4:current_idx_1[1]*4]
                s2 = sample["s2"][current_idx_2[0]*4:current_idx_2[1]*4]
                s1_matrix_end = sequence_length
                s2_matrix_end = sequence_length

                if len(s1) < sequence_length * 4:
                    # pad the sequence
                    padding_len1 = sequence_length * 4 - len(s1)
                    padding1 = np.array(DEFAULT_NOTE * int(padding_len1 / 4))#np.zeros(padding_len1)
                    s1_matrix_end = int(len(s1) / 4)
                    s1 = np.concatenate((s1,padding1)).astype(int)

                if len(s2) < sequence_length * 4:
                    # pad the sequence
                    padding_len2 = sequence_length * 4 - len(s2)
                    padding2 = np.array(DEFAULT_NOTE * int(padding_len2 / 4))#np.zeros(padding_len2)
                    s2_matrix_end = int(len(s2) / 4)
                    s2 = np.concatenate((s2,padding2)).astype(int)

                s1 = np.concatenate((np.zeros(4), s1)).astype(int)
                s2 = np.concatenate((np.zeros(4), s2)).astype(int)
                sequences = torch.from_numpy(np.concatenate((s1,s2)).astype(int)).contiguous().unsqueeze(0)
                
                # call the encoder
                sequences = sequences.to(encoder_model.device)
                confidence_matrix = encoder_model(sequences, return_confidence_matrix = True)
                confidence_matrix_segment = confidence_matrix[0,1:s1_matrix_end + 1,1:s2_matrix_end + 1].detach().cpu().numpy()
                full_similarity_matrix[current_idx_1[0]:current_idx_1[1],current_idx_2[0]:current_idx_2[1]] += confidence_matrix_segment
                
        now = time.time()
        (path, 
        starting_path, ending_path, 
        s1_exclusion_start, s1_exclusion_end) = get_local_path_from_confidence_matrix(full_similarity_matrix)
        then = time.time()
        print("DTW local path time: ", then - now)
        print("PATH length:", len(path), full_similarity_matrix.shape)
            
        s1_to_s2_map = get_input_to_ref_map(note_array,
                                            note_array_ref,
                                            path,
                                            return_callable = False)
        
        alignment = get_note_matches_with_updating_map(note_array,
                                        note_array_ref,
                                        s1_to_s2_map,
                                        onset_threshold = 1000,
                                        unmatched_idx = unmatched_idx)
        
        return alignment


    def match_midi(self,
                   midi_path_0,
                   midi_path_1):
        
        midi_0 = symusic.Score(midi_path_0)
        midi_1 = symusic.Score(midi_path_1)
        alignment = self.run_model_inference(
                        midi_0,
                        midi_1,
                        return_formatted_alignment = False)
        return alignment