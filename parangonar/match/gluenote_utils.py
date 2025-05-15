#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
utilities for the thegluenote
"""
import numpy as np
from symusic import Score, Note, Track
from ..dp.dtw import WDTW
from ..dp.metrics import invert_matrix
from collections import defaultdict
from scipy.interpolate import interp1d

DEFAULT_NOTE = [
    196,  # 'TimeShift_1.0.8'
    4,  # 'Pitch_21'
    107,  # 'Velocity_63'
    131,  # 'Duration_1.0.8'
]


def format_note_array_alignment(
    score_note_array, performance_note_array, alignment, unmatched_idx
):
    alignment_idx = list()
    score_note_name_to_index = {
        sid: idx for idx, sid in enumerate(score_note_array["id"])
    }
    performance_note_name_to_index = {
        pid: idx for idx, pid in enumerate(performance_note_array["id"])
    }
    for match in alignment:
        if match["label"] == "match":
            alignment_idx.append(
                [
                    score_note_name_to_index[match["score_id"]],
                    performance_note_name_to_index[match["performance_id"]],
                ]
            )
        elif match["label"] == "deletion":
            alignment_idx.append(
                [score_note_name_to_index[match["score_id"]], unmatched_idx]
            )
        elif match["label"] == "insertion":
            alignment_idx.append(
                [unmatched_idx, performance_note_name_to_index[match["performance_id"]]]
            )
    return alignment_idx


def format_score_performance_alignment(
    score_note_array, performance_note_array, alignment_idx, unmatched_idx
):
    alignment = list()
    for sidx, pidx in alignment_idx:
        if sidx < unmatched_idx and pidx < unmatched_idx:
            alignment.append(
                {
                    "label": "match",
                    "score_id": score_note_array["id"][sidx],
                    "performance_id": performance_note_array["id"][pidx],
                }
            )
        else:
            if sidx < unmatched_idx:
                alignment.append(
                    {"label": "deletion", "score_id": score_note_array["id"][sidx]}
                )
            if pidx < unmatched_idx:
                alignment.append(
                    {
                        "label": "insertion",
                        "performance_id": performance_note_array["id"][pidx],
                    }
                )

    return alignment


def get_shifted_and_stretched_and_agnostic_midis(midi1, midi2):
    note_info2 = midi2.tracks[0].notes.numpy()
    note_info1 = midi1.tracks[0].notes.numpy()
    # shift to zero
    note_info2["time"] -= note_info2["time"].min()
    # stretch to the same length
    note_info1 = stretch(note_info1, note_info2, factor=1.0)
    note_info1 = velocity_and_duration_agnostic_note_info(note_info1)
    note_info2 = velocity_and_duration_agnostic_note_info(note_info2)
    symusic_note_list1 = Note.from_numpy(**note_info1)
    symusic_note_list2 = Note.from_numpy(**note_info2)
    midi1.tracks[0].notes = symusic_note_list1
    midi2.tracks[0].notes = symusic_note_list2
    return midi1, midi2


def stretch(note_info_to_change, note_info_ref, factor=2.0):
    time_to_change = note_info_to_change["time"]
    dur_to_change = note_info_to_change["duration"]
    time_ref = note_info_ref["time"]

    min_to_change = time_to_change.min()
    max_to_change = time_to_change.max()
    min_ref = time_ref.min()
    max_ref = time_ref.max()

    stretch_factor = (max_ref - min_ref) * factor / (max_to_change - min_to_change)
    new_time = (time_to_change - min_to_change) * stretch_factor
    new_dur = dur_to_change * stretch_factor

    return {
        "time": new_time.astype(np.int32),
        "duration": new_dur.astype(np.int32),
        "pitch": note_info_to_change["pitch"].astype(np.int8),
        "velocity": note_info_to_change["velocity"].astype(np.int8),
    }


def velocity_and_duration_agnostic_note_info(note_info):
    new_duration = np.full_like(note_info["duration"], 100)
    new_velocity = np.full_like(note_info["velocity"], 63)
    note_info["velocity"] = new_velocity
    note_info["duration"] = new_duration
    return note_info


def minimal_note_array_from_symusic(score, fields=["pitch", "time"]):
    note_info = score.tracks[0].notes.numpy()
    note_array = np.column_stack([note_info[field] for field in fields])
    return note_array


def note_array_to_symusic_score(note_array):
    fields = set(note_array.dtype.fields)
    score_units = set(("onset_quarter", "onset_beat"))
    performance_units = set(("onset_tick", "onset_sec"))

    if len(score_units.intersection(fields)) > 0:
        onset_field = "onset_quarter"
        duration_field = "duration_quarter"
        time_conversion = 480
    elif len(performance_units.intersection(fields)) > 0:
        if len(set(("onset_tick")).intersection(fields)) > 0:
            onset_field = "onset_tick"
            duration_field = "duration_tick"
            time_conversion = 1
        else:
            onset_field = "onset_sec"
            duration_field = "duration_sec"
            time_conversion = 480

    symusic_container = Score()
    empty_track = Track()
    symusic_container.tracks.append(empty_track)
    time = note_array[onset_field] * time_conversion
    duration = note_array[duration_field] * time_conversion
    pitch = note_array["pitch"]
    if "velocity" in fields:
        velocity = note_array["velocity"]
    else:
        velocity = np.full_like(note_array[onset_field], 64)

    note_info = {
        "time": time.astype(np.int32),
        "duration": duration.astype(np.int32),
        "pitch": pitch.astype(np.int8),
        "velocity": velocity.astype(np.int8),
    }

    symusic_note_list = Note.from_numpy(**note_info)
    symusic_container.tracks[0].notes = symusic_note_list
    return symusic_container


###################### DTW ############################


def get_local_path_from_confidence_matrix(confidence_matrix):
    path = get_path_from_confidence_matrix(confidence_matrix)
    (new_path, starting_path, ending_path, startpoints, endpoints) = get_path_endpoints(
        path, confidence_matrix.shape, cutoff=1
    )

    return (new_path, starting_path, ending_path, startpoints, endpoints)


def get_path_from_confidence_matrix(mat, directional_weights=np.array([1, 2, 1])):
    wdtw = WDTW(directional_weights=directional_weights)
    dmat = invert_matrix(mat)
    path = wdtw.from_distance_matrix(dmat)[0]
    return path


def get_path_endpoints(path, size, cutoff=1):
    new_path = np.copy(path)
    starting_path = np.array([])
    ending_path = np.array([])
    startpoints = np.array([0, 0])
    endpoints = np.array(size) - 1

    left_path = path[:, 1] < 1
    right_path = path[:, 1] > size[1] - 2
    top_path = path[:, 0] < 1
    bottom_path = path[:, 0] > size[0] - 2

    new_path_mask_start = np.ones(len(path)) > 0
    new_path_mask_end = np.ones(len(path)) > 0

    s1_exclusion_start = True
    s1_exclusion_end = True

    if left_path.sum() > cutoff:
        # remove last path element from starting path
        last_start_entry_id = np.where(left_path == True)[0][-1]
        left_path[last_start_entry_id] = False
        # update output vars
        starting_path = path[left_path, :]
        new_path_mask_start = np.invert(left_path)
        s1_exclusion_start = True
    elif top_path.sum() > cutoff:
        # remove last path element from starting path
        last_start_entry_id = np.where(top_path == True)[0][-1]
        top_path[last_start_entry_id] = False
        # update output vars
        starting_path = path[top_path, :]
        new_path_mask_start = np.invert(top_path)
        s1_exclusion_start = False

    if right_path.sum() > cutoff:
        first_end_entry_id = np.where(right_path == True)[0][0]
        right_path[first_end_entry_id] = False
        # update output vars
        ending_path = path[right_path, :]
        new_path_mask_end = np.invert(right_path)
        s1_exclusion_end = True

    elif bottom_path.sum() > cutoff:
        first_end_entry_id = np.where(bottom_path == True)[0][0]
        bottom_path[first_end_entry_id] = False
        # update output vars
        ending_path = path[bottom_path, :]
        new_path_mask_end = np.invert(bottom_path)
        s1_exclusion_end = False

    new_path_mask = np.all((new_path_mask_start, new_path_mask_end), axis=0)
    new_path = new_path[new_path_mask]
    return new_path, starting_path, ending_path, s1_exclusion_start, s1_exclusion_end


def get_merging_idx(array, threshold=2):
    array = array[np.argsort(array)]

    mergers = defaultdict(list)
    prev_e = array[0] - threshold * 2
    prev_i = -1
    for i, e in enumerate(array):
        if abs(e - prev_e) < threshold:
            mergers[prev_i].append(i)
        else:
            prev_e = e
            prev_i = i
    return mergers


def get_input_to_ref_map(
    note_array,  # pitch, onset
    note_array_ref,  # pitch, onset
    alignment_idx,
    merge_close_onsets=5,
    return_callable=True,
):  # na_idx, na_ref_idx
    na_onset = note_array[alignment_idx[:, 0], 1]
    na_ref_onset = note_array_ref[alignment_idx[:, 1], 1]
    onsets = np.column_stack((na_onset, na_ref_onset))

    if merge_close_onsets:
        # onsets s1
        keep_idx = np.full(len(onsets), True)
        onsets = onsets[np.argsort(onsets[:, 0]), :]
        merging_ids = get_merging_idx(onsets[:, 0], threshold=merge_close_onsets)
        new_vals_list = list()
        for merging_idx in merging_ids.keys():
            mask = np.array([merging_idx] + merging_ids[merging_idx])
            new_vals = np.median(onsets[mask, :], axis=0)
            new_vals_list.append(new_vals)
            keep_idx[mask] = False

        if len(new_vals_list) > 0:
            new_vals_array = np.row_stack(new_vals_list)
            onsets = np.concatenate((onsets[keep_idx], new_vals_array), axis=0)

        # onsets s2
        keep_idx = np.full(len(onsets), True)
        onsets = onsets[np.argsort(onsets[:, 1]), :]
        merging_ids = get_merging_idx(onsets[:, 1], threshold=merge_close_onsets)
        new_vals_list = list()
        for merging_idx in merging_ids.keys():
            mask = np.array([merging_idx] + merging_ids[merging_idx])
            new_vals = np.median(onsets[mask, :], axis=0)
            new_vals_list.append(new_vals)
            keep_idx[mask] = False

        if len(new_vals_list) > 0:
            new_vals_array = np.row_stack(new_vals_list)
            onsets = np.concatenate((onsets[keep_idx], new_vals_array), axis=0)
        # sort again
        onsets = onsets[np.argsort(onsets[:, 0]), :]

    if return_callable:
        input_to_ref_map = interp1d(
            onsets[:, 0], onsets[:, 1], kind="linear", fill_value="extrapolate"
        )

        return input_to_ref_map
    else:
        return onsets
