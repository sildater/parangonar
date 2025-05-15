#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods to visualize and
evaluate note alignments of symbolic music data.
"""
from typing import List
import numpy as np
import partitura as pt
import matplotlib.pyplot as plt
import random
import os
from .io import save_pianoprecision_csv

def fscore_alignments(
    prediction: List[dict],
    ground_truth: List[dict],
    types: List[str],
    return_numbers=False,
) -> (float, float, float):
    """
    Parameters
    ----------
    prediction: List of dictionaries containing the predicted alignments
    ground_truth: List of dictionaries containing the ground truth alignments
    types: List of alignment types to consider for evaluation (e.g ['match', 'deletion', 'insertion']

    Returns
    -------
    precision, recall, f score
    """

    pred_filtered = list(filter(lambda x: x["label"] in types, prediction))
    gt_filtered = list(filter(lambda x: x["label"] in types, ground_truth))

    filtered_correct = [pred for pred in pred_filtered if pred in gt_filtered]

    n_pred_filtered = len(pred_filtered)
    n_gt_filtered = len(gt_filtered)
    n_correct = len(filtered_correct)

    if n_pred_filtered > 0 or n_gt_filtered > 0:
        precision = n_correct / n_pred_filtered if n_pred_filtered > 0 else 0.0
        recall = n_correct / n_gt_filtered if n_gt_filtered > 0 else 0
        f_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
    else:
        # no prediction and no ground truth for a given type -> correct alignment
        precision, recall, f_score = 1.0, 1.0, 1.0

    if return_numbers:
        return precision, recall, f_score, len(pred_filtered), len(gt_filtered)
    else:
        return precision, recall, f_score


def print_fscore_alignments(prediction: List[dict], ground_truth: List[dict]):
    print("------------------")
    types = ["match", "insertion", "deletion"]
    for alignment_type in types:
        precision, recall, f_score = fscore_alignments(
            prediction, ground_truth, alignment_type
        )
        print("Evaluate ", alignment_type)
        print(
            "Precision: ",
            format(precision, ".3f"),
            "Recall ",
            format(recall, ".3f"),
            "F-Score ",
            format(f_score, ".3f"),
        )
        print("------------------")


def plot_alignment(
    ppart_na,
    part_na,
    alignment,
    save_file=False,
    fname="note_alignment",
    random_color=False,
):
    first_note_midi = np.min(ppart_na["onset_sec"])
    last_note_midi = np.max(ppart_na["onset_sec"] + ppart_na["duration_sec"])
    first_note_start = np.min(part_na["onset_beat"])
    last_note_start = np.max(part_na["onset_beat"])
    length_of_midi = last_note_midi - first_note_midi
    length_of_xml = last_note_start - first_note_start

    length_of_pianorolls = max(10000, int(length_of_xml * 8))
    time_div_midi = int(np.floor(length_of_pianorolls / length_of_midi))
    time_div_xml = int(np.floor(length_of_pianorolls / length_of_xml))

    midi_piano_roll, perfidx = pt.utils.compute_pianoroll(
        ppart_na,
        time_unit="sec",
        time_div=time_div_midi,
        return_idxs=True,
        remove_drums=False,
    )
    xml_piano_roll, scoreidx = pt.utils.compute_pianoroll(
        part_na,
        time_unit="beat",
        time_div=time_div_xml,
        return_idxs=True,
        remove_drums=False,
    )

    plot_array = np.zeros((128 * 2 + 50, length_of_pianorolls + 800))
    dense_midi_pr = midi_piano_roll.todense()  # [:,:time_size]
    dense_midi_pr[dense_midi_pr > 0] = 1
    plot_array[:128, : xml_piano_roll.shape[1]] = xml_piano_roll.todense()
    plot_array[50 + 128 :, : midi_piano_roll.shape[1]] = dense_midi_pr

    f, axs = plt.subplots(1, 1, figsize=(100, 10))
    axs.matshow(plot_array, aspect="auto", origin="lower")
    hexadecimal_alphabets = "0123456789ABCDEF"

    if random_color:
        colors = [
            "#" + "".join([random.choice(hexadecimal_alphabets) for j in range(6)])
            for i in range(40)
        ]
    else:
        colors = ["#00FF00" for i in range(40)]

    score_dict = dict()
    for note, pos in zip(part_na, scoreidx):
        score_dict[note["id"]] = pos
    perf_dict = dict()
    for note, pos in zip(ppart_na, perfidx):
        perf_dict[note["id"]] = pos
    for i, line in enumerate(alignment):
        if line["label"] == "match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot(
                [score_pos[1], perf_pos[1]],
                [score_pos[0], 128 + 50 + perf_pos[0]],
                "o-",
                lw=2,
                c=colors[i % 40],
            )

    if save_file:
        plt.savefig(fname + ".png")
        plt.close(f)
    else:
        plt.show()


def plot_alignment_comparison(
    ppart_na,
    part_na,
    alignment1,
    alignment2,
    save_file=False,
    return_figure=False,
    figsize=(100, 10),
    fname="note_alignments",
):
    first_note_midi = np.min(ppart_na["onset_sec"])
    last_note_midi = np.max(ppart_na["onset_sec"] + ppart_na["duration_sec"])
    first_note_start = np.min(part_na["onset_beat"])
    last_note_start = np.max(part_na["onset_beat"])
    length_of_midi = last_note_midi - first_note_midi
    length_of_xml = last_note_start - first_note_start

    length_of_pianorolls = max(10000, int(length_of_xml * 8))
    time_div_midi = int(np.floor(length_of_pianorolls / length_of_midi))
    time_div_xml = int(np.floor(length_of_pianorolls / length_of_xml))

    midi_piano_roll, perfidx = pt.utils.compute_pianoroll(
        ppart_na,
        time_unit="sec",
        time_div=time_div_midi,
        return_idxs=True,
        remove_drums=False,
    )
    xml_piano_roll, scoreidx = pt.utils.compute_pianoroll(
        part_na,
        time_unit="beat",
        time_div=time_div_xml,
        return_idxs=True,
        remove_drums=False,
    )

    plot_array = np.zeros((128 * 2 + 50, length_of_pianorolls + 800))
    dense_midi_pr = midi_piano_roll.todense()  # [:,:time_size]
    dense_midi_pr[dense_midi_pr > 0] = 1
    plot_array[:128, : xml_piano_roll.shape[1]] = xml_piano_roll.todense()
    plot_array[50 + 128 :, : midi_piano_roll.shape[1]] = dense_midi_pr

    f, axs = plt.subplots(1, 1, figsize=figsize)
    axs.matshow(plot_array, aspect="auto", origin="lower")
    hexadecimal_alphabets = "0123456789ABCDEF"

    colors1 = ["#00FF00" for i in range(40)]
    colors2 = ["#0000FF" for i in range(40)]
    colors3 = ["#FF0000" for i in range(40)]
    colors4 = ["#FF00FF" for i in range(40)]

    n1_but_not_n2 = [al for al in alignment1 if not al in alignment2]
    n2_but_not_n1 = [al for al in alignment2 if not al in alignment1]

    score_dict = dict()
    for note, pos in zip(part_na, scoreidx):
        score_dict[note["id"]] = pos
    perf_dict = dict()
    for note, pos in zip(ppart_na, perfidx):
        perf_dict[note["id"]] = pos
    for i, line in enumerate(alignment1):
        if line["label"] == "match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot(
                [score_pos[1], perf_pos[1]],
                [score_pos[0], 128 + 50 + perf_pos[0]],
                "o-",
                lw=2,
                c=colors1[i % 40],
            )
    for i, line in enumerate(alignment2):
        if line["label"] == "match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot(
                [score_pos[1], perf_pos[1]],
                [score_pos[0], 128 + 50 + perf_pos[0]],
                "o-",
                lw=2,
                c=colors2[i % 40],
            )
    for i, line in enumerate(n1_but_not_n2):
        if line["label"] == "match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot(
                [score_pos[1], perf_pos[1]],
                [score_pos[0], 128 + 50 + perf_pos[0]],
                "o-",
                lw=2,
                c=colors3[i % 40],
            )
    for i, line in enumerate(n2_but_not_n1):
        if line["label"] == "match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot(
                [score_pos[1], perf_pos[1]],
                [score_pos[0], 128 + 50 + perf_pos[0]],
                "o-",
                lw=2,
                c=colors4[i % 40],
            )

    if save_file:
        plt.savefig(fname + ".png")
        plt.close(f)
    else:
        if return_figure:
            return f
        else:
            plt.show()


def plot_alignment_mappings(
    ppart_na,
    part_na,
    score_to_performance_mapping1,
    score_to_performance_mapping2,
    save_file=False,
    fname="onset_alignments",
):
    first_note_midi = np.min(ppart_na["onset_sec"])
    last_note_midi = np.max(ppart_na["onset_sec"] + ppart_na["duration_sec"])
    first_note_start = np.min(part_na["onset_beat"])
    last_note_start = np.max(part_na["onset_beat"])
    length_of_midi = last_note_midi - first_note_midi
    length_of_xml = last_note_start - first_note_start

    length_of_pianorolls = max(10000, int(length_of_xml * 8))
    time_div_midi = int(np.floor(length_of_pianorolls / length_of_midi))
    time_div_xml = int(np.floor(length_of_pianorolls / length_of_xml))

    midi_piano_roll, perfidx = pt.utils.compute_pianoroll(
        ppart_na,
        time_unit="sec",
        time_div=time_div_midi,
        return_idxs=True,
        remove_drums=False,
    )
    xml_piano_roll, scoreidx = pt.utils.compute_pianoroll(
        part_na,
        time_unit="beat",
        time_div=time_div_xml,
        return_idxs=True,
        remove_drums=False,
    )

    midi_to_pr_mapping = lambda x: int((x - first_note_midi) * time_div_midi)
    xml_to_pr_mapping = lambda x: int((x - first_note_start) * time_div_xml)

    plot_array = np.zeros((128 * 2 + 50, length_of_pianorolls + 800))
    dense_midi_pr = midi_piano_roll.todense()  # [:,:time_size]
    dense_midi_pr[dense_midi_pr > 0] = 1
    plot_array[:128, : xml_piano_roll.shape[1]] = xml_piano_roll.todense()
    plot_array[50 + 128 :, : midi_piano_roll.shape[1]] = dense_midi_pr

    f, axs = plt.subplots(1, 1, figsize=(100, 10))
    axs.matshow(plot_array, aspect="auto", origin="lower")
    for unique_onset in np.unique(part_na["onset_beat"]):
        perf_pos1 = midi_to_pr_mapping(score_to_performance_mapping1(unique_onset))
        perf_pos2 = midi_to_pr_mapping(score_to_performance_mapping2(unique_onset))
        score_pos = xml_to_pr_mapping(unique_onset)
        axs.plot([score_pos, perf_pos1], [128, 128 + 50], "o-", lw=2, c="#00FF00")
        axs.plot([score_pos, perf_pos2], [128, 128 + 50], "o-", lw=2, c="#0000FF")

    if save_file:
        plt.savefig(fname + ".png")
        plt.close(f)
    else:
        plt.show()


def evaluate_asynchrony(target_ponsets, tracked_ponsets):
    asynchrony = target_ponsets - tracked_ponsets
    abs_asynch = abs(asynchrony)
    mean_asynch = np.median(abs_asynch)
    lt_25ms = np.mean(abs_asynch <= 0.025)
    lt_50ms = np.mean(abs_asynch <= 0.05)
    lt_100ms = np.mean(abs_asynch <= 0.1)
    return mean_asynch, lt_25ms, lt_50ms, lt_100ms


def evaluate_score_following(
    performance_note_array=None,
    score_note_array=None,
    gt_alignment=None,
    alignment=None,
    out_dir="",
    file_suffix="",
    write_to_file=False,
    print_results=False,
):
    """
    Parameters
    ----------
    performance_note_array: np.ndarray

    score_note_array: np.ndarray

    gt_alignment: List[dict]

    alignment: List[dict]

    out_dir: str

    print_results: bool

    Returns
    -------
    mean_asynch: float

    lt_25ms: float

    lt_50ms: float

    lt_100ms: float

    """

    (
        _,
        stime_to_ptime_map_gt,
    ) = pt.musicanalysis.performance_codec.get_time_maps_from_alignment(
        performance_note_array, score_note_array, gt_alignment, remove_ornaments=True
    )

    (
        _,
        stime_to_ptime_map,
    ) = pt.musicanalysis.performance_codec.get_time_maps_from_alignment(
        performance_note_array, score_note_array, alignment, remove_ornaments=True
    )
    sid1 = set()
    sid2 = set()
    for al in alignment:
        # Get only matched notes (i.e., ignore inserted or deleted notes)
        if al["label"] == "match":
            sid1.add(al["score_id"])
    for al in gt_alignment:
        if al["label"] == "match":
            sid2.add(al["score_id"])

    all_sid = list(sid1.intersection(sid2))

    tracked_sonsets = np.array(
        [
            score_note["onset_beat"]
            for score_note in score_note_array
            if score_note["id"] in all_sid
        ]
    )
    tracked_sonsets = np.unique(tracked_sonsets)

    target_ponsets = stime_to_ptime_map_gt(tracked_sonsets)
    tracked_ponsets = stime_to_ptime_map(tracked_sonsets)

    mean_asynch, lt_25ms, lt_50ms, lt_100ms = evaluate_asynchrony(
        target_ponsets=target_ponsets,
        tracked_ponsets=tracked_ponsets,
    )

    if write_to_file:
        results_fn = os.path.join(out_dir, file_suffix + "results.csv")
        with open(results_fn, "w") as f:
            f.write("mean_asynch_ms,leq_25ms_%,leq_50ms_%,leq_100ms_%\n")

            if print_results:
                print(
                    f"Mean asynchrony (ms): {mean_asynch * 1000:.2f}\nAsynchrony <= 25ms (%): {lt_25ms * 100:.1f}\n"
                    f"Asynchrony <= 50ms (%): {lt_50ms * 100:.1f}\nAsynchrony <= 100ms (%): {lt_100ms * 100:.1f}\n"
                )
            f.write(
                f"{mean_asynch * 1000:.2f},{lt_25ms * 100:.1f},"
                f"{lt_50ms * 100:.1f},{lt_100ms * 100:.1f}"
            )

    else:
        return mean_asynch, lt_25ms, lt_50ms, lt_100ms
