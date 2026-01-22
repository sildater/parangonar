#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods to evaluate
- note matching via f scores for match, insertion, and deletion
- score following / temporal alignment via asynchrony
"""
from typing import List, Dict, Tuple, Union, Optional, Any
import numpy as np
import partitura as pt
import os


def fscore_alignments(
    prediction: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    types: List[str],
    return_numbers: bool = False,
) -> Union[Tuple[float, float, float], Tuple[float, float, float, int, int]]:
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


def print_fscore_alignments(
    prediction: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
) -> None:
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


def evaluate_asynchrony(
    target_ponsets: np.ndarray, tracked_ponsets: np.ndarray
) -> Tuple[float, float, float, float]:
    asynchrony = target_ponsets - tracked_ponsets
    abs_asynch = abs(asynchrony)
    mean_asynch = np.median(abs_asynch)
    lt_25ms = np.mean(abs_asynch <= 0.025)
    lt_50ms = np.mean(abs_asynch <= 0.05)
    lt_100ms = np.mean(abs_asynch <= 0.1)
    return mean_asynch, lt_25ms, lt_50ms, lt_100ms


def evaluate_score_following(
    performance_note_array: np.ndarray,
    score_note_array: np.ndarray,
    gt_alignment: List[Dict[str, Any]],
    alignment:List[Dict[str, Any]],
    out_dir: str = "",
    file_suffix: str = "",
    write_to_file: bool = False,
    print_results: bool = False,
) -> Optional[Tuple[float, float, float, float]]:
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
