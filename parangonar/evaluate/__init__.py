#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module
evaluate note alignments symbolic music data.
"""
from typing import List
import numpy as np
import partitura as pt
import matplotlib.pyplot as plt
import random

def fscore_alignments(prediction: List[dict], 
                        ground_truth: List[dict], 
                        types: List[str],
                        return_numbers= False) -> (float, float, float):
                        
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

    pred_filtered = list(filter(lambda x: x['label'] in types, prediction))
    gt_filtered = list(filter(lambda x: x['label'] in types, ground_truth))

    filtered_correct = [pred for pred in pred_filtered if pred in gt_filtered]

    n_pred_filtered = len(pred_filtered)
    n_gt_filtered = len(gt_filtered)
    n_correct = len(filtered_correct)

    if n_pred_filtered > 0 or n_gt_filtered > 0:
        precision = n_correct / n_pred_filtered if n_pred_filtered > 0 else 0.
        recall = n_correct / n_gt_filtered if n_gt_filtered > 0 else 0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        # no prediction and no ground truth for a given type -> correct alignment
        precision, recall, f_score = 1., 1., 1.

    if return_numbers:
        return precision, recall, f_score, len(pred_filtered), len(gt_filtered)
    else:
        return precision, recall, f_score



def plot_alignment(ppart_na, 
         part_na, 
         alignment, 
         save_file = False,
         fname = "note_alignment",
         random_color = False
         ):
    
    first_note_midi = np.min(ppart_na["onset_sec"])
    last_note_midi = np.max(ppart_na["onset_sec"]+ppart_na["duration_sec"])
    first_note_start = np.min(part_na["onset_beat"])
    last_note_start = np.max(part_na["onset_beat"])
    length_of_midi = last_note_midi - first_note_midi
    length_of_xml = last_note_start - first_note_start

    length_of_pianorolls = max(10000,int(length_of_xml*8))
    time_div_midi = int(np.floor(length_of_pianorolls/length_of_midi))
    time_div_xml = int(np.floor(length_of_pianorolls/length_of_xml))
    
    midi_piano_roll, perfidx = pt.utils.compute_pianoroll(ppart_na,
                                                time_unit = "sec",
                                                time_div = time_div_midi,
                                                return_idxs=True,
                                                remove_drums=False)
    xml_piano_roll, scoreidx = pt.utils.compute_pianoroll(part_na,
                                                time_unit = "beat",
                                                time_div = time_div_xml,
                                                return_idxs = True,
                                                remove_drums=False)
    
    plot_array = np.zeros((128*2+50, length_of_pianorolls+800))
    dense_midi_pr = midi_piano_roll.todense()#[:,:time_size]
    dense_midi_pr[dense_midi_pr>0] = 1
    plot_array[:128,:xml_piano_roll.shape[1]] = xml_piano_roll.todense()
    plot_array[50+128:,:midi_piano_roll.shape[1]] = dense_midi_pr

    f, axs = plt.subplots(1,1,figsize=(100, 10))
    axs.matshow(plot_array, aspect = "auto",  origin='lower')
    hexadecimal_alphabets = '0123456789ABCDEF'

    if random_color:
        colors = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in range(40)]
    else:
        colors = ["#00FF00" for i in range(40)]
        
    score_dict = dict()
    for note, pos in zip(part_na, scoreidx):
        score_dict[note["id"]] = pos
    perf_dict = dict()
    for note, pos in zip(ppart_na, perfidx):
        perf_dict[note["id"]] = pos
    for i, line in enumerate(alignment):
        if line["label"]=="match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot([score_pos[1], perf_pos[1]], [ score_pos[0], 128+50 +perf_pos[0]],'o-', lw=2, c = colors[i%40])

    if save_file:
        plt.savefig(fname+".png")
        plt.close(f)
    else:
        plt.show()
        
        
def plot_alignment_comparison(ppart_na, 
         part_na, 
         alignment1, 
         alignment2, 
         save_file = False,
         fname = "note_alignments",
         ):
    
    first_note_midi = np.min(ppart_na["onset_sec"])
    last_note_midi = np.max(ppart_na["onset_sec"]+ppart_na["duration_sec"])
    first_note_start = np.min(part_na["onset_beat"])
    last_note_start = np.max(part_na["onset_beat"])
    length_of_midi = last_note_midi - first_note_midi
    length_of_xml = last_note_start - first_note_start

    length_of_pianorolls = max(10000,int(length_of_xml*8))
    time_div_midi = int(np.floor(length_of_pianorolls/length_of_midi))
    time_div_xml = int(np.floor(length_of_pianorolls/length_of_xml))
    
    midi_piano_roll, perfidx = pt.utils.compute_pianoroll(ppart_na,
                                                time_unit = "sec",
                                                time_div = time_div_midi,
                                                return_idxs=True,
                                                remove_drums=False)
    xml_piano_roll, scoreidx = pt.utils.compute_pianoroll(part_na,
                                                time_unit = "beat",
                                                time_div = time_div_xml,
                                                return_idxs = True,
                                                remove_drums=False)
    
    plot_array = np.zeros((128*2+50, length_of_pianorolls+800))
    dense_midi_pr = midi_piano_roll.todense()#[:,:time_size]
    dense_midi_pr[dense_midi_pr>0] = 1
    plot_array[:128,:xml_piano_roll.shape[1]] = xml_piano_roll.todense()
    plot_array[50+128:,:midi_piano_roll.shape[1]] = dense_midi_pr

    f, axs = plt.subplots(1,1,figsize=(100, 10))
    axs.matshow(plot_array, aspect = "auto",  origin='lower')
    hexadecimal_alphabets = '0123456789ABCDEF'


    colors1 = ["#00FF00" for i in range(40)]
    colors2 = ["#0000FF" for i in range(40)]
    colors3 = ["#FF0000" for i in range(40)]
    colors4 = ["#FF00FF" for i in range(40)]


    n1_but_not_n2 = [al for al in alignment1 if not al in alignment2 ]
    n2_but_not_n1 = [al for al in alignment2 if not al in alignment1 ]
            
        
    score_dict = dict()
    for note, pos in zip(part_na, scoreidx):
        score_dict[note["id"]] = pos
    perf_dict = dict()
    for note, pos in zip(ppart_na, perfidx):
        perf_dict[note["id"]] = pos
    for i, line in enumerate(alignment1):
        if line["label"]=="match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot([score_pos[1], perf_pos[1]], [ score_pos[0], 128+50 +perf_pos[0]],'o-', lw=2, c = colors1[i%40])
    for i, line in enumerate(alignment2):
        if line["label"]=="match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot([score_pos[1], perf_pos[1]], [ score_pos[0], 128+50 +perf_pos[0]],'o-', lw=2, c = colors2[i%40])
    for i, line in enumerate(n1_but_not_n2):
        if line["label"]=="match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot([score_pos[1], perf_pos[1]], [ score_pos[0], 128+50 +perf_pos[0]],'o-', lw=2, c = colors3[i%40])
    for i, line in enumerate(n2_but_not_n1):
        if line["label"]=="match":
            perf_pos = perf_dict[line["performance_id"]]
            score_pos = score_dict[line["score_id"]]
            axs.plot([score_pos[1], perf_pos[1]], [ score_pos[0], 128+50 +perf_pos[0]],'o-', lw=2, c = colors4[i%40])



    if save_file:
        plt.savefig(fname+".png")
        plt.close(f)
    else:
        plt.show()





# def transform_labels(ids: List[str], 
#                      prediction: List[dict],
#                      ground_truth: List[dict], 
#                      field: str) -> (List[str], List[str]):
#     """
#     Parameters
#     ----------
#     ids: List of note ids
#     prediction: List of dictionaries containing the predicted alignments
#     ground_truth: List of dictionaries containing the ground truth alignments
#     field: which note ids to evaluate (score_id or performance_id)

#     Returns
#     -------
#     List of predicted labels, List of ground truth labels
#     """

#     assert field in ['score_id', 'performance_id']

#     y_pred, y_gt = [], []
#     for id in ids:

#         pred = list(filter(lambda x: field in x and str(x[field]) == id, prediction))
#         gt = list(filter(lambda x: field in x and str(x[field]) == id, ground_truth))
#         try:
#             assert len(pred) == len(gt) == 1
#             pred = pred[0]

#         except:
#             print(f'Missing note in predictions {id}')

#             if field == 'score_id':
#                 pred = dict(label='deletion',
#                             score_id=id)
#             elif field == 'performance_id':
#                 pred = dict(label='insertion',
#                             performance_id=id)
#             # import pdb
#             # pdb.set_trace()
            
#         gt = gt[0]

#         gt_label = gt['label']
#         pred_label = pred['label']

#         if gt_label == pred_label == 'match':
#             # matched to different note in performance -> mismatch
#             if pred['performance_id'] != gt['performance_id']:
#                 pred_label = 'mismatch'

#         y_gt.append(gt_label)
#         y_pred.append(pred_label)

#     return y_pred, y_gt









# def alignment_confusion_matrices(score_ids: List[str],
#                                  performance_ids: List[str],
#                                  prediction: List[dict],
#                                  ground_truth: List[dict],
#                                  normalize: str or None=None) -> (np.ndarray, np.ndarray):
#     """

#     Parameters
#     ----------
#     score_ids: List of score note ids
#     performance_ids: List of performance note ids
#     prediction: List of dictionaries containing the predicted alignments
#     ground_truth: List of dictionaries containing the ground truth alignments

#     Returns
#     -------
#     2 3x3 numpy array containing the confusion matrices for score and performance
#     (note that the last row (mismatch) should always be 0 for each column since does not occur in the ground truth
#     """
#     if normalize is not None:
#         assert normalize in ['true', 'pred', 'all']

#     y_pred_score, y_gt_score = transform_labels(score_ids, prediction, ground_truth, field='score_id')
#     y_pred_perf, y_gt_perf = transform_labels(performance_ids, prediction, ground_truth, field='performance_id')

#     conf_matrix_score = confusion_matrix(y_gt_score, y_pred_score,
#                                          labels=['match', 'deletion',  'mismatch'],
#                                          normalize=normalize)
#     conf_matrix_perf = confusion_matrix(y_gt_perf, y_pred_perf,
#                                         labels=['match', 'insertion', 'mismatch'],
#                                         normalize=normalize)

#     return conf_matrix_score, conf_matrix_perf