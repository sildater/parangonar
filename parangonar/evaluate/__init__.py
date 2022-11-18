#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module
evaluate note alignments symbolic music data.
"""
from typing import List

def fscore_alignments(prediction: List[dict], 
                        ground_truth: List[dict], 
                        types: List[str]) -> (float, float, float):
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

    return precision, recall, f_score









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