#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contain methods for (repeat) structure identification
"""
import partitura as pt
import numpy as np
from ..dp.nwtw import BoundedSmithWaterman
import matplotlib.pyplot as plt

class RepeatIdentifier(object):
    """
    method wrapper to compute the most likely and musically sensible
    (starts at the start, ends at the end, repeats a valid number of times) 
    sequence of score sections that correspond to an input performance.
    
    """

    def __init__(self, 
                 max_number_of_paths = 100000):
        self.directions =  np.array([[1,1],[1,0]])
        self.dists = np.array([1,1])
        self.matcher = BoundedSmithWaterman(threshold = 0.5,
                                       gamma_penalty = -1,
                                       gamma_match = 1,
                                       directions = self.directions,
                                       directional_distances = self.dists,
                                       gain_max_val = 10)
        self.max_number_of_paths = max_number_of_paths


    def prepare_score(self, score):
        # score representation
        # score = pt.load_musicxml(score_path)
        part = pt.score.merge_parts(score.parts)
        score_note_array = part.note_array()
        unique_onsets = np.unique(score_note_array["onset_beat"])
        # # create pitch set representation
        score_pitches_at_onsets = list()
        for onset in unique_onsets:
            score_pitches_at_onsets.append(set(score_note_array[score_note_array["onset_beat"] == onset]["pitch"]))
        return part, unique_onsets, score_pitches_at_onsets

    def prepare_performance(self, perf):
        # perf = pt.load_performance_midi(perf_path)
        # performance representation
        perf_note_array = perf.note_array()
        perf_pitches = perf_note_array["pitch"]
        return perf_note_array, perf_pitches
    
    def extract_segments(self, part, 
                        unique_onsets,
                        verbose = False):
        if verbose:
            print("*"*20)
            print("SEGMENTS")
            print(pt.score.pretty_segments(part))
            print("*"*20)
        # segments and paths
        pt.score.add_segments(part, force_new = True)
        segments = pt.score.get_segments(part)
        paths = pt.score.get_paths(part)
        # map segments to the score input to the alignment
        segment_onset_idx = {}
        segment_onsets = {}
        for seg_id in segments.keys():
            start = part.beat_map(segments[seg_id].start.t)
            end = part.beat_map(segments[seg_id].end.t)
            onset_mask = np.where(np.all((unique_onsets >= start, unique_onsets < end), axis = 0))
            segment_onset_idx[seg_id] = np.arange(len(unique_onsets))[onset_mask[0]]
            segment_onsets[seg_id] = unique_onsets[onset_mask[0]]

        if len(paths) > self.max_number_of_paths:
            pt.score.add_segments(part, force_new = True)
            max_paths_w_leap = pt.score.get_paths(part, 
                                    no_repeats= False, 
                                    all_repeats= True, 
                                    ignore_leap_info= False)
            pt.score.add_segments(part, force_new = True)
            max_paths_wt_leap = pt.score.get_paths(part, 
                                    no_repeats=False, 
                                    all_repeats=True, 
                                    ignore_leap_info=True)
            pt.score.add_segments(part, force_new = True)
            min_paths = pt.score.get_paths(part, 
                                    no_repeats=True, 
                                    all_repeats=False, 
                                    ignore_leap_info=True)
            
            paths = max_paths_w_leap + max_paths_wt_leap + min_paths
        return paths, segment_onset_idx, segment_onsets
    
    def partial_backtrack(self, 
                        B,
                        cost,
                        starting_n, 
                        starting_m,
                        ending_n,
                        ending_m,
                        directions = np.array([[1, 0],[1, 1],[0, 1]]),
                        unmatched_cost = -10,
                        matched_cost = 10
                        ):
        n = ending_n #N - 1
        m = ending_m #M - 1
        step = [m, n]
        path = [step]
        final_cost = cost[m, n]
        current_cost = cost[m, n]
        # initialize boolean variables for stopping decoding
        costs = [current_cost]
        n_start_below_1 = 0
        close_to_start = 10
        while (n > max(starting_n-1,0)) and \
            (m > max(starting_m-1,0)) and \
            (current_cost > 0) and \
            not (abs(n - starting_n) < close_to_start and n_start_below_1 > 2):# stop iterating if cost is low close to the start
            backtracking_pointer = B[m, n]
            bt_vector = directions[backtracking_pointer]
            m -= bt_vector[0]
            n -= bt_vector[1]
            step = [m, n]
            # append next step to the path
            path.append(step)
            # update current cost
            current_cost = cost[m, n]
            costs.append(current_cost)
            if current_cost < 1:
                n_start_below_1 += 1
            else:
                n_start_below_1 = 0

        # remove path appendix of cost below 1 if it is in the beginning
        if n_start_below_1 > 0:
            path = path[:-n_start_below_1]

        # check if the path roughly covers the segment
        if abs(n - starting_n) > close_to_start or len(path) <= 1:
            output_path = None
            path_cost = unmatched_cost

        elif len(path) > 10 and max(costs) > 7:
            output_path = np.array(path, dtype=np.int32)[-2::-1] # remove last path element
            path_cost = matched_cost

        elif len(path) <= 10:
            output_path = np.array(path, dtype=np.int32)[-2::-1] # remove last path element
            path_cost = matched_cost

        else:
            output_path = None
            path_cost = unmatched_cost

        return output_path, path_cost

    def compute_path_gain(
            self,
            cost,
            path,
            backtracking,
            segment_onset_idx,
            directions
        ):
        ending_m = cost.shape[0] - 1 # the end of the performance
        starting_m = 0
        path_gain = 0
        full_path = np.array([[cost.shape[0],cost.shape[1]]])
        full_path_list = list()
        for seg_id in path.path[::-1]:
            starting_n = segment_onset_idx[seg_id][0]
            ending_n = segment_onset_idx[seg_id][-1]
            partial_path, partial_gain = self.partial_backtrack(backtracking, 
                                                                cost, 
                                                                starting_n, 
                                                                starting_m,
                                                                ending_n,
                                                                ending_m,
                                                                directions,
                                                                unmatched_cost = -10)
            path_gain += partial_gain
            if partial_path is not None:
                ending_m = partial_path[0,0] - 1
                full_path = np.row_stack((partial_path, full_path))
                full_path_list.append((seg_id, partial_path))
    
        return path_gain, full_path, full_path_list[::-1]


    def __call__(self, score, performance, 
                 verbose = False, plot = False):
        """
        Parameters
        ----------
        score: object
            a score object
        performance : object
            a performance object
        """

        part, unique_onsets, score_pitches_at_onsets = self.prepare_score(score)
        perf_note_array, perf_pitches = self.prepare_performance(performance)
        # compute coste
        cost, backtracking = self.matcher(perf_pitches, score_pitches_at_onsets)

        # figure out the possible score segments
        paths, segment_onset_idx, segment_onsets = self.extract_segments(part, 
                                                                        unique_onsets, 
                                                                        verbose = verbose)
        
        if len(paths) < 2:
            print("no structural variations!")
            print("*"*20)
            return None

        path_gains = {}
        for path in paths:
            path_string = "".join(path.path)
            if verbose:
                print("Testing path:", path_string)
            path_gain, full_path, full_path_list = self.compute_path_gain(cost,path, 
                                        backtracking,
                                        segment_onset_idx,
                                        directions = self.directions)
            path_gains[path_gain] = (path_string, path)
            

        max_gain = max([k for k in path_gains.keys()])
        found_path, found_path_object = path_gains[max_gain] 
        if verbose:
            print("best fitting path: ", found_path)

        if plot:
            colors = ["r", "g", "b"]
            plt.imshow(cost, aspect="auto")
            for pp_no, pp_val in enumerate(full_path_list):
                partial_path_id, partial_path = pp_val
                plt.plot(partial_path[:,1], partial_path[:,0], c = colors[pp_no%3])
                plt.text(partial_path[0,1], partial_path[0,0], partial_path_id, c = colors[pp_no%3], fontsize=12)
            plt.savefig(performance + "_" + path_string + "_.png")
            plt.close()
        path_gains[path_gain] = path_string

        return found_path, found_path_object
    
