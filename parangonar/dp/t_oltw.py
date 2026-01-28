#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains TempoOLTW.
"""

from typing import Optional, List, Callable, Any, Tuple, Generator
import numpy as np
from numpy.typing import NDArray
from enum import IntEnum
from queue import Queue
from ..dp.metrics import tempo_and_pitch_metric
import progressbar

def accumulate_tester():
    score = [[0, {1, 2}], [1, {3, 4}], [2, {3, 4}], [3, {3, 4}]]  # onset_s, pitch set
    perf = [
        [0, 1],
        [0.1, 2],
        [2, 3],
        [2.9, 4],
        [4, 3],
        [4.02, 4],
        [6, 4],
        [6.01, 3],
    ]

    M = len(perf)
    N = len(score)
    # the distance matrix is initialized with INFINITY
    D = np.ones((M + 1, N + 1), dtype=float) * np.inf
    D[0, 0] = 0
    # Tempo collector
    init_tempo = 2  # sec / beat
    Tempos = np.ones((M, N), dtype=float) * init_tempo
    directions = np.array([[1, 0], [1, 1], [0, 1]])

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            mincost = np.inf
            besttempo = init_tempo
            for directionsidx, direction in enumerate(directions):
                (istep, jstep) = direction
                previ = i - istep
                prevj = j - jstep

                if previ > 0 and prevj > 0:
                    prev_onset_p = perf[previ - 1][0]
                    prev_onset_s = score[prevj - 1][0]
                    tempo = Tempos[previ - 1, prevj - 1]
                else:
                    prev_onset_p = 0  # smart?
                    prev_onset_s = 0  # smart?
                    tempo = init_tempo

                dist, tempo_new = tempo_and_pitch_metric(
                    pitch_set_s=score[j - 1][1],
                    pitch_p=perf[i - 1][1],
                    onset_s=score[j - 1][0],
                    onset_p=perf[i - 1][0],
                    prev_onset_s=prev_onset_s,
                    prev_onset_p=prev_onset_p,
                    tempo=tempo,  # sec / beat
                    time_weight=0.5,
                    tempo_factor=0.1,
                )

                if previ >= 0 and prevj >= 0:
                    cost = D[previ, prevj] + dist
                    if cost < mincost:
                        mincost = cost
                        besttempo = tempo_new

            D[i, j] = mincost
            Tempos[i - 1, j - 1] = besttempo

    return D, Tempos


class Direction(IntEnum):
    REF = 0
    INPUT = 1
    BOTH = 2

    def toggle(self) -> "Direction":
        return Direction(self ^ 1) if self != Direction.BOTH else Direction.INPUT


class T_OLTW(object):
    """
    On-line Dynamic Time Warping (OLTW) algorithm for aligning a input sequence to a reference sequence.

    Adapted from: https://github.com/laurenceyoon/python-match

    Parameters
    ----------
    reference_features : list
        list of score features: pitch set and onset

    queue : Queue
        A queue for storing the input features, pitch and onset

    window_size : int
        Size of the window (in steps) of the cost matrix.

    max_run_count : int
        Maximum number of times the class can run in the same direction.

    hop_size : int
        number of seq items that get added at step

    directional_weights: np.ndarray
        weights associated with each of the three possible steps

    Attributes
    ----------
    warping_path : np.ndarray [shape=(2, T)]
        Warping path with pairs of indices of the reference and input features.
        where warping_path[0] is the index of the reference feature and
        warping_path[1] is the index of the input feature.
    """

    def __init__(
        self,
        reference_features: Optional[List[Any]] = None,
        queue: Optional[Queue] = None,
        window_size: int = 10,  # shape of the acc cost matric
        max_run_count: int = 100,  # maximal number of steps
        hop_size: int = 1,  # number of seq items that get added at step
        directional_weights: np.ndarray = np.array([1, 1, 1]),  # down, diag, right
        directions: np.ndarray = np.array([[1, 0], [1, 1], [0, 1]]),
        metric: Callable = tempo_and_pitch_metric,
        init_tempo: float = 1,  # sec / beat
        time_weight: float = 0.5,
        tempo_factor: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.queue = queue
        self.metric = metric
        self.directional_weights = directional_weights
        self.directions = directions
        if reference_features is not None:
            self.set_feature_arrays(reference_features)
        else:
            self.reference_features = None
            self.N_ref = None
            self.input_features = None

        self.init_tempo = init_tempo
        self.time_weight = time_weight
        self.tempo_factor = tempo_factor
        self.w = window_size
        self.max_run_count = max_run_count
        self.hop_size = hop_size
        self.initialize()

    def set_feature_arrays(self, reference_features: List[Any]) -> None:
        self.reference_features = reference_features
        self.N_ref = len(reference_features)
        # prepend the first onset
        self.reference_features.insert(0, self.reference_features[0])
        self.input_features = list()

    def initialize(self) -> None:
        self.ref_pointer = 0
        self.input_pointer = 0
        self.run_count = 0
        self.previous_direction = None
        self.wp = np.array([[-1, -1]]).T  # [shape=(2, T)]

        self.ref_pointer += self.w  # window of ref shifted at start
        self.acc_dist_matrix = np.full((self.w, self.w), np.inf)
        self.acc_dist_matrix[0, :] = 0
        self.acc_len_matrix = np.zeros((self.w, self.w))
        self.queue_non_empty = True

    @property
    def warping_path(self) -> np.ndarray:  # [shape=(2, T)]
        return self.wp[:, 1:]

    def offset(self) -> np.ndarray:
        offset_x = max(self.ref_pointer - self.w, 0)
        offset_y = max(self.input_pointer - self.w, 0)
        return np.array([offset_x, offset_y])

    def update_ref_direction(
        self,
        new_tempo_acc,
        new_acc,
        new_len_acc,
        wx,
        wy,
        d,
        score_features,
        perf_features,
    ):
        update_x0 = wx - d
        for i in range(d):  # score direction
            for j in range(wy):  # performance direction
                mincost = np.inf
                besttempo = self.init_tempo
                bestpath = 1
                if j == 0:
                    prev_onset_p = perf_features[j + 1][
                        0
                    ]  # previous onset in p direction
                    prev_onset_s = score_features[i][0]
                    tempo = new_tempo_acc[update_x0 + i - 1, j]
                    local_dist, new_tempo = tempo_and_pitch_metric(
                        pitch_set_s=score_features[i + 1][1],
                        pitch_p=perf_features[j + 1][1],
                        onset_s=score_features[i + 1][0],
                        onset_p=perf_features[j + 1][0],
                        prev_onset_s=prev_onset_s,
                        prev_onset_p=prev_onset_p,
                        tempo=tempo,  # sec / beat
                        time_weight=self.time_weight,
                        tempo_factor=self.tempo_factor,
                    )

                    new_acc[update_x0 + i, j] = (
                        local_dist * self.directional_weights[0]
                        + new_acc[update_x0 + i - 1, j]
                    )
                    new_len_acc[update_x0 + i, j] = (
                        1 + new_len_acc[update_x0 + i - 1, j]
                    )
                    new_tempo_acc[update_x0 + i, j] = new_tempo

                else:
                    for d_idx, direction in enumerate(self.directions):
                        (istep, jstep) = direction
                        previ = i - istep
                        prevj = j - jstep
                        prev_onset_p = perf_features[prevj + 1][
                            0
                        ]  # previous onset in p direction
                        prev_onset_s = score_features[previ + 1][0]
                        tempo = new_tempo_acc[update_x0 + previ, prevj]
                        prev_path_len = new_len_acc[update_x0 + previ, prevj]
                        local_dist, new_tempo = tempo_and_pitch_metric(
                            pitch_set_s=score_features[i + 1][1],
                            pitch_p=perf_features[j + 1][1],
                            onset_s=score_features[i + 1][0],
                            onset_p=perf_features[j + 1][0],
                            prev_onset_s=prev_onset_s,
                            prev_onset_p=prev_onset_p,
                            tempo=tempo,  # sec / beat
                            time_weight=self.time_weight,
                            tempo_factor=self.tempo_factor,
                        )

                        cost = (
                            new_acc[update_x0 + previ, prevj]
                            + local_dist * self.directional_weights[d_idx]
                        )
                        if cost < mincost:
                            mincost = cost
                            besttempo = new_tempo
                            bestpath = 1 + prev_path_len

                    new_acc[update_x0 + i, j] = mincost
                    new_len_acc[update_x0 + i, j] = bestpath
                    new_tempo_acc[update_x0 + i, j] = besttempo

        return new_acc, new_len_acc, new_tempo_acc

    def update_input_direction(
        self,
        new_tempo_acc,
        new_acc,
        new_len_acc,
        wx,
        wy,
        d,
        score_features,
        perf_features,
    ):
        update_y0 = wy - d
        for i in range(wx):  # score direction
            for j in range(d):  # perfpormance direction
                mincost = np.inf
                besttempo = self.init_tempo
                bestpath = 1
                if i == 0:
                    prev_onset_p = perf_features[j][0]  # previous onset in p direction
                    prev_onset_s = score_features[i + 1][0]
                    tempo = new_tempo_acc[i, update_y0 + j - 1]
                    local_dist, new_tempo = tempo_and_pitch_metric(
                        pitch_set_s=score_features[i + 1][1],
                        pitch_p=perf_features[j + 1][1],
                        onset_s=score_features[i + 1][0],
                        onset_p=perf_features[j + 1][0],
                        prev_onset_s=prev_onset_s,
                        prev_onset_p=prev_onset_p,
                        tempo=tempo,  # sec / beat
                        time_weight=self.time_weight,
                        tempo_factor=self.tempo_factor,
                    )

                    new_acc[i, update_y0 + j] = (
                        local_dist * self.directional_weights[2]
                        + new_acc[i, update_y0 - 1 + j]
                    )
                    new_len_acc[i, update_y0 + j] = (
                        1 + new_len_acc[i, update_y0 - 1 + j]
                    )
                    new_tempo_acc[i, update_y0 + j] = new_tempo

                else:
                    for d_idx, direction in enumerate(self.directions):
                        (istep, jstep) = direction
                        previ = i - istep
                        prevj = j - jstep
                        prev_onset_p = perf_features[prevj + 1][
                            0
                        ]  # previous onset in p direction
                        prev_onset_s = score_features[previ + 1][0]
                        tempo = new_tempo_acc[previ, update_y0 + prevj]
                        prev_path_len = new_len_acc[previ, update_y0 + prevj]
                        local_dist, new_tempo = tempo_and_pitch_metric(
                            pitch_set_s=score_features[i + 1][1],
                            pitch_p=perf_features[j + 1][1],
                            onset_s=score_features[i + 1][0],
                            onset_p=perf_features[j + 1][0],
                            prev_onset_s=prev_onset_s,
                            prev_onset_p=prev_onset_p,
                            tempo=tempo,  # sec / beat
                            time_weight=self.time_weight,
                            tempo_factor=self.tempo_factor,
                        )

                        cost = (
                            new_acc[previ, update_y0 + prevj]
                            + local_dist * self.directional_weights[d_idx]
                        )
                        if cost < mincost:
                            mincost = cost
                            besttempo = new_tempo
                            bestpath = 1 + prev_path_len

                    new_acc[i, update_y0 + j] = mincost
                    new_len_acc[i, update_y0 + j] = bestpath
                    new_tempo_acc[i, update_y0 + j] = besttempo

        return new_acc, new_len_acc, new_tempo_acc

    def update_both_direction(
        self,
        new_tempo_acc,
        new_acc,
        new_len_acc,
        wx,
        wy,
        d,
        score_features,
        perf_features,
    ):
        for i in range(wx):  # score direction
            for j in range(wy):  # perfpormance direction
                mincost = np.inf
                besttempo = self.init_tempo
                bestpath = 1
                if i == 0 and j >= wy - d:
                    prev_onset_p = perf_features[j][0]  # previous onset in p direction
                    prev_onset_s = score_features[i + 1][0]
                    tempo = new_tempo_acc[i, j - 1]
                    local_dist, new_tempo = tempo_and_pitch_metric(
                        pitch_set_s=score_features[i + 1][1],
                        pitch_p=perf_features[j + 1][1],
                        onset_s=score_features[i + 1][0],
                        onset_p=perf_features[j + 1][0],
                        prev_onset_s=prev_onset_s,
                        prev_onset_p=prev_onset_p,
                        tempo=tempo,  # sec / beat
                        time_weight=self.time_weight,
                        tempo_factor=self.tempo_factor,
                    )

                    new_acc[i, j] = (
                        local_dist * self.directional_weights[2] + new_acc[i, j - 1]
                    )
                    new_len_acc[i, j] = 1 + new_len_acc[i, j - 1]
                    new_tempo_acc[i, j] = new_tempo

                elif i >= wx - d and j == 0:
                    prev_onset_p = perf_features[j + 1][
                        0
                    ]  # previous onset in p direction
                    prev_onset_s = score_features[i][0]
                    tempo = new_tempo_acc[i - 1, j]
                    local_dist, new_tempo = tempo_and_pitch_metric(
                        pitch_set_s=score_features[i + 1][1],
                        pitch_p=perf_features[j + 1][1],
                        onset_s=score_features[i + 1][0],
                        onset_p=perf_features[j + 1][0],
                        prev_onset_s=prev_onset_s,
                        prev_onset_p=prev_onset_p,
                        tempo=tempo,  # sec / beat
                        time_weight=self.time_weight,
                        tempo_factor=self.tempo_factor,
                    )

                    new_acc[i, j] = (
                        local_dist * self.directional_weights[0] + new_acc[i - 1, j]
                    )
                    new_len_acc[i, j] = 1 + new_len_acc[i - 1, j]
                    new_tempo_acc[i, j] = new_tempo

                elif i >= wx - d or j >= wy - d:
                    for d_idx, direction in enumerate(self.directions):
                        (istep, jstep) = direction
                        previ = i - istep
                        prevj = j - jstep
                        prev_onset_p = perf_features[prevj + 1][
                            0
                        ]  # previous onset in p direction
                        prev_onset_s = score_features[previ + 1][0]
                        tempo = new_tempo_acc[previ, prevj]
                        prev_path_len = new_len_acc[previ, prevj]
                        local_dist, new_tempo = tempo_and_pitch_metric(
                            pitch_set_s=score_features[i + 1][1],
                            pitch_p=perf_features[j + 1][1],
                            onset_s=score_features[i + 1][0],
                            onset_p=perf_features[j + 1][0],
                            prev_onset_s=prev_onset_s,
                            prev_onset_p=prev_onset_p,
                            tempo=tempo,  # sec / beat
                            time_weight=self.time_weight,
                            tempo_factor=self.tempo_factor,
                        )

                        cost = (
                            new_acc[previ, prevj]
                            + local_dist * self.directional_weights[d_idx]
                        )
                        if cost < mincost:
                            mincost = cost
                            besttempo = new_tempo
                            bestpath = 1 + prev_path_len

                    new_acc[i, j] = mincost
                    new_len_acc[i, j] = bestpath
                    new_tempo_acc[i, j] = besttempo

        return new_acc, new_len_acc, new_tempo_acc

    def update_cost_matrix(self, direction):
        # local cost matrix
        x, y = self.ref_pointer, self.input_pointer
        wx, wy = min(self.w, x), min(self.w, y)
        d = self.hop_size
        # storage matrices
        new_acc = np.full((wx, wy), np.inf, dtype=np.float32)
        new_len_acc = np.zeros((wx, wy))
        new_tempo_acc = np.full((wx, wy), self.init_tempo, dtype=float)

        if direction is Direction.REF:
            new_acc[:-d, :] = self.acc_dist_matrix[d:]
            new_len_acc[:-d, :] = self.acc_len_matrix[d:]
            new_tempo_acc[:-d, :] = self.acc_tempo_matrix[d:, :]
            x_seg = self.reference_features[x - d : x + 1]  # [d + 1]
            y_seg = self.input_features[y - wy : y + 1]  # [wy + 1]
            # x_seg = self.reference_features[x - d : x]  # [d, 12]
            # y_seg = self.input_features[y - wy : y]  # [wy, 12]

            new_acc, new_len_acc, new_tempo_acc = self.update_ref_direction(
                new_tempo_acc, new_acc, new_len_acc, wx, wy, d, x_seg, y_seg
            )

        elif direction is Direction.INPUT:
            overlap_y = wy - d
            new_acc[:, :-d] = self.acc_dist_matrix[:, -overlap_y:]
            new_len_acc[:, :-d] = self.acc_len_matrix[:, -overlap_y:]
            new_tempo_acc[:, :-d] = self.acc_tempo_matrix[:, -overlap_y:]
            # x_seg = self.reference_features[x - wx : x]  # [wx, 12]
            # y_seg = self.input_features[y - d : y]  # [d, 12]
            x_seg = self.reference_features[x - wx : x + 1]  # [wx + 1]
            y_seg = self.input_features[y - d : y + 1]  # [d + 1]
            new_acc, new_len_acc, new_tempo_acc = self.update_input_direction(
                new_tempo_acc, new_acc, new_len_acc, wx, wy, d, x_seg, y_seg
            )

        elif direction is Direction.BOTH:  # input, BOTH
            overlap_y = wy - d
            new_acc[:-d, :-d] = self.acc_dist_matrix[d:, -overlap_y:]
            new_len_acc[:-d, :-d] = self.acc_len_matrix[d:, -overlap_y:]
            new_tempo_acc[:-d, :-d] = self.acc_tempo_matrix[d:, -overlap_y:]
            # x_seg = self.reference_features[x - wx : x]  # [wx, 12]
            # y_seg = self.input_features[y - wy : y]  # [wy, 12]
            x_seg = self.reference_features[x - wx : x + 1]  # [wx + 1]
            y_seg = self.input_features[y - wy : y + 1]  # [wy  + 1]
            new_acc, new_len_acc, new_tempo_acc = self.update_both_direction(
                new_tempo_acc, new_acc, new_len_acc, wx, wy, d, x_seg, y_seg
            )

        self.acc_dist_matrix = new_acc
        self.acc_len_matrix = new_len_acc
        self.acc_tempo_matrix = new_tempo_acc

    def first_cost_matrix(self):
        # local cost matrix
        x, y = self.ref_pointer, self.input_pointer  # should be window, hop_size
        wx, wy = x, y + 1  # append an extra column
        d = self.hop_size
        # storage matrices
        new_acc = np.full((wx, wy), np.inf, dtype=np.float32)
        new_acc[0, 0] = 0  # starting point in corner
        new_len_acc = np.zeros((wx, wy))
        new_tempo_acc = np.full((wx, wy), self.init_tempo, dtype=float)

        # copy the first element and extract overlapping feature vectors
        self.input_features.insert(0, self.input_features[0])
        x_seg = self.reference_features[x - wx : x + 1]  # [wx + 1]
        y_seg = self.input_features[y - d : y + 1]  # [d + 1]

        new_acc, new_len_acc, new_tempo_acc = self.update_input_direction(
            new_tempo_acc, new_acc, new_len_acc, wx, wy, d, x_seg, y_seg
        )

        self.acc_dist_matrix = new_acc[:, 1:]
        self.acc_len_matrix = new_len_acc[:, 1:]
        self.acc_tempo_matrix = new_tempo_acc[:, 1:]

    def select_candidate(self):
        norm_x_edge = self.acc_dist_matrix[-1, :] / self.acc_len_matrix[-1, :]
        norm_y_edge = self.acc_dist_matrix[:, -1] / self.acc_len_matrix[:, -1]
        cat = np.concatenate((norm_x_edge, norm_y_edge))

        min_idx = np.argmin(cat)
        offset = self.offset()
        if min_idx < len(norm_x_edge):
            self.candidate = np.array([self.ref_pointer - offset[0] - 1, min_idx])
        else:
            self.candidate = np.array(
                [min_idx - len(norm_x_edge), self.input_pointer - offset[1] - 1]
            )

    def add_candidate_to_path(self):
        new_coordinate = np.expand_dims(
            self.offset() + self.candidate, axis=1
        )  # [2, 1]
        self.wp = np.concatenate((self.wp, new_coordinate), axis=1)

    def select_next_direction(self):
        if self.input_pointer < self.w:
            next_direction = Direction.INPUT
        elif self.run_count > self.max_run_count:
            next_direction = self.previous_direction.toggle()
        elif self.ref_pointer > (self.N_ref - self.hop_size):
            next_direction = Direction.INPUT
        else:
            offset = self.offset()
            x0, y0 = offset[0], offset[1]
            ref_bool = self.candidate[0] == self.ref_pointer - x0 - 1
            input_bool = self.candidate[1] == self.input_pointer - y0 - 1
            if ref_bool and input_bool:
                next_direction = Direction.BOTH
            elif ref_bool:
                next_direction = Direction.REF
            else:
                assert input_bool
                next_direction = Direction.INPUT
        return next_direction

    def get_new_input(self):
        try:
            input_feature = self.queue.get(block=False)
            self.input_features += input_feature
            self.input_pointer += self.hop_size

        except:
            print("empty queue")
            self.queue_non_empty = False

    def is_still_following(self):
        is_still_following = self.ref_pointer <= self.N_ref
        return is_still_following and self.queue_non_empty

    def handle_direction(self, direction):
        # get fitting features/windows
        if direction is not Direction.REF:
            self.get_new_input()
        if direction is not Direction.INPUT:
            self.ref_pointer += self.hop_size

        # accumulate run count
        if direction == self.previous_direction:
            self.run_count += 1
        else:
            self.run_count = 1
        self.previous_direction = direction

    def handle_first_input(self):
        direction = self.select_next_direction()
        self.handle_direction(direction)
        self.first_cost_matrix()
        self.select_candidate()
        self.add_candidate_to_path()

    def run(self, verbose: bool = False) -> np.ndarray:
        if verbose:
            print("Start running OLTW")
        self.initialize()
        self.handle_first_input()
        direction = self.select_next_direction()
        self.handle_direction(direction)
        while self.is_still_following():
            self.update_cost_matrix(direction)
            self.select_candidate()
            self.add_candidate_to_path()
            # select direction and
            direction = self.select_next_direction()
            if verbose:
                print("ACC DIST \n", self.acc_dist_matrix)
                print("ACC LEN \n", self.acc_len_matrix)
                print("TEMPO \n", self.acc_tempo_matrix)
                print(
                    "RECENT WARPING PATH (top s, bottom p) \n",
                    self.warping_path[:, -3:],
                )
                print("NEXT DIRECTION \n", direction)
                print("RUN COUNT\n", self.run_count)
                print("*" * 50)
            self.handle_direction(direction)

        if verbose:
            print("... and we're done.")
        return self.warping_path
    



class SLT_OLTW(object):
    """
    Single Loop T_OLTW version refactored
    for update with __call__ and .run() API.
    inspired by Matchmaker OnlineTimeWarpingArzt.
    """

    def __init__(
        self,
        reference_features: Optional[List[Any]] = None,
        queue: Optional[Queue] = None,
        window_size: int = 10,  # shape of the acc cost matric
        max_run_count: int = 100,  # maximal number of steps
        directional_weights: np.ndarray = np.array([1, 1, 1]),  # down, diag, right
        directions: np.ndarray = np.array([[1, 0], [1, 1], [0, 1]]),
        cdist_metric: Callable = tempo_and_pitch_metric,
        init_tempo: float = 1,  # sec / beat
        time_weight: float = 0.5,
        tempo_factor: float = 0.1,
    ) -> None:
        self.queue = queue
        self.cdist_metric = cdist_metric
        self.directional_weights = directional_weights
        self.directions = directions
        if reference_features is not None:
            self.set_feature_arrays(reference_features)
        else:
            self.reference_features = None
            self.N_ref = None
            self.input_features = None
        self.window_size = window_size
        self.max_run_count = max_run_count
        self.init_tempo = init_tempo
        self.time_weight = time_weight
        self.tempo_factor = tempo_factor
        self.initialize()

    def set_feature_arrays(self, reference_features: List[Any]) -> None:
        self.reference_features = reference_features
        self.N_ref = len(reference_features)
        self.input_features = list()

    def initialize(self) -> None:
        self.init_position: int = 0 # score/reference 
        self.current_position: int = 0 # score/reference/playhead pointer
        
        self.positions: List[int] = []
        self._warping_path: List = []
        self.global_cost_matrix: NDArray[np.float32] = (
            np.full((self.N_ref + 1, 2), np.inf, dtype=np.float32) 
        ).astype(np.float32)
        self.global_path_length_matrix: NDArray[np.float32] = (
            np.zeros((self.N_ref + 1, 2), dtype=np.float32) 
        ).astype(np.float32)
        self.global_tempo_matrix: NDArray[np.float32] = (
            np.full((self.N_ref + 1, 2), self.init_tempo, dtype=np.float32) 
        ).astype(np.float32)
 
        self.input_index: int = 0 # input pointer
        if self.queue is not None:
            self.queue_non_empty: bool = True
        else:
            self.queue_non_empty: bool = False


    @property
    def warping_path(self) -> np.ndarray:
        wp = (np.array(self._warping_path).T).astype(np.int32) # [shape=(2, T)]
        return wp

    def get_window(self) -> Tuple[int, int]:
        w_size = self.window_size
        window_start = max(self.window_index - w_size, 0)
        window_end = min(self.window_index + w_size, self.N_ref)
        return window_start, window_end

    def __call__(self, input: np.ndarray) -> int:
        self.step(input)
        return self.current_position
    
    def update_loop(self,
                    window_start,
                    window_end,
                    min_index
        ):
        i = window_start # score idx
        j = self.input_index # performance idx
        min_cost = np.inf


        if i == j == 0:
            # default cost to get started
            self.global_cost_matrix[1, 1] = 0.0
            self.global_path_length_matrix[1, 1] = 0
            min_cost = 0
            min_index = 0
            
        while i < window_end:
            if not (i == j == 0):
                min_local_cost = np.inf
                min_local_tempo = self.init_tempo
                min_local_path_len = 1
                for d_idx, direction in enumerate(self.directions):
                    (istep, jstep) = direction
                    previ = i - istep
                    prevj = j - jstep
                    jlocal = 1 - jstep
                    input_f = self.input_features[j]
                    ref_f = self.reference_features[i]
                    prev_onset_p = self.input_features[prevj][0]  # previous onset in p direction
                    prev_onset_s = self.reference_features[previ][0]
                    tempo = self.global_tempo_matrix[previ + 1, jlocal]
                    prev_path_len = self.global_path_length_matrix[previ + 1, jlocal] # global matrices are shifted by 1 in score direction
                    prev_cost = self.global_cost_matrix[previ + 1, jlocal]

                    if prev_cost < np.inf:
                    
                        local_dist, new_tempo = self.cdist_metric(
                            pitch_set_s=ref_f[1],
                            pitch_p=input_f[1],
                            onset_s=ref_f[0],
                            onset_p=input_f[0],
                            prev_onset_s=prev_onset_s,
                            prev_onset_p=prev_onset_p,
                            tempo=tempo,  # sec / beat
                            time_weight=self.time_weight,
                            tempo_factor=self.tempo_factor
                            )
        
                        cost = (
                            prev_cost * prev_path_len 
                            + local_dist * self.directional_weights[d_idx]
                            ) / (prev_path_len + 1)

                        if cost < min_local_cost:
                            min_local_cost = cost
                            min_local_tempo = new_tempo
                            min_local_path_len = 1 + prev_path_len

                self.global_cost_matrix[i+1, 1] = min_local_cost
                self.global_path_length_matrix[i+1, 1] = min_local_path_len
                self.global_tempo_matrix[i+1, 1] = min_local_tempo

                if min_local_cost < min_cost:
                    min_cost = min_local_cost
                    min_index = i
            
            i = i + 1

        # rotate the columns for reuse
        self.global_cost_matrix[:, 0] = self.global_cost_matrix[:, 1]
        self.global_cost_matrix[:, 1] = np.inf
        self.global_path_length_matrix[:, 0] = self.global_path_length_matrix[:, 1]
        self.global_path_length_matrix[:, 1] = np.inf
        self.global_tempo_matrix[:, 0] = self.global_tempo_matrix[:, 1]
        self.global_tempo_matrix[:, 1] = self.init_tempo

        return min_index
        
    def step(self, input_features):
        """
        Update the current position and the warping path.
        """
        self.input_features += input_features 
        window_start, window_end = self.get_window()
        min_index = window_start

        min_index = self.update_loop(
            window_start=window_start,
            window_end=window_end,
            min_index=min_index,
        )

        # adapt current_position: do not go backwards,
        # but also go a maximum of self.max_run_count steps forward
        if self.input_index == 0:
            pass
        else:
            self.current_position = min(
                max(self.current_position, min_index),
                self.current_position + self.max_run_count
            )

        self._warping_path.append((self.current_position, self.input_index))
        # update input index
        self.input_index += 1
    
    def run(self) -> np.ndarray: 
        """
        Run the online alignment process in an offline loop.
        """
        self.initialize()
        if self.queue_non_empty:
            new_features = self.get_new_input()
            while self.is_still_following():
                # for offline usage
                self.step(new_features)
                new_features = self.get_new_input()
            return self.warping_path
        else:
            print("standalone offline run requires a queue")

    def get_new_input(self):
        """
        queue.get wrapper for graceful exit when used offline.
        """
        try:
            input_feature = self.queue.get(block=False)
            return input_feature

        except:
            print("empty queue")
            self.queue_non_empty = False
            return None
    
    def is_still_following(self):
        is_still_following = self.current_position < self.N_ref
        return is_still_following and self.queue_non_empty

    @property
    def window_index(self) -> int:
        return self.current_position









def testfeatures_t_oltw():
    score = [
        [0, {1, 2}],  # onset_s, pitch set
        [1, {3, 4}],
        [2, {3, 4}],
        [3, {3, 4}],
        [4, {3, 4}],
        [5, {3, 4}],
        [6, {3, 4}],
        [7, {3, 4}],
    ]
    perf = [[0, 1], [0.1, 2], [2, 3], [2.1, 4], [4, 3], [4.02, 4], [6, 4], [6.01, 3]]
    return score, perf


if __name__ == "__main__":
    RANGE_L = 6
    REPEATS = 3
    HOP_SIZE = 1
    WINDOW_SIZE = 3
    import copy

    r, t = testfeatures_t_oltw()
    queue1 = Queue()
    
    for tt in t:
        queue1.put([tt])

    o1 = T_OLTW(
        reference_features=copy.copy(r),
        queue=queue1,
        frame_per_seg=HOP_SIZE,
        window_size=WINDOW_SIZE,
        max_run_count=8,
        init_tempo=2,
        tempo_factor=0.1,
        time_weight=0.1,
        directional_weights=np.array([1.0, 1.0, 1.0]),
    )
    p1 = o1.run(verbose = False)
    print("path T_OLTW \n", p1)

    # D, T = accumulate_tester()
    queue2 = Queue()

    for tt in t:
        queue2.put([tt])

    o2 = SLT_OLTW(
        reference_features=copy.copy(r),
        queue=queue2,
        window_size=WINDOW_SIZE,
        max_run_count=8,
        init_tempo=2,
        tempo_factor=0.1,
        time_weight=0.1,
        directional_weights=np.array([1.0, 1.0, 1.0])
        )


    p2 = o2.run(verbose = False)
    print("path single loop T_OLTW \n", p2)