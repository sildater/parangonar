#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains online note matcher classes.
"""
import logging
from typing import List, Dict, Any, Optional, Callable, Set
from .. import ALIGNMENT_TRANSFORMER_CHECKPOINT
import numpy as np
from collections import defaultdict
from .pretrained_models import AlignmentTransformer
import torch
from .matchers import na_within
from partitura.utils.generic import interp1d
from ..dp.t_oltw import T_OLTW, SLT_OLTW
from ..dp.oltw import OLTW, SL_OLTW
from ..dp.metrics import bounded_recursion
from queue import Queue
from scipy import ndimage
from collections import deque

logger = logging.getLogger(__name__)



################################### TEMPO MODELS ###################################


class TempoModel(object):
    """
    Base class for synchronization models

    Attributes
    ----------
    """

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        init_perf_onset: float = 0,
        lookback: int = 1,
    ) -> None:
        self.lookback = lookback
        self.beat_period = init_beat_period
        self.prev_score_onsets = [init_score_onset - 2 * lookback]
        self.prev_perf_onsets = [init_perf_onset - 2 * lookback * self.beat_period]
        self.prev_perf_onsets_at_score_onsets = defaultdict(list)
        self.prev_perf_onsets_at_score_onsets[self.prev_score_onsets[-1]].append(
            self.prev_perf_onsets[-1]
        )
        self.est_onset = None
        self.score_perf_map = None
        # Count how many times has the tempo model been called
        self.counter = 0
        self.update(
            init_perf_onset - lookback * self.beat_period, init_score_onset - lookback
        )

    def predict(self, score_onset: float) -> float:
        self.est_onset = (
            self.score_perf_map(score_onset - (self.lookback + 1)) 
            + (self.lookback + 1) * self.beat_period
        )
        return self.est_onset

    def predict_ratio(self, score_onset: float, perf_onset: float) -> float:
        self.est_onset = (
            self.score_perf_map(score_onset - (self.lookback + 1))
            + (self.lookback + 1) * self.beat_period
        )
        error = perf_onset - self.est_onset
        offset_score = score_onset - self.prev_score_onsets[-1]
        if offset_score > 0.0:
            return error / (offset_score * self.beat_period)
        else:
            return error

    def update(self, performed_onset: float, score_onset: float) -> None:
        self.prev_perf_onsets_at_score_onsets[score_onset].append(performed_onset)
        if score_onset == self.prev_score_onsets[-1]:
            #     self.prev_perf_onsets[-1] = 4/5 * self.prev_perf_onsets[-1] + 1/5* performed_onset
            self.prev_perf_onsets[-1] = np.median(
                self.prev_perf_onsets_at_score_onsets[score_onset]
            )
        else:
            self.prev_score_onsets.append(score_onset)
            self.prev_perf_onsets.append(performed_onset)

        self.score_perf_map = interp1d(
            self.prev_score_onsets[-100:],
            self.prev_perf_onsets[-100:],
            fill_value="extrapolate",
        )
        self.beat_period = np.clip(
            (
                self.score_perf_map(score_onset)
                - self.score_perf_map(score_onset - self.lookback)
            )
            / self.lookback,
            0.1,
            10.0,
        )
        self.counter += 1


class DummyTempoModel(object):

    """
    Base class for synchronization models

    Attributes
    ----------
    """

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        init_perf_onset: float = 0,
        lookback: int = 1,
        func: Optional[Callable] = None,
    ) -> None:
        self.lookback = lookback
        self.beat_period = init_beat_period
        self.score_perf_map = func
        # Count how many times has the tempo model been called
        self.counter = 0

    def predict(self, score_onset: float) -> float:
        self.est_onset = self.score_perf_map(score_onset)
        return self.est_onset

    def update(self, performed_onset: float, score_onset: float) -> None:
        self.counter += 1


################################### ONLINE MATCHERS ###################################


class OnlineTransformerMatcher(object):
    def __init__(
        self,
        score_note_array_full: np.ndarray,
        token_number: int = 91,
        dim_model: int = 64,
        dim_class: int = 2,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        dropout_p: float = 0.1,
        init_beat_period: float = 0.5,
        lookback: int = 1,
    ) -> None:
        self.token_number = token_number
        self.dim_model = dim_model
        self.dim_class = dim_class
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        self.init_beat_period = init_beat_period
        self.lookback = lookback

        self.score_note_array_full = np.sort(score_note_array_full, order="onset_beat")
        self.first_p_onset = None
        self.tempo_model = None

        self._prev_performance_notes = list()
        self._prev_score_onset = None
        self._snote_aligned = set()
        self._pnote_aligned = set()
        self._pnote_aligned_pitch = list()
        self.alignment = []
        self.note_alignments = []
        self.prepare_score()
        self.prepare_model()
        self.initialize()

    def initialize(self):
        # alias and utils for matchmaker
        self.unique_onsets = self._unique_score_onsets
        self.N_ref = len(self.unique_onsets)
        self.current_position = 0
        self.input_index = 0
        self._warping_path = list()
        self.time_since_nn_update = 0
        self.stuck_with_no_options = 0

    def prepare_score(self):
        self.score_note_array_no_grace = self.score_note_array_full[
            self.score_note_array_full["is_grace"] == False
        ]
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            self.score_by_pitch[pitch] = self.score_note_array_full[
                self.score_note_array_full["pitch"] == pitch
            ]

        self._prev_score_onset = self.score_note_array_full["onset_beat"][0]
        self._unique_score_onsets = np.unique(self.score_note_array_full["onset_beat"])

        # set of pitches at onset / map from onset to idx in unique onsets
        self.pitches_at_onset_by_id = list()
        self.id_by_onset = dict()

        for i, onset in enumerate(self._unique_score_onsets):
            self.pitches_at_onset_by_id.append(
                set(
                    self.score_note_array_no_grace[
                        self.score_note_array_no_grace["onset_beat"] == onset
                    ]["pitch"]
                )
            )
            self.id_by_onset[onset] = i

        # aligned notes at each onset
        self.aligned_notes_at_onset = defaultdict(list)

    def prepare_performance(self, first_onset: float, init_beat_period: Optional[float] = None) -> None:
        if init_beat_period is not None:
            beat_period = init_beat_period
        else:
            beat_period = self.init_beat_period
        self.tempo_model = TempoModel(
            init_beat_period=beat_period,
            init_score_onset=self.score_note_array_full["onset_beat"][0],
            init_perf_onset=first_onset,
            lookback=self.lookback,
        )

    def prepare_model(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "The 'OnlineTransformerMatcher' class requires torch, but it is not installed. "
                "Please install it with: pip install parangonar[accelerated]"
            )
        self.model = AlignmentTransformer(
            token_number=self.token_number,
            dim_model=self.dim_model,
            dim_class=self.dim_class,
            num_heads=self.num_heads,
            num_decoder_layers=self.num_decoder_layers,
            dropout_p=self.dropout_p,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            ALIGNMENT_TRANSFORMER_CHECKPOINT,
            weights_only=True,
            map_location=torch.device(self.device)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def offline(self, performance_note_array: np.ndarray) -> List[Dict[str, Any]]:
        self.prepare_performance(performance_note_array[0]["onset_sec"])

        for p_note in performance_note_array[:]:
            self.online(p_note)

        for s_ID, p_ID in self.alignment:
            self.note_alignments.append(
                {"label": "match", "score_id": s_ID, "performance_id": p_ID}
            )
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                self.note_alignments.append(
                    {"label": "deletion", "score_id": score_note["id"]}
                )

        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                self.note_alignments.append(
                    {"label": "insertion", "performance_id": performance_note["id"]}
                )

        return self.note_alignments

    def online_legacy(self, performance_note: np.ndarray, debug: bool = False) -> None:
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        self._prev_performance_notes.append(p_pitch)

        possible_score_notes = self.score_by_pitch[p_pitch]

        # align greedily if open note at current onset
        if (
            p_pitch
            in self.pitches_at_onset_by_id[self.id_by_onset[self._prev_score_onset]]
        ):
            best_notes = na_within(
                possible_score_notes,
                "onset_beat",
                self._prev_score_onset,
                self._prev_score_onset,
                exclusion_ids=self._snote_aligned,
            )
            if len(best_notes) > 0:
                best_note = best_notes[0]
                self.add_note_alignment(
                    p_id, best_note["id"], p_onset, best_note["onset_beat"]
                )
                return

        current_id = self.id_by_onset[self._prev_score_onset]
        s_slice = slice(np.max((current_id - 7, 0)), current_id + 9)
        p_slice = slice(-8, None)
        score_seq = self.pitches_at_onset_by_id[s_slice]
        perf_seq = self._prev_performance_notes[p_slice]

        tokenized_score_seq = tokenize(score_seq, perf_seq, dims=7)
        out = self.model(
            torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device)
        )
        pred_ids = (
            torch.argsort(torch.softmax(out.squeeze(1), dim=0)[:, 1], descending=True) 
            .cpu()
            .numpy()
        )

        top_three_notes = dict()
        for pred_id in pred_ids[:3]: 
            new_pred_id = (
                pred_id - len(perf_seq) - 1 - (current_id - np.max((current_id - 7, 0)))
            )

            pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
            possible_score_notes = self.score_by_pitch[p_pitch]
            possible_score_notes = na_within(
                possible_score_notes,
                "onset_beat",
                pred_score_onset,
                pred_score_onset,
                exclusion_ids=self._snote_aligned,
            )

            if len(possible_score_notes) > 0:
                dist = np.abs(
                    self.tempo_model.predict(possible_score_notes[0]["onset_beat"])
                    - p_onset
                )
                top_three_notes[dist] = possible_score_notes[0]

        dists = list(top_three_notes.keys())
        if len(dists) >= 1:
            best_note = top_three_notes[np.min(dists)]

            if best_note["is_grace"]:
                self.add_note_alignment(p_id, best_note["id"])
            else:
                self.add_note_alignment(
                    p_id, best_note["id"], p_onset, best_note["onset_beat"]
                )
   
    def online(self, performance_note: np.ndarray, debug: bool = False) -> None:
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

        # go through the model
        current_id = self.id_by_onset[self._prev_score_onset]
        s_slice = slice(np.max((current_id-7, 0)), current_id+9 )
        p_slice = slice(-8, None )
        score_seq = self.pitches_at_onset_by_id[s_slice]
        perf_seq = self._prev_performance_notes[p_slice]

        tokenized_score_seq =  tokenize(score_seq, perf_seq, dims = 7)
        out = self.model(torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device))
        pred_id = torch.argmax(torch.softmax(out.squeeze(1),dim=0)[:,1]).cpu().numpy()
        new_pred_id = pred_id - len(perf_seq) - 1 - (current_id - np.max((current_id-7, 0)))

        ## <----x-> window of sensibility
        if new_pred_id > -5 and new_pred_id < 2:
            pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
            possible_score_notes = self.score_by_pitch[p_pitch]
            possible_score_notes =  na_within(possible_score_notes, "onset_beat", 
                                          pred_score_onset, pred_score_onset,
                                          exclusion_ids=self._snote_aligned)

            if len(possible_score_notes) > 0:
                best_note = possible_score_notes[0]
                if best_note["is_grace"]:
                    self.add_note_alignment(p_id, best_note["id"])
                else:
                    self.add_note_alignment(p_id, best_note["id"], p_onset, best_note["onset_beat"])

        # do you really want to jump?
        elif new_pred_id >= 2:
            # check how many notes are implicitly unaligned
            pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
            implicitly_jumped_notes = 0
            for onset_id in np.arange(current_id, current_id + new_pred_id, 1) :
                implicitly_jumped_notes += len(self.pitches_at_onset_by_id[onset_id])

            # check whether the predicted note could be in the next onset
            if p_pitch in self.pitches_at_onset_by_id[current_id + 1]:
                # check whether the timing is not completely off
                possible_score_notes = self.score_by_pitch[p_pitch]
                possible_score_notes =  na_within(possible_score_notes, "onset_beat", 
                                           self._unique_score_onsets[current_id + 1], self._unique_score_onsets[current_id + 1],
                                          exclusion_ids=self._snote_aligned)
                if len(possible_score_notes) > 0:
                    dist = np.abs(self.tempo_model.predict(possible_score_notes[0]["onset_beat"]) - p_onset)
                    if dist < 1.0:
                        # print("aligned with tempo model:", p_id)
                        self.add_note_alignment(p_id, possible_score_notes[0]["id"], p_onset, possible_score_notes[0]["onset_beat"])
                        return

            if self.time_since_nn_update > 2 and implicitly_jumped_notes <= 10:
                
                possible_score_notes = self.score_by_pitch[p_pitch]
                possible_score_notes =  na_within(possible_score_notes, "onset_beat", 
                                          pred_score_onset, pred_score_onset,
                                          exclusion_ids=self._snote_aligned)

                if len(possible_score_notes) > 0:
                    self.time_since_nn_update = 0
                    best_note = possible_score_notes[0]
                    if best_note["is_grace"]:
                        self.add_note_alignment(p_id, best_note["id"])
                    else:
                        self.add_note_alignment(p_id, best_note["id"], p_onset, best_note["onset_beat"])
        
    def add_note_alignment(self, perf_id, score_id, perf_onset=None, score_onset=None):
        self.alignment.append((score_id, perf_id))
        self._snote_aligned.add(score_id)
        self._pnote_aligned.add(perf_id)
        if perf_onset is not None and score_onset is not None:
            self.aligned_notes_at_onset[score_onset].append(perf_onset)
            if score_onset >= self._prev_score_onset:
                self.tempo_model.update(perf_onset, score_onset)
                self._prev_score_onset = score_onset

    def get_current_score_onset(self) -> float:
        return self._prev_score_onset
    
    ### MATCHMAKER COMPATIBILITY
    
    @property
    def warping_path(self) -> np.ndarray:
        """
        utility forward for matchmaker
        """
        wp = (np.array(self._warping_path).T).astype(np.int32) # [shape=(2, T)]
        return wp
    
    def is_still_following(self, offset = 0):
        """
        utility forward for matchmaker
        """
        return self.current_position < self.N_ref - offset
    
    def call_legacy(self, performance_note) -> int:
        """
        main entrypoint for matchmaker.
        performance_note arrives as single row of a note array

        the tracker returns an index which can be transformed back to 
        score position using self.unique_onsets
        """
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        self._prev_performance_notes.append(p_pitch)

        if self.input_index == 0:
            self.prepare_performance(first_onset = p_onset, 
                                     init_beat_period = 0.5)

        current_id = self.current_position

        possible_score_notes = self.score_by_pitch[p_pitch]
        # align greedily if open note at current onset
        if (
            p_pitch
            in self.pitches_at_onset_by_id[current_id]
        ):
            best_notes = na_within(
                possible_score_notes,
                "onset_beat",
                self._prev_score_onset,
                self._prev_score_onset,
                exclusion_ids=self._snote_aligned,
            )
            if len(best_notes) > 0:
                # stay at location
                self._warping_path.append((self.current_position, self.input_index))
                self.input_index += 1
                # self._prev_score_onset = self._unique_score_onsets[self.current_position]
                return self.current_position


        # use the model prediction
        s_slice = slice(np.max((current_id - 7, 0)), current_id + 9)
        p_slice = slice(-8, None)
        score_seq = self.pitches_at_onset_by_id[s_slice]
        perf_seq = self._prev_performance_notes[p_slice]

        tokenized_score_seq = tokenize(score_seq, perf_seq, dims=7)
        out = self.model(
            torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device)
        )
        # softmax is along 0 dimension here, unlike pure transformer
        pred_ids = torch.argsort(torch.softmax(out.squeeze(1), dim=0)[:, 1], descending=True).cpu().numpy()
        # pred_id = torch.argmax(torch.softmax(out.squeeze(1), dim=1)[:, 1]).cpu().numpy()

        # use the tempo model
        top_three_notes = dict()
        for pred_id in pred_ids[:3]:
            new_pred_id = (
                pred_id - len(perf_seq) - 1 - (current_id - np.max((current_id - 7, 0)))
            )
            pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
            possible_score_notes = self.score_by_pitch[p_pitch]
            possible_score_notes = na_within(
                possible_score_notes,
                "onset_beat",
                pred_score_onset,
                pred_score_onset,
                exclusion_ids=self._snote_aligned,
            )
            if len(possible_score_notes) > 0:
                dist = np.abs(
                    self.tempo_model.predict(possible_score_notes[0]["onset_beat"])
                    - p_onset
                )
                top_three_notes[dist] = possible_score_notes[0]

        dists = list(top_three_notes.keys())
        if len(dists) >= 1:
            closest_note = top_three_notes[np.min(dists)]
            closest_score_onset = closest_note["onset_beat"]
            closest_note_id = self.id_by_onset[closest_score_onset]
            self.current_position = closest_note_id
            self._warping_path.append((self.current_position, self.input_index))
            self.input_index += 1
            # update tempo model
            if not closest_note["is_grace"] and closest_score_onset >= self._prev_score_onset:
                self.tempo_model.update(p_onset, closest_score_onset)
                self._prev_score_onset = closest_score_onset
        
        return self.current_position
    
    def __call__(self, performance_note) -> int:
        """
        main entrypoint for matchmaker.
        performance_note arrives as single row of a note array

        the tracker returns an index which can be transformed back to 
        score position using self.unique_onsets
        """
        self.time_since_nn_update += 1
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        self._prev_performance_notes.append(p_pitch)
        if self.input_index == 0:
            self.prepare_performance(first_onset = p_onset, 
                                     init_beat_period = 0.5)

        # align greedily if open note at current onset
        current_id = self.current_position
        possible_score_notes = self.score_by_pitch[p_pitch]

        if p_pitch in self.pitches_at_onset_by_id[current_id]:
            best_notes = na_within(possible_score_notes, "onset_beat", 
                                    self._prev_score_onset, self._prev_score_onset,
                                    exclusion_ids=self._snote_aligned)
            if len(best_notes) > 0:
                # stay at location
                self._snote_aligned.add(best_notes[0]["id"])
                self.tempo_model.update(p_onset, best_notes[0]["onset_beat"])
                self._warping_path.append((self.current_position, self.input_index))
                self.input_index += 1
                self.stuck_with_no_options = 0
                return self.current_position

        # use the model prediction
        s_slice = slice(np.max((current_id - 7, 0)), current_id + 9)
        p_slice = slice(-8, None)
        score_seq = self.pitches_at_onset_by_id[s_slice]
        perf_seq = self._prev_performance_notes[p_slice]
    
        # only top prediction
        tokenized_score_seq =  tokenize(score_seq, perf_seq, dims = 7)
        out = self.model(torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device))
        pred_id = torch.argmax(torch.softmax(out.squeeze(1),dim=0)[:,1]).cpu().numpy()
        new_pred_id = pred_id - len(perf_seq) - 1 - (current_id - np.max((current_id-7, 0)))
        ## <----x-> window of sensibility
        if new_pred_id > -5 and new_pred_id < 2:
            pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
            possible_score_notes = self.score_by_pitch[p_pitch]
            possible_score_notes =  na_within(possible_score_notes, "onset_beat", 
                                          pred_score_onset, pred_score_onset,
                                          exclusion_ids=self._snote_aligned)

            if len(possible_score_notes) > 0:
                best_note = possible_score_notes[0]
                current_onset = best_note["onset_beat"]
                self._snote_aligned.add(best_note["id"])
                # update tempo model + position
                if not best_note["is_grace"] and current_onset >= self._prev_score_onset:
                    self.tempo_model.update(p_onset, current_onset)
                    self._prev_score_onset = current_onset
                    self.current_position = current_id + new_pred_id
                self._warping_path.append((self.current_position, self.input_index))
                self.input_index += 1
                self.stuck_with_no_options = 0
                return self.current_position
    
        # do you really want to jump?
        elif new_pred_id >= 2:
            # check how many notes are implicitly unaligned
            pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
            implicitly_jumped_notes = 0
            for onset_id in np.arange(current_id, current_id + new_pred_id, 1) :
                implicitly_jumped_notes += len(self.pitches_at_onset_by_id[onset_id])

            # check whether the predicted note could be in the next onset
            if p_pitch in self.pitches_at_onset_by_id[current_id + 1]:
                # check whether the timing is not completely off
                possible_score_notes = self.score_by_pitch[p_pitch]
                possible_score_notes =  na_within(possible_score_notes, "onset_beat", 
                                           self._unique_score_onsets[current_id + 1], self._unique_score_onsets[current_id + 1],
                                          exclusion_ids=self._snote_aligned)
                if len(possible_score_notes) > 0:
                    dist = np.abs(self.tempo_model.predict(possible_score_notes[0]["onset_beat"]) - p_onset)
                    if dist < 1.0:
                        best_note = possible_score_notes[0]
                        current_onset = best_note["onset_beat"]
                        self._snote_aligned.add(best_note["id"])
                        # update tempo model + position
                        if not best_note["is_grace"] and current_onset >= self._prev_score_onset:
                            self.tempo_model.update(p_onset, current_onset)
                            self._prev_score_onset = current_onset
                            self.current_position = current_id + 1
                        self._warping_path.append((self.current_position, self.input_index))
                        self.input_index += 1
                        self.stuck_with_no_options = 0
                        return self.current_position

            # actually do the jump, cautiously            
            if self.time_since_nn_update > 2 and implicitly_jumped_notes <= 10:
                
                possible_score_notes = self.score_by_pitch[p_pitch]
                possible_score_notes =  na_within(possible_score_notes, "onset_beat", 
                                          pred_score_onset, pred_score_onset,
                                          exclusion_ids=self._snote_aligned)

                if len(possible_score_notes) > 0:
                    self.time_since_nn_update = 0
                    best_note = possible_score_notes[0]
                    current_onset = best_note["onset_beat"]
                    self._snote_aligned.add(best_note["id"])
                    # update tempo model + position
                    if not best_note["is_grace"] and current_onset >= self._prev_score_onset:
                        self.tempo_model.update(p_onset, current_onset)
                        self._prev_score_onset = current_onset
                        self.current_position = current_id + new_pred_id
                    self._warping_path.append((self.current_position, self.input_index))
                    self.input_index += 1
                    self.stuck_with_no_options = 0
                    return self.current_position
        
        # if all else fails, do nothing
        self._warping_path.append((self.current_position, self.input_index))
        self.input_index += 1
        self.stuck_with_no_options += 1
        if self.stuck_with_no_options >= 10 and self.stuck_with_no_options < 11:
            # self.stuck_with_no_options = 0
            # just jump forward
            self.current_position += 8
            self._warping_path[-1] = (self.current_position, self.input_index - 1)
            logger.warning("STUCK with no options for 10 inputs at input idx: %s", self.input_index)
        if self.stuck_with_no_options >= 30 and self.stuck_with_no_options < 31:
            # self.stuck_with_no_options = 0
            logger.warning("STUCK with no options for 30 inputs at input idx: %s", self.input_index)

        return self.current_position
    

#### TOKENIZATION


def perf_tokenizer(pitch: int, dims: int = 7) -> np.ndarray:
    return np.ones((1, dims), dtype=int) * (pitch - 20)


def score_tokenizer(pitch_set: Set[int], dims: int = 7) -> np.ndarray:
    token = np.zeros((1, dims), dtype=int)
    for no, pitch in enumerate(list(pitch_set)):
        if pitch >= 21 and pitch <= 108 and no < dims:
            token[0, no] = pitch - 20
    return token


def perf_to_score_tokenizer(dims: int = 7) -> np.ndarray:
    return np.ones((1, dims), dtype=int) * 89


def end_tokenizer(dims: int = 7, end_dims: int = 1) -> np.ndarray:
    return np.ones((end_dims, dims), dtype=int) * 90


def tokenize(score_segment: List[Set[int]], perf_segment: List[int], dims: int = 7) -> np.ndarray:
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

    return np.vstack(tokens)


### PURE TRANSFORMER

class OnlinePureTransformerMatcher(object):
    def __init__(self, 
                 score_note_array_full: np.ndarray,
                 allow_jump: bool = True,
                 backup_steps: int = 24,
                 jump_trigger: int = 24) -> None:
        self.score_note_array_full = np.sort(score_note_array_full, order="onset_beat")
        self.first_p_onset = None
        self._prev_performance_notes = list()
        self._prev_score_onset = None
        self._snote_aligned = set()
        self._pnote_aligned = set()
        self._pnote_aligned_pitch = list()
        self.alignment = []
        self.note_alignments = []
        self.prepare_score()
        self.prepare_model()
        self.initialize()

        # lostness tracker
        self.allow_jump = allow_jump
        self.backup_steps = backup_steps
        self.jump_trigger = jump_trigger
        self.prepare_backup()
        self.prepare_lostness_tracker()
        self.global_backup = list()
        self.buffer_len = self.jump_trigger + 4
        self.non_forward_buffer = deque(maxlen=self.buffer_len)
        self.non_matched_buffer = deque(maxlen=self.buffer_len)
        self.non_matched_theoretical_buffer = deque(maxlen=self.buffer_len)
        self.reset_buffers()

        self.tracked_non_forward_av = list()
        self.tracked_non_matched_av = list()
        self.tracked_non_matched_theoretical_av = list()


    def initialize(self):
        # alias and utils for matchmaker
        self.unique_onsets = self._unique_score_onsets
        self.N_ref = len(self.unique_onsets)
        self.current_position = 0
        self.input_index = 0
        self._warping_path = list()

    def prepare_score(self):
        self.score_note_array_no_grace = self.score_note_array_full[
            self.score_note_array_full["is_grace"] == False
        ]
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            self.score_by_pitch[pitch] = self.score_note_array_full[
                self.score_note_array_full["pitch"] == pitch
            ]

        self._prev_score_onset = self.score_note_array_full["onset_beat"][0]
        self._unique_score_onsets = np.unique(self.score_note_array_full["onset_beat"])

        # set of pitches at onset / map from onset to idx in unique onsets
        self.pitches_at_onset_by_id = list()
        self.pitch_class_at_onset_by_id = list()
        self.id_by_onset = dict()

        for i, onset in enumerate(self._unique_score_onsets):
            self.pitches_at_onset_by_id.append(
                set(
                    self.score_note_array_no_grace[
                        self.score_note_array_no_grace["onset_beat"] == onset
                    ]["pitch"]
                )
            )
            self.pitch_class_at_onset_by_id.append(
                set(
                    self.score_note_array_no_grace[
                        self.score_note_array_no_grace["onset_beat"] == onset
                    ]["pitch"] % 12
                )
            )
            self.id_by_onset[onset] = i

        # aligned notes at each onset
        self.aligned_notes_at_onset = defaultdict(list)

    def reset_buffers(self):
        for k in range(self.buffer_len):
            self.non_forward_buffer.append(0)
            self.non_matched_buffer.append(0)
            self.non_matched_theoretical_buffer.append(0)
        self.non_forward_running_av = 0
        self.non_matched_running_av = 0
        self.non_matched_theoretical_running_av = 0

    def prepare_backup(self):
        self.pitch_to_onset_mask = dict()
        for pitch in range(128):
            self.pitch_to_onset_mask[pitch] = np.array([pitch in pitches_at_onset  
                                                        for pitches_at_onset in self.pitches_at_onset_by_id])
        self.backup_cost_matrix = np.zeros(len(self._unique_score_onsets))

    def update_backup(self, pitch):
        """
        backup_cost_matrix is a single vector of score onset length
        it accumulates the gain of a bounded smith-waterman recursion for every performance note
                p_n-1   p_n
            \        \
        s_m-1   v1  -   v2  
            \        \
        s_m     v3  -   v4 
        don't jump from score to score for same p note
        only diagonal and in perf dir 
        -> max filter of size 2 -> m_filtered[n] = max(m[n], m[n-1])
        if accumulates the gain wherever the pitch matches
        and decreases it for mismatches
        -> rolling cross-similarity matrix for local sequences
        """
        # pitch_class = pitch % 12
        mask = self.pitch_to_onset_mask[pitch]
        max_filtered_last_step = ndimage.maximum_filter(self.backup_cost_matrix, size = 2)
        self.backup_cost_matrix[mask] = bounded_recursion(max_filtered_last_step[mask], 0, self.backup_steps)
        self.backup_cost_matrix[~mask] = np.clip(self.backup_cost_matrix[~mask] - 1, 0, self.backup_steps)
        self.global_backup.append(np.copy(self.backup_cost_matrix))
    
    def trigger_backup_and_jump(self, current_idx):
        # sorted_idx = np.argsort(self.backup_cost_matrix)
        # if np.max(self.backup_cost_matrix[current_idx - 8:current_idx + 8]) > self.backup_steps - 4:
        #     idx_max = np.argmax(self.backup_cost_matrix[current_idx - 8:current_idx + 8])
        #     print("jump index gain not very far:", self.backup_cost_matrix[current_idx + idx_max - 8])
        #     return current_idx + idx_max - 8
        # else:
        #     for idx in sorted_idx[::-1]:
        #         if abs(idx - current_idx) > 8:
        #             print("jump index gain:", self.backup_cost_matrix[idx])
        #             return idx
        # return current_idx
        max_jump_idx = 16
        max_backwards = 4
        logger.debug("slice of backup matrix %s", self.backup_cost_matrix[current_idx - max_backwards:current_idx + max_jump_idx])
        idx_max = np.argmax(self.backup_cost_matrix[current_idx - max_backwards:current_idx + max_jump_idx])
        logger.debug("jump index gain not very far: %s", self.backup_cost_matrix[current_idx + idx_max - max_backwards])
        return current_idx + idx_max - max_backwards
              
    def prepare_lostness_tracker(self):
        self.used_pitches_tracker = [paobi.copy() for paobi in self.pitches_at_onset_by_id]
        self.theoretical_pitches_tracker = [paobi.copy() for paobi in self.pitches_at_onset_by_id]
        self.untracked_note_at_id = np.zeros_like(self.pitches_at_onset_by_id)
    
    def update_lostness_tracker(self, score_idx, performance_pitch, p_id):
        """
        the lostness tracker receives information about the main tracker's predicted output
        and estimates whether the main tracker is lost
        it keeps a record of:
            whether the tracker is standing still or moving backwards 
                (buffer and running average of steps) 
            whether the tracker is using previously unaligned notes,
                i.e., possible pitch, and unused note of this pitch at this onset
                (buffer and running average of steps) 
            whether the tracker is matching at all
                i.e., not even possible pitch
        """
        set_of_pitches = self.used_pitches_tracker[score_idx]
        set_of_pitches_theoretical = self.theoretical_pitches_tracker[score_idx]
        if score_idx <= self.current_position:
            push_val0 = 1
        else:
            push_val0 = 0
        drop_val0 = self.non_forward_buffer[0]
        self.non_forward_buffer.append(push_val0)
        self.non_forward_running_av += push_val0 - drop_val0

        push_val = 0
        push_val_theory = 0
        if performance_pitch in set_of_pitches:
            self.used_pitches_tracker[score_idx].remove(performance_pitch)
        else:
            push_val = 1
            if performance_pitch not in set_of_pitches_theoretical:
                self.untracked_note_at_id[score_idx] += 1
                push_val_theory = 1
                
        drop_val = self.non_matched_buffer[0]
        self.non_matched_buffer.append(push_val)
        self.non_matched_running_av += push_val - drop_val

        drop_val_theory = self.non_matched_theoretical_buffer[0]
        self.non_matched_theoretical_buffer.append(push_val_theory)
        self.non_matched_theoretical_running_av += push_val_theory - drop_val_theory

        self.tracked_non_matched_theoretical_av.append(self.non_matched_theoretical_running_av)
        self.tracked_non_forward_av.append(self.non_forward_running_av)
        self.tracked_non_matched_av.append(self.non_matched_running_av)

        # if self.untracked_note_at_id[score_idx] > self.jump_trigger:
        if self.non_matched_theoretical_running_av >= self.jump_trigger:# and self.non_forward_running_av >= self.jump_trigger:
            logger.debug("jump was triggered at: %s %s", score_idx, self.unique_onsets[score_idx])
            logger.debug("by performance note id: %s", p_id)
            logger.debug("non matched buffer: %s", self.non_matched_buffer)
            logger.debug("non forward buffer: %s", self.non_forward_buffer)
            jump_idx = self.trigger_backup_and_jump(score_idx)
            logger.debug("jumping to: %s %s", jump_idx, self.unique_onsets[jump_idx])
            logger.debug("set of still unmatched pitches at current position: %s", self.used_pitches_tracker[score_idx])
            logger.debug("untracked notes counted at surrounding positions: %s", self.untracked_note_at_id[score_idx-3:score_idx+3])
            logger.debug("%s", "*"*50)
            # reset backup
            self.backup_cost_matrix = np.zeros(len(self._unique_score_onsets))
            # reset lostness
            self.prepare_lostness_tracker()
            # reset buffers
            self.reset_buffers()
            return jump_idx
        
        return score_idx

    def prepare_model(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "The 'OnlinePureTransformerMatcher' class requires torch, but it is not installed. "
                "Please install it with: pip install parangonar[accelerated]"
            )
        self.model = AlignmentTransformer(
            token_number=91,  # 21 - 108 + 2 for padding (start_score, end) + 1 for non_pitch
            dim_model=64,
            dim_class=2,
            num_heads=8,
            num_decoder_layers=6,
            dropout_p=0.1,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            ALIGNMENT_TRANSFORMER_CHECKPOINT,
            weights_only=True,
            map_location=torch.device(self.device)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def offline(
        self, performance_note_array: np.ndarray
    ) -> List[Dict[str, Any]]:

        for p_note in performance_note_array[:]:
            self.online(p_note)

        for s_ID, p_ID in self.alignment:
            self.note_alignments.append(
                {"label": "match", "score_id": s_ID, "performance_id": p_ID}
            )
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                self.note_alignments.append(
                    {"label": "deletion", "score_id": score_note["id"]}
                )

        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                self.note_alignments.append(
                    {"label": "insertion", "performance_id": performance_note["id"]}
                )

        return self.note_alignments

    def online(self, performance_note, debug=False):
        # directly align with NN without any cautionary measures
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        self._prev_performance_notes.append(p_pitch)

        current_id = self.id_by_onset[self._prev_score_onset]
        s_slice = slice(np.max((current_id - 7, 0)), current_id + 9)
        p_slice = slice(-8, None)
        score_seq = self.pitches_at_onset_by_id[s_slice]
        perf_seq = self._prev_performance_notes[p_slice]

        tokenized_score_seq = tokenize(score_seq, perf_seq, dims=7)
        out = self.model(
            torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device)
        )
        pred_id = torch.argmax(torch.softmax(out.squeeze(1), dim=1)[:, 1]).cpu().numpy()
        new_pred_id = (
            pred_id - len(perf_seq) - 1 - (current_id - np.max((current_id - 7, 0)))
        )

        pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
        possible_score_notes = self.score_by_pitch[p_pitch]
        possible_score_notes = na_within(
            possible_score_notes,
            "onset_beat",
            pred_score_onset,
            pred_score_onset,
            exclusion_ids=self._snote_aligned,
        )

        if len(possible_score_notes) > 0:
            best_note = possible_score_notes[0]
            if best_note["is_grace"]:
                self.add_note_alignment(p_id, best_note["id"])
            else:
                self.add_note_alignment(
                    p_id, best_note["id"], p_onset, best_note["onset_beat"]
                )

    def add_note_alignment(self, perf_id, score_id, perf_onset=None, score_onset=None):
        self.alignment.append((score_id, perf_id))
        self._snote_aligned.add(score_id)
        self._pnote_aligned.add(perf_id)
        if perf_onset is not None and score_onset is not None:
            self.aligned_notes_at_onset[score_onset].append(perf_onset)
            if score_onset >= self._prev_score_onset:
                self._prev_score_onset = score_onset

    def get_current_score_onset(self) -> float:
        return self._prev_score_onset
    
    ### MATCHMAKER COMPATIBILITY
    
    @property
    def warping_path(self) -> np.ndarray:
        """
        utility forward for matchmaker
        """
        wp = (np.array(self._warping_path).T).astype(np.int32) # [shape=(2, T)]
        return wp
    
    def is_still_following(self, offset = 0):
        """
        utility forward for matchmaker
        """
        return self.current_position < self.N_ref - offset
    
    def __call__(self, performance_note) -> int:
        """
        main entrypoint for matchmaker.
        performance_note arrives as single row of a note array

        the tracker returns an index which can be transformed back to 
        score position using self.unique_onsets
        """
        p_pitch = performance_note["pitch"]
        self._prev_performance_notes.append(p_pitch)
        self.update_backup(p_pitch)

        current_id = self.current_position
        s_slice = slice(np.max((current_id - 7, 0)), current_id + 9)
        p_slice = slice(-8, None)
        score_seq = self.pitches_at_onset_by_id[s_slice]
        perf_seq = self._prev_performance_notes[p_slice]

        tokenized_score_seq = tokenize(score_seq, perf_seq, dims=7)
        out = self.model(
            torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device)
        )
        pred_id = torch.argmax(torch.softmax(out.squeeze(1), dim=1)[:, 1]).cpu().numpy()
        new_pred_id = pred_id - len(perf_seq) - 1 + np.max((current_id - 7, 0))
        if self.allow_jump:
            new_pred_id = self.update_lostness_tracker(new_pred_id, p_pitch, performance_note["id"])
        self.current_position = new_pred_id
        self._warping_path.append((self.current_position, self.input_index))
        self.input_index += 1
        
        return self.current_position


################################### OLTW MATCHERS ###################################
    
class TOLTWMatcher(object):
    """
    T_OLTW online note alignment object
    or
    SLT_OLTW score follower object that plugs into matchmaker API
    """
    def __init__(
        self,
        score_note_array: np.ndarray,
        tracker_type: str = "T_OLTW",
        init_tempo: Optional[float] = None,
        hop_size: int = 1,
        window_size: int = 40,
        max_run_count: int = 10,
        tempo_factor: float = 0.1,
        time_weight: float = 2.0,
        directional_weights: np.ndarray = np.array([2.0, 1.0, 1.0]),
    ):
        
        self.score_note_array_full = np.sort(score_note_array, order="onset_beat")
        if init_tempo is not None:
            self.init_tempo = init_tempo
        else:
            # start with a rough estimate of 90 qpm, divided by initial bpq
            beat_per_quarter = self.score_note_array_full["duration_beat"][0]/ self.score_note_array_full["duration_quarter"][0]
            self.init_tempo = 60 / 90 / beat_per_quarter
        self.features_s = self.prepare_score(self.score_note_array_full)
        self.features_p = None
        self.performance_note_array = None
        self.queue = Queue()

        # standard T_OLTW as used for https://arxiv.org/abs/2505.05078v1
        if tracker_type == "T_OLTW":
            # best parameters according to https://arxiv.org/abs/2505.05078v1
            self.tracker = T_OLTW(
                    reference_features=self.features_s,
                    queue=self.queue,
                    hop_size=hop_size,
                    window_size=window_size,
                    max_run_count=max_run_count,
                    init_tempo=self.init_tempo,
                    tempo_factor=tempo_factor,
                    time_weight=time_weight,
                    directional_weights=directional_weights,
                )

        # single loop TempoOLTW which interfaces with matchmaker
        elif tracker_type == "SLT_OLTW":
            self.tracker = SLT_OLTW(
                reference_features=self.features_s,
                queue=self.queue,
                window_size=window_size,
                max_run_count=max_run_count,
                init_tempo=self.init_tempo,
                tempo_factor=tempo_factor,
                time_weight=time_weight,
                directional_weights=directional_weights,
            )

        # note alignment compatibility
        self._snote_aligned = set()
        self._pnote_aligned = set()
        self.alignment = []

    def prepare_score(self, s_array: np.ndarray):
        features = list()
        unique_onsets = np.unique(s_array["onset_beat"])
        self.unique_onsets = unique_onsets
        self.N_ref = len(self.unique_onsets)
        # create pitch set representation
        for onset in unique_onsets:
            features.append(
                [onset, set(s_array[s_array["onset_beat"] == onset]["pitch"])]
            )
        # score by pitch representation
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            self.score_by_pitch[pitch] = self.score_note_array_full[
                self.score_note_array_full["pitch"] == pitch
            ]
        return features
    
    ### MATCHMAKER COMPATIBILITY
    
    @property
    def warping_path(self) -> np.ndarray:
        """
        utility forward for matchmaker
        """
        return self.tracker.warping_path
    
    def is_still_following(self, offset = 1):
        """
        utility forward for matchmaker
        """
        return self.tracker.current_position < self.N_ref - offset
    
    def __call__(self, performance_note) -> int:
        """
        main entrypoint for matchmaker.
        performance_note arrives as single row of a note array
        and is converted to a tuple of onset time and pitch.
        (see prepare_performance for info on how to transform)

        the tracker returns an index which can be transformed back to 
        score position using self.unique_onsets
        """
        note_tuple = [[performance_note["onset_sec"], performance_note["pitch"]]]
        return self.tracker(note_tuple)
    
    ### PARANGONAR COMPATIBILITY

    def prepare_performance(self, performance_note_array: np.ndarray):
        self.performance_note_array = performance_note_array
        features = list()
        for note in performance_note_array:
            features.append([note["onset_sec"], note["pitch"]])
        return features

    def offline(self, performance_note_array: np.ndarray):
        tracking_path = self.compute_tracking_path(performance_note_array)
        # process tracking path into alignment
        path_perf_notes = self.performance_note_array[tracking_path[1]]
        predicted_score_times = self.unique_onsets[tracking_path[0]]
        for pred_score_onset, perf_note in zip(predicted_score_times, path_perf_notes):
            if perf_note["id"] not in self._pnote_aligned:
                p_pitch = perf_note["pitch"]
                possible_score_notes = self.score_by_pitch[p_pitch]
                possible_score_notes = na_within(
                    possible_score_notes,
                    "onset_beat",
                    pred_score_onset,
                    pred_score_onset,
                    exclusion_ids=self._snote_aligned,
                )
                if len(possible_score_notes) > 0:
                    best_note = possible_score_notes[0]
                    self.add_note_alignment(perf_note["id"], best_note["id"])

        # create output alignment list
        note_alignments = list()
        for s_ID, p_ID in self.alignment:
            note_alignments.append(
                {"label": "match", "score_id": s_ID, "performance_id": p_ID}
            )
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                note_alignments.append(
                    {"label": "deletion", "score_id": score_note["id"]}
                )

        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                note_alignments.append(
                    {"label": "insertion", "performance_id": performance_note["id"]}
                )

        return note_alignments

    def add_note_alignment(self, perf_id: str, score_id: str):
        self.alignment.append((score_id, perf_id))
        self._snote_aligned.add(score_id)
        self._pnote_aligned.add(perf_id)

    def compute_tracking_path(self, performance_note_array: np.ndarray):
        self.features_p = self.prepare_performance(performance_note_array)
        for feature in self.features_p:
            self.queue.put([feature])
        tracking_path = self.tracker.run()
        return tracking_path

class OLTWMatcher(object):
    def __init__(
        self,
        score_note_array: np.ndarray,
        tracker_type: str = "OLTW",
        hop_size: int = 1,
        window_size: int = 40,
        max_run_count: int = 10,
        directional_weights: np.ndarray = np.array([2.0, 1.0, 1.0]),
    ):
        self.score_note_array_full = np.sort(score_note_array, order="onset_beat")
        self.features_s = self.prepare_score(self.score_note_array_full)
        self.features_p = None
        self.performance_note_array = None
        self.queue = Queue()

        # standard OLTW 
        if tracker_type == "OLTW":
            self.tracker = OLTW(
                reference_features=self.features_s,
                queue=self.queue,
                hop_size=hop_size,
                window_size=window_size,
                max_run_count=max_run_count,
                directional_weights=directional_weights,
            )

        # single loop OLTW which interfaces with matchmaker
        elif tracker_type == "SL_OLTW":
            self.tracker = SL_OLTW(
                reference_features=self.features_s,
                window_size=window_size,
                max_run_count=max_run_count,
                directional_weights=directional_weights,
            )

        # alignment collectors
        self._snote_aligned = set()
        self._pnote_aligned = set()
        self.alignment = []
       
    def prepare_score(self, s_array: np.ndarray):
        features = list()
        unique_onsets = np.unique(s_array["onset_beat"])
        self.unique_onsets = unique_onsets
        self.N_ref = len(self.unique_onsets)
        # create pitch set representation
        for onset in unique_onsets:
            features.append(set(s_array[s_array["onset_beat"] == onset]["pitch"]))
        # score by pitch representation
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            self.score_by_pitch[pitch] = self.score_note_array_full[
                self.score_note_array_full["pitch"] == pitch
            ]
        return features

    
    ### MATCHMAKER COMPATIBILITY

    @property
    def warping_path(self) -> np.ndarray:
        """
        utility forward for matchmaker
        """
        return self.tracker.warping_path
    
    def is_still_following(self, offset = 1):
        """
        utility forward for matchmaker
        """
        return self.tracker.current_position < self.N_ref - offset
    
    def __call__(self, performance_note) -> int:
        """
        main entrypoint for matchmaker.
        performance_note arrives as single row of a note array
        and is converted to a an int for pitch.
        (see prepare_performance for info on how to transform)

        the tracker returns an index which can be transformed back to 
        score position using self.unique_onsets
        """
        note_pitch = [performance_note["pitch"]]
        return self.tracker(note_pitch)

    ### PARANGONAR COMPATIBILITY
    
    def prepare_performance(self, performance_note_array: np.ndarray):
        self.performance_note_array = performance_note_array
        features = list()
        for note in performance_note_array:
            features.append(note["pitch"])
        return features

    def offline(self, performance_note_array: np.ndarray):
        tracking_path = self.compute_tracking_path(performance_note_array)
        # process tracking path into alignment
        path_perf_notes = self.performance_note_array[tracking_path[1]]
        predicted_score_times = self.unique_onsets[tracking_path[0]]
        for pred_score_onset, perf_note in zip(predicted_score_times, path_perf_notes):
            if perf_note["id"] not in self._pnote_aligned:
                p_pitch = perf_note["pitch"]
                possible_score_notes = self.score_by_pitch[p_pitch]
                possible_score_notes = na_within(
                    possible_score_notes,
                    "onset_beat",
                    pred_score_onset,
                    pred_score_onset,
                    exclusion_ids=self._snote_aligned,
                )
                if len(possible_score_notes) > 0:
                    best_note = possible_score_notes[0]
                    self.add_note_alignment(perf_note["id"], best_note["id"])

        # create output alignment list
        note_alignments = list()
        for s_ID, p_ID in self.alignment:
            note_alignments.append(
                {"label": "match", "score_id": s_ID, "performance_id": p_ID}
            )
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                note_alignments.append(
                    {"label": "deletion", "score_id": score_note["id"]}
                )

        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                note_alignments.append(
                    {"label": "insertion", "performance_id": performance_note["id"]}
                )

        return note_alignments

    def add_note_alignment(self, perf_id: str, score_id: str):
        self.alignment.append((score_id, perf_id))
        self._snote_aligned.add(score_id)
        self._pnote_aligned.add(perf_id)

    def compute_tracking_path(self, performance_note_array: np.ndarray):
        self.features_p = self.prepare_performance(performance_note_array)
        for feature in self.features_p:
            self.queue.put([feature])
        tracking_path = self.tracker.run()
        return tracking_path
