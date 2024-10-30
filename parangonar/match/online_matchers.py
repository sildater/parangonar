from .. import ALIGNMENT_TRANSFORMER_CHECKPOINT
import numpy as np
from collections import defaultdict
from .pretrained_models import AlignmentTransformer
import torch
from .matchers import na_within
from scipy.interpolate import interp1d

################################### TEMPO MODELS ###################################

class TempoModel(object):
    """
    Base class for synchronization models

    Attributes
    ----------
    """
    def __init__(
        self,
        init_beat_period = 0.5,
        init_score_onset = 0,
        init_perf_onset = 0,
        lookback = 1    
        ):
        self.lookback = lookback
        self.beat_period = init_beat_period
        self.prev_score_onsets = [init_score_onset - 2 * lookback]
        self.prev_perf_onsets = [init_perf_onset - 2 * lookback * self.beat_period]
        self.prev_perf_onsets_at_score_onsets = defaultdict(list)
        self.prev_perf_onsets_at_score_onsets[self.prev_score_onsets[-1]].append(self.prev_perf_onsets[-1])
        self.est_onset = None
        self.score_perf_map = None
        # Count how many times has the tempo model been called
        self.counter = 0
        self.update(init_perf_onset - lookback * self.beat_period, init_score_onset - lookback)

    def predict(
        self,
        score_onset
        ):
        self.est_onset = self.score_perf_map(score_onset - (self.lookback+1)) + \
            (self.lookback+1) * self.beat_period 
        return self.est_onset

    
    def predict_ratio(
        self,
        score_onset,
        perf_onset
        ):
        self.est_onset = self.score_perf_map(score_onset - (self.lookback+1)) + \
            (self.lookback+1) * self.beat_period 
        error = perf_onset - self.est_onset
        offset_score =  score_onset  - self.prev_score_onsets[-1] 
        if offset_score > 0.0:
            return error/(offset_score * self.beat_period)
        else:
            return error

    def update(
        self,
        performed_onset,
        score_onset
        ):

        self.prev_perf_onsets_at_score_onsets[score_onset].append(performed_onset)
        if score_onset == self.prev_score_onsets[-1]:
            #     self.prev_perf_onsets[-1] = 4/5 * self.prev_perf_onsets[-1] + 1/5* performed_onset
            self.prev_perf_onsets[-1] = np.median(self.prev_perf_onsets_at_score_onsets[score_onset])
        else:
            self.prev_score_onsets.append(score_onset)
            self.prev_perf_onsets.append(performed_onset)
            
        self.score_perf_map = interp1d(self.prev_score_onsets[-100:], 
                                       self.prev_perf_onsets[-100:], 
                                       fill_value="extrapolate")
        self.beat_period = np.clip((self.score_perf_map(score_onset) - \
            self.score_perf_map(score_onset - self.lookback))/self.lookback, 0.1, 10.0)
        self.counter += 1

class DummyTempoModel(object):


    """
    Base class for synchronization models

    Attributes
    ----------
    """
    def __init__(
        self,
        init_beat_period = 0.5,
        init_score_onset = 0,
        init_perf_onset = 0,
        lookback = 1,
        func = None
    ):
        
        self.lookback = lookback
        self.beat_period = init_beat_period
        self.score_perf_map = func
        # Count how many times has the tempo model been called
        self.counter = 0

    def predict(
        self,
        score_onset
        ):
        self.est_onset = self.score_perf_map(score_onset)
        return self.est_onset
    
    # def predict_ratio(
    #     self,
    #     score_onset,
    #     perf_onset
    #     ):
    #     self.est_onset = self.score_perf_map(score_onset - self.lookback) + \
    #         self.lookback * self.beat_period 
    #     error = perf_onset - self.est_onset
    #     offset_score =  score_onset  - self.prev_score_onsets[-1] 
    #     if offset_score > 0.0:
    #         return error/(offset_score * self.beat_period)
    #     else:
    #         return error
    def update(
        self,
        performed_onset,
        score_onset
        ):
        self.counter += 1

################################### ONLINE MATCHERS ###################################

class OnlineTransformerMatcher(object):
    def __init__(self,
                 score_note_array_full
                 ):
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
        self.time_since_nn_update = 0
        self.prepare_score()
        self.prepare_model()

    def prepare_score(self):

        self.score_note_array_no_grace = self.score_note_array_full[self.score_note_array_full["is_grace"] == False]
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            self.score_by_pitch[pitch] = self.score_note_array_full[self.score_note_array_full["pitch"] == pitch]

        self._prev_score_onset = self.score_note_array_full["onset_beat"][0]
        self._unique_score_onsets = np.unique(self.score_note_array_full["onset_beat"])

        # onset range for forward backward view
        self.onset_range_at_onset = dict()
        for s_id, s_onset in enumerate(self._unique_score_onsets[1:-1]):
            self.onset_range_at_onset[s_onset] = [self._unique_score_onsets[s_id], self._unique_score_onsets[s_id+2]]
        self.onset_range_at_onset[self._unique_score_onsets[0]] = [self._unique_score_onsets[0], self._unique_score_onsets[1]]
        self.onset_range_at_onset[self._unique_score_onsets[-1]] = [self._unique_score_onsets[-2], self._unique_score_onsets[-1]]

        # set of pitches at onset / map from onset to idx in unique onsets
        self.pitches_at_onset_by_id = list()
        self.id_by_onset = dict()

        for i, onset in enumerate(self._unique_score_onsets):
            self.pitches_at_onset_by_id.append(
                set(self.score_note_array_no_grace[
                    self.score_note_array_no_grace["onset_beat"] == onset
                    ]["pitch"])
                )
            self.id_by_onset[onset] = i

        # aligned notes at each onset
        self.aligned_notes_at_onset = defaultdict(list)

    def prepare_performance(self, first_onset, init_beat_period = 0.5):
        self.tempo_model = TempoModel(init_beat_period = init_beat_period,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 1)
        
    def prepare_model(self):
        self.model = AlignmentTransformer(
            token_number = 91,
            dim_model = 64,
            dim_class = 2,
            num_heads = 8,
            num_decoder_layers = 6,
            dropout_p = 0.1
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ALIGNMENT_TRANSFORMER_CHECKPOINT, 
                                map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def offline(self, performance_note_array):
        self.prepare_performance(performance_note_array[0]["onset_sec"])

        for p_note in performance_note_array[:]:
            self.online(p_note)

        for s_ID, p_ID in self.alignment:
                self.note_alignments.append({'label': 'match', 
                                        "score_id": s_ID, 
                                        "performance_id": p_ID})
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                self.note_alignments.append({'label': 'deletion', 'score_id': score_note["id"]})
        
        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                self.note_alignments.append({'label': 'insertion', 'performance_id': performance_note["id"]})

        return self.note_alignments

    def online(self, performance_note, debug=False):
        self.time_since_nn_update += 1
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        self._prev_performance_notes.append(p_pitch)

        possible_score_notes = self.score_by_pitch[p_pitch]

        # align greedily if open note at current onset
        if p_pitch in self.pitches_at_onset_by_id[self.id_by_onset[self._prev_score_onset]]:
            best_notes = na_within(possible_score_notes, "onset_beat", 
                                    self._prev_score_onset, self._prev_score_onset,
                                    exclusion_ids=self._snote_aligned)
            if len(best_notes) > 0:
                best_note = best_notes[0]
                self.add_note_alignment(p_id, best_note["id"], p_onset, best_note["onset_beat"])
                return
        
        current_id = self.id_by_onset[self._prev_score_onset]
        s_slice = slice(np.max((current_id-7, 0)), current_id+9 )
        p_slice = slice(-8, None )
        score_seq = self.pitches_at_onset_by_id[s_slice]
        perf_seq = self._prev_performance_notes[p_slice]

        tokenized_score_seq =  tokenize(score_seq, perf_seq, dims = 7)
        out = self.model(torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device))
        pred_ids = torch.argsort(torch.softmax(out.squeeze(1),dim=0)[:,1], descending=True).cpu().numpy()

        top_three_notes = dict()
        for pred_id in pred_ids[:3]:
            new_pred_id = pred_id - len(perf_seq) - 1 - (current_id - np.max((current_id-7, 0)))

            pred_score_onset = self._unique_score_onsets[current_id + new_pred_id]
            possible_score_notes = self.score_by_pitch[p_pitch]
            possible_score_notes =  na_within(possible_score_notes, "onset_beat", 
                                          pred_score_onset, pred_score_onset,
                                          exclusion_ids=self._snote_aligned)

            if len(possible_score_notes) > 0:
                dist = np.abs(self.tempo_model.predict(possible_score_notes[0]["onset_beat"]) - p_onset)
                top_three_notes[dist] = possible_score_notes[0]
                
        dists = list(top_three_notes.keys())
        if len(dists) >= 1:

            best_note = top_three_notes[np.min(dists)]
            if best_note["is_grace"]:
                self.add_note_alignment(p_id, best_note["id"])
            else:
                self.add_note_alignment(p_id, best_note["id"], p_onset, best_note["onset_beat"])                        

    def add_note_alignment(self,
                           perf_id, score_id, 
                           perf_onset = None, score_onset = None
                           ):
        self.alignment.append((score_id, perf_id))
        self._snote_aligned.add(score_id)
        self._pnote_aligned.add(perf_id)
        if perf_onset is not None and score_onset is not None:
            self.aligned_notes_at_onset[score_onset].append(perf_onset)
            if score_onset >= self._prev_score_onset:
                self.tempo_model.update(perf_onset, score_onset)
                self._prev_score_onset = score_onset

    def __call__(self):
        return None
    
    def get_current_score_onset(self):
        return self._prev_score_onset

def perf_tokenizer(pitch, dims = 7):
    return np.ones((1,dims), dtype = int) * (pitch - 20)

def score_tokenizer(pitch_set, dims = 7):
    token = np.zeros((1,dims), dtype = int)
    for no, pitch in enumerate(list(pitch_set)):
        if pitch >= 21 and pitch <= 108 and no < dims:
            token[0,no] = pitch - 20
    return token

def perf_to_score_tokenizer(dims = 7):
    return np.ones((1,dims), dtype = int) *89

def end_tokenizer(dims = 7, end_dims=1):
    return np.ones((end_dims,dims), dtype = int) *90

def tokenize(score_segment, perf_segment, dims = 7):
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

    return np.row_stack(tokens)

class OnlinePureTransformerMatcher(object):
    def __init__(self,
                 score_note_array_full
                 ):
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
        self.time_since_nn_update = 0
        self.prepare_score()
        self.prepare_model()

    def prepare_score(self):

        self.score_note_array_no_grace = self.score_note_array_full[self.score_note_array_full["is_grace"] == False]
        self.score_by_pitch = defaultdict(list)
        unique_pitches = np.unique(self.score_note_array_full["pitch"])
        for pitch in unique_pitches:
            self.score_by_pitch[pitch] = self.score_note_array_full[self.score_note_array_full["pitch"] == pitch]

        self._prev_score_onset = self.score_note_array_full["onset_beat"][0]
        self._unique_score_onsets = np.unique(self.score_note_array_full["onset_beat"])

        # onset range for forward backward view
        self.onset_range_at_onset = dict()
        for s_id, s_onset in enumerate(self._unique_score_onsets[1:-1]):
            self.onset_range_at_onset[s_onset] = [self._unique_score_onsets[s_id], self._unique_score_onsets[s_id+2]]
        self.onset_range_at_onset[self._unique_score_onsets[0]] = [self._unique_score_onsets[0], self._unique_score_onsets[1]]
        self.onset_range_at_onset[self._unique_score_onsets[-1]] = [self._unique_score_onsets[-2], self._unique_score_onsets[-1]]

        # set of pitches at onset / map from onset to idx in unique onsets
        self.pitches_at_onset_by_id = list()
        self.id_by_onset = dict()

        for i, onset in enumerate(self._unique_score_onsets):
            self.pitches_at_onset_by_id.append(
                set(self.score_note_array_no_grace[
                    self.score_note_array_no_grace["onset_beat"] == onset
                    ]["pitch"])
                )
            self.id_by_onset[onset] = i

        # aligned notes at each onset
        self.aligned_notes_at_onset = defaultdict(list)

    def prepare_performance(self, first_onset, func = None):
        if func is None:
            self.tempo_model = TempoModel(init_beat_period = 0.5,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 3)
        else:
            self.tempo_model = DummyTempoModel(init_beat_period = 0.5,
                                    init_score_onset = self.score_note_array_full["onset_beat"][0],
                                    init_perf_onset = first_onset,
                                    lookback = 3,
                                    func = func)

    def prepare_model(self):
        self.model = AlignmentTransformer(
            token_number = 91,# 21 - 108 + 2 for padding (start_score, end) + 1 for non_pitch
            dim_model = 64,
            dim_class = 2,
            num_heads = 8,
            num_decoder_layers = 6,
            dropout_p = 0.1
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ALIGNMENT_TRANSFORMER_CHECKPOINT, 
                                map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def offline(self, performance_note_array, func = None):
        self.prepare_performance(performance_note_array[0]["onset_sec"], func)

        for p_note in performance_note_array[:]:
            self.online(p_note)

        for s_ID, p_ID in self.alignment:
                self.note_alignments.append({'label': 'match', 
                                        "score_id": s_ID, 
                                        "performance_id": p_ID})
        # add unmatched notes
        for score_note in self.score_note_array_full:
            if score_note["id"] not in self._snote_aligned:
                self.note_alignments.append({'label': 'deletion', 'score_id': score_note["id"]})
        
        for performance_note in performance_note_array:
            if performance_note["id"] not in self._pnote_aligned:
                self.note_alignments.append({'label': 'insertion', 'performance_id': performance_note["id"]})

        return self.note_alignments

    def online(self, performance_note, debug=False):
        # directly align with NN without any cautionary measures
        p_id = performance_note["id"]
        p_onset = performance_note["onset_sec"]
        p_pitch = performance_note["pitch"]
        self._prev_performance_notes.append(p_pitch)

        current_id = self.id_by_onset[self._prev_score_onset]
        s_slice = slice(np.max((current_id-7, 0)), current_id+9 )
        p_slice = slice(-8, None )
        score_seq = self.pitches_at_onset_by_id[s_slice]
        perf_seq = self._prev_performance_notes[p_slice]

        tokenized_score_seq =  tokenize(score_seq, perf_seq, dims = 7)
        out = self.model(torch.from_numpy(tokenized_score_seq).unsqueeze(0).to(self.device))
        pred_id = torch.argmax(
            torch.softmax(out.squeeze(1),dim=1)[:,1]
            ).cpu().numpy()
        new_pred_id = pred_id - len(perf_seq) - 1 - (current_id - np.max((current_id-7, 0)))

        
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

    def add_note_alignment(self,
                           perf_id, score_id, 
                           perf_onset = None, 
                           score_onset = None
                           ):
        self.alignment.append((score_id, perf_id))
        self._snote_aligned.add(score_id)
        self._pnote_aligned.add(perf_id)
        if perf_onset is not None and score_onset is not None:
            self.aligned_notes_at_onset[score_onset].append(perf_onset)
            if score_onset >= self._prev_score_onset:
                self.tempo_model.update(perf_onset, score_onset)
                self._prev_score_onset = score_onset

    def __call__(self):

        return None

################################### OLTW MATCHERS ###################################



