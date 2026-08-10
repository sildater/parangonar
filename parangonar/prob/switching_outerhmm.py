import os

from typing import List, Optional, Union, Set, Dict

import numpy as np
from numpy.typing import NDArray

import pandas as pd

import matplotlib.pyplot as plt

import partitura as pt
from partitura.score import Part, Score, ScoreLike

from tqdm import tqdm

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]

DEFAULT_PITCH_ERROR_PROBS = {
    "correct_pitch_prob": 0.9497,
    "semi_tone_error_prob": 0.0145 / 2.0,
    "whole_tone_error_prob": 0.0224 / 2.0,
    "octave_error_prob": 0.0047 / 2.0,
    "within_one_octave_error_prob": 0.0086 / 9.0 / 2.0,
}

DEFAULT_TRANSITIONS = [
    (-3, 0.00509),
    (-2, 0.00516),
    (-1, 0.00886),
    (0, 0.01342),
    (1, 0.94531),
    (2, 0.00610),
    (3, 0.00073),
]

# Create another set of transitions biasing previous states more heavily. 
# Use a ramp up distribution going back to -10 where 1 is still assigned the highest probability, 
# but the probabilities for -1, -2, -3, etc. are higher than in DEFAULT_TRANSITIONS.


DEFAULT_D1 = 3
DEFAULT_D2 = 3

IOI_THRESHOLD = 0.035  # seconds


def compute_OuterProductHMM_pitch_probabilities(
    chords: List[set],
    pitch_error_probs: dict = None,
    other_prob: float = 1e-6,
) -> NDArrayFloat:
    """
    Precompute emission probabilities corresponding to neighbouring pitches for OuterProductHMM states.
    This function takes into consideration pitch errors such as semitone, whole tone, octave, and within one octave errors.

    Parameters
    ----------
    chords : list of sets
        chords[i] contains MIDI pitches (0–127) for score chord at state i.
        A chord is defined as all notes with the same onset time.
    pitch_error_probs : dict or None
        If None, uses DEFAULT_PITCH_ERROR_PROBS. These are the probabilities assigned to different pitch error categories.
    other_prob : float
        Probability assigned to any pitch not falling into error categories. Default is 1e-6.

    Returns
    -------
    b_table : ndarray (N x 128)
        b_table[i, p] = probability of observing pitch p at state i.
    """

    if pitch_error_probs is None:
        pitch_error_probs = DEFAULT_PITCH_ERROR_PROBS

    N = len(chords)
    max_pitch = 128
    b_table = np.full((N, max_pitch), other_prob, dtype=float)

    for i, chord in enumerate(chords):
        if not chord:
            continue
        correct = set(chord)
        semi = {p + 1 for p in chord if 0 <= p + 1 < max_pitch} | {
            p - 1 for p in chord if 0 <= p - 1 < max_pitch
        }
        semi -= correct
        whole = {p + 2 for p in chord if 0 <= p + 2 < max_pitch} | {
            p - 2 for p in chord if 0 <= p - 2 < max_pitch
        }
        whole -= correct | semi
        octv = {p + 12 for p in chord if 0 <= p + 12 < max_pitch} | {
            p - 12 for p in chord if 0 <= p - 12 < max_pitch
        }
        octv -= correct | semi | whole
        within_oct = {
            x
            for p in chord
            for x in range(p - 11, p + 12)
            if 0 <= x < max_pitch
            and x not in correct
            and x not in semi
            and x not in whole
            and x not in octv
        }

        probs = pitch_error_probs
        for p in correct:
            b_table[i, p] = probs["correct_pitch_prob"] / len(correct)
        for p in semi:
            b_table[i, p] = probs["semi_tone_error_prob"] / max(1, len(semi))
        for p in whole:
            b_table[i, p] = probs["whole_tone_error_prob"] / max(1, len(whole))
        for p in octv:
            b_table[i, p] = probs["octave_error_prob"] / max(1, len(octv))
        for p in within_oct:
            b_table[i, p] = probs["within_one_octave_error_prob"] / max(
                1, len(within_oct)
            )
    return b_table


def get_chords_from_score(
    score: ScoreLike,
    return_unique_onsets: bool = False,
) -> List[set]:
    """
    Extract chords from a score-like object.
    A chord is defined as all notes with the same onset time.

    Parameters
    ----------
    score : ScoreLike
        The score-like object to extract chords from.

    return_unique_onsets : bool
        If True, also return the unique onset times.

    Returns
    -------
    List[set]
        A list of sets, each containing the MIDI pitches for a chord.
    """

    if isinstance(score, (Score, Part)):
        note_array = score.note_array()

    if isinstance(score, np.ndarray):
        note_array = score

        if "onset_beat" not in note_array.dtype.names:
            raise ValueError("`score` is not a valid note array")

    # This code does not handle ornaments
    # We are using score-like objects, but we might want to have this to be more general

    unique_onsets = np.unique(note_array["onset_beat"])

    unique_onset_idxs = [
        np.where(note_array["onset_beat"] == uo)[0] for uo in unique_onsets
    ]

    chords = [set(note_array["pitch"][ui]) for ui in unique_onset_idxs]

    if return_unique_onsets:
        return chords, unique_onsets
    else:
        return chords
    
def get_downbeats_from_score(
    score: ScoreLike,
    unique_onsets: Optional[NDArrayFloat] = None,
) -> NDArrayFloat:
    """
    Extract downbeat information from a score-like object.

    Parameters
    ----------
    score : ScoreLike
        The score-like object to extract downbeats from.

    unique_onsets : ndarray or None
        If provided, should be the unique onset times corresponding to the chords.
        If None, it will be computed from the score.

    Returns
    -------
    ndarray
        A boolean array indicating which unique onsets are downbeats.
    """

    if isinstance(score, (Score, Part)):
        note_array = score.note_array()

    if isinstance(score, np.ndarray):
        note_array = score

    if unique_onsets is None:
        unique_onsets = np.unique(note_array["onset_beat"])

    downbeats = np.zeros(len(unique_onsets), dtype=int)

    for i, unique_onset in enumerate(unique_onsets):
        # Check if any note with this onset is a downbeat
        onset_array_loc = np.where(note_array["onset_beat"] == unique_onset)[0][0]
        if note_array["is_downbeat"][onset_array_loc]:
            downbeats[i] = 1

    return downbeats


def compute_transition_matrix(
    N: int,
    transitions: list[tuple[int, float]] = None,
    D1: int = DEFAULT_D1,
    D2: int = DEFAULT_D2,
) -> tuple[NDArrayFloat, int, int]:
    """
    Construct banded transition matrix (α) from transition deltas and probabilities.

    Parameters
    ----------
    N : int
        Number of score states (chords)
    transitions : list of (delta, prob) or None
        If None, uses DEFAULT_TRANSITIONS.

    Returns
    -------
    alpha : ndarray (N x N)
        α[i,j] = probability of transitioning from state i -> j (banded structure)
    D1, D2 : int
        Fixed neighbourhood sizes (default 3)
    """
    if transitions is None:
        transitions = DEFAULT_TRANSITIONS

    # intialize transition matrix with epsilons
    alpha = np.full((N, N), 1e-6, dtype=float)
    for delta, prob in transitions:
        for i in range(N):
            j = i + delta
            if 0 <= j < N:
                alpha[i, j] = prob

    alpha += np.finfo(float).eps
    alpha /= alpha.sum(axis=1, keepdims=True)
    return alpha, D1, D2


def generate_hesitation_transitions(
    sigma: float = 3.0,
    gamma: float = np.log(2),
    D1: int = 10,
    D2: int = 2,
) -> List[tuple[int, float]]:
    """
    Construct a transition probability distribution for state offsets
    based on a Gaussian left side and exponential decay right side.

    Parameters
    ----------
    sigma : float
        Standard deviation controlling the Gaussian spread to the left.
    gamma : float
        Exponential decay rate to the right (≈ ln(2) gives halving each step).
    D1 : int
        Maximum backward offset.
    D2 : int
        Maximum forward offset.

    Returns
    -------
    List[Tuple[int, float]]
        List of (offset, probability) pairs normalized to sum to 1.
    """

    offsets = np.arange(-D1, D2 + 1)

    # peak at +1
    center = 1

    log_probs = (
        -((offsets - center) ** 2) / (2 * sigma**2)
        - gamma * np.maximum(0, offsets - center)
    )

    probs = np.exp(log_probs)

    # normalize
    probs /= probs.sum()

    return list(zip(offsets.tolist(), probs.tolist()))


class SwitchingOuterHMM():
    def __init__(
        self,
        reference_features: np.ndarray,
        transitions: Optional[List[tuple[int, float]]] = None,
        pitch_error_probs: Optional[dict[str, float]] = None,
        S: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
        resumption_type: str = "downbeat",
        other_prob: float = 1e-6,
        ioi_num: int = 8,
        hesitation_jump_back: int = 9,
        hesitation_jump_forward: int = 2,
        hesitation_from_avg_ioi: bool = True,
        hesitation_ioi_ratio_threshold: float = 2,
        hesitation_from_pitch_errors: bool = True,
        neighbourhood_range: int = 2,
        sigma: float = 4.0,
        gamma: float = np.log(10),
    ) -> None:
        """
        Outer-product Hidden Markov Model for score following.
        Note: This implementation aligns every performance note to a score note. i.e., there are no insertions possible.

        Parameters:
        -----------

            reference_features: A partitura Score Note Array.
            
            transitions: A list of tuples defining the neighbourhood transition probabilities to states around the current state. 
                Each tuple should be of the form (jump distance in sequential score states, normalised probability). 
                Refer to DEFAULT_TRANSITIONS for an example.
            
            pitch_error_probs: A dictionary defining the probabilities of pitch errors. Refer to DEFAULT_PITCH_ERROR_PROBS for more details.
            
            S: A numpy array defining the probability of skipping from each state. Should be of shape (num_states,).
            
            r: A numpy array defining the probability of resuming to each state after a skip. Should be of shape (num_states,).
            
            resumption_type : str
                Type of distribution to use for resumption probabilities (r).
                The Default is "downbeat", which assigns higher probabilities to resuming on downbeat states after a skip.
                If "manual", the user must provide r. 
                If "uniform", a uniform distribution is used.

            other_prob: A small probability to assign to all transitions and emissions to avoid zero probabilities. Default is 1e-6.

            ioi_num: The number of previous IOIs to consider for calculating the average IOI for hesitation modelling. Default is 8.

            hesitation_jump_back: The maximum number of previous states that are given a higher probability of jumping back when modelling a hesitation. 
                Default is 9.

            hesitation_jump_forward: The maximum number of future states that are given a higher probability of jumping forward when modelling a hesitation. 
                Default is 2.

            hesitation_from_avg_ioi: Whether to model hesitation jumps based on deviations from the average IOI (if True). Default is True.

            hesitation_ioi_ratio_threshold: The IOI ratio threshold for modelling hesitation jumps. 
                If the current IOI is greater than (average IOI * threshold), a hesitation jump is modelled. Default is 2.

            hesitation_from_pitch_errors: Whether to model hesitation jumps based on pitch errors. Default is True.

            neighbourhood_range: The number of states on either side of the current state to consider for pitch errors.
                If the current incoming pitch does not exist in the neighbourhood range of the current state, 
                the hesitation jump probabilities are applied. This is to account for multiple states within a chord.Default is 2.

            sigma: The standard deviation parameter for the Gaussian distribution used to model the backward jumps in the hesitation model.
                Default is 4.0.

            gamma: The scaling factor for the forward jump probabilities in the hesitation model. Default is log(10).

        """

        self.reference_features = reference_features

        chords, unique_onsets = get_chords_from_score(
            self.reference_features, return_unique_onsets=True
        )

        self.chords = chords
        self.n_states = len(chords)
        self.state_space = unique_onsets


        self.transitions = (
            transitions if transitions is not None else DEFAULT_TRANSITIONS
        )
        self.pitch_error_probs = (
            pitch_error_probs
            if pitch_error_probs is not None
            else DEFAULT_PITCH_ERROR_PROBS
        )
        self.other_prob = other_prob

        # Transition setup
        self.default_alpha, self.default_D1, self.default_D2 = compute_transition_matrix(
            self.n_states, self.transitions
        )
        self.alpha = self.default_alpha
        self.D1 = self.default_D1
        self.D2 = self.default_D2

        
        self.sigma = sigma
        self.gamma = gamma
        self.hesitation_jump_back = hesitation_jump_back
        self.hesitation_jump_forward = hesitation_jump_forward
        self.hesitation_transitions = generate_hesitation_transitions(
            sigma=self.sigma,
            gamma=self.gamma,
            D1=self.hesitation_jump_back, 
            D2=self.hesitation_jump_forward
        )
        self.hesitation_alpha, self.hesitation_D1, self.hesitation_D2 = compute_transition_matrix(
            self.n_states, self.hesitation_transitions, D1=self.hesitation_jump_back, D2=self.hesitation_jump_forward
        )
        
        self.neighbourhood_range = neighbourhood_range

        self.hesitation = False
        self.hesitation_from_avg_ioi = hesitation_from_avg_ioi
        self.hesitation_from_pitch_errors = hesitation_from_pitch_errors

        self.S = (
            np.ones(self.n_states) / self.n_states
            if S is None
            else np.array(S, dtype=float)
        )
        
        self.resumption_type = resumption_type
        if self.resumption_type == "manual":
            self.r = r
        elif self.resumption_type == "downbeat":
            downbeats = get_downbeats_from_score(
                self.reference_features, unique_onsets
            )
            # create a normalized distribution that assigns higher probabilities to downbeat states
            self.r = downbeats / downbeats.sum()
        else:
            self.r = np.ones(self.n_states) / self.n_states

        # Emission setup
        self.b_table = compute_OuterProductHMM_pitch_probabilities(
            chords, pitch_error_probs, other_prob
        )

        self.current_state = 0
        self.prev_state = 0
        self._warping_path = []
        self._current_chord = np.zeros(128, dtype=int)
        self.prev_pitch_obs = None

        self.state_probabilities = np.ones(self.n_states) / self.n_states
        # add some bias towards starting at the beginning of the score
        # self.state_probabilities[0:5] *= 2
        # self.state_probabilities /= self.state_probabilities.sum()

        self.hesitation_ioi_ratio_threshold = hesitation_ioi_ratio_threshold
        self.last_few_iois = [IOI_THRESHOLD for _ in range(ioi_num)]
        self.avg_ioi = np.mean(self.last_few_iois)
        self.prev_avg_ioi = np.mean(self.last_few_iois)

        self.input_pitch_spelling = None

    @property
    def warping_path(self) -> NDArrayInt:
        return (np.array(self._warping_path).T).astype(np.int32)
    
    def save_hyperparameters(self) -> dict:
        return {
            "Default Transitions": self.transitions,
            "Hesitation Transitions": self.hesitation_transitions,
            "Sigma": self.sigma,
            "Gamma": self.gamma,
            "Pitch Error Probs": self.pitch_error_probs,
            "Resumption Type": self.resumption_type,
            "Hesitation from avg IOI": self.hesitation_from_avg_ioi,
            "Hesitation IOI Ratio Threshold": self.hesitation_ioi_ratio_threshold,
            "Hesitation from Pitch Errors": self.hesitation_from_pitch_errors,
            "IOI Num": len(self.last_few_iois),
            "Hesitation Jump Back": self.hesitation_jump_back,
            "Hesitation Jump Forward": self.hesitation_jump_forward,
            "Neighbourhood Range": self.neighbourhood_range,
        }

    def check_hesitation_and_pitch(
            self, 
            ioi: float, 
        ) -> None:
        if self.input_pitch_spelling is None:
            self.hesitation = False
            return
        self.prev_avg_ioi = self.avg_ioi
        self.last_few_iois.pop(0)
        self.last_few_iois.append(ioi)
        self.avg_ioi = np.mean(self.last_few_iois)

        neighbourhood_left = max(0, self.current_state - self.neighbourhood_range)
        neighbourhood_right = min(self.n_states - 1, self.current_state + self.neighbourhood_range)
        neighbourhood_chord_pitches = set()
        for state in range(neighbourhood_left, neighbourhood_right + 1):
            for pitch in self.chords[state]:
                neighbourhood_chord_pitches.add(pitch)


        if self.avg_ioi > self.hesitation_ioi_ratio_threshold * self.prev_avg_ioi:
            if self.hesitation_from_avg_ioi:
                self.hesitation = True
            else:
                self.hesitation = False
        elif self.input_pitch_spelling not in neighbourhood_chord_pitches:
            if self.hesitation_from_pitch_errors:
                self.hesitation = True
            else:
                self.hesitation = False
        else:
            self.hesitation = False
    
    def set_alpha(self) -> None:
        if self.hesitation:
            self.alpha = self.hesitation_alpha
            self.D1 = self.hesitation_D1
            self.D2 = self.hesitation_D2
        else:
            self.alpha = self.default_alpha
            self.D1 = self.default_D1
            self.D2 = self.default_D2
    
    def __call__(
        self, input: tuple[np.ndarray, float], *args, **kwargs
    ) -> Optional[int]:
        pitch_obs, ioi = input

        if ioi < IOI_THRESHOLD:
            self._current_chord = np.maximum(self._current_chord, pitch_obs)
            return self.current_state
        else:
            self.check_hesitation_and_pitch(ioi)
            self.set_alpha()

            self._current_chord = pitch_obs
            self.state_probabilities = self.viterbi_step(
                self.state_probabilities, self._current_chord
            )
            self.current_state = np.argmax(self.state_probabilities)
            self._warping_path.append(self.current_state)
            self.prev_pitch_obs = pitch_obs             

            return self.current_state
    
    # Observation likelihood
    def compute_obs_likelihood(
        self,
        observation: np.ndarray,
    ) -> NDArrayFloat:
        """
        Given observed MIDI pitches, return likelihood vector b[i].

        Parameters
        ----------
        observation: iterable of MIDI note numbers

        Returns
        -------
        b : ndarray (N,)
            b[i] = likelihood of observing `observation` at state i.
        """

        log_b = np.log(np.maximum(self.b_table, 1e-300))  # (N, 88)
        log_em = log_b @ observation  # (N,): log-product over active pitches
        log_em -= log_em.max()  # shift for numerical stability
        return np.exp(log_em)  # (N,)

    # Viterbi update
    def viterbi_step(
        self,
        prev_probs: NDArrayFloat,
        observation: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        This function performs a fast outer-product Viterbi update.
        Parameters
        ----------
        prev_probs : ndarray (N,)
            Previous state probabilities.
        observation : ndarray (88,)
            Current observed MIDI pitches (88 keys from A0 to C8).
        Returns
        -------
        new_probs : ndarray (N,)
            Updated state probabilities after the Viterbi step.
        """

        b = self.compute_obs_likelihood(observation)

        skip_values = prev_probs * self.S
        global_skip_max = skip_values.max()
        new_probs = np.zeros(self.n_states, dtype=float)
        for i in range(self.n_states):
            j_start = max(0, i - self.D2)
            j_end = min(self.n_states, i + self.D1 + 1)
            local_max = 0.0
            for j in range(j_start, j_end):
                val = prev_probs[j] * self.alpha[j, i]
                if val > local_max:
                    local_max = val
            skip_contrib = self.r[i] * global_skip_max

            new_probs[i] = b[i] * (
                skip_contrib if skip_contrib >= local_max else local_max
            )
        if np.sum(new_probs) > 0:
            new_probs /= np.sum(new_probs)
        else:
            new_probs = np.ones(self.n_states) / self.n_states
        return new_probs

    def run(
        self,
        queue_input: Optional[tuple[np.ndarray, float]] = None,
        verbose: bool = False,
    ) -> NDArrayInt:
        same_state_counter = 0
        empty_counter = 0

        self.prev_state = self.current_state

        if queue_input is not None:
            self.input_pitch_spelling = np.where(queue_input[0] == 1)[0][0]
            self.current_state = self(queue_input)
            
            if verbose:
                print(
                    "Current State: ", self.current_state, 
                    "Chords at Current State: ", self.chords[self.current_state], 
                    "Input Pitch: ", self.input_pitch_spelling)
                
            input_pitch_probability_distribution = compute_OuterProductHMM_pitch_probabilities(
                [{self.input_pitch_spelling}], self.pitch_error_probs, self.other_prob   
            )

            aligned_beat = self.state_space[self.current_state]
            aligned_chord = self.chords[self.current_state]
            aligned_chord_probabilities = {}
            for score_pitch in aligned_chord:
                score_pitch_vector = np.zeros(128, dtype=int)
                score_pitch_vector[score_pitch] = 1
                aligned_chord_probabilities[score_pitch] = np.dot(input_pitch_probability_distribution[0], score_pitch_vector)

            max_prob_pitch = max(aligned_chord_probabilities, key=aligned_chord_probabilities.get)
            
            score_array_at_aligned_beat = self.reference_features[self.reference_features["onset_beat"] == aligned_beat]
            score_array_with_max_prob_pitch = score_array_at_aligned_beat[score_array_at_aligned_beat["pitch"] == max_prob_pitch]
            
            score_id_with_max_prob_pitch = str(score_array_with_max_prob_pitch["id"][0])
            
            return aligned_beat, score_id_with_max_prob_pitch

        else:
            raise NotImplementedError("Queue input is required for online alignment.")


class SwitchSnapOuterHMM(object):
    def __init__(
            self,
            reference_features: np.ndarray,
            performance_note_array: np.ndarray,
            score_measure_number_map: Dict[int, int] = None,
            transitions: Optional[List[tuple[int, float]]] = None,
            pitch_error_probs: Optional[dict[str, float]] = None,
            S: Optional[np.ndarray] = None,
            r: Optional[np.ndarray] = None,
            resumption_type: str = "downbeat",
            other_prob: float = 1e-6,
            ioi_num: int = 8,
            hesitation_jump_back: int = 9,
            hesitation_jump_forward: int = 4, # CHANGE BACK TO 2?
            hesitation_from_avg_ioi: bool = True,
            hesitation_ioi_ratio_threshold: float = 2,
            hesitation_from_pitch_errors: bool = True,
            neighbourhood_range: int = 2,
            sigma: float = 4.0, # CHANGE BACK TO 4.0
            gamma: float = np.log(10), # CHANGE BACK TO 2
            annotation_beat_dict: Optional[dict[str, float]] = None,
            annotation_dict: Optional[dict[str, str]] = None,
            evaluate_post_processed_alignment: bool = False,
            ids_association_dict: Optional[dict[str, Set[str]]] = None,
            minimum_ref_id_dict: Optional[dict[str, str]] = None,
            onset_beat_associations_dict: Optional[dict[str, Set[float]]] = None,
            min_ref_onset_beat_dict: Optional[dict[str, float]] = None,
            diagonals_beats_to_num_dict: Optional[dict[tuple[float, float], int]] = None,
            diagonal_borders_dict: Optional[dict[int, tuple[float, float]]] = None,
            average_notes_per_measure: int = None,
            section_omit_reason: Optional[str] = None,
        ) -> None:
        '''
        INITIALIZE THE OUTER HMM MATCHER

        Parameters:
        -----------

            reference_features: A partitura Score Note Array.

            performance_note_array: A performance note array derived from a MIDI performance.

            score_measure_number_map: A dictionary mapping score note onset in divisions to their corresponding measure numbers.
            
            transitions: A list of tuples defining the neighbourhood transition probabilities to states around the current state. 
                Each tuple should be of the form (jump, normalised probability).
            
            pitch_error_probs: A dictionary defining the probabilities of pitch errors. Refer to OuterProductHMM for more details.
            
            S: A numpy array defining the probability of skipping from each state. Should be of shape (num_states,).
            
            r: A numpy array defining the probability of resuming to each state after a skip. Should be of shape (num_states,).
            
            resumption_type : str
                Type of distribution to use for resumption probabilities (r).
                The Default is "downbeat", which assigns higher probabilities to resuming on downbeat states.
                If "manual", the user must provide r. 
                If "uniform", a uniform distribution is used.

            other_prob: A small probability to assign to all transitions and emissions to avoid zero probabilities.

            ioi_num: The number of previous IOIs to consider for calculating the average IOI for hesitation modelling.

            hesitation_jump_back: The maximum number of previous states that are given a higher probability of jumping back when modelling a hesitation. 

            hesitation_jump_forward: The maximum number of future states that are given a higher probability of jumping forward when modelling a hesitation. 

            hesitation_from_avg_ioi: Whether to model hesitation jumps based on deviations from the average IOI (if True).

            hesitation_ioi_ratio_threshold: The IOI ratio threshold for modelling hesitation jumps. 
                If the current IOI is greater than (average IOI * threshold), a hesitation jump is modelled.

            hesitation_from_pitch_errors: Whether to model hesitation jumps based on pitch errors.

            neighbourhood_range: The number of states on either side of the current state to consider for pitch errors.
                If the current incoming pitch does not exist in the neighbourhood range of the current state, 
                the hesitation jump probabilities are applied.

            sigma: The standard deviation parameter for the Gaussian distribution used to model the backward jumps in the hesitation model.

            gamma: The scaling factor for the forward jump probabilities in the hesitation model.

            annotation_beat_dict: A dictionary containing the annotated score beats for evaluation. 
                The keys should be performance note IDs and the values should be the corresponding annotated score beats.

            annotation_dict: A dictionary containing the annotated score ids for evaluation.
                The keys should be performance note IDs and the values should be the corresponding annotated score ids.

            evaluate_post_processed_alignment: Whether to evaluate the post-processed alignment after cleaning quick to-fro jumps. Default is False.

            ids_association_dict: A dictionary mapping reference score IDs to sets of equivalent score IDs.
                These reflect repeated passages in the score that occur at different places in the score, but are considered musically equivalent.
                The practice of one such passage can be considered as the practice of the other. They can therefore be aligned to each other.
                This contains a score ID as a key, and the value is a set of score IDs that are considered equivalent to the key score ID.

            minimum_ref_id_dict: A dictionary mapping equivalent score IDs to their corresponding minimum reference score IDs.
                This is the inverse mapping of ids_association_dict, where the keys are the equivalent score IDs 
                and the values are the minimum reference score IDs. This is to enable a quick lookup.

            onset_beat_associations_dict: A dictionary mapping reference score onset beats to sets of equivalent onset beats.

            min_ref_onset_beat_dict: A dictionary mapping equivalent score onset beats to their corresponding minimum reference score onset beats.    

            diagonals_beats_to_num_dict: A dictionary mapping diagonal borders (tuples of start and end beats) to their corresponding diagonal numbers.

            diagonal_borders_dict: A dictionary mapping diagonal numbers to their corresponding diagonal borders (tuples of start and end beats).

            average_notes_per_measure: The average number of notes per measure in the score. This is used for snapping to diagonals.

            section_omit_reason: A string indicating the reason for omitting a section in the performance.

        '''

        ######### INITIAL ALIGNMENT VARIABLES #########
        
        self.reference_features = reference_features
        self.score_measure_number_map = score_measure_number_map
        self.transitions = transitions
        self.pitch_error_probs = pitch_error_probs
        self.S = S
        self.r = r
        self.resumption_type = resumption_type
        self.other_prob = other_prob
        self.ioi_num = ioi_num
        self.hesitation_jump_back = hesitation_jump_back
        self.hesitation_jump_forward = hesitation_jump_forward
        self.hesitation_from_avg_ioi = hesitation_from_avg_ioi
        self.hesitation_ioi_ratio_threshold = hesitation_ioi_ratio_threshold
        self.hesitation_from_pitch_errors = hesitation_from_pitch_errors
        self.neighbourhood_range = neighbourhood_range
        self.sigma = sigma
        self.gamma = gamma
        self.performance_note_array = performance_note_array

        self.outerHMM = SwitchingOuterHMM(
            reference_features=self.reference_features,
            transitions=self.transitions,
            pitch_error_probs=self.pitch_error_probs,
            S=self.S,
            r=self.r,
            resumption_type=self.resumption_type,
            other_prob=self.other_prob,
            ioi_num=self.ioi_num,
            hesitation_jump_back=self.hesitation_jump_back,
            hesitation_jump_forward=self.hesitation_jump_forward,
            hesitation_from_avg_ioi=self.hesitation_from_avg_ioi,
            hesitation_ioi_ratio_threshold=self.hesitation_ioi_ratio_threshold,
            hesitation_from_pitch_errors=self.hesitation_from_pitch_errors,
            neighbourhood_range=self.neighbourhood_range,
            sigma=self.sigma,
            gamma=self.gamma,
        )

        ######### ALIGNMENT POST-PROCESSING VARIABLES #########
        self.pid_sid_map = {}

        self.alignment = []
        self.alignment_dict = {}

        self.pid_alignment_map = {}
        self.deletions_list = []

        self.parallel_alignment = []
        self.parallel_alignment_dict = {}
        

        self.unique_score_onsets = np.unique(self.reference_features['onset_beat'])

        self.processed_alignment_dict = {}
        self.processed_alignment = []
        self.post_processing_done = False

        self.processed_pid_alignment_map = {}
        self.processed_deletions_list = []

        self.diagonals_beats_to_num_dict = diagonals_beats_to_num_dict
        self.diagonal_borders_dict = diagonal_borders_dict

        ######### SNAP TO DIAGONALS VARIABLES #########
        self.snapped_alignment_dict = {}
        self.snapped_alignment = []
        self.snapped_pid_alignment_map = {}
        self.snapped_deletions_list = []
        self.average_notes_per_measure = average_notes_per_measure
        self.snapping_done = False

        ######### SECTION VARIABLES #########
        self.sections = []
        self.section_omit_reason = section_omit_reason

        ######### EVALUATION VARIABLES #########
        self.evaluate_post_processed_alignment = evaluate_post_processed_alignment
        self.annotation_beat_dict = annotation_beat_dict
        self.annotation_dict = annotation_dict
        self.beat_accuracy = None
        self.f1_matches = None
        self.f1_insertions = None
        self.f1_deletions = None

        self.alignment_eval_dict = {}
        self.processed_alignment_eval_dict = {}

        self.ids_association_dict = ids_association_dict
        self.minimum_ref_id_dict = minimum_ref_id_dict
        self.onset_beat_associations_dict = onset_beat_associations_dict
        self.min_ref_onset_beat_dict = min_ref_onset_beat_dict

    def create_match_alignment_dict(self):
        pid_alignment_map = {}
        s_aligned = []
        deletions_list = []
        refer = False
        if self.snapping_done:
            refer = True
            reference = self.processed_pid_alignment_map

        for i in range(len(self.performance_note_array) - 1):
            note1 = self.performance_note_array[i]
            note2 = self.performance_note_array[i+1]

            pid1 = str(note1["id"])
            pid2 = str(note2["id"])

            ppitch1 = self.pid_sid_map[pid1][1]
            ppitch2 = self.pid_sid_map[pid2][1]

            aligned_sids = [self.pid_sid_map[pid1][0], self.pid_sid_map[pid2][0]]
            set_aligned_sids = set(aligned_sids) # check if both aligned sids are the same or not
            if len(set_aligned_sids) == 2: # if both aligned sids are different
                sid1 = str(aligned_sids[0])
                sid2 = str(aligned_sids[1])

                sid1_pitch = self.reference_features[self.reference_features['id'] == sid1]['pitch'][0]
                sid2_pitch = self.reference_features[self.reference_features['id'] == sid2]['pitch'][0]

                if pid1 not in pid_alignment_map:
                    if ppitch1 == sid1_pitch:
                        pid_alignment_map[pid1] = {"label": "match", "score_id": sid1, "performance_id": str(pid1)}
                    else:
                        pid_alignment_map[pid1] = {"label": "match", "score_id": sid1, "performance_id": str(pid1), "score_attributes_list": ["pitch_error"]}
                    if str(aligned_sids[0]) not in s_aligned:
                        s_aligned.append(str(aligned_sids[0]))
                if refer: # if we need to refer to an earlier alignment to determine whether this was already deemed as an insertion
                    if not reference[pid2]['label'] == 'insertion': 
                        # if the second note was not already deemed as an insertion in the earlier alignment, consider it as a match
                        if ppitch2 == sid2_pitch:
                            pid_alignment_map[pid2] = {"label": "match", "score_id": sid2, "performance_id": str(pid2)}
                        else:
                            pid_alignment_map[pid2] = {"label": "match", "score_id": sid2, "performance_id": str(pid2), "score_attributes_list": ["pitch_error"]}
                        if str(aligned_sids[1]) not in s_aligned:
                            s_aligned.append(str(aligned_sids[1]))
                    else:
                        # if the second note was already deemed as an insertion in the earlier alignment, let it remain as an insertion
                        # the pid_sid_map always matches every performance note to a score note,
                        # and we do not want to change the aligned score note for a performance note that was already deemed as an insertion 
                        # in the earlier alignment, since it would not be a match in either case 
                        pid_alignment_map[pid2] = {"label": "insertion", "performance_id": str(pid2)}

                else:
                    # if we do not need to refer to an earlier alignment, consider the second note as a match as well
                    if ppitch2 == sid2_pitch:
                        pid_alignment_map[pid2] = {"label": "match", "score_id": sid2, "performance_id": str(pid2)}
                    else:
                        pid_alignment_map[pid2] = {"label": "match", "score_id": sid2, "performance_id": str(pid2), "score_attributes_list": ["pitch_error"]}
                    if str(aligned_sids[1]) not in s_aligned:
                        s_aligned.append(str(aligned_sids[1]))

            else: 
                # if both aligned sids are the same, 
                # we check the pitches of the two performance notes 
                # and the pitch of the aligned score note 
                # to determine which one is a match and which one is an insertion

                if ppitch1 == ppitch2: # if the same pitched note is played twice
                    if pid1 not in pid_alignment_map: 
                        # if the first two notes in the performance have the same pitch and are aligned to the same score note
                        # consider the first of the two notes as a match and the second one as an insertion
                        sid1 = str(self.pid_sid_map[pid1][0])
                        sid1_pitch = self.reference_features[self.reference_features['id'] == sid1]['pitch'][0]
                        if ppitch1 == sid1_pitch:
                            pid_alignment_map[pid1] = {"label": "match", "score_id": sid1, "performance_id": str(pid1)}
                        else:
                            pid_alignment_map[pid1] = {"label": "match", "score_id": sid1, "performance_id": str(pid1), "score_attributes_list": ["pitch_error"]}
                        if sid1 not in s_aligned:
                            s_aligned.append(sid1)
                    pid_alignment_map[pid2] = {"label": "insertion", "performance_id": str(pid2)}

                else: # if the pitches of the two notes are different
                    sid = str(aligned_sids[0])
                    score_pitch = self.reference_features[self.reference_features['id'] == sid]['pitch'][0]
                    if pid1 in pid_alignment_map:
                        if pid_alignment_map[pid1]['label'] == "insertion": 
                            # if the first of the two notes has already been marked as an insertion
                            if ppitch2 == score_pitch: # if the second note matches the score pitch, consider it as a match
                                if refer:
                                    if not reference[pid2]['label'] == 'insertion':     
                                        pid_alignment_map[pid2] = {"label": "match", "score_id": str(sid), "performance_id": str(pid2)}
                                        if sid not in s_aligned:
                                            s_aligned.append(sid)
                                    else:
                                        pid_alignment_map[pid2] = {"label": "insertion", "performance_id": str(pid2)}
                            else: 
                                # mark both as insertions 
                                # (since there would definitely be an earlier performance note that is aligned to this score note)
                                pid_alignment_map[pid2] = {"label": "insertion", "performance_id": str(pid2)}

                        else: # if the first of the two notes has already been marked as a match
                            if refer and not reference[pid2]['label'] == 'insertion':
                                ppitch1_distance = abs(ppitch1 - score_pitch)
                                ppitch2_distance = abs(ppitch2 - score_pitch)
                                    
                                if ppitch2_distance < ppitch1_distance: 
                                    # if the second note is closer in pitch to the score note, 
                                    # consider it as a match and the first note as an insertion
                                    if ppitch2 == score_pitch:
                                        pid_alignment_map[pid2] = {"label": "match", "score_id": str(sid), "performance_id": str(pid2)}
                                    else:
                                        pid_alignment_map[pid2] = {"label": "match", "score_id": str(sid), "performance_id": str(pid2), "score_attributes_list": ["pitch_error"]}
                                    if sid not in s_aligned:
                                        s_aligned.append(sid)
                                    pid_alignment_map[pid1] = {"label": "insertion", "performance_id": str(pid1)}
                                else:
                                    # consider the first note as a match and the second note as an insertion
                                    if ppitch1 == score_pitch:
                                        pid_alignment_map[pid1] = {"label": "match", "score_id": str(sid), "performance_id": str(pid1)}
                                    else:
                                        pid_alignment_map[pid1] = {"label": "match", "score_id": str(sid), "performance_id": str(pid1), "score_attributes_list": ["pitch_error"]}
                                    if sid not in s_aligned:
                                        s_aligned.append(sid)
                                    pid_alignment_map[pid2] = {"label": "insertion", "performance_id": str(pid2)}

                            else:
                                pid_alignment_map[pid2] = {"label": "insertion", "performance_id": str(pid2)}

                    else:
                        # if neither of the two notes have been aligned previously, 
                        # we consider the first one as a match and the second one as an insertion
                        if ppitch1 == score_pitch:
                            pid_alignment_map[pid1] = {"label": "match", "score_id": str(sid), "performance_id": str(pid1)}
                        else:
                            pid_alignment_map[pid1] = {"label": "match", "score_id": str(sid), "performance_id": str(pid1), "score_attributes_list": ["pitch_error"]}
                        if sid not in s_aligned:
                            s_aligned.append(sid)
                        pid_alignment_map[pid2] = {"label": "insertion", "performance_id": str(pid2)} 

        for s_note in self.reference_features:
            sid = str(s_note["id"])
            if sid not in s_aligned:
                deletions_list.append(
                    {"label": "deletion", "score_id": sid}
                )
        
        return pid_alignment_map, deletions_list

    def create_match(self, pid_alignment_map, deletions_list):
        alignment = []
        for p_note in self.performance_note_array:
            pid = str(p_note["id"])
            if pid in pid_alignment_map:
                alignment.append(pid_alignment_map[pid])
            else:
                alignment.append({"label": "insertion", "performance_id": pid})

        for deletion in deletions_list:
            alignment.append(deletion)

        return alignment     
    
    def run(
        self,
        ) -> tuple[List[dict], dict]:

        if self.performance_note_array is None:
            raise ValueError("performance_note_array must be provided for OuterHMM Matcher")

        last_onset = None
        for i, p_note in tqdm(enumerate(self.performance_note_array), total=len(self.performance_note_array)):
            pid = str(p_note["id"])
            p_pitch = p_note["pitch"]
            p_pitch_vector = np.zeros(128)
            p_pitch_vector[p_pitch] = 1
            p_onset = p_note["onset_sec"]
            if i == 0:
                last_onset = p_onset
            ioi = p_onset - last_onset
            last_onset = p_onset
            queue_input = (p_pitch_vector, ioi)
            alignment_beat, alignment_score_id = self.outerHMM.run(queue_input=queue_input, verbose=False)

            self.alignment_dict[pid] = alignment_beat
            self.pid_sid_map[pid] = (str(alignment_score_id), p_pitch)

        self.pid_alignment_map, self.deletions_list = self.create_match_alignment_dict()
        self.alignment = self.create_match(self.pid_alignment_map, self.deletions_list)

        self.alignment_eval_dict = self.alignment_dict.copy()
        for line in self.alignment:
            if line['label'] == 'insertion':
                pid_insertion = line['performance_id']
                self.alignment_eval_dict[pid_insertion] = -100

        return self.alignment, self.alignment_dict
    
    def clean_quick_to_fro_jumps(
            self,
            beat_diff_threshold: float = 10.0,
            max_future_states_to_check: int = 20,
            jump_back_beat_offset_allowance_ratio: float = 0.2,
        ) -> tuple[List[dict], dict]:
        '''
        Post-process the alignment to clean quick to-fro jumps that are likely to be misalignments. 
        This is done by identifying sequences of states in the alignment where the score IDs jump back and forth between 
        two or more states in a short period of time, and then removing these jumps from the alignment.

        Parameters:
        -----------

            beat_diff_threshold: The minimum allowed difference in beats between consecutive aligned states to consider it as a jump. 

            max_future_states_to_check: The maximum number of future states to check for a quick return from a previous jump state. 

            jump_back_beat_offset_allowance_ratio: The ratio of the beat difference to allow for a jump back in the alignment. 
                e.g. a jump forward by 200.0 beats might be followed by a jump back of 198.5 beats, and this should still be considered a jump back.
                This is because the HMM might continue to track a few states after jumping away before it returns.

        Returns:
        -----------
            processed_alignment: The cleaned alignment after removing quick to-fro jumps.
            processed_alignment_dict: A dictionary mapping performance note IDs to their processed alignment information.

        '''
        
        pp_alignment_df = pd.DataFrame(list(self.alignment_dict.items()), columns=['Performance Note ID', 'Predicted Score Beat'])

        # Add a column for the difference in predicted score beats between consecutive performance notes
        pp_alignment_df['Beat Difference'] = pp_alignment_df['Predicted Score Beat'].diff().fillna(0)

        jumps = []

        ######### Identify quick to-fro jumps in the alignment #########
        for i, row in pp_alignment_df.iterrows():
            if abs(row['Beat Difference']) > beat_diff_threshold:
                beat_dif = row['Beat Difference']
                for j in range(i+1, i+max_future_states_to_check):
                    if j >= len(pp_alignment_df):
                        break
                    if beat_dif < 0:
                        jump_back_beat_offset_allowance_ratio = -jump_back_beat_offset_allowance_ratio
                    else:
                        jump_back_beat_offset_allowance_ratio = jump_back_beat_offset_allowance_ratio
                    # check for returning jumps to score beat position before the jump
                    if pp_alignment_df.loc[j, 'Beat Difference'] > -int(beat_dif + jump_back_beat_offset_allowance_ratio * beat_dif) and pp_alignment_df.loc[j, 'Beat Difference'] < -int(beat_dif - jump_back_beat_offset_allowance_ratio * beat_dif):
                        jump_start_point = i
                        jump_end_point = j
                        beat_dif_2 = pp_alignment_df.loc[j, 'Beat Difference']
                        second_jump = False
                        for k in range(j+1, j+max_future_states_to_check):
                            if k >= len(pp_alignment_df):
                                break
                            if pp_alignment_df.loc[k, 'Beat Difference'] > -int(beat_dif_2 + jump_back_beat_offset_allowance_ratio * beat_dif_2) and pp_alignment_df.loc[k, 'Beat Difference'] < -int(beat_dif_2 - jump_back_beat_offset_allowance_ratio * beat_dif_2):        
                                second_jump = True
                                break # if there is another jump after the jump back, we do not consider it as a quick to-fro jump and we break out of the loop
                        if not second_jump:
                            jumps.append((jump_start_point, jump_end_point))
                            break

        ######### Clean the identified to-fro jumps #########
        modified_pids = []
        for jump in jumps:
            start = jump[0]
            end = jump[1]
            pre_jump_beat = pp_alignment_df.loc[start-1, 'Predicted Score Beat'] if start > 0 else 0
            post_jump_beat = pp_alignment_df.loc[end, 'Predicted Score Beat']
            # check if there is a unique score onset between pre_jump_beat and post_jump_beat
            unique_onsets_in_range = [onset for onset in self.unique_score_onsets if onset > pre_jump_beat and onset < post_jump_beat]
            num_points = end - start
            if len(unique_onsets_in_range) == 0:
                # divide the values in pp_alignment_df from start to end equally to take values of pre_jump_beat and post_jump_beat
                if num_points == 1:
                    pp_alignment_df.loc[start, 'Predicted Score Beat'] = post_jump_beat
                    pp_alignment_df.loc[start, 'Beat Difference'] = pp_alignment_df.loc[start, 'Predicted Score Beat'] - pre_jump_beat
                    modified_pids.append(pp_alignment_df.loc[start, 'Performance Note ID'])
                    score_pitches_at_beat = self.reference_features[self.reference_features['onset_beat'] == post_jump_beat]['pitch'].tolist()
                    p_id = pp_alignment_df.loc[start, 'Performance Note ID']
                    p_pitch = self.performance_note_array[self.performance_note_array['id'] == p_id]['pitch'][0]
                    self.assign_closest_score_id(post_jump_beat, score_pitches_at_beat, p_id, p_pitch)
                else:
                    # half the points take the value of pre_jump_beat and the other half take the value of post_jump_beat
                    half_point = start + num_points // 2
                    pp_alignment_df.loc[start:half_point, 'Predicted Score Beat'] = pre_jump_beat
                    pp_alignment_df.loc[half_point+1:end, 'Predicted Score Beat'] = post_jump_beat
                    modified_pids.extend(pp_alignment_df.loc[start:end, 'Performance Note ID'].tolist())
                    # recalculate the 'Beat Difference' column for the modified points
                    for i in range(start, end+1):
                        if i == 0:
                            pp_alignment_df.loc[i, 'Beat Difference'] = 0
                        else:
                            pp_alignment_df.loc[i, 'Beat Difference'] = pp_alignment_df.loc[i, 'Predicted Score Beat'] - pp_alignment_df.loc[i-1, 'Predicted Score Beat']
                    score_pitches_at_pre_jump_beat = self.reference_features[self.reference_features['onset_beat'] == pre_jump_beat]['pitch'].tolist()
                    score_pitches_at_post_jump_beat = self.reference_features[self.reference_features['onset_beat'] == post_jump_beat]['pitch'].tolist()
                    for i in range(start, end):
                        p_id = str(pp_alignment_df.loc[i, 'Performance Note ID'])
                        p_pitch = self.performance_note_array[self.performance_note_array['id'] == p_id]['pitch'][0]
                        if pp_alignment_df.loc[i, 'Predicted Score Beat'] == pre_jump_beat:
                            self.assign_closest_score_id(pre_jump_beat, score_pitches_at_pre_jump_beat, p_id, p_pitch)
                        else:
                            self.assign_closest_score_id(post_jump_beat, score_pitches_at_post_jump_beat, p_id, p_pitch)
            else:
                # divide the values in pp_alignment_df from start to end based on the number of unique onsets in range, and assign the values of the unique onsets to the points in pp_alignment_df
                onsets_to_assign = unique_onsets_in_range
                num_onsets = len(onsets_to_assign)
                points_per_onset = num_points // num_onsets
                for i in range(num_onsets):
                    onset = onsets_to_assign[i]
                    if i == num_onsets - 1:
                        pp_alignment_df.loc[start + i*points_per_onset:end, 'Predicted Score Beat'] = onset
                    else:
                        pp_alignment_df.loc[start + i*points_per_onset:start + (i+1)*points_per_onset, 'Predicted Score Beat'] = onset
                modified_pids.extend(pp_alignment_df.loc[start:end, 'Performance Note ID'].tolist())
                # recalculate the 'Beat Difference' column for the modified points
                for i in range(start, end+1):
                    if i == 0:
                        pp_alignment_df.loc[i, 'Beat Difference'] = 0
                    else:
                        pp_alignment_df.loc[i, 'Beat Difference'] = pp_alignment_df.loc[i, 'Predicted Score Beat'] - pp_alignment_df.loc[i-1, 'Predicted Score Beat']

                for i in range(start, end):
                    p_id = pp_alignment_df.loc[i, 'Performance Note ID']
                    p_pitch = self.performance_note_array[self.performance_note_array['id'] == p_id]['pitch'][0]
                    assigned_beat = pp_alignment_df.loc[i, 'Predicted Score Beat']
                    score_pitches_at_beat = self.reference_features[self.reference_features['onset_beat'] == assigned_beat]['pitch'].tolist()
                    self.assign_closest_score_id(assigned_beat, score_pitches_at_beat, p_id, p_pitch)
        
        self.processed_alignment_dict = dict(zip(pp_alignment_df['Performance Note ID'], pp_alignment_df['Predicted Score Beat']))

        self.post_processing_done = True
        self.processed_pid_alignment_map, self.processed_deletions_list = self.create_match_alignment_dict()
        self.processed_alignment = self.create_match(self.processed_pid_alignment_map, self.processed_deletions_list)

        self.processed_alignment_eval_dict = self.processed_alignment_dict.copy()
        
        for line in self.processed_alignment:
            if line['label'] == 'insertion':
                pid_insertion = line['performance_id']
                self.processed_alignment_eval_dict[pid_insertion] = -100
                    
        return self.processed_alignment, self.processed_alignment_dict

    def assign_closest_score_id(self, target_beat, score_pitches_at_beat, p_id, p_pitch):
        if len(score_pitches_at_beat) > 1:
            pitch_distances = {}
            for score_pitch in score_pitches_at_beat:
                pitch_distances[score_pitch] = abs(p_pitch - score_pitch)
            closest_score_pitch = min(pitch_distances, key=pitch_distances.get)
            closest_score_id = str(self.reference_features[(self.reference_features['onset_beat'] == target_beat) & (self.reference_features['pitch'] == closest_score_pitch)]['id'][0])
            self.pid_sid_map[p_id] = (closest_score_id, p_pitch)
        else:
            closest_score_id = str(self.reference_features[self.reference_features['onset_beat'] == target_beat]['id'][0])
            self.pid_sid_map[p_id] = (closest_score_id, p_pitch)

    
    def snap_to_most_likely_diagonal(self):
        if self.evaluate_post_processed_alignment:
            alignment_dict_to_process = self.processed_alignment_eval_dict.copy()
            alignment_to_process = self.processed_alignment.copy()
        else:
            alignment_dict_to_process = self.alignment_eval_dict.copy()
            alignment_to_process = self.alignment.copy()

        first_sna_beat = min(self.reference_features['onset_beat'])
        current_diagonal_num = None
        previous_beats = [first_sna_beat, first_sna_beat, first_sna_beat, first_sna_beat] # initialize a list to keep track of the previous four beats, starting with the first beat in the score
        epsilon = 1e-6
        buffer = 0
        if self.average_notes_per_measure is None or self.average_notes_per_measure < 8:
            buffer_limit = 8
        else:
            buffer_limit = self.average_notes_per_measure

        for line in alignment_to_process:
            if line['label'] != 'match':
                continue
            else:
                pid = line['performance_id']
                predicted_beat = alignment_dict_to_process[pid]
                possible_diagonal_nums = []

                for borders, diagonal_num in self.diagonals_beats_to_num_dict.items():
                    if predicted_beat >= borders[0] and predicted_beat <= borders[1]:
                        possible_diagonal_nums.append(diagonal_num)
                if len(possible_diagonal_nums) == 0:
                    if current_diagonal_num is not None:
                        if predicted_beat in self.min_ref_onset_beat_dict:
                            lookup_beat = self.min_ref_onset_beat_dict[predicted_beat]
                            associated_beats = self.onset_beat_associations_dict[lookup_beat]
                            parallel_to_current_diagonal = False
                            for associated_beat in associated_beats:
                                for borders, diagonal_num in self.diagonals_beats_to_num_dict.items():
                                    if associated_beat >= borders[0] and associated_beat <= borders[1]:
                                        if diagonal_num == current_diagonal_num:
                                            parallel_to_current_diagonal = True
                                            target_beat = associated_beat
                                            break
                                if parallel_to_current_diagonal:
                                    break
                            if parallel_to_current_diagonal:
                                predicted_beat_diff = abs(predicted_beat - np.mean(previous_beats))
                                target_beat_diff = abs(target_beat - np.mean(previous_beats))
                                if target_beat_diff == 0:
                                    target_beat_diff = epsilon
                                beat_diff_ratio = predicted_beat_diff / target_beat_diff
                                if beat_diff_ratio > 2.0:  
                                    score_pitches_at_beat = self.reference_features[self.reference_features['onset_beat'] == target_beat]['pitch'].tolist()
                                    p_pitch = self.performance_note_array[self.performance_note_array['id'] == pid]['pitch'][0]
                                    alignment_dict_to_process[pid] = target_beat
                                    self.assign_closest_score_id(target_beat, score_pitches_at_beat, pid, p_pitch)
                                    previous_beats.pop(0)
                                    previous_beats.append(target_beat)
                                    buffer = 0
                                    continue
   
                        if buffer < buffer_limit:
                            buffer += 1

                        else:
                            previous_beats.pop(0)
                            previous_beats.append(predicted_beat)
                            buffer = 0
                            current_diagonal_num = None
                        continue
                    previous_beats.pop(0)
                    previous_beats.append(predicted_beat)
                    buffer = 0
                    continue
                else:
                    # possible diagonals exist
                    if current_diagonal_num is not None:
                        if current_diagonal_num not in possible_diagonal_nums:
                            if predicted_beat in self.min_ref_onset_beat_dict:
                                lookup_beat = self.min_ref_onset_beat_dict[predicted_beat]
                                associated_beats = self.onset_beat_associations_dict[lookup_beat]
                                parallel_to_current_diagonal = False
                                for associated_beat in associated_beats:
                                    for borders, diagonal_num in self.diagonals_beats_to_num_dict.items():
                                        if associated_beat >= borders[0] and associated_beat <= borders[1]:
                                            if diagonal_num == current_diagonal_num:
                                                parallel_to_current_diagonal = True
                                                target_beat = associated_beat
                                                break
                                    if parallel_to_current_diagonal:
                                        break
                                if parallel_to_current_diagonal:
                                    predicted_beat_diff = abs(predicted_beat - np.mean(previous_beats))
                                    target_beat_diff = abs(target_beat - np.mean(previous_beats))
                                    if target_beat_diff == 0:
                                        target_beat_diff = epsilon
                                    beat_diff_ratio = predicted_beat_diff / target_beat_diff
                                    if beat_diff_ratio > 2.0:
                                        score_pitches_at_beat = self.reference_features[self.reference_features['onset_beat'] == target_beat]['pitch'].tolist()
                                        p_pitch = self.performance_note_array[self.performance_note_array['id'] == pid]['pitch'][0]
                                        alignment_dict_to_process[pid] = target_beat
                                        self.assign_closest_score_id(target_beat, score_pitches_at_beat, pid, p_pitch)
                                        previous_beats.pop(0)
                                        previous_beats.append(target_beat)
                                        buffer = 0
                                        continue
                        else:
                            buffer = 0
                            previous_beats.pop(0)
                            previous_beats.append(predicted_beat)
                            continue    
                            
                        if buffer < buffer_limit:
                            buffer += 1
                        else:
                            buffer = 0
                            current_diagonal_num = None
                            previous_beats.pop(0)
                            previous_beats.append(predicted_beat)

                    else:
                        # sometimes the HMM jumps to a parallel diagonal just before it enters a diagonal, 
                        # which can cause it to be snapped to the wrong diagonal. 
                        # To account for this, we check if among the associated beats of the current predicted beat,
                        # there is a beat that is close to any of the last four beats. If so, we declare the diagonal num 
                        # associated with that beat as the current diagonal, even if the current predicted beat is not in that diagonal.
                        check_for_pre_diagonal = False
                        if predicted_beat in self.min_ref_onset_beat_dict:
                            lookup_beat = self.min_ref_onset_beat_dict[predicted_beat]
                            associated_beats = self.onset_beat_associations_dict[lookup_beat]
                            for associated_beat in associated_beats:
                                if associated_beat == predicted_beat:
                                    continue
                                for previous_beat in previous_beats:
                                    if abs(associated_beat - previous_beat) < 2.0:
                                        check_for_pre_diagonal = True
                                        break
                                if check_for_pre_diagonal:
                                    break
                        if check_for_pre_diagonal:
                            possible_diagonal_nums_pre = []
                            possible_diagonal_lengths_pre = []
                            for borders, diagonal_num in self.diagonals_beats_to_num_dict.items():
                                if associated_beat >= borders[0] and associated_beat <= borders[1]:
                                    possible_diagonal_nums_pre.append(diagonal_num)
                                    possible_diagonal_lengths_pre.append(borders[1] - borders[0])
                            if len(possible_diagonal_nums_pre) > 0:
                                # get the idx of the diagonal with the longest length
                                longest_diagonal_idx_pre = possible_diagonal_lengths_pre.index(max(possible_diagonal_lengths_pre))
                                current_diagonal_num = possible_diagonal_nums_pre[longest_diagonal_idx_pre]
                                score_pitches_at_beat = self.reference_features[self.reference_features['onset_beat'] == associated_beat]['pitch'].tolist()
                                p_pitch = self.performance_note_array[self.performance_note_array['id'] == pid]['pitch'][0]
                                alignment_dict_to_process[pid] = associated_beat
                                self.assign_closest_score_id(associated_beat, score_pitches_at_beat, pid, p_pitch)
                                buffer = 0
                                previous_beats.pop(0)
                                previous_beats.append(associated_beat)
                                continue

                        if len(possible_diagonal_nums) == 1:
                            current_diagonal_num = possible_diagonal_nums[0]
                            buffer = 0
                            
                        else:
                            possible_diagonal_midpoint_beats = []
                            for diagonal_num in possible_diagonal_nums:
                                diagonal_borders = self.diagonal_borders_dict[diagonal_num]
                                diagonal_midpoint_beat = (diagonal_borders[0] + diagonal_borders[1]) / 2
                                possible_diagonal_midpoint_beats.append(diagonal_midpoint_beat)
                            
                            # assign the entry_diagonal_num as the diagonal with the closest midpoint beat to the previous beat
                            beat_distances = {}
                            for idx, diagonal_num in enumerate(possible_diagonal_nums):
                                beat_distances[diagonal_num] = abs(possible_diagonal_midpoint_beats[idx] - np.mean(previous_beats))
                            current_diagonal_num = min(beat_distances, key=beat_distances.get)
                            buffer = 0

                        previous_beats.pop(0)
                        previous_beats.append(predicted_beat)

        self.snapping_done = True
        self.snapped_pid_alignment_dict, self.snapped_deletions_list = self.create_match_alignment_dict()
        self.snapped_alignment = self.create_match(self.snapped_pid_alignment_dict, self.snapped_deletions_list)

        self.snapped_alignment_eval_dict = alignment_dict_to_process.copy()
        self.snapped_alignment_dict = alignment_dict_to_process.copy()

        for line in self.snapped_alignment:
            if line['label'] == 'insertion':
                pid_insertion = line['performance_id']
                self.snapped_alignment_eval_dict[pid_insertion] = -100
                                
        return self.snapped_alignment, alignment_dict_to_process
    
    def create_parallel_alignment(self):
        if self.evaluate_post_processed_alignment:
            alignment_to_process = self.snapped_alignment.copy()
        else:
            alignment_to_process = self.alignment.copy()

        for alignment_item in alignment_to_process:
            self.parallel_alignment.append(alignment_item)
            label = alignment_item['label']
            if label == 'match':
                performance_id = alignment_item['performance_id']
                score_id = str(alignment_item['score_id'])

                self.parallel_alignment_dict[(score_id, performance_id)] = alignment_item

                if score_id in self.minimum_ref_id_dict:
                    parallel_score_lookup_id = self.minimum_ref_id_dict[score_id]
                    set_of_parallel_score_ids = self.ids_association_dict[parallel_score_lookup_id]
                    # remove the score_id itself from the set of parallel score ids if it is in there
                    if score_id in set_of_parallel_score_ids:
                        set_of_parallel_score_ids.remove(score_id)
                    if len(set_of_parallel_score_ids) > 0:
                        for parallel_score_id in set_of_parallel_score_ids:
                            parallel_alignment_item = alignment_item.copy()
                            parallel_alignment_item['score_id'] = parallel_score_id
                            parallel_alignment_item['score_attributes_list'] = [f'musically_identical_to_score_id_{score_id}']
                            self.parallel_alignment.append(parallel_alignment_item)
                            self.parallel_alignment_dict[(parallel_score_id, performance_id)] = parallel_alignment_item

    def create_omitted_section_lines(self):
        '''
        Create a list of omitted section lines based on the alignment. 
        Each omitted section line is represented as a dictionary with the following keys:
        'start_in_beats_unfolded'
        'end_in_beats_unfolded'
        'start_in_beats_original'
        'end_in_beats_original'
        'section_attr_list'
        '''

        if self.evaluate_post_processed_alignment:
            alignment_dict_to_process = self.snapped_alignment_dict.copy()
        else:
            alignment_dict_to_process = self.alignment_dict.copy()

        # get all the values of alignment_dict_to_process and sort them
        aligned_beats = sorted(set(alignment_dict_to_process.values()))
        omitted_beats = self.unique_score_onsets.copy()
        omitted_beats = omitted_beats[~np.isin(omitted_beats, aligned_beats)]
        omitted_sections = []

        last_beat = None
        section_number = 1
        next_unique_onset = None
        omitted_section = None

        if self.section_omit_reason is None:
            self.section_omit_reason = 'not_performed'

        for i, beat in enumerate(omitted_beats):
            sid = str(self.reference_features[self.reference_features['onset_beat'] == beat]['id'][0])
            sid_duration_beat_array = self.reference_features[self.reference_features['onset_beat'] == beat]['duration_beat']
            sid_duration_beat = max(sid_duration_beat_array) if len(sid_duration_beat_array) > 0 else 0
            beat_offset = beat + sid_duration_beat
            if '-2' in sid:
                sid_original = sid.replace('-2', '-1')
                sid_original_onset_beat = self.reference_features[self.reference_features['id'] == sid_original]['onset_beat'].item()
                sid_original_duration_beat_array = self.reference_features[self.reference_features['onset_beat'] == sid_original_onset_beat]['duration_beat']
                sid_original_duration_beat = max(sid_original_duration_beat_array) if len(sid_original_duration_beat_array) > 0 else 0
                sid_original_offset_beat = sid_original_onset_beat + sid_original_duration_beat
                start_in_beats_original = sid_original_onset_beat
                end_in_beats_original = sid_original_offset_beat
            else:
                start_in_beats_original = beat
                duration_beat_array = self.reference_features[self.reference_features['onset_beat'] == beat]['duration_beat']
                duration_beat = max(duration_beat_array) if len(duration_beat_array) > 0 else 0
                end_in_beats_original = beat + duration_beat
            
            if i == 0:
                omitted_section = dict()
                omitted_section['id'] = f'os{section_number}'
                omitted_section['start_in_beats_unfolded'] = beat
                omitted_section['end_in_beats_unfolded'] = beat_offset
                omitted_section['start_in_beats_original'] = start_in_beats_original
                omitted_section['end_in_beats_original'] = end_in_beats_original
                omitted_section['section_attr_list'] = [self.section_omit_reason]
                last_beat = beat
                next_unique_onset = self.unique_score_onsets[self.unique_score_onsets > beat][0] if len(self.unique_score_onsets[self.unique_score_onsets > beat]) > 0 else self.unique_score_onsets[-1]
                continue

            if beat >= last_beat and beat <= next_unique_onset:
                omitted_section['end_in_beats_unfolded'] = beat_offset
                omitted_section['end_in_beats_original'] = end_in_beats_original
                last_beat = beat
                next_unique_onset = self.unique_score_onsets[self.unique_score_onsets > beat][0] if len(self.unique_score_onsets[self.unique_score_onsets > beat]) > 0 else self.unique_score_onsets[-1]
            else:
                if omitted_section['end_in_beats_unfolded'] - omitted_section['start_in_beats_unfolded'] >= 4: # minimum section length of 4 beats
                    omitted_sections.append(omitted_section)
                    section_number += 1

                omitted_section = dict()
                omitted_section['id'] = f'os{section_number}'
                omitted_section['start_in_beats_unfolded'] = beat
                omitted_section['end_in_beats_unfolded'] = beat_offset
                omitted_section['start_in_beats_original'] = start_in_beats_original
                omitted_section['end_in_beats_original'] = end_in_beats_original
                omitted_section['section_attr_list'] = [self.section_omit_reason]
                last_beat = beat
                next_unique_onset = self.unique_score_onsets[self.unique_score_onsets > beat][0] if len(self.unique_score_onsets[self.unique_score_onsets > beat]) > 0 else self.unique_score_onsets[-1]

        if omitted_section is not None and omitted_section['end_in_beats_unfolded'] - omitted_section['start_in_beats_unfolded'] >= 4:
            omitted_sections.append(omitted_section)

        return omitted_sections
                
    
    def create_section_lines(self):
        '''
        Create a list of section lines based on the alignment.
        Each section line is represented as a dictionary with the following keys:
        'start_in_beats_unfolded'
        'end_in_beats_unfolded'
        'start_in_beats_original'
        'end_in_beats_original'
        'start_in_perf_time'
        'end_in_perf_time'
        'section_attr_list'
        '''

        if self.evaluate_post_processed_alignment:
            alignment_list = self.snapped_alignment.copy()
        else:
            alignment_list = self.alignment.copy()

        sna = self.reference_features.copy()
        pna = self.performance_note_array.copy()

        sections = []
        section = None
        section_number = 1
        encountered_notes = 0
        unencountered_notes = 0

        encountered_sids = []

        for i, line in enumerate(alignment_list):
            if line['label'] == 'deletion':
                continue

            elif line['label'] == 'match':
                sid = line['score_id']
                pid = line['performance_id']
                score_onset_beat = sna[sna['id'] == sid]['onset_beat'][0]
                score_duration_beat = sna[sna['id'] == sid]['duration_beat'][0]
                score_offset_beat = score_onset_beat + score_duration_beat
                score_onset_div = sna[sna['id'] == sid]['onset_div'][0]
                perf_onset_tick = pna[pna['id'] == pid]['onset_tick'][0]
                perf_duration_tick = pna[pna['id'] == pid]['duration_tick'][0]
                perf_offset_tick = perf_onset_tick + perf_duration_tick

                if '-2' in sid:
                    sid_original = sid.replace('-2', '-1')
                    sid_original_onset_beat = sna[sna['id'] == sid_original]['onset_beat'].item()
                    sid_original_duration_beat = sna[sna['id'] == sid_original]['duration_beat'].item()
                    sid_original_offset_beat = sid_original_onset_beat + sid_original_duration_beat
                    start_in_beats_original = sid_original_onset_beat
                    end_in_beats_original = sid_original_offset_beat
                else:
                    start_in_beats_original = score_onset_beat
                    end_in_beats_original = score_offset_beat

                if section is None:
                    section = dict()
                    section['id'] = f's{section_number}'
                    section['start_in_beats_unfolded'] = score_onset_beat
                    section['end_in_beats_unfolded'] = score_offset_beat
                    section['start_in_beats_original'] = start_in_beats_original
                    section['end_in_beats_original'] = end_in_beats_original
                    section['start_in_perf_time'] = perf_onset_tick
                    section['end_in_perf_time'] = perf_offset_tick
                    section['section_attr_list'] = []

                    current_score_onset = score_onset_beat
                    current_measure_no = self.score_measure_number_map(score_onset_div)
                    next_unique_onsets = self.unique_score_onsets[self.unique_score_onsets > current_score_onset]
                    if len(next_unique_onsets) > 0:
                        next_score_onset = next_unique_onsets[0]
                    else:
                        next_score_onset = self.unique_score_onsets[-1]

                    next_score_onset_div = sna[sna['onset_beat'] == next_score_onset]['onset_div'][0] if next_score_onset is not None else None
                    next_score_measure_no = self.score_measure_number_map(next_score_onset_div) if next_score_onset_div is not None else None

                    if sid in encountered_sids:
                        encountered_notes += 1
                    else:
                        unencountered_notes += 1
                        encountered_sids.append(sid)

                    alignment_entry = self.parallel_alignment_dict.get((sid, pid), None)
                    if alignment_entry is not None:
                        # get the index in self.parallel_alignment which is equal to alignment_entry
                        parallel_alignment_index = self.parallel_alignment.index(alignment_entry)
                        if 'score_attributes_list' not in self.parallel_alignment[parallel_alignment_index]:
                            self.parallel_alignment[parallel_alignment_index]['score_attributes_list'] = []
                        self.parallel_alignment[parallel_alignment_index]['score_attributes_list'].append(f'section_s{section_number}')

                else:
                    if score_onset_beat >= current_score_onset - 1 and score_onset_beat <= next_score_onset:
                        section['end_in_beats_unfolded'] = score_offset_beat
                        section['end_in_beats_original'] = end_in_beats_original
                        section['end_in_perf_time'] = perf_offset_tick
                        current_score_onset = score_onset_beat
                        current_measure_no = self.score_measure_number_map(score_onset_div)
                        next_unique_onsets = self.unique_score_onsets[self.unique_score_onsets > current_score_onset]
                        if len(next_unique_onsets) > 0:
                            next_score_onset = next_unique_onsets[0]
                        else:
                            next_score_onset = self.unique_score_onsets[-1]

                        next_score_onset_div = sna[sna['onset_beat'] == next_score_onset]['onset_div'][0] if next_score_onset is not None else None
                        next_score_measure_no = self.score_measure_number_map(next_score_onset_div) if next_score_onset_div is not None else None

                        if sid in encountered_sids:
                            encountered_notes += 1
                        else:
                            unencountered_notes += 1
                            encountered_sids.append(sid)

                        alignment_entry = self.parallel_alignment_dict.get((sid, pid), None)
                        if alignment_entry is not None:
                            # get the index in self.parallel_alignment which is equal to alignment_entry
                            parallel_alignment_index = self.parallel_alignment.index(alignment_entry)
                            if 'score_attributes_list' not in self.parallel_alignment[parallel_alignment_index]:
                                self.parallel_alignment[parallel_alignment_index]['score_attributes_list'] = []
                            self.parallel_alignment[parallel_alignment_index]['score_attributes_list'].append(f'section_s{section_number}')

                    else:
                        measure_no = self.score_measure_number_map(score_onset_div)
                        if measure_no in [current_measure_no, next_score_measure_no] and score_onset_beat >= current_score_onset:
                            section['end_in_beats_unfolded'] = score_offset_beat
                            section['end_in_beats_original'] = end_in_beats_original
                            section['end_in_perf_time'] = perf_offset_tick
                            current_score_onset = score_onset_beat
                            current_measure_no = self.score_measure_number_map(score_onset_div)
                            next_unique_onsets = self.unique_score_onsets[self.unique_score_onsets > current_score_onset]
                            if len(next_unique_onsets) > 0:
                                next_score_onset = next_unique_onsets[0]
                            else:
                                next_score_onset = self.unique_score_onsets[-1]

                            next_score_onset_div = sna[sna['onset_beat'] == next_score_onset]['onset_div'][0] if next_score_onset is not None else None
                            next_score_measure_no = self.score_measure_number_map(next_score_onset_div) if next_score_onset_div is not None else None

                            if sid in encountered_sids:
                                encountered_notes += 1
                            else:
                                unencountered_notes += 1
                                encountered_sids.append(sid)

                            alignment_entry = self.parallel_alignment_dict.get((sid, pid), None)
                            if alignment_entry is not None:
                                # get the index in self.parallel_alignment which is equal to alignment_entry
                                parallel_alignment_index = self.parallel_alignment.index(alignment_entry)
                                if 'score_attributes_list' not in self.parallel_alignment[parallel_alignment_index]:
                                    self.parallel_alignment[parallel_alignment_index]['score_attributes_list'] = []
                                self.parallel_alignment[parallel_alignment_index]['score_attributes_list'].append(f'section_s{section_number}')
                                
                        else:
                            if encountered_notes > unencountered_notes:
                                section['section_attr_list'].append('rehearsal_repetition')
                            else:
                                section['section_attr_list'].append('first_rehearsal_run_of_section')

                            alignment_entry = self.parallel_alignment_dict.get((sid, pid), None)
                            if alignment_entry is not None:
                                # get the index in self.parallel_alignment which is equal to alignment_entry
                                parallel_alignment_index = self.parallel_alignment.index(alignment_entry)
                                if 'score_attributes_list' not in self.parallel_alignment[parallel_alignment_index]:
                                    self.parallel_alignment[parallel_alignment_index]['score_attributes_list'] = []
                                self.parallel_alignment[parallel_alignment_index]['score_attributes_list'].append(f'section_s{section_number}')
                            
                            sections.append(section)
                            section = None
                            section_number += 1
                            encountered_notes = 0
                            unencountered_notes = 0
            else:
                if section is not None:
                    pid = line['performance_id']
                    perf_onset_tick = pna[pna['id'] == pid]['onset_tick'][0]
                    perf_duration_tick = pna[pna['id'] == pid]['duration_tick'][0]
                    perf_offset_tick = perf_onset_tick + perf_duration_tick
                    section['end_in_perf_time'] = perf_offset_tick
        
        self.sections = sections
        return sections

    
    def evaluate_alignment_at_beat(self, ablation=None):
        '''
        Evaluate the alignment by comparing the predicted score beats in the alignment_dict with the annotated score beats in the annotation_beat_dict. 
        The annotation_beat_dict should be a dictionary where the keys are performance note IDs and the values are the corresponding annotated score beats.

        Parameters:
        -----------

            evaluate_processed_alignment: Whether to evaluate the processed alignment after cleaning quick to-fro jumps. 
                If False, the original alignment is evaluated. Default is True.

        Returns:
        -----------

            Accuracy of the alignment, calculated as the percentage of performance notes for which the predicted score beat is the annotated score beat.

        '''

        if self.annotation_beat_dict is None:
            raise ValueError("annotation_beat_dict must be provided for evaluation")

        if ablation == 'vanilla':
            alignment_to_evaluate = self.alignment_eval_dict.copy()
        elif ablation == 'switching_OPHMM':
            alignment_to_evaluate = self.processed_alignment_eval_dict.copy()
        else:
            alignment_to_evaluate = self.snapped_alignment_eval_dict.copy()

        correct_matches = 0
        total_matches = 0

        for perf_id, predicted_score_beat in alignment_to_evaluate.items():
            if predicted_score_beat != -100:
                total_matches += 1

        for perf_id, predicted_score_beat in alignment_to_evaluate.items():
            annot_score_beat = self.annotation_beat_dict.get(perf_id)
            if predicted_score_beat != -100:
                if predicted_score_beat == annot_score_beat:
                    correct_matches += 1

        self.beat_accuracy = correct_matches / total_matches if total_matches > 0 else 0
        
        return self.beat_accuracy
    
    def evaluate_alignment_at_beat_with_parallels(self, ablation=None):
        '''
        Evaluate the alignment by comparing the predicted score beats in the alignment_dict with the annotated score beats in the annotation_beat_dict. 
        The annotation_beat_dict should be a dictionary where the keys are performance note IDs and the values are the corresponding annotated score beats.

        Parameters:
        -----------

            evaluate_processed_alignment: Whether to evaluate the processed alignment after cleaning quick to-fro jumps. 
                If False, the original alignment is evaluated. Default is True.

        Returns:
        -----------

            Accuracy of the alignment, calculated as the percentage of performance notes for which the predicted score beat is the annotated score beat.

        '''

        if self.annotation_beat_dict is None:
            raise ValueError("annotation_beat_dict must be provided for evaluation")

        if ablation == 'vanilla':
            alignment_to_evaluate = self.alignment_eval_dict.copy()
        elif ablation == 'switching_OPHMM':
            alignment_to_evaluate = self.processed_alignment_eval_dict.copy()
        else:
            alignment_to_evaluate = self.snapped_alignment_eval_dict.copy()

        correct_matches = 0
        total_matches = 0

        for perf_id, predicted_score_beat in alignment_to_evaluate.items():
            if predicted_score_beat != -100:
                total_matches += 1

        for perf_id, predicted_score_beat in alignment_to_evaluate.items():
            annot_score_beat = self.annotation_beat_dict.get(perf_id)
            if annot_score_beat in self.min_ref_onset_beat_dict:
                lookup_beat = self.min_ref_onset_beat_dict[annot_score_beat]
                set_of_associated_beats = self.onset_beat_associations_dict[lookup_beat]
                if predicted_score_beat != -100:
                    if predicted_score_beat in set_of_associated_beats:
                        correct_matches += 1
            else:
                if predicted_score_beat != -100:
                    if predicted_score_beat == annot_score_beat:
                        correct_matches += 1


        parallel_beat_accuracy = correct_matches / total_matches if total_matches > 0 else 0
        
        return parallel_beat_accuracy
    
    def evaluate_match_alignment(self, ablation=None):
        '''
        Evaluate the alignment by comparing the predicted score beats in the alignment_dict with the annotated score beats in the annotation_dict. 
        The annotation_dict should be a dictionary where the keys are performance note IDs and the values are the corresponding annotated score beats.
        '''
        if self.annotation_dict is None:
            raise ValueError("annotation_dict must be provided for evaluation")

        if ablation == 'vanilla':
            alignment_to_evaluate = self.alignment.copy()
        elif ablation == 'switching_OPHMM':
            alignment_to_evaluate = self.processed_alignment.copy()
        else:
            alignment_to_evaluate = self.snapped_alignment.copy()

        total_annotated_matches = 0
        total_annotated_insertions = 0
        total_annotated_deletions = 0
        annotated_deletions_sid_list = []

        found_matches = 0
        found_insertions = 0
        found_deletions = 0

        correctly_labeled_matches = 0
        correctly_labeled_insertions = 0
        correctly_labeled_deletions = 0

        for annot_pid, annot_sid in self.annotation_dict.items():
            if annot_pid == 'n-1':
                total_annotated_deletions = len(annot_sid)
                annotated_deletions_sid_list = annot_sid
            elif annot_sid == 'n-1':
                total_annotated_insertions += 1
            else:
                total_annotated_matches += 1

        for alignment_item in alignment_to_evaluate:
            label = alignment_item['label']
            if label == 'match':
                found_matches += 1
                perf_id = str(alignment_item['performance_id'])
                score_id = str(alignment_item['score_id'])
                annot_score_id = str(self.annotation_dict.get(perf_id))
                if annot_score_id == score_id:
                    correctly_labeled_matches += 1

            elif label == 'insertion':
                found_insertions += 1
                perf_id = str(alignment_item['performance_id'])
                annot_sid = str(self.annotation_dict.get(perf_id))
                if annot_sid == 'n-1':
                    correctly_labeled_insertions += 1
            elif label == 'deletion':
                found_deletions += 1
                score_id = str(alignment_item['score_id'])
                if score_id in annotated_deletions_sid_list:
                    correctly_labeled_deletions += 1

        precision_matches = correctly_labeled_matches / found_matches if found_matches > 0 else 0
        recall_matches = correctly_labeled_matches / total_annotated_matches if total_annotated_matches > 0 else 0
        self.f1_matches = 2 * (precision_matches * recall_matches) / (precision_matches + recall_matches) if (precision_matches + recall_matches) > 0 else 0

        precision_insertions = correctly_labeled_insertions / found_insertions if found_insertions > 0 else 0
        recall_insertions = correctly_labeled_insertions / total_annotated_insertions if total_annotated_insertions > 0 else 0
        self.f1_insertions = 2 * (precision_insertions * recall_insertions) / (precision_insertions + recall_insertions) if (precision_insertions + recall_insertions) > 0 else 0

        precision_deletions = correctly_labeled_deletions / found_deletions if found_deletions > 0 else 0
        recall_deletions = correctly_labeled_deletions / total_annotated_deletions if total_annotated_deletions > 0 else 0
        self.f1_deletions = 2 * (precision_deletions * recall_deletions) / (precision_deletions + recall_deletions) if (precision_deletions + recall_deletions) > 0 else 0    
                    
        
        return self.f1_matches, self.f1_insertions, self.f1_deletions
    
    def evaluate_match_alignment_with_parallels(self, ablation=None):
        '''
        Evaluate the alignment by comparing the predicted score beats in the alignment_dict with the annotated score beats in the annotation_dict. 
        The annotation_dict should be a dictionary where the keys are performance note IDs and the values are the corresponding annotated score beats.
        '''
        if self.annotation_dict is None:
            raise ValueError("annotation_dict must be provided for evaluation")

        if ablation == 'vanilla':
            alignment_to_evaluate = self.alignment.copy()
        elif ablation == 'switching_OPHMM':
            alignment_to_evaluate = self.processed_alignment.copy()
        else:
            alignment_to_evaluate = self.snapped_alignment.copy()

        total_annotated_matches = 0
        total_annotated_insertions = 0
        total_annotated_deletions = 0
        annotated_deletions_sid_list = []

        found_matches = 0
        found_insertions = 0
        found_deletions = 0

        correctly_labeled_matches = 0
        correctly_labeled_insertions = 0
        correctly_labeled_deletions = 0

        for annot_pid, annot_sid in self.annotation_dict.items():
            if annot_pid == 'n-1':
                total_annotated_deletions = len(annot_sid)
                annotated_deletions_sid_list = annot_sid
            elif annot_sid == 'n-1':
                total_annotated_insertions += 1
            else:
                total_annotated_matches += 1

        for alignment_item in alignment_to_evaluate:
            label = alignment_item['label']
            if label == 'match':
                found_matches += 1
                perf_id = str(alignment_item['performance_id'])
                score_id = str(alignment_item['score_id'])
                annot_score_id = str(self.annotation_dict.get(perf_id))

                if annot_score_id in self.minimum_ref_id_dict:
                    lookup_score_id = self.minimum_ref_id_dict[annot_score_id]
                    set_of_associated_score_ids = self.ids_association_dict[lookup_score_id]
                    if score_id in set_of_associated_score_ids:
                        correctly_labeled_matches += 1
                else:
                    if annot_score_id == score_id:
                        correctly_labeled_matches += 1

            elif label == 'insertion':
                found_insertions += 1
                perf_id = str(alignment_item['performance_id'])
                annot_sid = str(self.annotation_dict.get(perf_id))
                if annot_sid == 'n-1':
                    correctly_labeled_insertions += 1
            elif label == 'deletion':
                found_deletions += 1
                score_id = str(alignment_item['score_id'])
                if score_id in annotated_deletions_sid_list:
                    correctly_labeled_deletions += 1

        precision_matches = correctly_labeled_matches / found_matches if found_matches > 0 else 0
        recall_matches = correctly_labeled_matches / total_annotated_matches if total_annotated_matches > 0 else 0
        parallel_f1_matches = 2 * (precision_matches * recall_matches) / (precision_matches + recall_matches) if (precision_matches + recall_matches) > 0 else 0

        precision_insertions = correctly_labeled_insertions / found_insertions if found_insertions > 0 else 0
        recall_insertions = correctly_labeled_insertions / total_annotated_insertions if total_annotated_insertions > 0 else 0
        parallel_f1_insertions = 2 * (precision_insertions * recall_insertions) / (precision_insertions + recall_insertions) if (precision_insertions + recall_insertions) > 0 else 0

        precision_deletions = correctly_labeled_deletions / found_deletions if found_deletions > 0 else 0
        recall_deletions = correctly_labeled_deletions / total_annotated_deletions if total_annotated_deletions > 0 else 0
        parallel_f1_deletions = 2 * (precision_deletions * recall_deletions) / (precision_deletions + recall_deletions) if (precision_deletions + recall_deletions) > 0 else 0    

        
        return parallel_f1_matches, parallel_f1_insertions, parallel_f1_deletions

    def save_parangonada_csv(
        self,
        results_dir: str
    ):
        '''
        Save the parangonada csv file for the alignment.
        Parameters:
        -----------
            results_dir: The directory where the parangonada csv should be saved.
        '''
        # Check if results directory exists, if not create it
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Create a subdirectory for the parangonada csv files within the results directory
        parangonada_dir = os.path.join(results_dir, "parangonada")
        if not os.path.exists(parangonada_dir):
            os.makedirs(parangonada_dir)

        if self.evaluate_post_processed_alignment:
            match_alignment_to_save = self.snapped_alignment
        else:
            match_alignment_to_save = self.alignment

        # print("Saving Parangonada CSV...")
        # print("---------")

        pt.io.exportparangonada.save_parangonada_csv(match_alignment_to_save, self.performance_note_array, self.reference_features, outdir=parangonada_dir)

        # print("Parangonada CSV Saved at: ", parangonada_dir)
        # print("---------")
        # print()
        # print("Parangonada csv files saved successfully!")
    
    def save_hyperparameters(
        self,
        results_dir: str,
        rehearsal_file_name: str,
        score_file_name: str,
    ):
        '''
        Save the hyperparameters of the Outer HMM model to a text file for future reference.
        Parameters:
        -----------
            results_dir: The directory where the hyperparameters should be saved.

            rehearsal_file_name: The name of the rehearsal file, used for naming the hyperparameters file.

            score_file_name: The name of the score file, used for naming the hyperparameters file
        '''
        
        # Check if results directory exists, if not create it
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        print("Saving Hyperparameters...")
        print("---------")
        print()

        hyperparameters = self.outerHMM.save_hyperparameters()

        if self.annotation_beat_dict is not None and self.annotation_dict is not None:
            if self.beat_accuracy is not None:
                hyperparameters['Alignment Beat Accuracy'] = self.beat_accuracy
            else:
                print("Evaluation not yet run. Running evaluation before saving...")
                self.beat_accuracy = self.evaluate_alignment_at_beat(evaluate_processed_alignment=self.evaluate_post_processed_alignment)
                hyperparameters['Alignment Beat Accuracy'] = self.beat_accuracy

            if self.f1_matches is not None and self.f1_insertions is not None and self.f1_deletions is not None:
                hyperparameters['F1 Score Matches'] = self.f1_matches
                hyperparameters['F1 Score Insertions'] = self.f1_insertions
                hyperparameters['F1 Score Deletions'] = self.f1_deletions
            else:
                print("Evaluation not yet run. Running evaluation before saving...")
                self.f1_matches, self.f1_insertions, self.f1_deletions = self.evaluate_match_alignment(evaluate_processed_alignment=self.evaluate_post_processed_alignment)
                hyperparameters['F1 Score Matches'] = self.f1_matches
                hyperparameters['F1 Score Insertions'] = self.f1_insertions
                hyperparameters['F1 Score Deletions'] = self.f1_deletions

            config_filename = "config_and_results.txt"

        else:
            config_filename = "config.txt"

        with open(os.path.join(results_dir, config_filename), "w") as f:
            f.write(f"Rehearsal File: {rehearsal_file_name}\n")
            f.write(f"Score File: {score_file_name}\n")
            f.write("-------------------------------\n")
            f.write("Hyperparameters:\n")
            for key, value in hyperparameters.items():
                f.write(f"{key}: {value}\n")

        print("Hyperparameters Saved at: ", results_dir)
        print("---------")
        print()

    def save_alignment_plot(
        self,
        performance_file_name: str,
        results_dir: str = None,
        ):
        '''
        Save only the alignment plot.
        Parameters:
        -----------
            results_dir: The directory where the plot should be saved.

            performance_file_name: The name of the performance file, used for naming the plot.
        '''
        
        # Check if results directory exists, if not create it
        if results_dir is not None and not os.path.exists(results_dir):
            os.makedirs(results_dir)

        alignment_to_plot = self.snapped_alignment_dict.copy()

        # convert the Performance Note IDs in alignment_to_plot to integers
        alignment_to_plot_int = {int(pid[1:]): beat for pid, beat in alignment_to_plot.items()}

        print("Saving plot...")

        plt.figure(figsize=(20, 15))

        plt.xlim(0, max(alignment_to_plot_int.keys()) + 10)
        plot_filename = f"{performance_file_name}_alignment_plot.png"


        # --- Blue predicted alignment ---
        plt.scatter(
            list(alignment_to_plot_int.keys()),
            list(alignment_to_plot_int.values()),
            label='Predicted Alignment',
            color='blue',
            marker='x',
            s=10,
            alpha=1
        )

        plt.ylim(0, max(alignment_to_plot_int.values()) + 10)
        plt.xlabel('Rehearsal Note Event', fontsize=32)
        plt.ylabel('Score Time in Beats', fontsize=32)
        plt.legend(fontsize=32, markerscale=4, loc='upper right')
        plt.grid()

        if results_dir is not None:
            plt.savefig(os.path.join(results_dir, plot_filename))
        else:
            plt.show()
