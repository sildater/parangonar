#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods for note and sequence alignment.

sequence alignment:
- Dynamic Time Warping
- Needleman-Wunsch 

note alignment:
- Hidden Markov Model
- Greedy note alignment
- Path-augmented combinatorial alignment

most of the hidden markov machinery comes from 
https://github.com/neosatrapahereje/hiddenmarkov.git
this file contains specific variations and additions.

"""


from .dtw import DTW, DTWSL
from .nwtw import NW_DTW, NW
from .matchers import (AnchorPointNoteMatcher, 
                       AutomaticNoteMatcher,
                       ChordEncodingMatcher,
                       CleanOnsetMatcher,
                       CleanMatcher,
                       CleanOrnamentMatcher,
                       OnlineMatcher,
                       OnlineTransformerMatcher,
                       OnlineTransformerMatcher2,
                       OnlinePureTransformerMatcher,
                       pitch_and_onset_wise_times,
                       pitch_and_onset_wise_times_ornament,
                       get_score_to_perf_map)
from .utils import (node_array,
                    save_parangonada_csv)
from .pretrained_models import (AlignmentTransformer)