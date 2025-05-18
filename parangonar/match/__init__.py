#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods for note alignment.
"""

from .matchers import (
    AnchorPointNoteMatcher,
    AutomaticNoteMatcher,
    CleanOrnamentMatcher,
    DualDTWNoteMatcher,
    TheGlueNoteMatcher,
    pitch_and_onset_wise_times,
    pitch_and_onset_wise_times_ornament,
    get_score_to_perf_map,
)
from .online_matchers import (
    OnlineTransformerMatcher, 
    OnlinePureTransformerMatcher,
    TOLTWMatcher
)
from .utils import node_array, save_parangonada_csv
from .pretrained_models import AlignmentTransformer, TheGlueNote
