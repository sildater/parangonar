#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods for note and sequence alignment.

"""

from .dtw import DTW, DTWSL
from .nwtw import NW_DTW, NW
from .matchers import (AnchorPointNoteMatcher, 
                       AutomaticNoteMatcher,
                       CleanOrnamentMatcher,
                       DualDTWNoteMatcher,
                       pitch_and_onset_wise_times,
                       pitch_and_onset_wise_times_ornament,
                       get_score_to_perf_map)
from .online_matchers import (OnlineTransformerMatcher, 
                              OnlinePureTransformerMatcher)
from .utils import (node_array,
                    save_parangonada_csv)
from .pretrained_models import (AlignmentTransformer)