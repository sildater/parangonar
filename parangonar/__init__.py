#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The top level of the package contains functions to
note-align symbolic music data.
"""
import pkg_resources

EXAMPLE = pkg_resources.resource_filename("parangonar", "assets/mozart_k265_var1.match")
ALIGNMENT_TRANSFORMER_CHECKPOINT = pkg_resources.resource_filename(
    "parangonar", "assets/alignment_transformer_checkpoint.pt"
)
THEGLUENOTE_CHECKPOINT = pkg_resources.resource_filename(
    "parangonar", "assets/thegluenote_small_checkpoint.pt"
)
from .match import (
    AnchorPointNoteMatcher,
    AutomaticNoteMatcher,
    DualDTWNoteMatcher,
    TheGlueNoteMatcher,
)
from .match import (
    OnlineTransformerMatcher, 
    OnlinePureTransformerMatcher,
    TOLTWMatcher,
    OLTWMatcher
)
from .mismatch import (
    RepeatIdentifier,
    SubPartMatcher
)
from .evaluate import (
    fscore_alignments,
    print_fscore_alignments,
    plot_alignment,
    plot_alignment_comparison,
    save_pianoprecision_csv,
    save_sonic_visualizer_csvs
)

__all__ = [
    "AnchorPointNoteMatcher",
    "AutomaticNoteMatcher",
    "DualDTWNoteMatcher",
    "TheGlueNoteMatcher",
    "OnlineTransformerMatcher",
    "OnlinePureTransformerMatcher",
    "TOLTWMatcher"
    "fscore_alignments",
    "plot_alignment_comparison",
    "plot_alignment",
    "save_pianoprecision_csv"
]
