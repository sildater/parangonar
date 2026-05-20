#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The top level of the package
contains functions to
align music data.
"""

import logging

from importlib.metadata import version
from importlib.resources import files

# define a version variable
__version__ = version("parangonar")

# package-level logger; library users can configure it via
#   import logging; logging.getLogger("parangonar").setLevel(logging.DEBUG)
logger = logging.getLogger("parangonar")

#: An example MusicXML file for didactic purposes
EXAMPLE = str(files("parangonar") / "assets" / "mozart_k265_var1.match")
ALIGNMENT_TRANSFORMER_CHECKPOINT = str(files("parangonar") / "assets" / "alignment_transformer_checkpoint.pt")
THEGLUENOTE_CHECKPOINT = str(files("parangonar") / "assets" / "thegluenote_small_checkpoint.pt")

from .match import (
    AnchorPointNoteMatcher,
    AutomaticNoteMatcher,
    DualDTWNoteMatcher,
    TheGlueNoteMatcher,
    AudioToScoreMatcher,
    AudioToScoreMatcherLimited,
)

from .match import (
    OnlineTransformerMatcher,
    OnlinePureTransformerMatcher,
    TOLTWMatcher,
    OLTWMatcher,
) 


from .mismatch import RepeatIdentifier, SubPartMatcher
from .evaluate import (
    fscore_alignments,
    print_fscore_alignments,
    plot_alignment,
    plot_alignment_comparison,
    save_piano_precision_csv,
    save_sonic_visualizer_csvs,
    save_maps,
    match_midis
)

__all__ = [
    "AnchorPointNoteMatcher",
    "AutomaticNoteMatcher",
    "DualDTWNoteMatcher",
    "TheGlueNoteMatcher",
    "AudioToScoreMatcher",
    "AudioToScoreMatcherLimited",
    "OnlineTransformerMatcher",
    "OnlinePureTransformerMatcher",
    "TOLTWMatcher",
    "OLTWMatcher",
    "RepeatIdentifier",
    "SubPartMatcher",
    "fscore_alignments",
    "print_fscore_alignments",
    "plot_alignment_comparison",
    "plot_alignment",
    "save_piano_precision_csv",
    "save_sonic_visualizer_csvs",
    "save_maps",
    "match_midis"
]
