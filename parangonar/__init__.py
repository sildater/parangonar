#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The top level of the package 
contains functions to
align music data.
"""

import sys
# Use importlib.metadata and importlib.resources for modern Python versions
if sys.version_info >= (3, 9):
    from importlib.metadata import version
    from importlib.resources import files
else:
    # Backport for Python 3.7-3.8
    try:
        from importlib_metadata import version
    except ImportError:
        from importlib.metadata import version
    try:
        from importlib_resources import files
    except ImportError:
        from importlib.resources import files

# define a version variable
__version__ = version("parangonar")

#: An example MusicXML file for didactic purposes
EXAMPLE = str(files("parangonar") / "assets" / "mozart_k265_var1.match")
ALIGNMENT_TRANSFORMER_CHECKPOINT = str(files("parangonar") / "assets" / "alignment_transformer_checkpoint.pt")
THEGLUENOTE_CHECKPOINT = str(files("parangonar") / "assets" / "thegluenote_small_checkpoint.pt")

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
    OLTWMatcher,
    SLTOLTWMatcher,
    SLOLTWMatcher,
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
    "OnlineTransformerMatcher",
    "OnlinePureTransformerMatcher",
    "TOLTWMatcher",
    "OLTWMatcher",
    "SLTOLTWMatcher",
    "SLOLTWMatcher",
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
