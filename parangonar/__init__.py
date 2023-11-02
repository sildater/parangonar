#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The top level of the package contains functions to
note-align symbolic music data.
"""
import pkg_resources
EXAMPLE = pkg_resources.resource_filename("parangonar", 
                                          "assets/mozart_k265_var1.match")
ALIGNMENT_TRANSFORMER_CHECKPOINT = pkg_resources.resource_filename("parangonar", 
                                          "assets/alignment_transformer_checkpoint.pt")


from .match import AnchorPointNoteMatcher, AutomaticNoteMatcher, DualDTWNoteMatcher
from .evaluate import fscore_alignments, plot_alignment

__all__ = [
    "AnchorPointNoteMatcher",
    "AutomaticNoteMatcher",
    "fscore_alignments",
]
