#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The top level of the package contains functions to
note-align symbolic music data.
"""
import pkg_resources
EXAMPLE = pkg_resources.resource_filename("parangonar", 
                                          "data/mozart_k265_var1.match")


from .match import AnchorPointNoteMatcher, AutomaticNoteMatcher
from .evaluate import fscore_alignments

__all__ = [
    "AnchorPointNoteMatcher",
    "AutomaticNoteMatcher",
    "fscore_alignments",
]
