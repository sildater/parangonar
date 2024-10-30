#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: skip-file
"""
This module contains tests.
"""

import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
MATCH_FILES = [os.path.join(DATA_PATH, fn) for fn in ["mozart_k265_var1.match"]]
