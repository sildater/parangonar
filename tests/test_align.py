#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for alignment utilities.
"""
import unittest
import numpy as np
from parangonar.match.dtw import DTW
from parangonar.match.nwtw import NW_DTW, NW


RNG = np.random.RandomState(1984)

array1 = np.array([[0,1,2,3,6]]).T
array2 = np.array([[0,1,2,3,4,5,6]]).T

class TestAlignment(unittest.TestCase):
    def test_DTW_align(self, **kwargs):

        vanillaDTW = DTW()
        print(vanillaDTW(array1, array2))
        self.assertTrue(True)
        
    def test_NWDTW_align(self, **kwargs):

        vanillaNW_DTW = NW_DTW()
        print(vanillaNW_DTW(array1, array2))
        self.assertTrue(True)
        
    def test_NW_align(self, **kwargs):
        
        vanillaNW = NW()
        print(vanillaNW(array1, array2))
        self.assertTrue(True)
        
        
if __name__ == "__main__":
    unittest.main()
        