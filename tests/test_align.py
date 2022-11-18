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

result_dtw = np.array([[ 0,  0],
       [ 1,  1],
       [ 2,  2],
       [ 3,  3],
       [ 3,  4],
       [ 4,  5],
       [ 4,  6]])

result_nw = np.array([[ 0,  0],
       [ 1,  1],
       [ 2,  2],
       [ 3,  3],
       [-1,  4],
       [-1,  5],
       [ 4,  6]])

class TestAlignment(unittest.TestCase):
    def test_DTW_align(self, **kwargs):

        vanillaDTW = DTW()
        _, path = vanillaDTW(array1, array2)
        self.assertTrue(np.all(result_dtw == path))
        
    def test_NWDTW_align(self, **kwargs):

        vanillaNW_DTW = NW_DTW()
        _, path = vanillaNW_DTW(array1, array2)
        self.assertTrue(np.all(result_nw == path))
        
    def test_NW_align(self, **kwargs):
        
        vanillaNW = NW()
        _, path = vanillaNW(array1, array2)
        self.assertTrue(np.all(result_nw == path))
        
        
if __name__ == "__main__":
    unittest.main()
        