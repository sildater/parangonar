#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for alignment utilities.
"""
import unittest
import numpy as np
from parangonar.match import AutomaticNoteMatcher


RNG = np.random.RandomState(1984)

array1 = np.array([[0,1,2,3,6]]).T
array2 = np.array([[0,1,2,3,4,5,6]]).T


class TestNoteAlignment(unittest.TestCase):
    def test_DTW_align(self, **kwargs):

        vanillaDTW = DTW()
        _, path = vanillaDTW(array1, array2)
        self.assertTrue(np.all(result_dtw == path))
        
   
        
        
if __name__ == "__main__":
    unittest.main()
        