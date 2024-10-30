#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for alignment utilities.
"""
import unittest
import numpy as np
from parangonar.dp.dtw import DTW, WDTW, DTWSL, FDTW, l2
from parangonar.dp.nwtw import NW_DTW, NW, WNWTW, ONW

RNG = np.random.RandomState(1984)

array1 = np.array([[0, 1, 2, 3, 6]]).T
array2 = np.array([[0, 1, 2, 3, 4, 5, 6]]).T

result_dtw = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [3, 4], [4, 5], [4, 6]])

result_nw = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [-1, 4], [-1, 5], [4, 6]])

result_wnw = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [3, 4], [3, 5], [4, 6]])


class TestDTWAlignment(unittest.TestCase):
    def test_DTW_align(self, **kwargs):
        vanillaDTW = DTW()
        _, path = vanillaDTW(array1, array2, return_path=True)
        self.assertTrue(np.all(result_dtw == path))

    def test_DTWSL_align(self, **kwargs):
        vanillaDTWSL = DTWSL(metric=l2)
        _, path = vanillaDTWSL(array1, array2, return_path=True)
        self.assertTrue(np.all(result_dtw == path))

    def test_WDTW_align(self, **kwargs):
        weightedDTW = WDTW()
        path = weightedDTW(array1, array2)[0]
        self.assertTrue(np.all(result_dtw == path))

    def test_FDTW_align(self, **kwargs):
        FlexDTW = FDTW()
        path = FlexDTW(array1, array2)[0]
        self.assertTrue(np.all(result_dtw == path))


class TestNWDTWAlignment(unittest.TestCase):
    def test_NWDTW_align(self, **kwargs):
        vanillaNW_DTW = NW_DTW()
        _, path = vanillaNW_DTW(array1, array2)
        self.assertTrue(np.all(result_nw == path))

    def test_NW_align(self, **kwargs):
        vanillaNW = NW()
        _, path = vanillaNW(array1, array2)
        self.assertTrue(np.all(result_nw == path))

    def test_WNWTW_align(self, **kwargs):
        weightedNW = WNWTW()
        path = weightedNW(array1, array2)[0]
        self.assertTrue(np.all(result_wnw == path))

    def test_ONW_align(self, **kwargs):
        originalNW = ONW()
        path = originalNW(array1, array2)[0]
        self.assertTrue(np.all(result_wnw == path))


if __name__ == "__main__":
    unittest.main()
