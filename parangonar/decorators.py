#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Optional dependency decorators.

Provides fallback decorators when optional dependencies like numba are not installed.
"""

from typing import Any, Callable
from functools import wraps
import warnings

try:
    from numba import jit as numba_jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def numba_jit(func, *args, **kwargs):
        """
        Fallback decorator when numba is not available.

        This decorator acts as an identity function - it returns the original
        function unchanged. This allows code to run without numba acceleration,
        albeit slower.

        Usage:
            @jit(nopython=True)
            def my_function(x):
                return x + 1
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"numba is not installed. Function '{func.__name__}' will run "
                f"without JIT compilation, which may be slower. Install with: "
                f"pip install parangonar[accelerated]",
                RuntimeWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper
