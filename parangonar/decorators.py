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

    def numba_jit(*jit_args, **jit_kwargs):
        """
        Fallback for numba.jit when numba is not installed.
        Acts like a decorator factory so it matches @jit(nopython=True).
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"Numba is not installed. Function '{func.__name__}' will run "
                    f"without JIT compilation (slower). Install with: "
                    f"pip install numba",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator