#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from setuptools import find_packages, setup
import numpy as np

# Package meta-data.
NAME = 'parangonar'
DESCRIPTION = 'Symbolic music alignment'
KEYWORDS = 'match alignment midi performance score'
URL = "https://github.com/sildater/parangonar"
AUTHOR = 'Silvan Peter'
REQUIRES_PYTHON = '>=3.7'
VERSION = '0.0.2'

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy',
    'scipy',
    'partitura',
    'python-hiddenmarkov'
]

include_dirs = [np.get_include()]
here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=KEYWORDS,
    author=AUTHOR,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license="Apache 2.0",
)
