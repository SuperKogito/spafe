#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Top-level module for spafe
"""
import sys
import warnings

from . import fbanks
from . import features
from . import frequencies
from . import utils

# Throw a deprecation warning if we're on legacy python
if sys.version_info < (3,):
    warnings.warn('You are using spafe with Python 2.'
                  'Please note that spafe requires Python 3 or later.',
                  FutureWarning)
