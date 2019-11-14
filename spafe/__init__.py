#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Top-level module for spafe
"""
__version__ = '0.1.0'

import sys
import warnings

# Throw a deprecation warning if we're on legacy python
if sys.version_info < (3,):
    warnings.warn('You are using spafe with Python 2.'
                  'Please note that spafe requires Python 3 or later.',
                  FutureWarning)
