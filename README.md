<p align="center">
<img src="logo.png">
</p>

spafe: Simplified Python Audio-Features Extraction
==================================================

[![Build Status](https://travis-ci.org/SuperKogito/spafe.svg?branch=master)](https://travis-ci.org/SuperKogito/spafe) [![Documentation Status](https://readthedocs.org/projects/spafe/badge/?version=latest)](https://spafe.readthedocs.io/en/latest/?badge=latest) [![License](https://img.shields.io/badge/license-BSD%203--Clause%20License%20(Revised)%20-blue)](https://github.com/SuperKogito/spafe/blob/master/LICENSE) [![Python](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue)](https://www.python.org/doc/versions/) [![Coverage Status](https://coveralls.io/repos/github/SuperKogito/spafe/badge.svg?branch=master)](https://coveralls.io/github/SuperKogito/spafe?branch=master) [![codecov](https://codecov.io/gh/SuperKogito/spafe/branch/master/graph/badge.svg)](https://codecov.io/gh/SuperKogito/spafe) [![PyPI version](https://badge.fury.io/py/spafe.svg)](https://badge.fury.io/py/spafe) [![anaconda](https://anaconda.org/superkogito/spafe/badges/version.svg)](https://anaconda.org/SuperKogito/spafe) [![Downloads](https://pepy.tech/badge/spafe)](https://pepy.tech/project/spafe) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/e94b18b0e9a040d4bc30d478879f86eb)](https://www.codacy.com/manual/SuperKogito/spafe?utm_source=github.com&utm_medium=referral&utm_content=SuperKogito/spafe&utm_campaign=Badge_Grade) [![codebeat badge](https://codebeat.co/badges/97f81ec3-b8a3-42ff-a9f5-f6cf165f4448)](https://codebeat.co/projects/github-com-superkogito-spafe-master)

spafe aims to simplify features extractions from mono audio files. The library can extract of the following features: ***BFCC, LFCC, LPC, LPCC, MFCC, IMFCC, MSRCC, NGCC, PNCC, PSRCC, PLP, RPLP, Frequency-stats*** etc. It also provides various filterbank modules (Mel, Bark and Gammatone filterbanks) and other spectral statistics.

Installation
============

Dependencies
------------

spafe requires:

-	Python (>= 3.5)
-	NumPy (>= 1.17.2)
-	SciPy (>= 1.3.1)

User installation
-----------------

If you already have a working installation of numpy and scipy, you can simply install spafe using pip:

```
pip install -U spafe
```

or conda (*not available at the moment*\):

```
conda install spafe
```

How to use
==========

Various examples on how to use spafe filter banks or feature extraction techniques are available under [examples](https://github.com/SuperKogito/spafe/tree/master/examples).

Contributing
============

Contributions are welcome and encouraged. To learn more about how to contribute to spafe please refer to the [Contributing guidelines](https://github.com/SuperKogito/spafe/blob/master/CONTRIBUTING.md)
