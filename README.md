![](media/logo.png)

Spafe: Simplified Python Audio Features Extraction
==================================================

[![Build Status](https://travis-ci.org/SuperKogito/spafe.svg?branch=master)](https://travis-ci.org/SuperKogito/spafe) [![Documentation Status](https://readthedocs.org/projects/spafe/badge/?version=latest)](https://spafe.readthedocs.io/en/latest/?badge=latest) [![License](https://img.shields.io/badge/license-BSD%203--Clause%20License%20(Revised)%20-blue)](https://github.com/SuperKogito/spafe/blob/master/LICENSE) [![Python](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue)](https://www.python.org/doc/versions/) [![Coverage Status](https://coveralls.io/repos/github/SuperKogito/spafe/badge.svg?branch=master)](https://coveralls.io/github/SuperKogito/spafe?branch=master) [![codecov](https://codecov.io/gh/SuperKogito/spafe/branch/master/graph/badge.svg)](https://codecov.io/gh/SuperKogito/spafe) [![PyPI version](https://badge.fury.io/py/spafe.svg)](https://badge.fury.io/py/spafe) [![anaconda](https://anaconda.org/superkogito/spafe/badges/version.svg)](https://anaconda.org/SuperKogito/spafe) [![Downloads](https://pepy.tech/badge/spafe)](https://pepy.tech/project/spafe) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/e94b18b0e9a040d4bc30d478879f86eb)](https://www.codacy.com/manual/SuperKogito/spafe?utm_source=github.com&utm_medium=referral&utm_content=SuperKogito/spafe&utm_campaign=Badge_Grade) [![codebeat badge](https://codebeat.co/badges/97f81ec3-b8a3-42ff-a9f5-f6cf165f4448)](https://codebeat.co/projects/github-com-superkogito-spafe-master)


#  Table of Contents

- [Spafe](#Spafe)
- [Installation](#Installation)
  - [Dependencies](#Dependencies)
  - [Installation guide](#Installation-guide)
- [How to use](#How-to-use)
- [Contributing](#Contributing)
- [Citing](#citing)

# Spafe
spafe aims to simplify features extractions from **mono audio** files.
Spafe includes various computations related to filter banks, spectrograms, frequencies and cepstral features .
The library has the following structure:
![](media/spafe-structure.png)

## Filter banks
![](media/bark_fbanks.png)
  - ***Bark filter banks***
  - ***Gammatone filter banks***
  - ***Linear filter banks***
  - ***Mel filter banks***

## Spectrograms
![](media/melspectrogram.png)  
  - ***Bark spectrogram***
  - ***CQT spectrogram***
  - ***Erb spectrogram***
  - ***Mel spectrogram***

## Features
![](media/gfcc.png)
  - ***Bark Frequency Cepstral Coefﬁcients (BFCCs)***
  - ***Constant Q-transform Cepstral Coeﬃcients (CQCCs)***
  - ***Gammatone Frequency Cepstral Coefﬁcients (GFCCs)***
  - ***Linear Frequency Cepstral Coefﬁcients (LFCCs)***
  - ***Linear Prediction Components (LPCs)***
  - ***Mel Frequency Cepstral Coefﬁcients (MFCCs)***
  - ***Inverse Mel Frequency Cepstral Coefﬁcients (IMFCCs)***
  - ***Magnitude based Spectral Root Cepstral Coefficients (MSRCCs)***
  - ***Normalized Gammachirp Cepstral Coefficients (NGCCs)***
  - ***Power-Normalized Cepstral Coefficients (PNCCs)***
  - ***Phase based Spectral Root Cepstral Coefficients (PSRCCs)***
  - ***Perceptual Linear Prediction Coefficents (PLPs)***
  - ***Rasta Perceptual Linear Prediction Coefficents (RPLPs)***

The theory behind features computed using spafe can be summmarized in the following graph:
![](media/features-extraction-algorithms.png)

## Frequencies
![](media/dominant_frequencies.png)
  - ***Dominant frequencies***
  - ***Fundamental frequencies***

## Installation
### Dependencies

spafe requires:

-	[Python](https://www.python.org/) (>= 3.5)
-	[NumPy](https://numpy.org/) (>= 1.18.1)
-	[SciPy](https://scipy.org/) (>= 1.4.1)

if you want to use the visualization module/ functions of spafe, you will need to install:

- [Matplotlib](https://matplotlib.org/) (>= 3.5.2)


### Installation guide
Once you have the Dependencies installed, use one of the following install options.

#### Install from PyPI
- To freshly install spafe:
```
pip install -U spafe
```
-  To update an existing installation:
```
pip install -U spafe
```

#### Install from Anaconda
- Spafe is also available on anaconda:
```
conda install spafe
```

### Install from source
- You can build spafe from source, by following:
```
git clone git@github.com:SuperKogito/spafe.git
cd spafe
python setup.py install
```

## How to use

Various examples on how to use spafe are present in the documentation [https://superkogito.github.io/spafe/dev/](https://superkogito.github.io/spafe/dev/).

**<!>** Please make sure you are referring to the correct documentation version.

## Contributing

Contributions are welcome and encouraged. To learn more about how to contribute to spafe please refer to the [Contributing guidelines](https://github.com/SuperKogito/spafe/blob/master/CONTRIBUTING.md)

## Citing

If you want to cite spafe in your work, use the following:
```
@software{ayoub_malek_2020,
    author  = {Ayoub Malek},
    title   = {spafe/spafe: 0.1.2},
    month   = Apr,
    year    = 2020,
    version = {0.1.2},
    url     = {https://github.com/SuperKogito/spafe}
}
```
