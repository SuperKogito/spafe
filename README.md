![](https://github.com/SuperKogito/spafe/blob/master/media/logo.png?raw=true)

# Spafe

Simplified Python Audio Features Extraction

[![Build Status](https://github.com/SuperKogito/spafe/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperKogito/spafe/actions)
[![docs.rs](https://img.shields.io/docsrs/docs)](https://superkogito.github.io/spafe/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause%20License%20(Revised)%20-blue)](https://github.com/SuperKogito/spafe/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/doc/versions/)
[![codecov](https://codecov.io/gh/SuperKogito/spafe/branch/master/graph/badge.svg)](https://codecov.io/gh/SuperKogito/spafe)
[![codebeat badge](https://codebeat.co/badges/97f81ec3-b8a3-42ff-a9f5-f6cf165f4448)](https://codebeat.co/projects/github-com-superkogito-spafe-master)
[![PyPI version](https://badge.fury.io/py/spafe.svg)](https://badge.fury.io/py/spafe)
[![anaconda](https://anaconda.org/superkogito/spafe/badges/version.svg)](https://anaconda.org/SuperKogito/spafe)
[![Downloads](https://pepy.tech/badge/spafe)](https://pepy.tech/project/spafe)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6824667.svg)](https://doi.org/10.5281/zenodo.6824667)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04739/status.svg)](https://doi.org/10.21105/joss.04739)

#  Table of Contents

- [Structure](#Structure)
  - [Filter banks](#Filter-banks)
  - [Spectrograms](#Spectrograms)
  - [Features](#Features)
  - [Frequencies](#Frequencies)
- [Installation](#Installation)
  - [Dependencies](#Dependencies)
  - [Installation guide](#Installation-guide)
- [How to use](#How-to-use)
- [Contributing](#Contributing)
- [Citing](#Citing)

# Structure
spafe aims to simplify feature extractions from **mono audio** files.
Spafe includes various computations related to filter banks, spectrograms, frequencies and cepstral features .
The library has the following structure:
![](https://github.com/SuperKogito/spafe/raw/master/media/spafe-structure.png)

## Filter banks
![](https://github.com/SuperKogito/spafe/blob/master/media/bark_fbanks.png?raw=true)
  - Bark filter banks
  - Gammatone filter banks
  - Linear filter banks
  - Mel filter banks

## Spectrograms
![](https://github.com/SuperKogito/spafe/blob/master/media/melspectrogram.png?raw=true)  
  - Bark spectrogram
  - CQT spectrogram
  - Erb spectrogram
  - Mel spectrogram

## Features
![](https://github.com/SuperKogito/spafe/blob/master/media/gfcc.png?raw=true)
  - Bark Frequency Cepstral Coefﬁcients (BFCCs)
  - Constant Q-transform Cepstral Coeﬃcients (CQCCs)
  - Gammatone Frequency Cepstral Coefﬁcients (GFCCs)
  - Linear Frequency Cepstral Coefﬁcients (LFCCs)
  - Linear Prediction Components (LPCs)
  - Mel Frequency Cepstral Coefﬁcients (MFCCs)
  - Inverse Mel Frequency Cepstral Coefﬁcients (IMFCCs)
  - Magnitude based Spectral Root Cepstral Coefficients (MSRCCs)
  - Normalized Gammachirp Cepstral Coefficients (NGCCs)
  - Power-Normalized Cepstral Coefficients (PNCCs)
  - Phase based Spectral Root Cepstral Coefficients (PSRCCs)
  - Perceptual Linear Prediction Coefficents (PLPs)
  - Rasta Perceptual Linear Prediction Coefficents (RPLPs)

The theory behind features computed using spafe can be summmarized in the following graph:
![](https://github.com/SuperKogito/spafe/blob/master/media/features-extraction-algorithms.png?raw=true)

## Frequencies
![](https://github.com/SuperKogito/spafe/blob/master/media/dominant_frequencies.png?raw=true)
  - Dominant frequencies
  - Fundamental frequencies

## Installation
### Dependencies

spafe requires:

-	[Python](https://www.python.org/) (>= 3.5)
-	[NumPy](https://numpy.org/) (>= 1.22.0)
-	[SciPy](https://scipy.org/) (>= 1.8.0)

if you want to use the visualization module/ functions of spafe, you will need to install:

- [Matplotlib](https://matplotlib.org/) (>= 3.5.2)


### Installation guide
Once you have the Dependencies installed, use one of the following install options.

#### Install from PyPI
- To freshly install spafe:
```
pip install spafe
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

## Why use Spafe?

Unlike most existing audio feature extraction libraries ([python_speech_features](https://github.com/jameslyons/python_speech_features), [SpeechPy](https://github.com/astorfi/speechpy), [surfboard](https://github.com/novoic/surfboard) and [Bob](https://gitlab.idiap.ch/bob)), Spafe provides more options for spectral features extraction algorithms, notably:
- Bark Frequency Cepstral Coefﬁcients (BFCCs)
- Constant Q-transform Cepstral Coeﬃcients (CQCCs)
- Gammatone Frequency Cepstral Coefﬁcients (GFCCs)
- Power-Normalized Cepstral Coefficients (PNCCs)
- Phase based Spectral Root Cepstral Coefficients (PSRCCs)

Most existing libraries and to their credits provide great implementations for features extraction but are unfortunately limited to the Mel Frequency Features (MFCC) and at best have Bark frequency and linear predictive coefficients additionally. [Librosa](https://github.com/librosa/librosa) for example includes great implementation of various algorithms (only MFCC and LPC are included), based on the **Short Time Fourrier Transform (STFT)**, which is theoretically more accurate but slower than the **Discret Fourrier Transform used in Spafe**'s implementation.


## How to use

Various examples on how to use spafe are present in the documentation [https://superkogito.github.io/spafe](https://superkogito.github.io/spafe).

**<!>** Please make sure you are referring to the correct documentation version.

## Contributing

Contributions are welcome and encouraged. To learn more about how to contribute to spafe please refer to the [Contributing guidelines](https://github.com/SuperKogito/spafe/blob/master/CONTRIBUTING.md)

## Citing

-  **If you want to cite spafe as a software, please cite the version used as indexed in** [Zenodo](https://zenodo.org/):

   *Ayoub Malek. (2023). SuperKogito/spafe: Spafe: Simplified python audio features extraction (v0.3.1). Zenodo.* https://doi.org/10.5281/zenodo.7533946

   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6824667.svg)](https://doi.org/10.5281/zenodo.6824667)

- **You can also site spafe's paper as follows:**

  *Malek, A., (2023). Spafe: Simplified python audio features extraction. Journal of Open Source Software, 8(81), 4739,* https://doi.org/10.21105/joss.04739

  [![DOI](https://joss.theoj.org/papers/10.21105/joss.04739/status.svg)](https://doi.org/10.21105/joss.04739)
