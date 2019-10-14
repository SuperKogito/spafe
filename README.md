[![Build Status](https://travis-ci.org/SuperKogito/spafe.svg?branch=master)](https://travis-ci.org/SuperKogito/spafe)
[![Documentation Status](https://readthedocs.org/projects/spafe/badge/?version=latest)](https://spafe.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-BSD%203--Clause%20License%20(Revised)%20-blue)](https://github.com/SuperKogito/spafe/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python%20-3.5%2B-blue)](https://www.python.org/downloads/release/python-350/)

<p align="center">
<img src="logo.jpg">
</p>

# spafe: Simplified Python Audio-Features Extraction
spafe is a wrapper to simplify various types of features extractions. The library covers: MFCC, IMFCC, GFCC, LFCC, PNCC, PLP etc.
It also provides various filterbank modules (Mel, Bark and Gammatone filterbanks) and other spectral statistics.

# Contributions, feedback and suggestions are welcome
Contributions are encouraged as the library and any help is appreciated.



# Examples
## spafe.fbanks
### Bark filterbanks

    import matplotlib.pyplot as plt
    from spafe.fbanks import bark_fbanks

    # compute fbanks
    fbanks = bark_fbanks.bark_filter_banks(nfilts=24, nfft=512, fs=16000)

    # plot fbanks
    for i in range(len(fbanks)):
        plt.plot(fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()

<p align="center">
<img src="docs/source/fbanks/images/bark_fbanks.png">
</p>

### Gammatone filterbanks

    import matplotlib.pyplot as plt
    from spafe.fbanks import gammatone_fbanks

    # compute fbanks
    fbanks = gammatone_fbanks.gammatone_filter_banks(nfilts=24, nfft=512, fs=16000)

    # plot fbanks
    for i in range(len(fbanks)):
       plt.plot(fbanks[i])
       plt.ylim(0, 1.1)
       plt.grid(True)
       plt.ylabel(ylabel)
       plt.xlabel(xlabel)
       plt.show()

<p align="center">
<img src="docs/source/fbanks/images/gammatone_fbanks.png">
</p>

### Mel filterbanks
    import matplotlib.pyplot as plt
    from spafe.fbanks import mel_fbanks

    # compute fbanks
    fbanks = mel_fbanks.mel_filter_banks(nfilts=24, nfft=512, fs=16000)

    # plot fbanks
    for i in range(len(fbanks)):
        plt.plot(fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()


<p align="center">
<img src="docs/source/fbanks/images/mel_fbanks.png">
</p>

## spafe.features
### MFCC, IMFCC, MFE
    import scipy.io.wavfile
    import spafe.utils.vis as vis
    from spafe.features.mfcc import mfcc, imfcc, mfe


    # read wave file
    fs, sig = scipy.io.wavfile.read('../test.wav')

    # compute mfccs and mfes
    mfccs  = mfcc(sig, 13)
    imfccs = imfcc(sig, 13)
    mfes   = mfe(sig, fs)

    # visualize features
    vis.visualize(mfccs, 'MFCC Coefficient Index','Frame Index')
    vis.visualize(imfccs, 'IMFCC Coefficient Index','Frame Index')
    vis.plot(mfes,  'MFE Coefficient Index','Frame Index')


<p align="center">
<img src="docs/source/features/images/mfcc.png">
</p>

<p align="center">
<img src="docs/source/features/images/imfcc.png">
</p>

<p align="center">
<img src="docs/source/features/images/mfe.png">
</p>
