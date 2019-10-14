##############################################################################################
#                           linear-filter-banks implementation
##############################################################################################
import numpy as np


def linear_filter_banks(nfilts=20, nfft= 512, fs=16000, lowfreq=0, highfreq=None):
    """
    Compute linear-filterbanks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilt    (int) : the number of filters in the filterbank. (Default 20)
        nfft     (int) : the FFT size. (Default is 512)
        fs       (int) : sample rate/ sampling frequency of the signal.
                         (Default 16000 Hz)
        lowfreq  (int) : lowest band edge of mel filters. (Default 0 Hz)
        highfreq (int) : highest band edge of mel filters.(Default samplerate/2)

    Returns:
        (numpy array) array of size nfilt * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    highfreq  = highfreq or fs/2

    # compute points evenly spaced in mels (points are in Hz)
    melpoints = np.linspace(lowfreq, highfreq, nfilts + 2)

    # we use fft bins, so we have to convert from Hz to fft bin number
    bin   = np.floor((nfft + 1) * melpoints / fs)
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # compute amps of fbanks
    for j in range(0, nfilts):
        b0, b1, b2 = bin[j],  bin[j+1], bin[j+2]
        fbank[j, int(b0): int(b1)] = (np.arange(int(b0), int(b1)) - int(b0)) / (b1-b0)
        fbank[j, int(b1): int(b2)] = (int(b2) - np.arange(int(b1), int(b2))) / (b2-b1)

    return np.abs(fbank)
