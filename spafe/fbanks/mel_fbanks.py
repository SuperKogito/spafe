##############################################################################################
#                           Mel-filter-banks implementation
##############################################################################################
import numpy as np
from ..utils.converters import hz2mel, mel2hz


def mel_filter_banks(nfilts=20, nfft=512, fs=16000, lowfreq=0, highfreq=None, scale="const"):
    """
    Compute Mel-filterbanks.The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilt    (int) : the number of filters in the filterbank. (Default 20)
        nfft     (int) : the FFT size. (Default is 512)
        fs       (int) : sample rate/ sampling frequency of the signal.
                         (Default 16000 Hz)
        lowfreq  (int) : lowest band edge of mel filters. (Default 0 Hz)
        highfreq (int) : highest band edge of mel filters.(Default samplerate/2)

    Returns:
        a numpy array of size nfilt * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    highfreq  = highfreq or fs/2

    # compute points evenly spaced in mels (ponts are in Hz)
    lowmel    = hz2mel(lowfreq)
    highmel   = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilts + 2)

    # we use fft bins, so we have to convert from Hz to fft bin number
    bin   = np.floor((nfft + 1) * mel2hz(melpoints) / fs)
    fbank = np.zeros([nfilts, nfft // 2 + 1])
    c = 1
    # compute amps of fbanks
    for j in range(0, nfilts):
        b0, b1, b2 = bin[j],  bin[j+1], bin[j+2]
        if scale == "desc" : c -= 1 / nfilts
        else               : c  = 1        
        fbank[j, int(b0): int(b1)] = c*(np.arange(int(b0), int(b1)) - int(b0)) / (b1-b0)
        fbank[j, int(b1): int(b2)] = c*(int(b2) - np.arange(int(b1), int(b2))) / (b2-b1)
  
    return np.abs(fbank)


def inverse_mel_filter_banks(nfilts=20, nfft=512, fs=16000, lowfreq=0, highfreq=None):
    """
    Compute inverse Mel-filterbanks.The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilt    (int) : the number of filters in the filterbank. (Default 20)
        nfft     (int) : the FFT size. (Default is 512)
        fs       (int) : sample rate/ sampling frequency of the signal.
                         (Default 16000 Hz)
        lowfreq  (int) : lowest band edge of mel filters. (Default 0 Hz)
        highfreq (int) : highest band edge of mel filters.(Default samplerate/2)

    Returns:
        a numpy array of size nfilt * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    # generate inverse mel fbanks by inversing regular mel fbanks 
    imel_fbanks = mel_filter_banks(nfilts=24, nfft=512, fs=16000, scale="const")
    for i , pts in enumerate(imel_fbanks):
        imel_fbanks[i] = pts[::-1]  
    return np.abs(imel_fbanks)
