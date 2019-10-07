################################################################################
#                      Mel-filter-banks implementation
################################################################################
import numpy as np
from spafe.utils.converters import hz2mel, mel2hz


def mel_filter_banks(nfilts=20, nfft= 512, fs=16000, lowfreq=0, highfreq=None):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    Args:
        nfilt    : the number of filters in the filterbank, default 20.
        nfft     : the FFT size. Default is 512.
        fs       : the sample rate of the signal we are working with, in Hz. Affects mel spacing.
        lowfreq  : lowest band edge of mel filters, default 0 Hz
        highfreq : highest band edge of mel filters, default samplerate/2

    Returns:
        A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq  = highfreq or fs/2
    
    # compute points evenly spaced in mels
    lowmel    = hz2mel(lowfreq)
    highmel   = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilts + 2)
    
    # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
    bin   = np.floor((nfft + 1) * mel2hz(melpoints) / fs)
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    for j in range(0, nfilts):
        for i in range(int(bin[j]),   int(bin[j+1])): 
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        
        for i in range(int(bin[j+1]), int(bin[j+2])): 
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank
