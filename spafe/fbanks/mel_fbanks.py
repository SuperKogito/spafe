################################################################################
#                      Mel-filter-banks implementation
################################################################################
import numpy as np


def hz2mel(hz):
    """
    Convert a value in Hertz to Mels

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.)

def mel2hz(mel):
    """
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10**(mel / 2595.0) - 1)

def mel_filter_banks(nfilt=20, nfft= 512, fs=16000, lowfreq=0, highfreq=None):
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
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    
    # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
    bin   = np.floor((nfft + 1) * mel2hz(melpoints) / fs)
    fbank = np.zeros([nfilt, nfft // 2 + 1])

    for j in range(0, nfilt):
        for i in range(int(bin[j]),   int(bin[j+1])): fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])): fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank
