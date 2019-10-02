################################################################################
#                      Bark-filter-banks implementation
################################################################################
import numpy as np


def hz2bark(f):
    """
    Convert Hz frequencies to Bark.
    
    Args:
    -----
    f (np.array) : input frequencies [Hz].
    
    Returns:
    --------
    fb (np.array): frequencies in Bark [Bark].
    """
    return 6. * np.arcsinh(f / 600. )  

def bark2hz(fb):
    """
    Convert Bark frequencies to Hz.
    
    Args:
    -----
    fb (np.array) : input frequencies [Bark].
    
    Returns:
    --------
    f (np.array)  : frequencies in Hz [Hz].
    """
    return 600. * np.sinh( fb / 6.)

def fft2bark(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.
    
    Args:
    -----
    fft (np.array) : fft bin numbers.
    
    Returns:
    --------
    fb (np.array): frequencies in Bark [Bark].
    """
    return hz2bark((fft * fs) / (nfft + 1))

def bark2fft(fb, fs=16000, nfft=512):   
    """
    Convert Bark frequencies to fft bins.
    
    Args:
    -----
    fb (np.array): frequencies in Bark [Bark].
    
    Returns:
    --------
    fft (np.array) : fft bin numbers.
    """
    return (nfft + 1) * bark2hz(fb) / fs

def Fm(fb, fc):
    """
    Compute bark filter around a certain frequency in bark

    Args:
    -----
    fb (int): frequency in Bark [Bark].
    fc (int): center frequency in Bark [Bark].

    Returns:
    --------
    (float) : associated Bark filter value/amplitude.
    """
    if   fc - 2.5 <= fb <= fc - 0.5 : return 10**(2.5 * (fb - fc + 0.5))
    elif fc - 0.5 <  fb <  fc + 0.5 : return 1
    elif fc + 0.5 <= fb <= fc + 1.3 : return 10**(-2.5 * (fb - fc - 0.5))
    else                            : return 0
    
def bark_filter_banks(nfilt=20, nfft=512, fs=16000, lowfreq=0, highfreq=None):
    """
    Compute a Bark-filterbank. The filters are stored in the rows, the columns correspond
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
   
    # compute points evenly spaced in bark
    lowbark    = hz2bark(lowfreq)
    highbark   = hz2bark(highfreq)
    barkpoints = np.linspace(lowbark, highbark, nfilt + 4)

    # The frequencies array/ points are in Bark, but we use fft bins, so we 
    # have to convert from Bark to fft bin number
    bin   = np.floor(bark2fft(barkpoints))
    fbank = np.zeros([nfilt, nfft // 2 + 1])

    for j in range(2, nfilt + 2):
        for i in range(int(bin[j-2]), int(bin[j+2])):
            fc            = barkpoints[j]
            fb            = fft2bark(i)
            fbank[j-2, i] = Fm(fb, fc)
    return fbank