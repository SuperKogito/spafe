################################################################################
#                      Bark-filter-banks implementation
################################################################################
import numpy as np
from spafe.utils.converters import hz2erb, erb2hz 
from spafe.utils.converters import fft2erb, erb2fft
from spafe.utils.converters import hz2bark, bark2hz
from spafe.utils.converters import fft2hz, hz2fft


def Hz(f, fc, fs):
    T = 1/fs
    z = np.exp(1j * 2 * np.pi * f)
    # pre-computations for simplification
    b    = hz2erb(fc) * 1.019
    K    = np.exp(-2*np.pi*b*T)
    S    = np.sqrt(3+2**(3/2))
    Cos  = np.cos(2*np.pi*fc*T)
    Sin  = np.sin(2*np.pi*fc*T)
    # compute H(z)
    nominator   = -2 * T + (2 * T * K * Cos  + 2 * S * T * K * Sin) * z**-1
    denominator = -2 + 4 * K * Cos - 2 * K**2 * z**-2
    return nominator / denominator

def generate_center_frequencies(fl, fh, nfilt):
    c    = 1000/4.37
    M    = nfilt
    fcm  = lambda m, c, fl, fh, M: (-1*c) + (fh + c) * np.exp((m / M) * np.log((fl + c) / (fh + c)))
    return np.array([fcm(m, c, fl, fh, M) for m in range(1, M)])  

def _filter_banks(nfilt=20, nfft=512, fs=16000, lowfreq=0, highfreq=None):
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
    center_freqs = generate_center_frequencies(lowfreq, highfreq, 10*nfilt + 2)
    print([round(f, 3) for f in center_freqs])
    # The frequencies array/ points are in Bark, but we use fft bins, so we 
    # have to convert from Bark to fft bin number
    bin   = np.floor(hz2fft(center_freqs))
    fbank = np.zeros([nfilt, nfft // 2 + 1])


    for j in range(2, nfilt-2, 10):
        for i in range(j-2, j+2):
            fc          = fft2hz(bin[j])
            f           = fft2hz(bin[i])
            fbank[j, i] = np.abs(Hz(f, fc, fs)) / np.abs(np.max(Hz(f, fc, fs)))
    return fbank


import matplotlib.pyplot as plt 

fbanks = _filter_banks(nfilt=49, nfft=512, fs=16000)  

# plot the gammatone filter banks 
for i in range(0, len(fbanks), 2):
    plt.plot(fbanks[i],  linewidth=2,)
    plt.ylim(0, 1.1)
    plt.grid(True)
plt.show()
