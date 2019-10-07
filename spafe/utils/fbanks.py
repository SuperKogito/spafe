import numpy as np
import functions, conversion
from __future__ import division

################################################################################
#                      Mel-filter-banks implementation
################################################################################

def hz2mel(hz):
    """
    Convert a value in Hertz to Mels

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.)

def mel2hz(mel):
    """
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10**(mel / 2595.0) - 1)

def mel_filter_banks(nfilt=20, nfft= 512, samplerate=16000, lowfreq=0, highfreq=None):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    Args:
        nfilt: the number of filters in the filterbank, default 20.
        nfft: the FFT size. Default is 512.
        samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
        lowfreq: lowest band edge of mel filters, default 0 Hz
        highfreq: highest band edge of mel filters, default samplerate/2

    Returns:
        A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq  = highfreq or samplerate/2
    # compute points evenly spaced in mels
    lowmel    = hz2mel(lowfreq)
    highmel   = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
    bin   = numpy.floor((nfft + 1) * conversion.mel2hz(melpoints) / samplerate)
    fbank = numpy.zeros([nfilt, nfft // 2 + 1])
ear_q
    for j in range(0, nfilt):
        for i in range(int(bin[j]),   int(bin[j+1])): fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])): fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

################################################################################
#                    Gammatone-filter-banks implementation
################################################################################
"""
based on:
    https://github.com/jthiem/gtfblib
"""

def Hz2ERBnum(Hz):
    """
    Return the ERB filter number for a given frequency.

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    """
    return 21.4 * np.log10(Hz * 4.37e-3 + 1.0)

def ERBnum2Hz(ERB):
    """
    Return the frequency of given ERB filter.
    """
    return (10**(ERB / 21.4) - 1.0) / 4.37e-3

def ERBspacing_given_N(cf_first, cf_last, N):
    ERB_first = Hz2ERBnum(cf_first)
    ERB_last  = Hz2ERBnum(cf_last)
    cfs       = ERBnum2Hz(np.linspace(ERB_first, ERB_last, N))
    return cfs

def ERBspacing_given_spacing(cf_first, cf_last, ERBstep):
    ERB_first = Hz2ERBnum(cf_first)
    ERB_last  = Hz2ERBnum(cf_last)
    cfs       = ERBnum2Hz(np.arange(ERB_first, ERB_last, ERBstep))
    return cfs

def gammatone_filter_banks(nfilt=20, nfft= 512, fs=16000, lowfreq=0, highfreq=None,
                           cfs=None, EarQ=(1/0.108), Bfact=1.0186):
    """
    Compute the Gammatone filter banks

    Args:
        nfilt: the number of filters in the filterbank, default 20.
        nfft: the FFT size. Default is 512.
        samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
        lowfreq: lowest band edge of mel filters, default 0 Hz
        highfreq: highest band edge of mel filters, default samplerate/2

    Returns:
        X
    """
    if cfs is None:
        cfs = ERBspacing_given_N(80, 0.9*fs/2, 32)

    ERB = cfs/EarQ+24.7
    nfilt = cfs.shape[0]

    # shortcuts used by derived methods
    _omega = 2*np.pi*cfs/fs
    _normB = 2*np.pi*Bfact*self.ERB/fs

################################################################################
#                      Gammatone-filter-banks implementation
################################################################################
"""
based on https://github.com/mcusi/gammatonegram/
"""
import numpy as np

# Slaney's ERB Filter constants
EarQ  = 9.26449 
minBW = 24.7 

def generate_center_frequencies(fl, fh, nfilt):
    # init vars
    m  = np.array(range(nfilt)) + 2
    c  = EarQ * minBW
    M  = nfilt

    # compute center frequencies
    cfreqs = (fh + c) * np.exp((m/M) * np.log((fl + c) / (fh + c))) - c
    return cfreqs[::-1]

def compute_gain(fcs, B, wT, T):
    # pre-computations for simplification
    K    = np.exp(B * T)
    Cos  = np.cos(2 * fcs * np.pi * T)
    Sin  = np.sin(2 * fcs * np.pi * T)
    Smax = np.sqrt(3 + 2**(3/2))
    Smin = np.sqrt(3 - 2**(3/2))
        
    # define A matrix rows
    A11 = (Cos + Smax * Sin) / K
    A12 = (Cos - Smax * Sin) / K
    A13 = (Cos + Smin * Sin) / K
    A14 = (Cos - Smin * Sin) / K
    
    # Compute gain (vectorized)
    A     = np.array([A11, A12, A13, A14])
    Kj    = np.exp(1j*wT)
    Kjmat = np.array([Kj, Kj, Kj, Kj]).T
    G     =  2 * T * Kjmat * (A.T - Kjmat)
    Coe   = -2 / K**2 - 2*Kj**2 +  2*(1 + Kj**2) / K
    Gain  = np.abs(G[:,0] * G[:,1] * G[:,2] * G[:, 3] * Coe**-4)
    return A, Gain

def gammatone_filter_banks(nfilts=20, nfft=512, fs=16000, fmin=0, fmax=10000, order=1):
    # define custom difference func
    Dif = lambda u, a: u - a.reshape(nfilts, 1) 

    # init vars
    fbank  = np.zeros([nfilts, nfft])
    width  = 1.0 
    maxlen = nfft // 2 + 2
    T      = 1 / fs
    n      = 4
    u      = np.exp(1j * 2 * np.pi * np.array(range(nfft // 2 + 1)) / nfft)
    idx    = range(nfft // 2 + 1)  
   
    # computer center frequencies, convert to ERB scale and compute bandwidths
    fcs  = generate_center_frequencies(fmin, fmax, nfilts)
    ERB  = width * ((fcs / EarQ)**order + minBW**order)**(1 / order)
    B    = 1.019 * 2 * np.pi * ERB
    
    # compute input vars
    wT   = 2 * fcs * np.pi * T  
    pole = np.exp(1j*wT) / np.exp(B*T)
    
    # compute gain and A matrix 
    A, Gain = compute_gain(fcs, B, wT, T) 
    
    # compute fbank    
    fbank[:, idx] = (
                      (T**4 / Gain.reshape(nfilts, 1)) * 
                      np.abs(Dif(u, A[0]) * Dif(u, A[1]) * Dif(u, A[2]) * Dif(u, A[3])) * 
                      np.abs(Dif(u, pole) * Dif(u, pole.conj()))**(-n)
                    )           
    # make sure all filters has max value = 1.0
    fbs = np.array([ f/np.max(f) for f in fbank[:, range(maxlen)] ])
    return fbs
