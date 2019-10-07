################################################################################
#                      Gammatone-filter-banks implementation
################################################################################
"""
Created on Sat May 27 15:37:50 2017
Python version of:
D. P. W. Ellis (2009). "Gammatone-like spectrograms", web resource. http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/
On the corresponding webpage, Dan notes that he would be grateful if you cited him if you use his work (as above).
This python code does not contain all features present in MATLAB code.
Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017
"""
import numpy as np
import scipy.signal as sps
import scipy.io.wavfile as wf
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
    #after Slaney's MakeERBFilters
    EarQ  = 9.26449
    minBW = 24.7
    m     = np.array(range(nfilt)) + 1
    c     = EarQ * minBW
    M     = nfilt
    # compute center frequencies
    cfreqs = (fh + c) * np.exp((m/M) * np.log((fl + c) / (fh + c))) - c
    cfreqs = cfreqs[::-1]
    return cfreqs

def H(n, fc, T, B):
    K    = np.exp(B * T)
    Cos  = 2 * T * np.cos(2 * fc * np.pi * T)
    Sin  = 2 * T * np.sin(2 * fc * np.pi * T)
    if n <= 2 : S = np.sqrt(3 + 2**(3/2))
    else      : S = np.sqrt(3 - 2**(3/2))
    return -0.5*((Cos / K) + (-1)**n * ((S * Sin) / K))


def F(fs, fl, fh, nfilts=64):
    # init vars
    EarQ  = 9.26449
    minBW = 24.7
    order = 1
    width = 1.0
    
    # compute center frequencies
    fcs   = generate_center_frequencies(fl, fh, nfilts)
    ERB   = width*((fcs / EarQ)**order + minBW**order)**(1 / order)
     
    # pre-computations for simplification
    T    = 1 / fs
    B    = 1.019 * 2 * np.pi * ERB
    wT   = 2 * fcs * np.pi * T  
    K    = np.exp(B * T)
    Cos  = 2 * T * np.cos(2 * fcs * np.pi * T)
    Sin  = 2 * T * np.sin(2 * fcs * np.pi * T)
    Smax = np.sqrt(3 + 2**(3/2))
    Smin = np.sqrt(3 - 2**(3/2))

    # compute filters
    A11 = -0.5*(Cos / K + Smax * Sin / K) 
    A12 = -0.5*(Cos / K - Smax * Sin / K)
    A13 = -0.5*(Cos / K + Smin * Sin / K) 
    A14 = -0.5*(Cos / K - Smin * Sin / K)

    zros = -np.array([A11, A12, A13, A14]) / T
       
    gain = np.abs(
                    (-2*T*np.exp(2j * wT) + (2*T*np.exp(1j * wT) / K) * (np.cos(wT) - Smin * np.sin(wT))) * 
                    (-2*T*np.exp(2j * wT) + (2*T*np.exp(1j * wT) / K) * (np.cos(wT) + Smin * np.sin(wT))) * 
                    (-2*T*np.exp(2j * wT) + (2*T*np.exp(1j * wT) / K) * (np.cos(wT) - Smax * np.sin(wT))) * 
                    (-2*T*np.exp(2j * wT) + (2*T*np.exp(1j * wT) / K) * (np.cos(wT) + Smax * np.sin(wT))) * (-2 / np.exp(2*B*T) - 2*np.exp(2j * wT) + 2*(1 + np.exp(2j * wT)) / np.exp(B*T))**-4
                 )
    return zros, gain

def fft2gammatonemx(nfft, sr=20000, nfilts=64, width=1.0, minfreq=100, maxfreq=10000, maxlen=1024):    
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero. 
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.
    """
    wts = np.zeros([nfilts,nfft])
    
    #after Slaney's MakeERBFilters
    EarQ = 9.26449; minBW = 24.7; order = 1;
    
    nFr = np.array(range(nfilts)) + 1
    em = EarQ*minBW
    cfreqs = (maxfreq+em)*np.exp(nFr*(-np.log(maxfreq + em)+np.log(minfreq + em))/nfilts)-em
    cfreqs = cfreqs[::-1]
    
    GTord = 4
    ucircArray = np.array(range(int(nfft/2 + 1)))
    ucirc = np.exp(1j*2*np.pi*ucircArray/nfft);
    #justpoles = 0 :taking out the 'if' corresponding to this. 

    ERB = width*np.power(np.power(cfreqs/EarQ,order) + np.power(minBW,order),1/order);
    B = 1.019 * 2 * np.pi * ERB;
    r = np.exp(-B/sr)
    theta = 2*np.pi*cfreqs/sr
    pole = r*np.exp(1j*theta)
    T = 1/sr
    ebt = np.exp(B*T); cpt = 2*cfreqs*np.pi*T;  
    ccpt = 2*T*np.cos(cpt); scpt = 2*T*np.sin(cpt);



    # pre-computations for simplification
    fcs  = cfreqs
    fs   = sr
    T    = 1 / fs
    B    = 1.019 * 2 * np.pi * ERB
    wT   = 2 * fcs * np.pi * T  
    K    = np.exp(B * T)
    Cos  = 2 * T * np.cos(2 * fcs * np.pi * T)
    Sin  = 2 * T * np.sin(2 * fcs * np.pi * T)
    Smax = np.sqrt(3 + 2**(3/2))
    Smin = np.sqrt(3 - 2**(3/2))

    zros, gain = F(sr, minfreq, maxfreq, nfilts=20)
    wIdx = range(int(nfft/2 + 1))  
    #in MATLAB, there used to be 64 where here it says nfilts:
    wts[:, wIdx] =  ((T**4) / np.reshape(gain, (nfilts, 1))) * \
                    np.abs(ucirc - np.reshape(zros[0], (nfilts,1))) * \
                    np.abs(ucirc - np.reshape(zros[1], (nfilts,1))) * \
                    np.abs(ucirc - np.reshape(zros[2], (nfilts,1))) * \
                    np.abs(ucirc - np.reshape(zros[3], (nfilts,1))) * \
                    np.abs(np.power(np.multiply(np.reshape(pole, (nfilts,1)) - ucirc, np.conj(np.reshape(pole, (nfilts,1))) - ucirc), -GTord))       
                    
    wts = wts[:,range(maxlen)];   
    
    return wts, cfreqs, gain

def _filter_banks(nfilt=20, nfft=512, fs=16000, fmin=0, fmax=None):
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
    fmax  = fmax or fs/2
   
    # compute points evenly spaced in bark
    center_freqs = generate_center_frequencies(fmin, fmax, 10*nfilt + 2)
    # The frequencies array/ points are in Bark, but we use fft bins, so we 
    # have to convert from Bark to fft bin number
    bin   = np.floor(hz2fft(center_freqs))
    fbank = np.zeros([nfilt, nfft // 2 + 1])

    width   = 2.0
    twinmod = 1
    nfft = int(2**(np.ceil(np.log(2*twinmod*fs)/np.log(2))))
    [gtm,f, gain] = fft2gammatonemx(nfft, fs, nfilt, width, fmin, fmax, int(nfft/2+1))
    
    for g in gtm[::5]:
        plt.plot(g)
    plt.show()

    for j in range(2, nfilt-2, 10):
        for i in range(j-2, j+2):
            fc          = fft2hz(bin[j])
            f           = fft2hz(bin[i])
            fbank[j, i] = np.abs(Hz(f, fc, fs)) / np.abs(np.max(Hz(f, fc, fs)))
    return gtm, gain


import matplotlib.pyplot as plt 

fbanks, gain = _filter_banks(nfilt=49, nfft=512, fs=16000)  

# plot the gammatone filter banks 
for i in range(0, len(fbanks), 2):
    plt.plot(fbanks[i],  linewidth=2,)
    plt.ylim(0, 1.1)
    plt.grid(True)
plt.show()
