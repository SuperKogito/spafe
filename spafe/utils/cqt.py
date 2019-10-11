"""
based on: https://github.com/scoreur/cqt/blob/master/cqt.py
"""
import scipy
import numpy as np
import scipy.io.wavfile
from scipy.sparse import hstack, vstack, coo_matrix

# read wave file 
fs, sig = scipy.io.wavfile.read('../test.wav')    
length  = fs
x       = sig
    
def cqt(sig, fs=16000, fmin=0, fmax=3000, b=512) :
    """
    Compute the constant Q-transform.
    """
    # define lambda funcs for clarity
    f  = lambda     k: fmin * 2**((k-1) / b)
    w  = lambda     N: np.hamming(N) 
    nk = lambda     k: np.ceil(Q * fs / f(k))
    t  = lambda Nk, k: (1 / Nk) * w(Nk) * np.exp(2 * np.pi * 1j * Q * np.arange(Nk) / Nk)
    
    # init vars 
    Q    = 1 / (2**(1/b) - 1)
    K    = int(np.ceil(b * np.log2(fmax / fmin)))
    nfft = int( 2**np.ceil(np.log2(Q * fs / fmin)) )
    
    # define temporal kernal and sparse kernal variables
    S = [scipy.sparse.coo_matrix(np.fft.fft(t(nk(k), k), nfft)) for k in range(K, 0, -1)]
    S = scipy.sparse.vstack(S[::-1]).tocsc().transpose().conj() / nfft
 
    # compute the constant Q-transform 
    xcq = (np.fft.fft(sig, nfft).reshape(1, nfft) * S)[0]
    return xcq



xcq = cqt( x, 40, 22050, 12, fs)

import matplotlib.pyplot as plt
ampxcq = np.abs(xcq)**2
plt.plot(ampxcq)
plt.show()