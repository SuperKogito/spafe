"""
based on: https://github.com/scoreur/cqt/blob/master/cqt.py
"""
import scipy
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.sparse import hstack, vstack, coo_matrix


def cqt(sig, fs=16000, low_freq=0, high_freq=3000, b=512) :
    """
    Compute the constant Q-transform.
    """
    # define lambda funcs for clarity
    f  = lambda     k: low_freq * 2**((k-1) / b)
    w  = lambda     N: np.hamming(N)
    nk = lambda     k: np.ceil(Q * fs / f(k))
    t  = lambda Nk, k: (1 / Nk) * w(Nk) * np.exp(2 * np.pi * 1j * Q * np.arange(Nk) / Nk)

    # init vars
    Q    = 1 / (2**(1/b) - 1)
    K    = int(np.ceil(b * np.log2(high_freq / low_freq)))
    nfft = int( 2**np.ceil(np.log2(Q * fs / low_freq)) )

    # define temporal kernal and sparse kernal variables
    S = [scipy.sparse.coo_matrix(np.fft.fft(t(nk(k), k), nfft)) for k in range(K, 0, -1)]
    S = scipy.sparse.vstack(S[::-1]).tocsc().transpose().conj() / nfft

    # compute the constant Q-transform
    xcq = (np.fft.fft(sig, nfft).reshape(1, nfft) * S)[0]
    return xcq


def test_cqt():
    """
    Test constant Q-transform function.
    """
    # read wave file
    fs, sig = scipy.io.wavfile.read('../test.wav')
    xcq     = cqt(sig, fs, 40, 22050, 12)
    ampxcq  = np.abs(xcq)**2
    plt.plot(ampxcq)
    plt.show()
