#############################################################################################
#                           Gammatone-filter-banks implementation
#############################################################################################
"""
based on https://github.com/mcusi/gammatonegram/
"""
import numpy as np
from ..utils.exceptions import ParameterError, ErrorMsgs

# Slaney's ERB Filter constants
EarQ = 9.26449
minBW = 24.7


def generate_center_frequencies(min_freq, max_freq, nfilts):
    """
    Compute center frequencies in the ERB scale.

    Args:
        min_freq (int) : minimum frequency of the center frequencies domain.
        max_freq (int) : maximum frequency of the center frequencies domain.
        nfilts   (int) : number of filters, that is equivalent to the number of
                        center frequencies to compute.

    Returns:
        an array of center frequencies.
    """
    # init vars
    m = np.array(range(nfilts)) + 2
    c = EarQ * minBW
    M = nfilts

    # compute center frequencies
    cfreqs = (max_freq + c) * np.exp((m / M) * np.log(
        (min_freq + c) / (max_freq + c))) - c
    return cfreqs[::-1]


def compute_gain(fcs, B, wT, T):
    """
    Compute Gaina and matrixify computation for speed purposes.

    Args:
        fcs (array) : center frequencies in
        B   (array) : bandwidths of the filters.
        wT  (array) : corresponds to `(omega) * T = 2 * pi * freq * T` used for
                      the frequency domain computations.
        T   (float) : periode in seconds aka inverse of the sampling rate.

    Returns:
        a 2d numpy array representing the filter gains.
        a 2d array A used for final computations.
    """
    # pre-computations for simplification
    K = np.exp(B * T)
    Cos = np.cos(2 * fcs * np.pi * T)
    Sin = np.sin(2 * fcs * np.pi * T)
    Smax = np.sqrt(3 + 2**(3 / 2))
    Smin = np.sqrt(3 - 2**(3 / 2))

    # define A matrix rows
    A11 = (Cos + Smax * Sin) / K
    A12 = (Cos - Smax * Sin) / K
    A13 = (Cos + Smin * Sin) / K
    A14 = (Cos - Smin * Sin) / K

    # Compute gain (vectorized)
    A = np.array([A11, A12, A13, A14])
    Kj = np.exp(1j * wT)
    Kjmat = np.array([Kj, Kj, Kj, Kj]).T
    G = 2 * T * Kjmat * (A.T - Kjmat)
    Coe = -2 / K**2 - 2 * Kj**2 + 2 * (1 + Kj**2) / K
    Gain = np.abs(G[:, 0] * G[:, 1] * G[:, 2] * G[:, 3] * Coe**-4)
    return A, Gain


def gammatone_filter_banks(nfilts=20,
                           nfft=512,
                           fs=16000,
                           low_freq=None,
                           high_freq=None,
                           scale="contsant",
                           order=4):
    """
    Compute Gammatone-filterbanks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilts    (int) : the number of filters in the filterbank.
                          (Default 20)
        nfft      (int) : the FFT size.
                          (Default is 512)
        fs        (int) : sample rate/ sampling frequency of the signal.
                          (Default 16000 Hz)
        low_freq  (int) : lowest band edge of mel filters.
                          (Default 0 Hz)
        high_freq (int) : highest band edge of mel filters.
                          (Default samplerate/2)
        scale     (str) : choose if max bins amplitudes ascend, descend or are constant (=1).
                          Default is "constant"
        order     (int) : order of the gammatone filter.
                          Default is 4.

    Returns:
        a numpy array of size nfilts * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # define custom difference func
    def Dif(u, a):
        return u - a.reshape(nfilts, 1)

    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # init vars
    fbank = np.zeros([nfilts, nfft])
    width = 1.0
    maxlen = nfft // 2 + 1
    T = 1 / fs
    n = 4
    u = np.exp(1j * 2 * np.pi * np.array(range(nfft // 2 + 1)) / nfft)
    idx = range(nfft // 2 + 1)

    # computer center frequencies, convert to ERB scale and compute bandwidths
    fcs = generate_center_frequencies(low_freq, high_freq, nfilts)
    ERB = width * ((fcs / EarQ)**order + minBW**order)**(1 / order)
    B = 1.019 * 2 * np.pi * ERB

    # compute input vars
    wT = 2 * fcs * np.pi * T
    pole = np.exp(1j * wT) / np.exp(B * T)

    # compute gain and A matrix
    A, Gain = compute_gain(fcs, B, wT, T)

    # compute fbank
    fbank[:, idx] = (
        (T**4 / Gain.reshape(nfilts, 1)) *
        np.abs(Dif(u, A[0]) * Dif(u, A[1]) * Dif(u, A[2]) * Dif(u, A[3])) *
        np.abs(Dif(u, pole) * Dif(u, pole.conj()))**(-n))

    # make sure all filters has max value = 1.0
    try:
        fbs = np.array([f / np.max(f) for f in fbank[:, range(maxlen)]])
    except BaseException:
        fbs = fbank[:, idx]
        
    # compute scaler
    if scale == "ascendant":
        c = [0, ]
        for i in range(1, nfilts):
            x = c[i-1] + 1 / nfilts
            c.append(x * (x < 1) + 1 * (x > 1))
    elif scale == "descendant":
        c = [1, ]
        for i in range(1, nfilts):
            x = c[i-1] - 1 / nfilts
            c.append(x * (x > 0) + 0 * (x < 0))
    else:
        c = [1 for i in range(nfilts)]

    # apply scaler
    c = np.array(c).reshape(nfilts, 1)
    fbs = c * np.abs(fbs)
    return fbs
