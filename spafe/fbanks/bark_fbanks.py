##############################################################################################
#                             Bark-filter-banks implementation
##############################################################################################
import numpy as np
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.converters import hz2bark, fft2bark, bark2fft


def Fm(fb, fc):
    """
    Compute a Bark filter around a certain center frequency in bark.

    Args:
        fb (int): frequency in Bark.
        fc (int): center frequency in Bark.

    Returns:
        (float) : associated Bark filter value/amplitude.
    """
    if fc - 2.5 <= fb <= fc - 0.5:
        return 10**(2.5 * (fb - fc + 0.5))
    elif fc - 0.5 < fb < fc + 0.5:
        return 1
    elif fc + 0.5 <= fb <= fc + 1.3:
        return 10**(-2.5 * (fb - fc - 0.5))
    else:
        return 0


def bark_filter_banks(nfilts=20,
                      nfft=512,
                      fs=16000,
                      low_freq=0,
                      high_freq=None,
                      scale="constant"):
    """
    Compute Bark-filterbanks. The filters are stored in the rows, the columns
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
        scale    (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                          Default is "constant"

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

    # compute points evenly spaced in Bark scale (points are in Bark)
    low_bark = hz2bark(low_freq)
    high_bark = hz2bark(high_freq)
    bark_points = np.linspace(low_bark, high_bark, nfilts + 4)

    # we use fft bins, so we have to convert from Bark to fft bin number
    bins = np.floor(bark2fft(bark_points))
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # init scaler
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c  = 0

    for j in range(2, nfilts + 2):
        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)

        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        for i in range(int(bins[j - 2]), int(bins[j + 2])):
            fc = bark_points[j]
            fb = fft2bark(i)
            fbank[j - 2, i] = c * Fm(fb, fc)
    return np.abs(fbank)
