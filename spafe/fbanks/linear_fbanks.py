##############################################################################################
#                           linear-filter-banks implementation
##############################################################################################
import numpy as np
from ..utils.exceptions import ParameterError, ErrorMsgs


def linear_filter_banks(nfilts=20,
                        nfft=512,
                        fs=16000,
                        low_freq=None,
                        high_freq=None,
                        scale="constant"):
    """
    Compute linear-filterbanks. The filters are stored in the rows, the columns
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
        (numpy array) array of size nfilts * (nfft/2 + 1) containing filterbank.
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

    # compute points evenly spaced in mels (points are in Hz)
    mel_points = np.linspace(low_freq, high_freq, nfilts + 2)

    # we use fft bins, so we have to convert from Hz to fft bin number
    bins = np.floor((nfft + 1) * mel_points / fs)
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # init scaler
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    # compute amps of fbanks
    for j in range(0, nfilts):
        b0, b1, b2 = bins[j], bins[j + 1], bins[j + 2]

        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)

        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        # compute fbanks
        fbank[j, int(b0):int(b1)] = c * (np.arange(int(b0), int(b1)) -
                                     int(b0)) / (b1 - b0)
        fbank[j, int(b1):int(b2)] = c * (int(b2) -
                                     np.arange(int(b1), int(b2))) / (b2 - b1)

    return np.abs(fbank)
