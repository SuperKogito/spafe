"""

- Description : Gammatone filter banks implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Optional, Tuple

import numpy as np

from ..utils.converters import hz2erb, ErbConversionApproach
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.filters import scale_fbank, ScaleType

# Slaney's ERB Filter constants
EarQ = 9.26449
minBW = 24.7


def generate_center_frequencies(
    min_freq: float, max_freq: float, nfilts: int
) -> np.ndarray:
    """
    Compute center frequencies in the ERB scale.

    Args:
        min_freq (float) : minimum frequency of the center frequencies' domain.
        max_freq (float) : maximum frequency of the center frequencies' domain.
        nfilts     (int) : number of filters <=> number of center frequencies to compute.

    Returns:
        (numpy.ndarray) : array of center frequencies.
    """
    # init vars
    m = np.array(range(nfilts)) + 1
    c = EarQ * minBW

    # compute center frequencies
    center_freqs = (max_freq + c) * np.exp(
        (m / nfilts) * (np.log(min_freq + c) - np.log(max_freq + c))
    ) - c
    return center_freqs[::-1]


def compute_gain(
    fcs: np.ndarray, B: np.ndarray, wT: np.ndarray, T: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gain and matrixify computation for speed purposes [Ellis-spectrogram]_.

    Args:
        fcs (numpy.ndarray) : center frequencies in
        B   (numpy.ndarray) : bandwidths of the filters.
        wT  (numpy.ndarray) : corresponds to :code:`(omega)*T = 2*pi*freq*T`.
        T           (float) : periode in seconds aka inverse of the sampling rate.

    Returns:
        (tuple):
            - (numpy.ndarray) : a 2d numpy array representing the filter gains.
            - (numpy.ndarray) : a 2d array A used for final computations.
    """
    # pre-computations for simplification
    K = np.exp(B * T)
    Cos = np.cos(2 * fcs * np.pi * T)
    Sin = np.sin(2 * fcs * np.pi * T)
    Smax = np.sqrt(3 + 2 ** (3 / 2))
    Smin = np.sqrt(3 - 2 ** (3 / 2))

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


def gammatone_filter_banks(
    nfilts: int = 24,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    order: int = 4,
    conversion_approach: ErbConversionApproach = "Glasberg",
):
    """
    Compute Gammatone-filter banks. The filters are stored in the rows, the columns
    correspond to fft bins [Ellis-spectrogram]_ and [Cusimano]_ .

    Args:
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default 20).
        nfft                (int) : the FFT size.
                                    (Default is 512).
        fs                  (int) : sample rate/ sampling frequency of the signal.
                                    (Default is 16000 Hz).
        low_freq          (float) : lowest band edge of mel filters.
                                    (Default is 0 Hz).
        high_freq         (float) : highest band edge of mel filters.
                                    (Default samplerate/2).
        scale               (str) : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        order               (int) : order of the gammatone filter.
                                    (Default is 4).
        conversion_approach (str) : erb scale conversion approach.
                                    (Default is "Glasberg").

    Returns:
        (tuple) :
            - (numpy.ndarray) : array of size nfilts * (nfft/2 + 1) containing filter bank. Each row holds 1 filter.
            - (numpy.ndarray) : array of center frequencies in Erb.

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`conversion_approach` : can take the following options ["Glasberg"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    References:
        .. [Ellis-spectrogram] : Ellis, D.P.W.  (2009). "Gammatone-like spectrograms",
                                 web resource. http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/
        .. [Cusimano] : Cusimano, M., gammatonegram, https://github.com/mcusi/gammatonegram/blob/master/gtg.py

    Examples:
        .. plot::

            import numpy as np
            from spafe.utils.converters import erb2hz
            from spafe.utils.vis import show_fbanks
            from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks

            # init var
            fs = 8000
            nfilts = 7
            nfft = 1024
            low_freq = 0
            high_freq = fs / 2

            # compute freqs for xaxis
            ghz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

            for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
                # gamma fbanks
                gamma_fbanks_mat, gamma_freqs = gammatone_filter_banks(nfilts=nfilts,
                                                                       nfft=nfft,
                                                                       fs=fs,
                                                                       low_freq=low_freq,
                                                                       high_freq=high_freq,
                                                                       scale=scale,
                                                                       order=4)
                # visualize filter bank
                show_fbanks(
                    gamma_fbanks_mat,
                    [erb2hz(freq) for freq in gamma_freqs],
                    ghz_freqs,
                    label + "Gamma Filter Bank",
                    ylabel="Weight",
                    x1label="Frequency / Hz",
                    x2label="Frequency / erb",
                    figsize=(14, 5),
                    fb_type="gamma")

    See Also:
        - :py:func:`spafe.fbanks.bark_fbanks.bark_filter_banks`
        - :py:func:`spafe.fbanks.linear_fbanks.linear_filter_banks`
        - :py:func:`spafe.fbanks.mel_fbanks.mel_filter_banks`
    """
    # init freqs
    high_freq = high_freq or fs / 2

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])

    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # define custom difference func
    def Dif(u, a):
        return u - a.reshape(nfilts, 1)

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
    ERB = width * ((fcs / EarQ) ** order + minBW**order) ** (1 / order)
    B = 1.019 * 2 * np.pi * ERB

    # compute input vars
    wT = 2 * fcs * np.pi * T
    pole = np.exp(1j * wT) / np.exp(B * T)

    # compute gain and A matrix
    A, Gain = compute_gain(fcs, B, wT, T)

    # compute fbank
    fbank[:, idx] = (
        (T**4 / Gain.reshape(nfilts, 1))
        * np.abs(Dif(u, A[0]) * Dif(u, A[1]) * Dif(u, A[2]) * Dif(u, A[3]))
        * np.abs(Dif(u, pole) * Dif(u, pole.conj())) ** (-n)
    )

    # make sure all filters has max value = 1.0
    try:
        fbank = np.array([f / np.max(f) for f in fbank[:, range(maxlen)]])

    except BaseException:
        fbank = fbank[:, idx]

    # compute scaling
    scaling = scale_fbank(scale=scale, nfilts=nfilts)
    fbank = fbank * scaling
    return fbank, np.array([hz2erb(freq, conversion_approach) for freq in fcs])
