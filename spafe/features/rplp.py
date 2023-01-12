"""

- Description : (Rasta) Perceptual linear prediction coefficents (RPLPs/PLPs) extraction algorithm implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Optional

import numpy as np

from ..fbanks.bark_fbanks import bark_filter_banks
from ..features.lpc import __lpc_helper, lpc2lpcc
from ..utils.cepstral import normalize_ceps, lifter_ceps, NormalizationType
from ..utils.converters import BarkConversionApproach
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.filters import rasta_filter, ScaleType
from ..utils.preprocessing import (
    pre_emphasis,
    framing,
    windowing,
    SlidingWindow,
)


def __rastaplp(
    sig: np.ndarray,
    fs: int = 16000,
    order: int = 13,
    pre_emph: bool = False,
    pre_emph_coeff: float = 0.97,
    window: Optional[SlidingWindow] = None,
    do_rasta: bool = False,
    nfilts: int = 24,
    nfft: int = 512,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    lifter: Optional[int] = None,
    normalize: Optional[NormalizationType] = None,
    fbanks: Optional[np.ndarray] = None,
    conversion_approach: BarkConversionApproach = "Wang",
) -> np.ndarray:
    """
    Compute Perceptual Linear Prediction coefficients with or without rasta filtering.

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        order               (int) : number of cepstra to return.
                                    (Default is 13).
        pre_emph           (bool) : apply pre-emphasis if True.
                                    (Default is False).
        pre_emph_coeff    (float) : pre-emphasis filter coefﬁcient.
                                    (Default is 0.97).
        window    (SlidingWindow) : sliding window object.
                                    (Default is None).
        do_rasta           (bool) : apply Rasta filtering if True.
                                    (Default is False).
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq          (float) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq         (float) : highest band edge of mel filters (Hz).
                                    (Default is samplerate/2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        lifter              (int) : apply liftering if specified.
                                    (Default is None).
        normalize           (str) : apply normalization if approach specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (numpy.ndarray) : 2d array of the PLP or RPLP coefficients.
        (Matrix of features, row = feature, col = frame).

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.

    """
    high_freq = high_freq or fs / 2
    num_ceps = order

    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    #  compute fbanks
    if fbanks is None:
        bark_fbanks_mat, _ = bark_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=high_freq,
            scale=scale,
            conversion_approach=conversion_approach,
        )
        fbanks = bark_fbanks_mat

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # init window
    if window is None:
        window = SlidingWindow()

    # -> framing
    frames, frame_length = framing(
        sig=sig, fs=fs, win_len=window.win_len, win_hop=window.win_hop
    )

    # -> windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=window.win_type)

    # -> FFT -> |.|
    ## Magnitude of the FFT
    fourrier_transform = np.absolute(np.fft.fft(windows, nfft))
    fourrier_transform = fourrier_transform[:, : int(nfft / 2) + 1]

    ##  -> |.|^2 (Power Spectrum)
    abs_fft_values = (1.0 / nfft) * np.square(fourrier_transform)

    # -> x filter bank = auditory spectrum
    auditory_spectrum = np.dot(a=abs_fft_values, b=fbanks.T)

    # rasta filtering
    if do_rasta:
        # put in log domain
        nl_aspectrum = np.log(auditory_spectrum)

        # next do rasta filtering
        ras_nl_aspectrum = rasta_filter(nl_aspectrum)

        # do inverse log
        auditory_spectrum = np.exp(ras_nl_aspectrum)

    # equal loudness pre_emphasis
    E = lambda w: ((w**2 + 56.8 * 10**6) * w**4) / (
        (w**2 + 6.3 * 10**6)
        * (w**2 + 0.38 * 10**9)
        * (w**6 + 9.58 * 10**26)
    )
    Y = [E(w) for w in auditory_spectrum]

    # intensity loudness compression
    L = np.abs(Y) ** (1 / 3)

    # ifft
    inverse_fourrier_transform = np.absolute(np.fft.ifft(L, nfft))

    # compute lpcs and lpccs
    lpcs = np.zeros((L.shape[0], order))
    lpccs = np.zeros((L.shape[0], order))
    for i in range(L.shape[0]):
        a, e = __lpc_helper(inverse_fourrier_transform[i, :], order - 1)
        lpcs[i, :] = a
        lpcc_coeffs = lpc2lpcc(a, e, order)
        lpccs[i, :] = np.array(lpcc_coeffs)

    # liftering
    if lifter is not None:
        lpccs = lifter_ceps(lpccs, lifter)

    # normalize
    if normalize:
        lpccs = normalize_ceps(lpccs, normalize)

    return lpccs


def plp(
    sig: np.ndarray,
    fs: int = 16000,
    order: int = 13,
    pre_emph: bool = False,
    pre_emph_coeff: float = 0.97,
    window: Optional[SlidingWindow] = None,
    nfilts: int = 24,
    nfft: int = 512,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    lifter: Optional[int] = None,
    normalize: Optional[NormalizationType] = None,
    fbanks: Optional[np.ndarray] = None,
    conversion_approach: BarkConversionApproach = "Wang",
) -> np.ndarray:
    """
    Compute Perceptual linear prediction coefficents according to [Hermansky]_
    and [Ajibola]_.

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        order               (int) : number of cepstra to return.
                                    (Default is 13).
        pre_emph           (bool) : apply pre-emphasis if 1.
                                    (Default is 1).
        pre_emph_coeff    (float) : pre-emphasis filter coefﬁcient.
                                    (Default is 0.97).
        window    (SlidingWindow) : sliding window object.
                                    (Default is None).
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq          (float) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq         (float) : highest band edge of mel filters (Hz).
                                    (Default is samplerate/2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        lifter              (int) : apply liftering if specified.
                                    (Default is None).
        normalize           (str) : apply normalization if approach specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (numpy.ndarray) : 2d array of PLP features (num_frames x order)

    Tip:
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/plps.png

           Architecture of perceptual linear prediction coefﬁcients extraction algorithm.

    Examples:
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.rplp import plp
            from spafe.utils.preprocessing import SlidingWindow
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../tests/data/test.wav"
            fs, sig = read(fpath)

            # compute plps
            plps = plp(sig,
                       fs=fs,
                       pre_emph=0,
                       pre_emph_coeff=0.97,
                       window=SlidingWindow(0.03, 0.015, "hamming"),
                       nfilts=128,
                       nfft=1024,
                       low_freq=0,
                       high_freq=fs/2,
                       lifter=0.9,
                       normalize="mvn")

            # visualize features
            show_features(plps, "Perceptual linear predictions", "PLP Index", "Frame Index")
    """
    return __rastaplp(
        sig,
        fs=fs,
        order=order,
        pre_emph=pre_emph,
        pre_emph_coeff=pre_emph_coeff,
        window=window,
        do_rasta=False,
        nfilts=nfilts,
        nfft=nfft,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
        lifter=lifter,
        normalize=normalize,
        fbanks=fbanks,
        conversion_approach=conversion_approach,
    )


def rplp(
    sig: np.ndarray,
    fs: int = 16000,
    order: int = 13,
    pre_emph: bool = False,
    pre_emph_coeff: float = 0.97,
    window: Optional[SlidingWindow] = None,
    nfilts: int = 24,
    nfft: int = 512,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    lifter: Optional[int] = None,
    normalize: Optional[NormalizationType] = None,
    fbanks: Optional[np.ndarray] = None,
    conversion_approach: BarkConversionApproach = "Wang",
) -> np.ndarray:
    """
    Compute rasta Perceptual linear prediction coefficents according to [Hermansky]_
    and [Ajibola]_.

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        order               (int) : number of cepstra to return.
                                    (Default is 13).
        pre_emph           (bool) : apply pre-emphasis if 1.
                                    (Default is True).
        pre_emph_coeff    (float) : pre-emphasis filter coefﬁcient.
                                    (Default is 0.97).
        window    (SlidingWindow) : sliding window object.
                                    (Default is None).
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq          (float) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq         (float) : highest band edge of mel filters (Hz).
                                    (Default is samplerate/2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        lifter              (int) : apply liftering if specified.
                                    (Default is None).
        normalize           (str) : apply normalization if approach specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (numpy.ndarray) : 2d array of rasta PLP features (num_frames x order)


    Tip:
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.
    Note:
        .. figure:: ../_static/architectures/rplps.png

           Architecture of rasta perceptual linear prediction coefﬁcients extraction algorithm.

    Examples:
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.rplp import rplp
            from spafe.utils.preprocessing import SlidingWindow
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../tests/data/test.wav"
            fs, sig = read(fpath)

            # compute rplps
            rplps = rplp(sig,
                         fs=fs,
                         pre_emph=0,
                         pre_emph_coeff=0.97,
                         window=SlidingWindow(0.03, 0.015, "hamming"),
                         nfilts=128,
                         nfft=1024,
                         low_freq=0,
                         high_freq=fs/2,
                         lifter=0.9,
                         normalize="mvn")

            # visualize features
            show_features(rplps, "Rasta perceptual linear predictions", "PLP Index", "Frame Index")

    References:

        .. [Ajibola] : Ajibola Alim, S., & Khair Alang Rashid, N. (2018). Some
                       Commonly Used Speech Feature Extraction Algorithms.
                       From Natural to Artificial Intelligence - Algorithms and
                       Applications. doi:10.5772/intechopen.80419
    """
    return __rastaplp(
        sig,
        fs=fs,
        order=order,
        pre_emph=pre_emph,
        pre_emph_coeff=pre_emph_coeff,
        window=window,
        do_rasta=True,
        nfilts=nfilts,
        nfft=nfft,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
        lifter=lifter,
        normalize=normalize,
        fbanks=fbanks,
        conversion_approach=conversion_approach,
    )
