# -*- coding: utf-8 -*-
"""

- Description : Spectral and frequency stats and features extraction algorithms implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import scipy
import numpy as np
from scipy import stats


def spectral_centroid(sig, fs, spectrum, i=1):
    """
    Compute the spectral centroid (which is the barycenter of the spectrum) as
    described in [Peeters]_.

    Args:
        sig      (numpy.ndarray) : input mono signal.
        fs                 (int) : input signal sampling rate.
        spectrum (numpy.ndarray) : signal spectrum.
        i                  (int) : centroid order.
                                   (Default is 1).

    Returns:
        (float) : spectral centroid.

    Note:
        .. math::

            \mu_{i} &= \\frac{\sum_{n=0}^{N}f_{k}^{i}*a_{k}}{\sum_{n=0}^{N}a_{k}}\\\\
            S_{centroid} &= \mu_{1}\\\\
    """
    # compute magnitude spectrum
    magnitude_spectrum = np.abs(spectrum)

    # compute positive frequencies
    freqs = np.abs(np.fft.fftfreq(len(spectrum), 1.0 / fs))

    # return weighted mean
    sc = np.sum(magnitude_spectrum * freqs**i) / np.sum(magnitude_spectrum)
    return sc


def spectral_skewness(sig, fs, spectrum):
    """
    Compute the spectral skewness (which is a measure of the asymmetry of a
    distribution around its mean) as described in [Peeters]_.

    Args:
        sig      (numpy.ndarray) : input mono signal.
        fs                 (int) : input signal sampling rate.
        spectrum (numpy.ndarray) : signal spectrum.

    Returns:
        (float) : spectral skewness.

    Note:
        .. math::

            S_{skewness} = \mu_{3} &= \\frac{\sum_{n=0}^{N}(f_{k} - \\mu_{1})^{3} . a_k}{\\mu_{2}^{3} . \sum_{n=0}^{N}a_{k}}\\\\
    """
    # compute magnitude spectrum, and centroids
    magnitude_spectrum = np.abs(spectrum)
    mu1 = spectral_centroid(sig, fs, spectrum, i=1)
    mu2 = spectral_centroid(sig, fs, spectrum, i=2)

    # compute positive frequencies
    freqs = np.abs(np.fft.fftfreq(len(spectrum), 1.0 / fs))

    # return weighted mean
    sk = np.sum(magnitude_spectrum * (freqs - mu1) ** 3) / (
        np.sum(magnitude_spectrum) * mu2**3
    )
    return sk


def spectral_kurtosis(sig, fs, spectrum):
    """
    Compute the spectral kurtosis (which is a measure of the flatness of a
    distribution around its mean) as described in [Peeters]_.

    Args:
        sig      (numpy.ndarray) : input mono signal.
        fs                 (int) : input signal sampling rate.
        spectrum (numpy.ndarray) : signal spectrum.

    Returns:
        (float) : spectral kurtosis.

    Note:
        .. math::

            S_{kurtosis} = \mu_{4} &= \\frac{\sum_{n=0}^{N}(f_{k} - \\mu_{1})^{4} . a_k}{\\mu_{2}^{4} . \sum_{n=0}^{N}a_{k}}\\\\
    """
    # compute magnitude spectrum, and centroids
    magnitude_spectrum = np.abs(spectrum)
    mu1 = spectral_centroid(sig, fs, spectrum, i=1)
    mu2 = spectral_centroid(sig, fs, spectrum, i=2)

    # compute positive frequencies
    freqs = np.abs(np.fft.fftfreq(len(spectrum), 1.0 / fs))

    # return weighted mean
    sk = np.sum(magnitude_spectrum * (freqs - mu1) ** 4) / (
        np.sum(magnitude_spectrum) * mu2**4
    )
    return sk


def spectral_entropy(sig, fs, spectrum):
    """
    Compute the spectral entropy as described in [Misra]_.

    Args:
        sig      (numpy.ndarray) : input mono signal.
        fs                 (int) : input signal sampling rate.
        spectrum (numpy.ndarray) : signal spectrum.

    Returns:
        (float) : spectral skewness.

    Note:
        .. math::

            S_{entropy} &= \\frac{\sum_{n=0}^{N}(f_{k} - \\mu_{1})^{3} . a_k}{\\mu_{2}^{3} . \sum_{n=0}^{N}a_{k}}\\\\

    References:
        .. [Misra] : Misra, H., S. Ikbal, H. Bourlard, and H. Hermansky.
                    "Spectral Entropy Based Feature for Robust ASR." 2004 IEEE
                    International Conference on Acoustics, Speech, and Signal Processing.
    """
    # compute magnitude spectrum, and centroids
    magnitude_spectrum = np.abs(spectrum)
    en = (
        -1
        * np.sum(magnitude_spectrum * np.log(magnitude_spectrum))
        / np.log(len(magnitude_spectrum))
    )
    return en


def spectral_spread(sig, fs, spectrum):
    """
    Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)
    as described in [Peeters]_.

    Args:
        sig      (numpy.ndarray) : input mono signal.
        fs                 (int) : input signal sampling rate.
        spectrum (numpy.ndarray) : signal spectrum.

    Returns:
        (numpy.array) : spectral spread.

    Note:
        .. math::
            S_{spread} = \\sqrt{\\sum{0}{N}\\frac{(f_k - \\mu_{1}) . s_k}{\\sum{0}{N}a_k}}
    """
    # compute magnitude spectrum
    magnitude_spectrum = np.abs(spectrum)

    # compute positive frequencies
    freqs = np.abs(np.fft.fftfreq(len(spectrum), 1.0 / fs))

    # get centroid
    mu1 = spectral_centroid(sig, fs, spectrum)

    # compute spread
    spread = np.sqrt(
        ((np.sum(freqs - mu1) ** 2) * magnitude_spectrum) / np.sum(magnitude_spectrum)
    )
    return spread


def spectral_flatness(sig, spectrum):
    """
    Compute spectral flatness.

    Args:
        sig      (numpy.ndarray) : audio signal array.
        spectrum (numpy.ndarray) : signal spectrum.

    Returns:
        (float) : spectral flatness value.

    Note:
        .. math::

            S_{flatness} = \\frac{exp(\\frac{1}{N}\sum_{k}log(a_{k}))}{\\frac{1}{N}\sum_{k}a_{k}}
    """
    # compute magnitude spectrum
    magnitude_spectrum = np.abs(spectrum)

    # compute spectral flatness
    sf = stats.mstats.gmean(magnitude_spectrum) / np.mean(magnitude_spectrum)
    return sf


def spectral_rolloff(sig, fs, spectrum, k=0.85):
    """
    Compute the spectral roll-off point which measures the bandwidth of the audio
    signal by determining the frequency bin under which a specified k percentage
    of the total spectral energy is contained below. see [Scheirer]_.

    Args:
        sig      (numpy.ndarray) : input mono signal array.
        fs                 (int) : input signal sampling rate.
        spectrum (numpy.ndarray) : signal spectrum.
        k                (float) : constant.
                                   (Default is 0.85).

    Returns:
        (float) : spectral rolloff point.
    """
    # convert to frequency domain
    magnitude_spectrum = np.abs(spectrum)

    # compute the spectral rolloff point
    i = k * np.sum(magnitude_spectrum)
    return i


def spectral_flux(sig, fs, spectrum, p=2):
    """
    Compute the spectral flux, which measures how quickly the power spectrum
    of a signal is changing. This implementation computes the spectral flux
    using the L2-norm per default i.e. the square root of the sum of absolute
    differences squared [Scheirer]_.

    Args:
        sig      (numpy.ndarray) : input mono signal.
        fs                 (int) : input signal sampling rate.
        spectrum (numpy.ndarray) : signal spectrum.
        p                  (int) : norm type.
                                   (Default is 2).

    Returns:
        (float) : spectral flux.

    Note:
        .. math::
            S_{flux} = (\sum_{k}(|a_{k}(t) - a_{k}(t-1))^2}{\sqrt{\sum_{k}a_{k}(t-1)|^p)^{\\frac{1}{p}}

    References:
        .. [Scheirer] : Scheirer, E., and M. Slaney. “Construction and Evaluation
                        of a Robust Multifeature Speech/Music Discriminator.” 1997
                        IEEE International Conference on Acoustics, Speech, and Signal
                        Processing, 1997.
    """
    # convert to frequency domain
    magnitude_spectrum = np.abs(spectrum)
    sf = (np.sum(np.abs(np.diff(magnitude_spectrum)) ** p)) ** (1 / p)
    return sf


def extract_feats(sig, fs, nfft=512):
    """
    Compute various spectral features [Peeters]_.

    Args:
        sig (numpy.ndarray) : input mono signal.
        fs            (int) : input signal sampling rate.
        nfft          (int) : number of FFT points.
                              (Default is 512).

    Returns:
        (dict) : a dictionary including various frequency and spectral features (see Notes).

    Note:
        The resulting dictionary includes the following elements:

        - spectral features
            - :code:`spectral_centroid`  : spectral centroid.
            - :code:`spectral_skewness`  : frequencies skewness.
            - :code:`spectral_kurtosis`  : frequencies kurtosis.
            - :code:`spectral_entropy`   : spectral entropy.
            - :code:`spectral_spread`    : spectral spread.
            - :code:`spectral_flatness`  : spectral flatness.
            - :code:`spectral_rolloff`   : spectral rolloff.
            - :code:`spectral_flux`      : spectral flux.
            - :code:`spectral_mean`      : spectral mean.
            - :code:`spectral_rms`       : spectral root mean square.
            - :code:`spectral_std`       : spectral standard deviation.
            - :code:`spectral_variance`  : spectral variance.

    Examples:
        .. plot::

            from spafe.features.spfeats import extract_feats
            from scipy.io.wavfile import read
            from pprint import pprint

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute erb spectrogram
            spectral_feats = extract_feats(sig, fs)
            pprint(spectral_feats)

    References:
        .. [Peeters] : Peeters, G. "A Large Set of Audio Features for Sound Description
                       (Similarity and Classification) in the CUIDADO Project."
                       Technical Report; IRCAM: Paris, France, 2004.
    """
    # init features dictionary
    feats = {}

    # compute the fft
    spectrum = np.fft.rfft(a=sig, n=nfft)

    # spectral features
    feats["spectral_centroid"] = spectral_centroid(sig, fs, spectrum)
    feats["spectral_skewness"] = spectral_skewness(sig, fs, spectrum)
    feats["spectral_kurtosis"] = spectral_kurtosis(sig, fs, spectrum)
    feats["spectral_entropy"] = spectral_entropy(sig, fs, spectrum)
    feats["spectral_spread"] = spectral_spread(sig, fs, spectrum)
    feats["spectral_flatness"] = spectral_flatness(sig, spectrum)
    feats["spectral_rolloff"] = spectral_rolloff(sig, fs, spectrum)
    feats["spectral_flux"] = spectral_flux(sig, fs, spectrum)
    feats["spectral_mean"] = np.mean(spectrum)
    feats["spectral_rms"] = np.sqrt(np.mean(spectrum**2))
    feats["spectral_std"] = np.std(spectrum)
    feats["spectral_variance"] = np.var(spectrum)
    return feats
