"""

- Description : Bark filter banks implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Optional

import numpy as np

from ..utils.converters import hz2bark, bark2hz, BarkConversionApproach
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.filters import scale_fbank, ScaleType


def Fm(fb: float, fc: float) -> float:
    """
    Compute a Bark filter around a certain center frequency in bark [Hermansky]_.

    Args:
        fb (float): frequency in Bark.
        fc (float): center frequency in Bark.

    Returns:
        (float) : associated Bark filter value/amplitude.
    """
    if (fb - fc < -1.3) or (2.5 < fb - fc):
        return 0

    elif -1.3 <= fb - fc <= -0.5:
        return 10 ** (2.5 * (fb - fc + 0.5))

    elif -0.5 < fb - fc < 0.5:
        return 1

    else:
        # if 0.5 <= fb - fc <= 2.5:
        return 10 ** (-1 * (fb - fc - 0.5))


def bark_filter_banks(
    nfilts: int = 24,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    conversion_approach: BarkConversionApproach = "Wang",
):
    """
    Compute Bark filter banks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 20).
        nfft                (int) : the FFT size.
                                    (Default is 512).
        fs                  (int) : sample rate/ sampling frequency of the signal.
                                    (Default 16000 Hz).
        low_freq          (float) : lowest band edge of mel filters.
                                    (Default 0 Hz).
        high_freq         (float) : highest band edge of mel filters.
                                    (Default is fs/2).
        scale               (str) : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (tuple) :
            - (numpy.ndarray) : array of size nfilts * (nfft/2 + 1) containing filter bank. Each row holds 1 filter.
            - (numpy.ndarray) : array of center frequencies

    Raises:
        ParameterError
            - if low_freq < 0 OR high_freq > (fs / 2)

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.

    References:
        .. [Hermansky] Hermansky, H. “Perceptual linear predictive (PLP) analysis of speech.”
                       The Journal of the Acoustical Society of America 87 4 (1990): 1738-52
                       doi: 10.1121/1.399423. PMID: 2341679.

    Examples:
        .. plot::

            import numpy as np
            from spafe.utils.converters import bark2hz
            from spafe.utils.vis import show_fbanks
            from spafe.fbanks.bark_fbanks import bark_filter_banks

            # init var
            fs = 8000
            nfilt = 7
            nfft = 1024
            low_freq = 0
            high_freq = fs / 2

            # compute freqs for xaxis
            bhz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

            for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
                # bark fbanks
                bark_fbanks_mat, bark_freqs = bark_filter_banks(nfilts=nfilt,
                                                                nfft=nfft,
                                                                fs=fs,
                                                                low_freq=low_freq,
                                                                high_freq=high_freq,
                                                                scale=scale)

                # visualize filter bank
                show_fbanks(
                    bark_fbanks_mat,
                    [bark2hz(freq) for freq in bark_freqs],
                    bhz_freqs,
                    label + "Bark Filter Bank",
                    ylabel="Weight",
                    x1label="Frequency / Hz",
                    x2label="Frequency / bark",
                    figsize=(14, 5),
                    fb_type="bark",
                )

    See Also:
        - :py:func:`spafe.fbanks.gammatone_fbanks.gammatone_filter_banks`
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

    # compute points evenly spaced in Bark scale (points are in Bark)
    low_bark = hz2bark(low_freq, conversion_approach)
    high_bark = hz2bark(high_freq, conversion_approach)
    bark_center_freqs = np.linspace(low_bark, high_bark, nfilts)

    # we use fft bins, so we have to convert from Bark to fft bin number
    bins = np.floor(
        np.array(
            [
                (nfft + 1) * (bark2hz(freq, conversion_approach) / fs)
                for freq in bark_center_freqs
            ]
        )
    )
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    for j in range(0, nfilts):
        for i in range(int(bins[0]), int(bins[nfilts - 1])):
            fc = bark_center_freqs[j]
            fb = hz2bark((i * fs) / (nfft + 1), conversion_approach)
            fbank[j, i] = Fm(fb, fc)

    # compute scaling
    scaling = scale_fbank(scale=scale, nfilts=nfilts)
    fbank = fbank * scaling

    return fbank, bark_center_freqs
