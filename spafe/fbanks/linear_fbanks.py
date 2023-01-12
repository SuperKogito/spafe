"""

- Description : Linear filter banks implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Optional

import numpy as np

from .mel_fbanks import mel_filter_banks_helper
from ..utils.filters import ScaleType


def linear_filter_banks(
    nfilts: int = 24,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
):
    """
    Compute linear-filter banks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilts      (int) : the number of filters in the filter bank.
                            (Default 20).
        nfft        (int) : the FFT size.
                            (Default is 512).
        fs          (int) : sample rate/ sampling frequency of the signal.
                            (Default 16000 Hz).
        low_freq  (float) : lowest band edge of linear filters.
                            (Default 0 Hz).
        high_freq (float) : highest band edge of linear filters.
                            (Default samplerate/2).
        scale       (str) : monotonicity behavior of the filter banks.
                            (Default is "constant").

    Returns:
        (tuple) :
            - (numpy.ndarray) : array of size nfilts * (nfft/2 + 1) containing filter bank. Each row holds 1 filter.
            - (numpy.ndarray) : array of center frequencies

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].

    Examples:

        .. plot::

            from spafe.utils.vis import show_fbanks
            from spafe.fbanks.linear_fbanks import linear_filter_banks


            # init var
            fs = 8000
            nfilt = 7
            nfft = 1024
            low_freq = 0
            high_freq = fs / 2

            # compute freqs for xaxis
            lhz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

            for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
                # ascendant linear fbank
                linear_fbanks_mat, lin_freqs = linear_filter_banks(nfilts=nfilt,
                                                                   nfft=nfft,
                                                                   fs=fs,
                                                                   low_freq=low_freq,
                                                                   high_freq=high_freq,
                                                                   scale=scale)

                # visualize fbanks
                show_fbanks(
                    linear_fbanks_mat,
                    lin_freqs,
                    lhz_freqs,
                    label + "Linear Filter Bank",
                    ylabel="Weight",
                    x1label="Frequency / Hz",
                    figsize=(14, 5),
                    fb_type="lin")

    See Also:
        - :py:func:`spafe.fbanks.bark_fbanks.bark_filter_banks`
        - :py:func:`spafe.fbanks.gammatone_fbanks.gammatone_filter_banks`
        - :py:func:`spafe.fbanks.mel_fbanks.mel_filter_banks`
    """
    # generate linear fbanks by inversing regular mel fbanks
    fbank, lin_freqs = mel_filter_banks_helper(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
        fb_type="lin",
    )
    return np.abs(fbank), lin_freqs
