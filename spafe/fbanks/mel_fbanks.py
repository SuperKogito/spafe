"""

- Description : Mel filter banks implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Optional

import numpy as np

from ..utils.converters import hz2mel, mel2hz, MelConversionApproach
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.filters import scale_fbank, ScaleType


def mel_filter_banks_helper(
    nfilts: int = 24,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    fb_type="mel",
    conversion_approach: MelConversionApproach = "Oshaghnessy",
):
    """
    Compute Mel-filter banks.The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default 20).
        nfft                (int) : the FFT size.
                                    (Default is 512).
        fs                  (int) : sample rate/ sampling frequency of the signal.
                                    (Default 16000 Hz).
        low_freq          (float) : lowest band edge of mel filters.
                                    (Default 0 Hz).
        high_freq         (float) : highest band edge of mel filters.
                                    (Default samplerate/2).
        scale               (str) : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        fb_type             (str) : the filter banks type.
                                    (Default is "mel")
        conversion_approach (str) : mel scale conversion approach.
                                    (Default is "Oshaghnessy").

    Returns:
        (tuple) :
            - (numpy.ndarray) : array of size nfilts * (nfft/2 + 1) containing filter bank. Each row holds 1 filter.
            - (numpy.ndarray) : array of center frequencies


    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`fb_type` : can take the following options ["mel", "lin"].
        - :code:`conversion_approach` : can take the following options ["Oshaghnessy", "Beranek", "Lindsay"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    """
    # init freqs
    high_freq = high_freq or fs / 2

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])

    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    if fb_type == "mel":
        # convert bounding freqs
        low_mel = hz2mel(low_freq, conversion_approach)
        high_mel = hz2mel(high_freq, conversion_approach)

        # compute points evenly spaced in mels (ponts are in Hz)
        delta_mel = abs(high_mel - low_mel) / (nfilts + 1)
        scale_freqs = low_mel + delta_mel * np.arange(0, nfilts + 2)

        # assign freqs
        lower_edges_mel = scale_freqs[:-2]
        upper_edges_mel = scale_freqs[2:]
        center_freqs_mel = scale_freqs[1:-1]

        # assign freqs in
        center_freqs_hz = mel2hz(center_freqs_mel, conversion_approach)
        lower_edges_hz = mel2hz(lower_edges_mel, conversion_approach)
        upper_edges_hz = mel2hz(upper_edges_mel, conversion_approach)
        center_freqs = center_freqs_mel
    else:
        # compute points evenly spaced in frequency (points are in Hz)
        delta_hz = abs(high_freq - low_freq) / (nfilts + 1)
        scale_freqs = low_freq + delta_hz * np.arange(0, nfilts + 2)

        # assign freqs
        lower_edges_hz = scale_freqs[:-2]
        upper_edges_hz = scale_freqs[2:]
        center_freqs_hz = scale_freqs[1:-1]
        center_freqs = center_freqs_hz

    freqs = np.linspace(low_freq, high_freq, nfft // 2 + 1)
    fbank = np.zeros((nfilts, nfft // 2 + 1))

    for j, (center, lower, upper) in enumerate(
        zip(center_freqs_hz, lower_edges_hz, upper_edges_hz)
    ):
        left_slope = (freqs >= lower) == (freqs <= center)
        fbank[j, left_slope] = (freqs[left_slope] - lower) / (center - lower)

        right_slope = (freqs >= center) == (freqs <= upper)
        fbank[j, right_slope] = (upper - freqs[right_slope]) / (upper - center)

    # compute scaling
    scaling = scale_fbank(scale=scale, nfilts=nfilts)
    fbank = fbank * scaling

    return fbank, center_freqs


def mel_filter_banks(
    nfilts: int = 24,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    conversion_approach: MelConversionApproach = "Oshaghnessy",
):
    """
    Compute Mel-filter banks.The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default 20).
        nfft                (int) : the FFT size.
                                    (Default is 512).
        fs                  (int) : sample rate/ sampling frequency of the signal.
                                    (Default 16000 Hz).
        low_freq          (float) : lowest band edge of mel filters.
                                    (Default 0 Hz).
        high_freq         (float) : highest band edge of mel filters.
                                    (Default samplerate/2).
        scale               (str) : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        conversion_approach (str) : mel scale conversion approach.
                                    (Default is "Oshaghnessy").

    Returns:
        (tuple) :
            - (numpy.ndarray) : array of size nfilts * (nfft/2 + 1) containing filter bank. Each row holds 1 filter.
            - (numpy.ndarray) : array of center frequencies

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`conversion_approach` : can take the following options ["Oshaghnessy", "Beranek", "Lindsay"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Examples:

        .. plot::

            import numpy as np
            from spafe.utils.converters import mel2hz
            from spafe.utils.vis import show_fbanks
            from spafe.fbanks.mel_fbanks import mel_filter_banks

            # init var
            fs = 8000
            nfilt = 7
            nfft = 1024
            low_freq = 0
            high_freq = fs / 2

            # compute freqs for xaxis
            mhz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

            for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
                # compute fbank
                mel_fbanks_mat, mel_freqs = mel_filter_banks(nfilts=nfilt,
                                                             nfft=nfft,
                                                             fs=fs,
                                                             low_freq=low_freq,
                                                             high_freq=high_freq,
                                                            scale=scale)

                # visualize fbank
                show_fbanks(
                    mel_fbanks_mat,
                    [mel2hz(freq) for freq in mel_freqs],
                    mhz_freqs,
                    label + "Mel Filter Bank",
                    ylabel="Weight",
                    x1label="Frequency / Hz",
                    x2label="Frequency / Mel",
                    figsize=(14, 5),
                    fb_type="mel")
    """
    # generate mel fbanks by inversing regular mel fbanks
    mel_fbanks, mel_freqs = mel_filter_banks_helper(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
        fb_type="mel",
        conversion_approach=conversion_approach,
    )

    return mel_fbanks, mel_freqs


def inverse_mel_filter_banks(
    nfilts: int = 24,
    nfft: int = 512,
    fs: int = 16000,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    conversion_approach: MelConversionApproach = "Oshaghnessy",
):
    """
    Compute inverse Mel-filter banks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilt               (int) : the number of filters in the filter bank.
                                    (Default 20).
        nfft                (int) : the FFT size.
                                    (Default is 512).
        fs                  (int) : sample rate/ sampling frequency of the signal.
                                    (Default 16000 Hz).
        low_freq          (float) : lowest band edge of mel filters.
                                    (Default 0 Hz).
        high_freq         (float) : highest band edge of mel filters.
                                    (Default samplerate/2).
        scale               (str) : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        conversion_approach (str) : mel scale conversion approach.
                                    (Default is "Oshaghnessy").

    Returns:
        (tuple) :
            - (numpy.ndarray) : array of size nfilts * (nfft/2 + 1) containing filter bank. Each row holds 1 filter.
            - (numpy.ndarray) : array of center frequencies

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`conversion_approach` : can take the following options ["Oshaghnessy", "Beranek", "Lindsay"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Examples:

        .. plot::

            import numpy as np
            from spafe.utils.converters import mel2hz
            from spafe.utils.vis import show_fbanks
            from spafe.fbanks.mel_fbanks import inverse_mel_filter_banks

            # init var
            fs = 8000
            nfilt = 7
            nfft = 1024
            low_freq = 0
            high_freq = fs / 2

            # compute freqs for xaxis
            mhz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

            for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
                # compute fbank
                imel_fbanks_mat, imel_freqs = inverse_mel_filter_banks(nfilts=nfilt,
                                                                       nfft=nfft,
                                                                       fs=fs,
                                                                       low_freq=low_freq,
                                                                       high_freq=high_freq,
                                                                       scale=scale)

                # visualize fbank
                show_fbanks(
                    imel_fbanks_mat,
                    [mel2hz(freq) for freq in imel_freqs],
                    mhz_freqs,
                    label + "Inverse Mel Filter Bank",
                    ylabel="Weight",
                    x1label="Frequency / Hz",
                    x2label="Frequency / Mel",
                    figsize=(14, 5),
                    fb_type="mel")

    See Also:
        - :py:func:`spafe.fbanks.gammatone_fbanks.gammatone_filter_banks`
        - :py:func:`spafe.fbanks.linear_fbanks.linear_filter_banks`
        - :py:func:`spafe.fbanks.bark_fbanks.bark_filter_banks`
    """
    # inverse scaler value
    scales = {
        "ascendant": "descendant",
        "descendant": "ascendant",
        "constant": "constant",
    }

    # generate inverse mel fbanks by inversing regular mel fbanks
    imel_fbanks, mel_freqs = mel_filter_banks_helper(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scales[scale],
        fb_type="mel",
        conversion_approach=conversion_approach,
    )

    # inverse regular filter banks
    for i, pts in enumerate(imel_fbanks):
        imel_fbanks[i] = pts[::-1]

    # get inverse freqs
    imel_center_freqs = [
        hz2mel(freq, conversion_approach)
        for freq in (high_freq - mel2hz(mel_freqs, conversion_approach))
    ]

    return np.abs(imel_fbanks), imel_center_freqs
