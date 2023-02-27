"""

- Description : Normalized Gammachirp Cepstral Coefficients (NGCCs) extraction algorithm implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Optional

import numpy as np
from scipy.fftpack import dct

from ..fbanks.gammatone_fbanks import gammatone_filter_banks
from ..utils.cepstral import normalize_ceps, lifter_ceps, NormalizationType
from ..utils.converters import ErbConversionApproach
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.filters import ScaleType
from ..utils.preprocessing import (
    pre_emphasis,
    framing,
    windowing,
    zero_handling,
    SlidingWindow,
)


def ngcc(
    sig: np.ndarray,
    fs: int = 16000,
    num_ceps=13,
    pre_emph: bool = True,
    pre_emph_coeff: float = 0.97,
    window: Optional[SlidingWindow] = None,
    nfilts: int = 24,
    nfft: int = 512,
    low_freq: Optional[float] = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    dct_type: int = 2,
    use_energy: bool = False,
    lifter: Optional[int] = None,
    normalize: Optional[NormalizationType] = None,
    fbanks: Optional[np.ndarray] = None,
    conversion_approach: ErbConversionApproach = "Glasberg",
) -> np.ndarray:
    """
    Compute the normalized gammachirp cepstral coefﬁcients (NGCC features) from
    an audio signal according to [Zouhir]_.

    Args:
        sig       (numpy.ndarray) : input mono audio signal (Nx1).
        fs                  (int) : signal sampling frequency.
                                    (Default is 16000).
        num_ceps            (int) : number of cepstra to return.
                                    (Default is 13).
        pre_emph           (bool) : apply pre-emphasis if 1.
                                    (Default is True).
        pre_emph_coeff    (float) : pre-emphasis filter coefficient.
                                    (Default is 0.97).
        window    (SlidingWindow) : sliding window object.
                                    (Default is None).
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40.
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq          (float) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq         (float) : highest band edge of mel filters (Hz).
                                    (Default is samplerate / 2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        dct_type            (int) : type of DCT used.
                                    (Default is 2).
        use_energy         (bool) : overwrite C0 with true log energy
                                    (Default is False).
        lifter              (int) : apply liftering if specified.
                                    (Default is None).
        normalize           (str) : apply normalization if specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : erb scale conversion approach.
                                    (Default is "Glasberg").

    Returns:
        (numpy.ndarray) : 2d array of NGCC features (num_frames x num_ceps)

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Glasberg"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/ngccs.png

           Architecture of normalized gammachirp cepstral coefﬁcients extraction algorithm.

    Examples:
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.ngcc import ngcc
            from spafe.utils.preprocessing import SlidingWindow
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../tests/data/test.wav"
            fs, sig = read(fpath)

            # compute ngccs
            ngccs  = ngcc(sig,
                          fs=fs,
                          pre_emph=1,
                          pre_emph_coeff=0.97,
                          window=SlidingWindow(0.03, 0.015, "hamming"),
                          nfilts=128,
                          nfft=2048,
                          low_freq=0,
                          high_freq=8000,
                          normalize="mvn")

            # visualize features
            show_features(ngccs, "Normalized Gammachirp Cepstral Coefficients", "NGCC Index", "Frame Index")

    References:
        .. [Zouhir] : Zouhir, Y., & Ouni, K. (2016).
                     Feature Extraction Method for Improving Speech Recognition in Noisy Environments.
                     J. Comput. Sci., 12, 56-61.
    """
    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # get fbanks
    if fbanks is None:
        # compute fbanks
        gamma_fbanks_mat, _ = gammatone_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=high_freq,
            scale=scale,
            conversion_approach=conversion_approach,
        )
        fbanks = gamma_fbanks_mat

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

    # -> FFT -> |.|**2
    fourrier_transform = np.absolute(np.fft.rfft(windows, nfft))
    abs_fft_values = (1.0 / nfft) * np.square(fourrier_transform)

    #  -> x Gammatone fbanks
    features = np.dot(abs_fft_values, fbanks.T)

    # -> log(.)
    # handle zeros: if feat is zero, we get problems with log
    features_no_zero = zero_handling(x=features)
    log_features = np.log(features_no_zero)

    #  -> DCT(.)
    ngccs = dct(x=log_features, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        # compute the # Magnitude of the FFT and then the Power Spectrum
        magnitude_frames = np.absolute(fourrier_transform)
        power_frames = (1.0 / nfft) * ((magnitude_frames) ** 2)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        ngccs[:, 0] = np.log(energy)

    # liftering
    if lifter:
        ngccs = lifter_ceps(ngccs, lifter)

    # normalization
    if normalize:
        ngccs = normalize_ceps(ngccs, normalize)

    return ngccs
