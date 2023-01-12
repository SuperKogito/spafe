"""

- Description : Bark Frequency Cepstral Coefﬁcients (BFCCs) extraction algorithm implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Optional

import numpy as np
from scipy.fftpack import dct

from ..fbanks.bark_fbanks import bark_filter_banks
from ..utils.cepstral import normalize_ceps, lifter_ceps, NormalizationType
from ..utils.converters import BarkConversionApproach
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.filters import ScaleType
from ..utils.preprocessing import (
    pre_emphasis,
    framing,
    windowing,
    zero_handling,
    SlidingWindow,
)


def intensity_power_law(w: np.ndarray) -> np.ndarray:
    """
    Apply the intensity power law based on [Hermansky]_ .

    Args:
        w (numpy.ndarray) : signal information.

    Returns:
        (numpy.ndarray) : array after intensity power law.

    Note:
        .. math::

            E(\\omega) = \\frac{(\\omega^{2}+56.8 \\times 10^{6}) \\omega^{4}}{(\\omega^{2}+6.3 \\times 10^{6})^{2} \\times (\\omega^{2}+0.38 \\times 10^{9})}
    """

    f = lambda w, c, p: w**2 + c * 10**p
    E = (f(w, 56.8, 6) * w**4) / (f(w, 6.3, 6) * f(w, 0.38, 9) * f(w**3, 9.58, 26))
    return E ** (1 / 3)


def bark_spectrogram(
    sig: np.ndarray,
    fs: int = 16000,
    pre_emph: float = 0,
    pre_emph_coeff: float = 0.97,
    window: Optional[SlidingWindow] = None,
    nfilts: int = 24,
    nfft: int = 512,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    fbanks: Optional[np.ndarray] = None,
    conversion_approach: BarkConversionApproach = "Wang",
):
    """
    Compute the bark scale spectrogram.

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        num_ceps          (float) : number of cepstra to return.
                                    (Default is 13).
        pre_emph            (bool) : apply pre-emphasis if 1.
                                    (Default is True).
        pre_emph_coeff    (float) : pre-emphasis filter coefficient).
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
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (tuple) :
            - (numpy.ndarray) : spectrogram matrix.
            - (numpy.ndarray) : fourrier transform.

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/bark_spectrogram.png

           Architecture of Bark spectrogram computation algorithm.

    Examples:
        .. plot::

            from spafe.features.bfcc import bark_spectrogram
            from spafe.utils.vis import show_spectrogram
            from spafe.utils.preprocessing import SlidingWindow
            from scipy.io.wavfile import read

            # read audio
            fpath = "../../../tests/data/test.wav"
            fs, sig = read(fpath)

            # compute bark spectrogram
            bSpec, bfreqs = bark_spectrogram(sig,
                                            fs=fs,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.03, 0.015, "hamming"),
                                            nfilts=128,
                                            nfft=2048,
                                            low_freq=0,
                                            high_freq=fs/2)

            # visualize spectrogram
            show_spectrogram(bSpec.T,
                             fs=fs,
                             xmin=0,
                             xmax=len(sig)/fs,
                             ymin=0,
                             ymax=(fs/2)/1000,
                             dbf=80.0,
                             xlabel="Time (s)",
                             ylabel="Frequency (kHz)",
                             title="Bark spectrogram (dB)",
                             cmap="jet")
    """
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
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

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
    fourrier_transform = np.absolute(np.fft.rfft(windows, nfft))

    ## Power Spectrum
    abs_fft_values = (1.0 / nfft) * np.square(fourrier_transform)

    # get features
    features = np.dot(abs_fft_values, fbanks.T)  # dB
    return features, fourrier_transform


def bfcc(
    sig: np.ndarray,
    fs: int = 16000,
    num_ceps: int = 13,
    pre_emph: bool = True,
    pre_emph_coeff: float = 0.97,
    window: Optional[SlidingWindow] = None,
    nfilts: int = 26,
    nfft: int = 512,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    dct_type: int = 2,
    use_energy: bool = False,
    lifter: Optional[int] = None,
    normalize: Optional[NormalizationType] = None,
    fbanks: Optional[np.ndarray] = None,
    conversion_approach: BarkConversionApproach = "Wang",
) -> np.ndarray:
    """
    Compute the Bark Frequency Cepstral Coefﬁcients (BFCCs) from an audio
    signal as described in [Kaminska]_.

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        num_ceps          (float) : number of cepstra to return.
                                    (Default is 13).
        pre_emph            (bool) : apply pre-emphasis if 1.
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
        dct_type            (int) : type of DCT used.
                                    (Default is 2).
        use_energy          (int) : overwrite C0 with true log energy.
                                    (Default is 0).
        lifter              (int) : apply liftering if specified.
                                    (Default is None).
        normalize           (str) : apply normalization if approach specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (numpy.ndarray) : 2d array of BFCC features (num_frames x num_ceps).

    Raises:
        ParameterError
            if nfilts < num_ceps

    Tip:
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/bfccs.png

           Architecture of Bark frequency cepstral coefﬁcients extraction algorithm.

    References:
        .. [Kaminska] : Kamińska, D. & Sapiński, T. & Anbarjafari, G. (2017).
                        Efficiency of chosen speech descriptors in relation to emotion recognition.
                        EURASIP Journal on Audio Speech and Music Processing. 2017. 10.1186/s13636-017-0100-x.

    Examples
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.bfcc import bfcc
            from spafe.utils.preprocessing import SlidingWindow
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../tests/data/test.wav"
            fs, sig = read(fpath)

            # compute bfccs
            bfccs  = bfcc(sig,
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
            show_features(bfccs, "Bark Frequency Cepstral Coefﬁcients", "BFCC Index", "Frame Index")
    """
    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # compute features
    features, fourrier_transform = bark_spectrogram(
        sig=sig,
        fs=fs,
        pre_emph=pre_emph,
        pre_emph_coeff=pre_emph_coeff,
        window=window,
        nfilts=nfilts,
        nfft=nfft,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
        fbanks=fbanks,
        conversion_approach=conversion_approach,
    )

    # Equal-loudness power law (.) -> Intensity-loudness power law
    ipl_features = intensity_power_law(w=features)

    # -> log(.)
    # handle zeros: if feat is zero, we get problems with log
    features_no_zero = zero_handling(x=ipl_features)
    log_features = np.log(features_no_zero)

    #  -> DCT(.)
    bfccs = dct(x=log_features, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        # compute the # Magnitude of the FFT and then the Power Spectrum
        magnitude_frames = np.absolute(fourrier_transform)
        power_frames = (1.0 / nfft) * ((magnitude_frames) ** 2)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        bfccs[:, 0] = np.log(energy)

    # liftering
    if lifter:
        bfccs = lifter_ceps(bfccs, lifter)

    # normalization
    if normalize:
        bfccs = normalize_ceps(bfccs, normalize)

    return bfccs
