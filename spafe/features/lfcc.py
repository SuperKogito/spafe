"""

- Description : Linear Frequency Cepstral Coefﬁcients (LFCCs) extraction algorithm implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy.fftpack import dct
from ..fbanks.linear_fbanks import linear_filter_banks
from ..utils.cepstral import normalize_ceps, lifter_ceps
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling


def linear_spectrogram(
    sig,
    fs=16000,
    pre_emph=0,
    pre_emph_coeff=0.97,
    win_len=0.025,
    win_hop=0.01,
    win_type="hamming",
    nfilts=24,
    nfft=512,
    low_freq=0,
    high_freq=None,
    scale="constant",
    fbanks=None,
):
    """
    Compute the mel scale spectrogram.

    Args:
        sig    (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 (Default is 16000).
        pre_emph         (int) : apply pre-emphasis if 1.
                                 (Default is 1).
        pre_emph_coeff (float) : pre-emphasis filter coefficient.
                                 (Default is 0.97).
        win_len        (float) : window length in sec.
                                 (Default is 0.025).
        win_hop        (float) : step between successive windows in sec.
                                 (Default is 0.01).
        win_type       (float) : window type to apply for the windowing.
                                 (Default is "hamming").
        nfilts           (int) : the number of filters in the filter bank.
                                 (Default is 40).
        nfft             (int) : number of FFT points.
                                 (Default is 512).
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 (Default is 0).
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 (Default is samplerate / 2).
        scale           (str)  : monotonicity behavior of the filter banks.
                                 (Default is "constant").
        fbanks (numpy.ndarray) : filter bank matrix.
                                 (Default is None).

    Returns:
        (numpy.ndarray) : features - the LFFC features: num_frames x num_ceps


    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].

    Note:
        .. figure:: ../_static/architectures/linear_spectrogram.png

           Architecture of linear scale spectrogram extraction algorithm.

    Examples:
        .. plot::

            from spafe.features.lfcc import linear_spectrogram
            from spafe.utils.vis import show_spectrogram
            from scipy.io.wavfile import read

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute spectrogram
            lSpec, lfreqs = linear_spectrogram(sig,
                                               fs=fs,
                                               pre_emph=0,
                                               pre_emph_coeff=0.97,
                                               win_len=0.030,
                                               win_hop=0.015,
                                               win_type="hamming",
                                               nfilts=128,
                                               nfft=2048,
                                               low_freq=0,
                                               high_freq=fs/2)

            # visualize spectrogram
            show_spectrogram(lSpec.T,
                             fs,
                             xmin=0,
                             xmax=len(sig)/fs,
                             ymin=0,
                             ymax=(fs/2)/1000,
                             dbf=80.0,
                             xlabel="Time (s)",
                             ylabel="Frequency (kHz)",
                             title="Linear spectrogram (dB)",
                             cmap="jet")
    """
    # get fbanks
    if fbanks is None:
        # compute fbanks
        lin_fbanks_mat, lin_freqs = linear_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=high_freq,
            scale=scale,
        )
        fbanks = lin_fbanks_mat

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    # -> FFT -> |.|
    ## Magnitude of the FFT
    fourrier_transform = np.absolute(np.fft.fft(windows, nfft))
    fourrier_transform = fourrier_transform[:, : int(nfft / 2) + 1]

    ## Power Spectrum
    abs_fft_values = (1.0 / nfft) * np.square(fourrier_transform)

    #  -> x Mel-fbanks
    features = np.dot(abs_fft_values, fbanks.T)  # dB
    return features, fourrier_transform


def lfcc(
    sig,
    fs=16000,
    num_ceps=13,
    pre_emph=0,
    pre_emph_coeff=0.97,
    win_len=0.025,
    win_hop=0.01,
    win_type="hamming",
    nfilts=24,
    nfft=512,
    low_freq=0,
    high_freq=None,
    scale="constant",
    dct_type=2,
    use_energy=False,
    lifter=None,
    normalize=None,
    fbanks=None,
):
    """
    Compute the linear-frequency cepstral coefﬁcients (GFCC features) from an
    audio signal as described in [Sehili]_.

    Args:
        sig      (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 (Default is 16000).
        num_ceps       (float) : number of cepstra to return.
                                 (Default is 13).
        pre_emph         (int) : apply pre-emphasis if 1.
                                 (Default is 1).
        pre_emph_coeff (float) : pre-emphasis filter coefficient.
                                 (Default is 0.97).
        win_len        (float) : window length in sec.
                                 (Default is 0.025).
        win_hop        (float) : step between successive windows in sec.
                                 (Default is 0.01).
        win_type       (float) : window type to apply for the windowing.
                                 (Default is "hamming").
        nfilts           (int) : the number of filters in the filter bank.
                                 (Default is 40).
        nfft             (int) : number of FFT points.
                                 (Default is 512).
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 (Default is 0).
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 (Default is samplerate / 2).
        scale           (str)  : monotonicity behavior of the filter banks.
                                 (Default is "constant").
        dct_type         (int) : type of DCT used.
                                 (Default is 2).
        use_energy       (int) : overwrite C0 with true log energy.
                                 (Default is 0).
        lifter           (int) : apply liftering if specified.
                                 (Default is None).
        normalize        (str) : apply normalization if approach provided.
                                 (Default is None).
        fbanks (numpy.ndarray) : filter bank matrix.
                                 (Default is None).

    Returns:
        (numpy.ndarray) : 2d array of LFCC features (num_frames x num_ceps).

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].

    Note:
        .. figure:: ../_static/architectures/lfccs.png

           Architecture of linear frequency cepstral coefﬁcients extraction algorithm.

    Examples:
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.lfcc import lfcc
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute lfccs
            lfccs  = lfcc(sig,
                          fs=fs,
                          pre_emph=1,
                          pre_emph_coeff=0.97,
                          win_len=0.030,
                          win_hop=0.015,
                          win_type="hamming",
                          nfilts=128,
                          nfft=2048,
                          low_freq=0,
                          high_freq=8000,
                          normalize="mvn")

            # visualize features
            show_features(lfccs, "Linear Frequency Cepstral Coefﬁcients", "LFCC Index", "Frame Index")

    References:
        .. [Sehili] : Sehili, A. & Istrate, B. & Boudy, J. (2022).
                      Primary Investigation of Sound Recognition for a domotic
                      application using Support Vector.
    """
    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # get features
    features, fourrier_transform = linear_spectrogram(
        sig=sig,
        fs=fs,
        pre_emph=pre_emph,
        pre_emph_coeff=pre_emph_coeff,
        win_len=win_len,
        win_hop=win_hop,
        win_type=win_type,
        nfilts=nfilts,
        nfft=nfft,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
        fbanks=fbanks,
    )

    # -> log(.)
    # handle zeros: if feat is zero, we get problems with log
    features_no_zero = zero_handling(features)
    log_features = np.log(features_no_zero)

    #  -> DCT(.)
    lfccs = dct(x=log_features, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        # compute the # Magnitude of the FFT and then the Power Spectrum
        magnitude_frames = np.absolute(fourrier_transform)
        power_frames = (1.0 / nfft) * ((magnitude_frames) ** 2)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        lfccs[:, 0] = np.log(energy)

    # liftering
    if lifter:
        lfccs = lifter_ceps(lfccs, lifter)

    # normalization
    if normalize:
        lfccs = normalize_ceps(lfccs, normalize)
    return lfccs
