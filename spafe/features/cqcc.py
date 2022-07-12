"""

- Description : Constant Q-transform Cepstral Coeﬃcients (CQCCs) extraction algorithm implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy.fftpack import dct
from scipy.signal import resample
from ..utils.spectral import compute_constant_qtransform
from ..utils.cepstral import normalize_ceps, lifter_ceps
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling


def cqt_spectrogram(
    sig,
    fs=16000,
    pre_emph=0,
    pre_emph_coeff=0.97,
    win_len=0.025,
    win_hop=0.01,
    win_type="hamming",
    nfft=512,
    low_freq=0,
    high_freq=None,
    number_of_octaves=7,
    number_of_bins_per_octave=24,
    spectral_threshold=0.005,
    f0=120,
    q_rate=1.0,
):
    """
    Compute the Constant-Q Cepstral spectrogram from an audio signal as in [Todisco]_.

    Args:
        sig             (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                        (int) : the sampling frequency of the signal we are working with.
                                          (Default is 16000).
        pre_emph                  (int) : apply pre-emphasis if 1.
                                          (Default is 1).
        pre_emph_coeff          (float) : pre-emphasis filter coefficient.
                                          (Default is 0.97).
        win_len                 (float) : window length in sec.
                                          (Default is 0.025).
        win_hop                 (float) : step between successive windows in sec.
                                          (Default is 0.01).
        win_type                (float) : window type to apply for the windowing.
                                          (Default is "hamming").
        nfft                      (int) : number of FFT points.
                                          (Default is 512).
        low_freq                  (int) : lowest band edge of mel filters (Hz).
                                          (Default is 0).
        high_freq                 (int) : highest band edge of mel filters (Hz).
                                          (Default is samplerate/2).
        number_of_octaves         (int) : number of occtaves.
                                          (Default is 7).
        number_of_bins_per_octave (int) : numbers of bins oer occtave.
                                          (Default is 24).
        spectral_threshold        (int) : spectral threshold.
                                          (Default is 0.005).
        f0                        (int) : fundamental frequency.
                                          (Default is 28).
        q_rate                    (int) : number of FFT points.
                                          (Default is 1.0).

    Returns:
        (numpy.ndarray) : 2d array of the spectrogram matrix (num_frames x num_ceps)

    Note:
        .. figure:: ../_static/architectures/cqt_spectrogram.png

           Architecture of Constant q-transform spectrogram computation algorithm.

    Examples:
        .. plot::

            from spafe.features.cqcc import cqt_spectrogram
            from spafe.utils.vis import show_spectrogram
            from scipy.io.wavfile import read

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute spectrogram
            qSpec = cqt_spectrogram(sig,
                                    fs=fs,
                                    pre_emph=0,
                                    pre_emph_coeff=0.97,
                                    win_len=0.030,
                                    win_hop=0.015,
                                    win_type="hamming",
                                    nfft=2048,
                                    low_freq=0,
                                    high_freq=fs/2)

            # visualize spectrogram
            show_spectrogram(qSpec,
                             fs=fs,
                             xmin=0,
                             xmax=len(sig)/fs,
                             ymin=0,
                             ymax=(fs/2)/1000,
                             dbf=80.0,
                             xlabel="Time (s)",
                             ylabel="Frequency (kHz)",
                             title="CQT spectrogram (dB)",
                             cmap="jet")
    """
    # init freqs
    high_freq = high_freq or fs / 2
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    # -> compute constant Q-transform
    constant_qtransform = compute_constant_qtransform(
        windows,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq,
        nfft=nfft,
        number_of_octaves=number_of_octaves,
        number_of_bins_per_octave=number_of_bins_per_octave,
        win_type=win_type,
        spectral_threshold=spectral_threshold,
        f0=f0,
        q_rate=q_rate,
    )

    return constant_qtransform


def cqcc(
    sig,
    fs=16000,
    num_ceps=13,
    pre_emph=1,
    pre_emph_coeff=0.97,
    win_len=0.02,
    win_hop=0.01,
    win_type="hamming",
    nfft=512,
    low_freq=0,
    high_freq=None,
    dct_type=2,
    lifter=None,
    normalize=None,
    number_of_octaves=7,
    number_of_bins_per_octave=24,
    resampling_ratio=0.95,
    spectral_threshold=0.005,
    f0=120,
    q_rate=1.0,
):
    """
    Compute the Constant-Q Cepstral Coeﬃcients (CQCC features) from an audio signal
    as described in [Todisco]_.

    Args:
        sig             (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                        (int) : the sampling frequency of the signal we are working with.
                                          (Default is 16000).
        num_ceps                (float) : number of cepstra to return.
                                          (Default is 13).
        pre_emph                  (int) : apply pre-emphasis if 1.
                                          (Default is 1).
        pre_emph_coeff          (float) : pre-emphasis filter coefficient.
                                          (Default is 0.97).
        win_len                 (float) : window length in sec.
                                          (Default is 0.025).
        win_hop                 (float) : step between successive windows in sec.
                                          (Default is 0.01).
        win_type                (float) : window type to apply for the windowing.
                                          (Default is "hamming").
        nfft                      (int) : number of FFT points.
                                          (Default is 512).
        low_freq                  (int) : lowest band edge of mel filters (Hz).
                                          (Default is 0).
        high_freq                 (int) : highest band edge of mel filters (Hz).
                                          (Default is samplerate/2).
        dct_type                  (int) : type of DCT used.
                                          (Default is 2).
        lifter                    (int) : apply liftering if value given.
                                          (Default is None).
        normalize                 (int) : normalization approach.
                                          (Default is None).
        number_of_octaves         (int) : number of occtaves.
                                          (Default is 7).
        number_of_bins_per_octave (int) : numbers of bins oer occtave.
                                          (Default is 24).
        resampling_ratio          (int) : ratio to use for the uniform resampling.
                                          (Default is 0.95).
        spectral_threshold        (int) : spectral threshold.
                                          (Default is 0.005).
        f0                        (int) : fundamental frequency.
                                          (Default is 28).
        q_rate                    (int) : number of FFT points.
                                          (Default is 1.0).

    Returns:
        (numpy.ndarray) : 2d array of BFCC features (num_frames x num_ceps).

    Tip:
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].

    References:
        .. [Todisco] : Todisco M., Héctor Delgado H., Evans N., Constant Q cepstral
                       coefficients: A spoofing countermeasure for automatic speaker verification,
                       Computer Speech & Language, Volume 45, 2017, Pages 516-535,
                       ISSN 0885-2308, https://doi.org/10.1016/j.csl.2017.01.001.

    Note:
        .. figure:: ../_static/architectures/cqccs.png

           Architecture of constant q-transform cepstral coefﬁcients extraction algorithm.

    Examples
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.cqcc import cqcc
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute cqccs
            cqccs  = cqcc(sig,
                          fs=fs,
                          pre_emph=1,
                          pre_emph_coeff=0.97,
                          win_len=0.030,
                          win_hop=0.015,
                          win_type="hamming",
                          nfft=2048,
                          low_freq=0,
                          high_freq=fs/2,
                          normalize="mvn")

            # visualize features
            show_features(cqccs, "Constant Q-Transform Cepstral Coefﬁcients", "CQCC Index", "Frame Index")
    """
    # get cqt
    constant_qtransform = cqt_spectrogram(
        sig,
        fs=fs,
        pre_emph=pre_emph,
        pre_emph_coeff=pre_emph_coeff,
        win_len=win_len,
        win_hop=win_hop,
        win_type=win_type,
        nfft=nfft,
        low_freq=low_freq,
        high_freq=high_freq,
        number_of_octaves=number_of_octaves,
        number_of_bins_per_octave=number_of_bins_per_octave,
        spectral_threshold=spectral_threshold,
        f0=f0,
        q_rate=q_rate,
    )

    # |Xcq|**2
    power_spectrum = np.absolute(constant_qtransform) ** 2

    # -> log(.)
    # handle zeros: if feat is zero, we get problems with log
    features_no_zero = zero_handling(x=power_spectrum)
    log_features = np.log(features_no_zero)

    # uniform resampling
    resampled_features = resample(
        log_features, int(len(log_features) * resampling_ratio)
    )

    #  -> DCT(.)
    cqccs = dct(x=resampled_features, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]

    # apply filter
    if lifter:
        cqccs = lifter_ceps(cqccs, lifter)

    # normalization
    if normalize:
        cqccs = normalize_ceps(cqccs, normalize)
    return cqccs
