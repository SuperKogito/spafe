"""

- Description : Mel and inverse Mel Features Cepstral Coefﬁcients (MFCCs and IMFCCs) extraction algorithm implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy.fftpack import dct
from ..utils.cepstral import normalize_ceps, lifter_ceps
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..fbanks.mel_fbanks import inverse_mel_filter_banks, mel_filter_banks
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling


def mel_spectrogram(
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
    conversion_approach="Oshaghnessy",
):
    """
    Compute the mel scale spectrogram.

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        pre_emph            (int) : apply pre-emphasis if 1.
                                    (Default is 1).
        pre_emph_coeff    (float) : pre-emphasis filter coefficient.
                                    (Default is 0.97).
        win_len           (float) : window length in sec.
                                    (Default is 0.025).
        win_hop           (float) : step between successive windows in sec.
                                    (Default is 0.01).
        win_type          (float) : window type to apply for the windowing.
                                    (Default is "hamming").
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq            (int) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq           (int) : highest band edge of mel filters (Hz).
                                    (Default is samplerate / 2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        fbanks    (numpy.ndarray) : filter bank matrix).
                                    (Default is None).
        conversion_approach (str) : approach to use for conversion to the erb scale.
                                    (Default is "Oshaghnessy").

    Returns:
        (numpy.ndarray) : the mel spectrogram (num_frames x nfilts).
        (numpy.ndarray) : the mel center frequencies.

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Oshaghnessy", "Lindsay"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/mel_spectrogram.png

           Architecture of Mel spectrogram extraction algorithm.

    Examples:
        .. plot::

            from spafe.features.mfcc import mel_spectrogram
            from spafe.utils.vis import show_spectrogram
            from scipy.io.wavfile import read

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            mSpec, _ = mel_spectrogram(sig,
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


            show_spectrogram(mSpec.T,
                             fs,
                             xmin=0,
                             xmax=len(sig)/fs,
                             ymin=0,
                             ymax=(fs/2)/1000,
                             dbf=80.0,
                             xlabel="Time (s)",
                             ylabel="Frequency (kHz)",
                             title="Mel spectrogram (dB)",
                             cmap="jet")

    """
    # get fbanks
    if fbanks is None:
        # compute fbank
        mel_fbanks_mat, _ = mel_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=high_freq,
            scale=scale,
            conversion_approach=conversion_approach,
        )
        fbanks = mel_fbanks_mat

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


def mfcc(
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
    conversion_approach="Oshaghnessy",
):
    """
    Compute MFCC features (Mel-frequency cepstral coefficients) from an audio
    signal. This function offers multiple approaches to features extraction
    depending on the input parameters. Implemenation is using FFT and based on
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.8029&rep=rep1&type=pdf

          - take the absolute value of the FFT
          - warp to a Mel frequency scale
          - take the DCT of the log-Mel-spectrum
          - return the first <num_ceps> components

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        num_ceps          (float) : number of cepstra to return.
                                    (Default is 13).
        pre_emph            (int) : apply pre-emphasis if 1.
                                    (Default is 1).
        pre_emph_coeff    (float) : pre-emphasis filter coefficient.
                                    (Default is 0.97).
        win_len           (float) : window length in sec.
                                    (Default is 0.025).
        win_hop           (float) : step between successive windows in sec.
                                    (Default is 0.01).
        win_type          (float) : window type to apply for the windowing.
                                    (Default is "hamming").
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq            (int) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq           (int) : highest band edge of mel filters (Hz).
                                    (Default is samplerate / 2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        dct_type            (int) : type of DCT used.
                                    (Default is 2).
        use_energy          (int) : overwrite C0 with true log energy.
                                    (Default is 0).
        lifter              (int) : apply liftering if not None.
                                    (Default is None).
        normalize           (int) : apply normalization if approach specified.
                                    (Default is 0).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : approach to use for conversion to the erb scale.
                                    (Default is "Oshaghnessy").

    Returns:
        (numpy.ndarray) : features - the MFFC features: num_frames x num_ceps

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Oshaghnessy", "Lindsay"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/mfccs.png

           Architecture of Mel frequency cepstral coefﬁcients extraction.

    Examples
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.mfcc import mfcc
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute mfccs and mfes
            mfccs  = mfcc(sig,
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
            show_features(mfccs, "Mel Frequency Cepstral Coefﬁcients", "MFCC Index", "Frame Index")
    """
    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # get features
    features, fourrier_transform = mel_spectrogram(
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
        fbanks=fbanks,
        scale=scale,
        conversion_approach=conversion_approach,
    )

    # -> log(.)
    features_no_zero = zero_handling(features)
    log_features = np.log(features_no_zero)

    # -> DCT(.)
    mfccs = dct(x=log_features, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        # compute the # Magnitude of the FFT and then the Power Spectrum
        magnitude_frames = np.absolute(fourrier_transform)
        power_frames = (1.0 / nfft) * ((magnitude_frames) ** 2)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        mfccs[:, 0] = np.log(energy)

    # liftering
    if lifter:
        mfccs = lifter_ceps(mfccs, lifter)

    # normalization
    if normalize:
        mfccs = normalize_ceps(mfccs, normalize)

    return mfccs


def imfcc(
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
    lifter=0,
    normalize=None,
    fbanks=None,
    conversion_approach="Oshaghnessy",
):
    """
    Compute Inverse MFCC features from an audio signal.

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        num_ceps          (float) : number of cepstra to return.
                                    (Default is 13).
        pre_emph            (int) : apply pre-emphasis if 1.
                                    (Default is 1).
        pre_emph_coeff    (float) : pre-emphasis filter coefficient.
                                    (Default is 0.97).
        win_len           (float) : window length in sec.
                                    (Default is 0.025).
        win_hop           (float) : step between successive windows in sec.
                                    (Default is 0.01).
        win_type          (float) : window type to apply for the windowing.
                                    (Default is "hamming").
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq            (int) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq           (int) : highest band edge of mel filters (Hz).
                                    (Default is samplerate / 2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        dct_type            (int) : type of DCT used.
                                    (Default is 2).
        use_energy          (int) : overwrite C0 with true log energy.
                                    (Default is 0).
        lifter              (int) : apply liftering if not None.
                                    (Default is None).
        normalize           (int) : apply normalization approach specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : approach to use for conversion to the erb scale.
                                    (Default is "Oshaghnessy").

    Returns:
        (numpy.ndarray) : features - the inverse MFFC features: num_frames x num_ceps

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Oshaghnessy", "Lindsay"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/imfccs.png

           Architecture of inverse Mel frequency cepstral coefﬁcients extraction algorithm.

    Examples
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.mfcc import imfcc
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute mfccs and mfes
            imfccs  = imfcc(sig,
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
            show_features(imfccs, "Inverse Mel Frequency Cepstral Coefﬁcients", "IMFCC Index","Frame Index")
    """
    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    #  get inverse Mel-fbank
    if fbanks is None:
        imel_fbanks_mat, _ = inverse_mel_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=high_freq,
            scale=scale,
            conversion_approach=conversion_approach,
        )
        fbanks = imel_fbanks_mat

    # compute imfcc
    imfccs = mfcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
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
        dct_type=dct_type,
        use_energy=use_energy,
        lifter=lifter,
        normalize=normalize,
        fbanks=fbanks,
        conversion_approach=conversion_approach,
    )
    return imfccs
