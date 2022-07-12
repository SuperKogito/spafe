"""

- Description : Gammatone Frequency Cepstral Coefﬁcients (GFCCs) extraction algorithm implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy.fftpack import dct
from ..utils.cepstral import normalize_ceps, lifter_ceps
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..fbanks.gammatone_fbanks import gammatone_filter_banks
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling


def erb_spectrogram(
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
    conversion_approach="Glasberg",
):
    """
    Compute the Gammatone/ erb scale spectrogram also known as Cochleagram.

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
                                    (Default is "hamming".
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512.
        low_freq            (int) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq           (int) : highest band edge of mel filters (Hz).
                                    (Default is samplerate / 2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : approach to use for conversion to the erb scale.
                                    (Default is "Glasberg").

    Returns:
        (tuple) :
            - (numpy.ndarray) : the erb spectrogram (num_frames x nfilts)
            - (numpy.ndarray) : the fourrier transform matrix.

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`conversion_approach` : can take the following options ["Glasberg"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/gammatone_spectrogram.png

           Architecture of the Gammatone spectrogram computation algorithm.

    Examples:
        .. plot::

            from spafe.features.gfcc import erb_spectrogram
            from spafe.utils.vis import show_spectrogram
            from scipy.io.wavfile import read

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute erb spectrogram
            gSpec, gfreqs = erb_spectrogram(sig,
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
            show_spectrogram(gSpec.T,
                             fs=fs,
                             xmin=0,
                             xmax=len(sig)/fs,
                             ymin=0,
                             ymax=(fs/2)/1000,
                             dbf=80.0,
                             xlabel="Time (s)",
                             ylabel="Frequency (kHz)",
                             title="Erb spectrogram (dB)",
                             cmap="jet")
    """
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

    #  -> x Gammatone-fbanks
    features = np.dot(abs_fft_values, fbanks.T)
    return features, fourrier_transform


def gfcc(
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
    conversion_approach="Glasberg",
):
    """
    Compute the Gammatone-Frequency Cepstral Coefﬁcients (GFCC features) from an
    audio signal as described in [Jeevan]_ and [Xu]_.

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        num_ceps          (float) : number of cepstra to return).
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
                                    (Default is "hamming".
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
        lifter              (int) : apply liftering if value given.
                                    (Default is None).
        normalize           (int) : apply normalization if type specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : erb scale conversion approach.
                                    (Default is "Glasberg").

    Returns:
        (numpy.ndarray) : 2d array of GFCC features (num_frames x num_ceps)

    Raises:
        ParameterError
            if nfilts < num_ceps

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Glasberg"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/gfccs.png

           Architecture of the Gammatone frequency cepstral coefﬁcients extraction algorithm.

    References:
        .. [Jeevan] : Jeevan, M., Dhingra, A., Hanmandlu, M., & Panigrahi, B. K. (2016).
                      Robust Speaker Verification Using GFCC Based i-Vectors.
                      Proceedings of the International Conference on Signal,
                      Networks, Computing, and Systems, 85–91. doi:10.1007/978-81-322-3592-7_9

        .. [Xu] : Xu, H., Lin, L., Sun, X., & Jin, H. (2012).
                  A New Algorithm for Auditory Feature Extraction.
                  2012 International Conference on Communication Systems
                  and Network Technologies. doi:10.1109/csnt.2012.57

    Examples
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.gfcc import gfcc
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute mfccs and mfes
            gfccs  = gfcc(sig,
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
            show_features(gfccs, "Gammatone Frequency Cepstral Coefﬁcients", "GFCC Index", "Frame Index")
    """
    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # get features
    features, fourrier_transform = erb_spectrogram(
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
        conversion_approach=conversion_approach,
        fbanks=fbanks,
    )

    # compute the filter bank energies
    nonlin_rect_features = np.power(features, 1 / 3)
    gfccs = dct(x=nonlin_rect_features, type=dct_type, axis=1, norm="ortho")[
        :, :num_ceps
    ]

    # use energy for 1st features column
    if use_energy:
        # compute the # Magnitude of the FFT and then the Power Spectrum
        magnitude_frames = np.absolute(fourrier_transform)
        power_frames = (1.0 / nfft) * ((magnitude_frames) ** 2)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        gfccs[:, 0] = np.log(energy)

    # liftering
    if lifter:
        gfccs = lifter_ceps(gfccs, lifter)

    # normalization
    if normalize:
        gfccs = normalize_ceps(gfccs, normalize)
    return gfccs
