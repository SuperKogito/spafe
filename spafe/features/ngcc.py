"""

- Description : Normalized Gammachirp Cepstral Coefficients (NGCCs) extraction algorithm implementation.
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


def ngcc(
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
    low_freq=None,
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
    Compute the normalized gammachirp cepstral coefﬁcients (NGCC features) from
    an audio signal according to [Zouhir]_.

    Args:
        sig         (numpy.ndarray) : input mono audio signal (Nx1).
        fs                  (int) : signal sampling frequency.
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
                                    (Default is 40.
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
        use_energy          (int) : overwrite C0 with true log energy
                                    (Default is 0).
        lifter              (int) : apply liftering if specifid.
                                    (Default is None).
        normalize           (int) : apply normalization if specifid.
                                    (Default is 0).
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
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute ngccs
            ngccs  = ngcc(sig,
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
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

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
