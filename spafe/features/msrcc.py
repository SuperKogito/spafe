"""

- Description : Magnitude based Spectral Root Cepstral Coefficients (MSRCCs) extraction algorithm implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy.fftpack import dct
from ..features.mfcc import mel_spectrogram
from ..utils.cepstral import normalize_ceps, lifter_ceps
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling


def msrcc(
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
    gamma=-1 / 7,
    dct_type=2,
    use_energy=False,
    lifter=None,
    normalize=None,
    fbanks=None,
    conversion_approach="Oshaghnessy",
):
    """
    Compute the Magnitude-based Spectral Root Cepstral Coefﬁcients (MSRCC) from
    an audio signal according to [Tapkir]_.

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
        gamma             (float) : power coefficient for resulting energies
                                    (Default -1/7).
        dct_type            (int) : type of DCT used.
                                    (Default is 2).
        use_energy          (int) : overwrite C0 with true log energy.
                                    (Default is 0).
        lifter              (int) : apply liftering if specified.
                                    (Default is None).
        normalize           (int) : apply normalization if specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : approach to use for conversion to the erb scale.
                                    (Default is "Oshaghnessy").

    Returns:
        (numpy.ndarray) : 2d array of MSRCC features (num_frames x num_ceps)

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Oshaghnessy", "Lindsay"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/msrccs.png

           Architecture of magnitude based spectral root cepstral coefﬁcients extraction algorithm.

    Examples:
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.msrcc import msrcc
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute msrccs
            msrccs  = msrcc(sig,
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
            show_features(msrccs, "Magnitude based Spectral Root Cepstral Coefficients", "MSRCC Index", "Frame Index")

    References:
        .. [Tapkir] : P. A. Tapkir, A. T. Patil, N. Shah and H. A. Patil,
                      "Novel Spectral Root Cepstral Features for Replay Spoof Detection,"
                      2018 Asia-Pacific Signal and Information Processing Association Annual
                      Summit and Conference (APSIPA ASC), 2018, pp. 1945-1950,
                      doi: 10.23919/APSIPA.2018.8659746.
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

    # -> (.)^(gamma)
    features = features**gamma

    # -> DCT(.)
    msrccs = dct(x=features, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        # compute the # Magnitude of the FFT and then the Power Spectrum
        magnitude_frames = np.absolute(fourrier_transform)
        power_frames = (1.0 / nfft) * ((magnitude_frames) ** 2)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        msrccs[:, 0] = np.log(energy)

    # liftering
    if lifter:
        msrccs = lifter_ceps(msrccs, lifter)

    # normalization
    if normalize:
        msrccs = normalize_ceps(msrccs, normalize)

    return msrccs
