"""

- Description : Phase based Spectral Root Cepstral Coefficients (PSRCCs) extraction algorithm implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy.fftpack import dct
from ..fbanks.mel_fbanks import mel_filter_banks
from ..utils.cepstral import normalize_ceps, lifter_ceps
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling


def psrcc(
    sig,
    fs=16000,
    num_ceps=13,
    pre_emph=0,
    pre_emph_coeff=0.97,
    win_len=0.025,
    win_hop=0.01,
    win_type="hamming",
    nfilts=26,
    nfft=512,
    low_freq=None,
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
    Compute the Phase-based Spectral Root Cepstral Coefﬁcients (PSRCC) from an
    audio signal according to [Tapkir]_.

    Args:
        sig         (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
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
        lifter              (int) : apply liftering if specifid.
                                    (Default is None).
        normalize           (int) : apply normalization if specifid.
                                    (Default is 0).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : mel scale conversion scale.
                                    (Default is "Oshaghnessy").

    Returns:
        (numpy.ndarray) : 2d array of PSRCC features (num_frames x num_ceps)

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Oshaghnessy", "Lindsay"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/psrccs.png

           Architecture of phase based spectral root cepstral coefﬁcients extraction algorithm.

    Examples:
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.psrcc import psrcc
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute psrccs
            psrccs  = psrcc(sig,
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
            show_features(psrccs, "Phase based Spectral Root Cepstral Coefficients", "PSRCC Index", "Frame Index")
    """
    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

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

    # -> FFT ->
    fourrier_transform = np.fft.rfft(windows, nfft)
    fft_phases = np.angle(z=fourrier_transform, deg=True)
    fft_phases = (360 + fft_phases) * (fft_phases < 0) + fft_phases * (fft_phases > 0)

    # -> x Mel-fbanks
    features = np.dot(fft_phases, fbanks.T)

    # -> (.)^(gamma)
    features = features**gamma

    # assign 0 to values to be computed based on negative phases (otherwise results in nan)
    features[np.isnan(features)] = 0
    # assign max to values to be computed based on 0 phases (otherwise results in inf)
    features[np.isinf(features)] = features.max()

    # -> DCT(.)
    psrccs = dct(x=features, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        # compute the # Magnitude of the FFT and then the Power Spectrum
        magnitude_frames = np.absolute(fourrier_transform)
        power_frames = (1.0 / nfft) * ((magnitude_frames) ** 2)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        psrccs[:, 0] = np.log(energy)

    # liftering
    if lifter:
        psrccs = lifter_ceps(psrccs, lifter)

    # normalization
    if normalize:
        psrccs = normalize_ceps(psrccs, normalize)

    return psrccs
