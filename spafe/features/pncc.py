"""

- Description : Power-Normalized Cepstral Coefficients (PNCCs) extraction algorithm implementation.
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
    SlidingWindow,
)


def medium_time_power_calculation(P: np.ndarray, M: int = 2) -> np.ndarray:
    """
    Compute the medium time power calulations according to [Kim]_ .

    Args:
        P (numpy.ndarray) : signal stft power.
        M           (int) : the temporal integration factor.

    Returns:
        (numpy.ndarray) : medium time power values.

    Note:
        .. math::
            \\tilde{Q}[m, l]=\\frac{1}{2 M+1} \\sum_{m^{\\prime}=m-M}^{m+M} P[m^{\\prime}]

        where :math:`\\tilde{Q}` is the medium time power, :math:`P` is the power, :math:`m` is the
        frame index, :math:`l` is te channel index and :math:`M` is the temporal integration factor.
    """
    Q_tilde = np.array(
        [
            [
                (1 / (2 * M + 1))
                * sum(P[max(0, m - M) : min(P.shape[0], m + M + 1), l])
                for l in range(P.shape[1])
            ]
            for m in range(P.shape[0])
        ]
    )
    return Q_tilde


def asymmetric_lowpass_filtering(
    Q_tilde_in: np.ndarray, lm_a: float = 0.999, lm_b: float = 0.5
) -> np.ndarray:
    """
    Apply asymmetric lowpass filter according to [Kim]_ .

    Args:
        Q_tilde_in (numpy.ndarray) : rectified signal.
        lm_a               (float) : filter parameter; lambda a.
        lm_b               (float) : filter parameter; lambda b.

    Returns:
        (numpy.ndarray) : filtered signal.

    Note:
        .. math::
            \\tilde{Q}_{out}[m, l]=\\left\\{\\begin{array}{l}
            \\lambda_{a}\\tilde{Q}_{out}[m-1, l]+(1-\\lambda_{a})\\tilde{Q}_{in}[m, l], & \\text{if } \\tilde{Q}_{in}[m, l]\\geq \\tilde{Q}_{out}[m-1, l] \\\\
            \\lambda_{b}\\tilde{Q}_{out}[m-1, l]+(1-\\lambda_{b})\\tilde{Q}_{in}[m, l], & \\text{if } \\tilde{Q}_{in}[m, l]<\\tilde{Q}_{out}[m-1, l] \\end{array}\\right.

        where :math:`\\tilde{Q}_{in}` and :math:`\\tilde{Q}_{out}` are arbitrary input and output,
        :math:`\\lambda_{a}` and :math:`\\lambda_{b}` are filter related parameters, :math:`m` is the
        frame index, :math:`l` is the channel index.
    """
    Q_tilde_out = np.zeros_like(Q_tilde_in)
    Q_tilde_out[0,] = 0.9 * Q_tilde_in[0,]

    # compute asymmetric nonlinear filter
    for m in range(Q_tilde_out.shape[0]):
        Q1 = lm_a * Q_tilde_out[m - 1, :] + (1 - lm_a) * Q_tilde_in[m, :]
        Q2 = lm_b * Q_tilde_out[m - 1, :] + (1 - lm_b) * Q_tilde_in[m, :]

        Q_tilde_out[m, :] = np.where(
            Q_tilde_in[m,] >= Q_tilde_out[m - 1, :],
            Q1,
            Q2,
        )
    return Q_tilde_out


def temporal_masking(
    Q_tilde_0: np.ndarray, lam_t: float = 0.85, myu_t: float = 0.2
) -> np.ndarray:
    """

    Args:
        Q_tilde_0 (numpy.ndarray) : rectified signal.
        lam_t             (float) : the forgetting factor-
        myu_t             (float) : the recognition accuracy.

    Returns:
        (numpy.ndarray) : Q_tilde_tm = temporal_masking(Q_tilde_0)

    Note:
        The previous steps can be summarised in the following graph from [Kim]_.

        .. figure:: ../_static/architectures/pncc_temporal_masking.png

            Block diagram of the components that accomplish temporal masking [Kim]_
    """
    # rectified_signal[m, l]
    Q_tilde_tm = np.zeros_like(Q_tilde_0)
    online_peak_power = np.zeros_like(Q_tilde_0)

    Q_tilde_tm[0, :] = Q_tilde_0[0,]
    online_peak_power[0, :] = Q_tilde_0[0, :]

    for m in range(1, Q_tilde_0.shape[0]):
        online_peak_power[m, :] = np.maximum(
            lam_t * online_peak_power[m - 1, :], Q_tilde_0[m, :]
        )
        Q_tilde_tm[m, :] = np.where(
            Q_tilde_0[m, :] >= lam_t * online_peak_power[m - 1, :],
            Q_tilde_0[m, :],
            myu_t * online_peak_power[m - 1, :],
        )

    return Q_tilde_tm


def weight_smoothing(
    R_tilde: np.ndarray, Q_tilde: np.ndarray, nfilts: int = 128, N: int = 4
) -> np.ndarray:
    """
    Apply spectral weight smoothing according to [Kim]_.

    Args:
        R_tilde (numpy.ndarray) :
        Q_tilde (numpy.ndarray) : medium time power
        nfilts            (int) : total number of channels / filters
        N                 (int) :

    Returns:
        (numpy.ndarray) : time-averaged frequency-averaged transfer function.

    Note:
        .. math::
            \\tilde{S}[m, l]=(\\frac{1}{l_{2}-l_{1}+1} \\sum_{l^{\\prime}=l_{1}}^{l_{2}} \\frac{\\tilde{R}[m, l^{\\prime}]}{\\tilde{Q}[m, l^{\\prime}]})

        where :math:`l_{2}=\\min (l+N, L)` and :math:`l_{1}=\\max (l-N, 1)`, and :math:`L` is the total number of channels,
        and :math:`\\tilde{R}` is the output of the asymmetric noise suppression and temporal masking modules
        and :math:`\\tilde{S}` is the time-averaged, frequency-averaged transfer function.
    """
    L = nfilts
    S_tilde = np.zeros_like(R_tilde)
    for m in range(R_tilde.shape[0]):
        for l in range(R_tilde.shape[1]):
            l1 = max(l - N, 1)
            l2 = min(l + N, L)
            # compute smoothing output
            S_tilde[m, l] = (1 / float(l2 - l1 + 1)) * sum(
                [R_tilde[m, lprime] / Q_tilde[m, lprime] for lprime in range(l1, l2)]
            )

    return S_tilde


def mean_power_normalization(
    T: np.ndarray, lam_myu: float = 0.999, nfilts: int = 80, k: int = 1
) -> np.ndarray:
    """
    Apply mean power normalization according to [Kim]_.

    Args:
        T (numpy.ndarray) : represents the transfer function.
        lam_myu   (float) : time constant.
        nfilts      (int) : total number of channels / filters.
        k           (int) : arbitrary constant.


    Returns:
        (numpy.ndarray) normalized mean power.

    Note:
        .. math::
            \\mu[m]=\\lambda_{\\mu} \\mu[m-1]+\\frac{(1-\\lambda_{\\mu})}{L} \\sum_{l=0}^{L-1} T[m, l]

            U[m, l]=k \\frac{T[m, l]}{\\mu[m]}

        where :math:`\\lambda_{\\mu}` is the time constant.
    """
    L = nfilts
    myu = np.zeros(shape=(T.shape[0]))
    myu[0] = 0.0001

    # compute the power estimate
    for m in range(1, T.shape[0]):
        myu[m] = lam_myu * myu[m - 1] + ((1 - lam_myu) / L) * sum(
            [T[m, l] for l in range(0, L)]
        )

    # compute normalized power: U
    U = k * T / myu[:, None]
    return U


def asymmetric_noise_suppression_with_temporal_masking(
    Q_tilde: np.ndarray, threshold: float = 0
) -> np.ndarray:
    """
    Apply asymmetric noise suppression with temporal masking according to [Kim]_.

    Args:
        Q_tilde (numpy.ndarray) : array representing the "medium-time power".
        threshold       (float) : threshold for the half wave rectifier.

    Returns:
        (numpy.ndarray) array after asymmetric noise sup and temporal masking.

    Note:
        - 2.1 Apply asymmetric lowpass filtering.
            .. math::
                \\tilde{Q}_{l e}[m, l]=\\mathcal{A} \\mathcal{F}_{0.999,0.5}[\\tilde{Q}[m, l]]

        - 2.2 Substract from the input medium-time power.
            .. math::
                \\tilde{Q}[m, l] - \\tilde{Q}_{l e}[m, l]

        - 2.3 Pass through an ideal half wave linear rectifier.
        - 2.4 Re-apply asymmetric lowpass filtering.
        - 2.5 Apply temporal masking.
        - 2.6 Switch excitation.

        - The previous steps can be summarised in the following graph from [Kim]_.

        .. figure:: ../_static/architectures/pncc_medium_processing.png

            Functional block diagram of the modules for asymmetric noise
            suppression (ANS) and temporal masking in PNCC processing [Kim]_.
    """
    # 2.1. asymmetric low pass filtering (Q_tilde_le : lowpass filter output)
    Q_tilde_le = asymmetric_lowpass_filtering(Q_tilde, 0.999, 0.5)

    # 2.2. Subtract filtering output from the input
    subtracted_lower_envelope = Q_tilde - Q_tilde_le

    # 2.3. half wave rectification (Q_tilde_0 : rectifier output/ rectified signal)
    Q_tilde_0 = np.where(
        subtracted_lower_envelope < threshold,
        np.zeros_like(subtracted_lower_envelope),
        subtracted_lower_envelope,
    )

    # loor level (Q_tilde_f : lower envelope of the rectifier output/ rectified signal)
    Q_tilde_f = asymmetric_lowpass_filtering(Q_tilde_0)

    # 2.5. temporal masking (Q_tilde_tm : temporal masked signal)
    Q_tilde_tm = temporal_masking(Q_tilde_0)

    # 2.6. switch excitation or non-excitation
    c = 2
    R_tilde = np.where(Q_tilde >= c * Q_tilde_le, Q_tilde_tm, Q_tilde_f)
    return R_tilde


def pncc(
    sig: np.ndarray,
    fs: int = 16000,
    num_ceps: int = 13,
    pre_emph: bool = True,
    pre_emph_coeff: float = 0.97,
    power=2,
    window: Optional[SlidingWindow] = None,
    nfilts: int = 24,
    nfft: int = 512,
    low_freq: Optional[float] = 0,
    high_freq: Optional[float] = None,
    scale: ScaleType = "constant",
    dct_type: int = 2,
    lifter: Optional[int] = None,
    normalize: Optional[NormalizationType] = None,
    fbanks: Optional[np.ndarray] = None,
    conversion_approach: ErbConversionApproach = "Glasberg",
) -> np.ndarray:
    """
    Compute the Power-Normalized Cepstral Coefficients (PNCCs) from an audio signal,
    based on [Kim]_ [Nakamura]_ .

    Args:
        sig       (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        num_ceps            (int) : number of cepstra to return.
                                    (Default is 13).
        pre_emph           (bool) : apply pre-emphasis if 1.
                                    (Default is True).
        pre_emph_coeff    (float) : pre-emphasis filter coefﬁcient.
                                    (Default is 0.97).
        power               (int) : power value to use .
                                    (Default is 2).
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
        lifter              (int) : apply liftering if specified.
                                    (Default is None).
        normalize           (str) : apply normalization if approach specified.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : erb scale conversion approach.
                                    (Default is "Glasberg").

    Returns:
        (numpy.ndarray) : 2d array of PNCC features (num_frames x num_ceps)

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`dct` : can take the following options [1, 2, 3, 4].
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Glasberg"].
          Note that the use of different options than the default can lead to unexpected behavior/issues.

    References:
        .. [Kim] : Kim C. and Stern R. M., "Power-Normalized Cepstral Coefficients (PNCC) for Robust Speech Recognition,"
                   in IEEE/ACM Transactions on Audio, Speech, and Language Processing,
                   vol. 24, no. 7, pp. 1315-1329, July 2016, doi: 10.1109/TASLP.2016.2545928.
        .. [Nakamura] : Nakamura T., An implementation of Power Normalized Cepstral Coefficients: PNCC
                        <https://github.com/supikiti/PNCC>

    Note:
        .. figure:: ../_static/architectures/pnccs.png

           Architecture of power normalized cepstral coefﬁcients extraction algorithm.

    Examples
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.pncc import pncc
            from spafe.utils.preprocessing import SlidingWindow
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../tests/data/test.wav"
            fs, sig = read(fpath)

            # compute pnccs
            pnccs  = pncc(sig,
                          fs=fs,
                          pre_emph=0,
                          pre_emph_coeff=0.97,
                          window=SlidingWindow(0.03, 0.015, "hamming"),
                          nfilts=128,
                          nfft=1024,
                          low_freq=0,
                          high_freq=fs/2,
                          lifter=0.7,
                          normalize="mvn")

            # visualize features
            show_features(pnccs, "Power-Normalized Cepstral Coefficients", "PNCC Index", "Frame Index")
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

    # -> FFT -> |.|
    ## Magnitude of the FFT
    fourrier_transform = np.absolute(np.fft.fft(windows, nfft))
    fourrier_transform = fourrier_transform[:, : int(nfft / 2) + 1]

    ##  -> |.|^2 (Power Spectrum)
    abs_fft_values = (1.0 / nfft) * (fourrier_transform**power)

    # -> x filter bank
    P = np.dot(a=abs_fft_values, b=fbanks.T)

    # medium_time_processing
    ## 1. medium time power caculations (Q_tilde : medium_time_power)
    Q_tilde = medium_time_power_calculation(P)

    ## 2. asymmetric noise suppression with temporal masking
    R_tilde = asymmetric_noise_suppression_with_temporal_masking(Q_tilde)

    ## 3. weight smoothing
    S_tilde = weight_smoothing(R_tilde, Q_tilde, nfilts=nfilts)

    # -> time frequency normalization
    T = P * S_tilde

    # -> mean power normalization
    U = mean_power_normalization(T, nfilts=nfilts)

    # -> power law non-linearity
    V = U ** (1 / 15)

    # DCT(.)
    pnccs = dct(x=V, type=dct_type, axis=1, norm="ortho")[:, :num_ceps]

    # liftering
    if lifter:
        pnccs = lifter_ceps(pnccs, lifter)

    # normalization
    if normalize:
        pnccs = normalize_ceps(pnccs, normalize)

    return pnccs
