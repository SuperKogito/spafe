"""

- Description : Linear Prediction Components and Cepstral Coefﬁcients (LPCs and LPCCs) extraction algorithm implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Optional

import numpy as np
from scipy import linalg

from ..utils.cepstral import normalize_ceps, lifter_ceps, NormalizationType
from ..utils.preprocessing import (
    pre_emphasis,
    framing,
    windowing,
    zero_handling,
    SlidingWindow,
)


def __lpc_helper(frame, order):
    """
    Computes for each given sequence the LPC ( Linear predictive components ) as
    described in . Further references are [Draconi]_ and [Cournapeau] and [Menares]_.

    Args:
        sig    (numpy.ndarray) : input mono audio signal (Nx1).
        order            (int) : Size of the cepstral components/ model order. If None is given,
                                 we use len(seq) as default, otherwise order+1.
                                 (Default is 13).

    Returns:
        - (numpy.ndarray) : linear prediction coefficents (lpc coefficents: a).
        - (numpy.ndarray) : the error term is the square root of the squared prediction error (e**2).

    Note:
        The premis of linear predictive analysis is that the nth sample can be estimated by a
        linear combination of the previous p samples:

        .. math::
            xp[n] = -a[1] * x[n-1] - ... -a[k] * x[n-k] ... - a[p] * x[n-p] = - \\sum_{k=1}^{p+1} a_{k} . x[n-k]

        where xp is the predicted signal. a_{1},.., a_{p} are known as the predictor
        coefficents and p is called the model order and n is the sample index.
        Based on the previous equation, we can estimate the prediction error as follows [Ucl-brain]_:

        .. math::
            e[n] = x[n] - xp[n] \\implies  x[n] = e[n] - \\sum_{k=1}^{p+1} a_{k} . x[n-k]

        The unknown here are the LP coefficients a, hence we need to minimize e to find those.
        We can further rewrite the previous equations for all samples [Collomb]_:

        .. math::
            E = \\sum_{i=1}^{N} (x[i] - (-\\sum_{k=1}^{p+1} a_{k} . x[i-k])) \\text{for x\\in[1,p]}


        All the previous steps can be presented in a matrix, which is a toeplitz matrix: R.A = 0
                           _          _
            -r[1] = r[0]   r[1]   ... r[p-1]    a[1]
             :      :      :          :         :
             :      :      :          _      *  :
            -r[p] = r[p-1] r[p-2] ... r[0]      a[p]

        To solve this, one can use the Levinson-Durbin, which is a well-known
        algorithm to solve the Hermitian toeplitz with respect to a. Using the
        special symmetry in the matrix, the inversion can be done in O(p^2)
        instead of O(p^3).

    References:
        .. [Darconis] : Draconi, Replacing Levinson implementation in scikits.talkbox,
                        Stackoverflow, https://stackoverflow.com/a/43457190/6939324
        .. [Cournapeau] : David Cournapeau D. talkbox, https://github.com/cournape/talkbox
        .. [Menares] : Menares E. F. M., ML-experiments, https://github.com/erickfmm/ML-experiments
        .. [Collomb] : Collomb C. Linear Prediction and Levinson-Durbin Algorithm, 03.02.2009,
                       <https://www.academia.edu/8479430/Linear_Prediction_and_Levinson-Durbin_Algorithm_Contents>
        .. [Ucl-brain] : Ucl psychology and language sciences, Faculty of brain Sciences, Unit 8 linear prediction
                         <https://www.phon.ucl.ac.uk/courses/spsci/dsp/lpc.html>
    """
    p = order + 1
    r = np.zeros(p, frame.dtype)

    # Number of non zero values in autocorrelation one needs for p LPC coefficients
    nx = np.min([p, frame.size])
    auto_corr = np.correlate(frame, frame, "full")
    r[:nx] = auto_corr[frame.size - 1 : frame.size + order]

    phi = np.dot(linalg.inv(linalg.toeplitz(r[:-1])), -r[1:])
    a = np.concatenate(([1.0], phi))
    e = auto_corr[0] + sum(ac_k * a_k for ac_k, a_k in zip(auto_corr[1:], a))
    return a, np.sqrt(e**2)


def lpc(
    sig,
    fs: int = 16000,
    order=13,
    pre_emph: bool = True,
    pre_emph_coeff: float = 0.97,
    window: Optional[SlidingWindow] = None,
):
    """
    Compute the Linear prediction coefficents (LPC) from an audio signal.

    Args:
        sig    (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 (Default is 16000).
        order            (int) : order of the LP model and number of cepstral components.
                                 (Default is 13).
        pre_emph        (bool) : apply pre-emphasis if 1.
                                 (Default is 1).
        pre_emph_coeff (float) : pre-emphasis filter coefficient.
                                 (Default is 0.97).
        window (SlidingWindow) : sliding window object.
                                 (Default is None).

    Returns:
        (tuple) :
            - (numpy.ndarray) : 2d array of LPC features (num_frames x num_ceps).
            - (numpy.ndarray) : The error term is the sqare root of the squared prediction error.

    Note:
        .. figure:: ../_static/architectures/lpcs.png

           Architecture of linear prediction components extraction algorithm.

    Examples
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.lpc import lpc
            from spafe.utils.preprocessing import SlidingWindow
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../tests/data/test.wav"
            fs, sig = read(fpath)

            # compute lpcs
            lpcs, _ = lpc(sig,
                          fs=fs,
                          pre_emph=0,
                          pre_emph_coeff=0.97,
                          window=SlidingWindow(0.030, 0.015, "hamming"))

            # visualize features
            show_features(lpcs, "Linear prediction coefficents", "LPCs Index", "Frame Index")
    """
    order = order - 1
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

    a_mat = np.zeros((len(windows), order + 1))
    e_vec = np.zeros((len(windows), 1))

    for i, windowed_frame in enumerate(frames):
        a, e = __lpc_helper(windowed_frame, order)
        a_mat[i, :] = a
        e_vec[i] = e

    return np.array(a_mat), np.sqrt(e_vec)


def lpc2lpcc(a, e, nceps):
    """
    Convert linear prediction coefficents (LPC) to linear prediction cepstral coefﬁcients (LPCC)
    as described in [Rao]_ and [Makhoul]_.

    Args:
        a (numpy.ndarray) : linear prediction coefficents.
        order       (int) : linear prediction model order.
        nceps       (int) : number of cepstral coefficients.

    Returns:
        (numpy.ndarray) : linear prediction cepstrum coefficents (LPCC).

    Note:
        .. math::

            C_{m}=\\left\\{\\begin{array}{l}
            log_{e}(p), & \\text{if } m = 0 \\\\
            a_{m} + \\sum_{k=1}^{m-1} \\frac{k}{m} C_{m} a_{m-k} , & \\text{if } 1 < m < p \\\\
            \\sum_{k=m-p}^{m-1} \\frac{k}{m} C_{m} a_{m-k} , & \\text{if } m > p \\end{array}\\right.

    References:
        .. [Makhoul] : Makhoul, J. (1975). Linear prediction: A tutorial review.
                       Proceedings of the IEEE, 63(4), 561–580. doi:10.1109/proc.1975.9792
        .. [Rao] : Rao, K. S., Reddy, V. R., & Maity, S. (2015). 
                   Language Identification Using Spectral and Prosodic Features. 
                   SpringerBriefs in Electrical and Computer Engineering. doi:10.1007/978-3-319-17163-0
    """
    p = len(a)
    c = [0 for i in range(nceps)]

    c[0] = np.log(zero_handling(e))
    c[1:p] = [
        a[m] + sum([(k / m) * c[k] * a[m - k] for k in range(1, m)])
        for m in range(1, p)
    ]

    if nceps > p:
        c[p:nceps] = [
            sum([(k / m) * c[k] * a[m - k] for k in range(m - p, m)])
            for m in range(p, nceps)
        ]

    return c


def lpcc(
    sig: np.ndarray,
    fs: int = 16000,
    order=13,
    pre_emph: bool = True,
    pre_emph_coeff: float = 0.97,
    window: Optional[SlidingWindow] = None,
    lifter: Optional[int] = None,
    normalize: Optional[NormalizationType] = None,
) -> np.ndarray:
    """
    Computes the linear predictive cepstral components / coefficents from an
    audio signal.

    Args:
        sig    (numpy.ndarray) : input mono audio signal (Nx1).
        fs               (int) : the sampling frequency of the signal.
                                 (Default is 16000).
        order            (int) : order of the LP model and number of cepstral components.
                                 (Default is 13).
        pre_emph        (bool) : apply pre-emphasis if 1.
                                 (Default is 1).
        pre_emph_coeff (float) : pre-emphasis filter coefficient.
                                 (Default is 0.97).
        window (SlidingWindow) : sliding window object.
                                 (Default is None).
        lifter           (int) : apply liftering if specified.
                                 (Default is None).
        normalize        (str) : apply normalization if provided.
                                 (Default is None).

    Returns:
        (numpy.ndarray) : 2d array of LPCC features (num_frames x num_ceps)

    Tip:
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].

    Note:
        Returned values are in the frequency domain

        .. figure:: ../_static/architectures/lpccs.png

           Architecture of linear prediction cepstral coefﬁcients extraction algorithm.

    Examples
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.lpc import lpcc
            from spafe.utils.preprocessing import SlidingWindow
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../tests/data/test.wav"
            fs, sig = read(fpath)

            # compute lpccs
            lpccs = lpcc(sig,
                         fs=fs,
                         pre_emph=0,
                         pre_emph_coeff=0.97,
                         window=SlidingWindow(0.03, 0.015, "hamming"))

            # visualize features
            show_features(lpccs, "Linear Prediction Cepstral Coefﬁcients", "LPCCs Index","Frame Index")
    """
    order = order - 1
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

    # compute lpccs
    lpccs = np.zeros((len(windows), order + 1))
    for i, windowed_frame in enumerate(frames):
        a, e = __lpc_helper(windowed_frame, order)
        lpcc_coeffs = lpc2lpcc(a, e, order + 1)
        lpccs[i, :] = np.array(lpcc_coeffs)

    # liftering
    if lifter:
        lpccs = lifter_ceps(lpccs, lifter)

    # normalization
    if normalize:
        lpccs = normalize_ceps(lpccs, normalize)

    return lpccs
