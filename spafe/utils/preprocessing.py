"""

- Description : Preprocessing utils implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
from typing import Tuple

import numpy as np
from dataclasses import dataclass
from typing_extensions import Literal

from .exceptions import ParameterError, ErrorMsgs

WindowType = Literal["hanning", "bartlet", "kaiser", "blackman", "hamming"]


@dataclass
class SlidingWindow:
    """
    Sliding widow class.

    Args:
        win_len (float) : window length in sec.
                          (Default is 0.025).
        win_hop (float) : step between successive windows in sec.
                          (Default is 0.01).
        win_type (float) : window type to apply for the windowing.
                          (Default is "hamming").
    """

    win_len: float = 0.025
    win_hop: float = 0.010
    win_type: WindowType = "hamming"


def zero_handling(x: np.ndarray) -> np.ndarray:
    """
    handle the issue with zero values if they are exposed to become an argument
    for any log function.

    Args:
        x (numpy.ndarray): input vector.

    Returns:
        (numpy.ndarray) : vector with zeros substituted with epsilon values.
    """
    return np.where(x == 0, np.finfo(float).eps, x)


def pre_emphasis(sig: np.ndarray, pre_emph_coeff: float = 0.97) -> np.ndarray:
    """
    perform preemphasis on the input signal.

    Args:
        sig (numpy.ndarray) : input signal.
        coeff       (float) : preemphasis coefficient. 0 is no filter.
                              (Default is 0.97).

    Returns:
        (numpy.ndarray) : pre-empahsised signal.

    Note:
        .. math::

            y[t] = x[t] - \\alpha \\times x[t-1]
    """
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])


def stride_trick(a: np.ndarray, stride_length: int, stride_step: int) -> np.ndarray:
    """
    apply framing using the stride trick from numpy.

    Args:
        a   (numpy.ndarray) : signal array.
        stride_length (int) : length of the stride.
        stride_step   (int) : stride step.

    Returns:
        (numpy.ndarray) : blocked/framed array.

    Note:
        You can refer to numpy documentation of this stride trick here:
        https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    """
    a = np.array(a)
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, stride_length), strides=(stride_step * n, n)
    )


def framing(
    sig: np.ndarray, fs: int = 16000, win_len: float = 0.025, win_hop: float = 0.01
) -> Tuple[np.ndarray, int]:
    """
    transform a signal into a series of overlapping frames (= Frame blocking)
    as described in [Malek-framing-blog]_.

    Args:
        sig (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs            (int) : the sampling frequency of the signal we are working with.
                              (Default is 16000).
        win_len     (float) : window length in sec.
                              (Default is 0.025).
        win_hop     (float) : step between successive windows in sec.
                              (Default is 0.01).

    Returns:
        (tuple) :
            - (numpy.ndarray) : array of frames.
            - (int)           : frame length.

    Note:
        Uses the stride trick to accelerate the processing.

    References:
        .. [Malek-framing-blog] : Malek A., Signal framing, 25.01.2022,
                                  https://superkogito.github.io/blog/2020/01/25/signal_framing.html
    """
    # run checks and assertions
    if win_len < win_hop:
        raise ParameterError(ErrorMsgs["win_len_win_hop_comparison"])

    # compute frame length and frame step (convert from seconds to samples)
    frame_length = int(win_len * fs)
    frame_step = int(win_hop * fs)

    # make sure to use integers as indices
    frames = stride_trick(sig, frame_length, frame_step)

    if len(frames[-1]) < frame_length:
        frames[-1] = np.append(
            frames[-1], np.array([0] * (frame_length - len(frames[0])))
        )

    return frames, frame_length


def windowing(
    frames: np.ndarray, frame_len: int, win_type: WindowType = "hamming"
) -> np.ndarray:
    """
    generate and apply a window function to avoid spectral leakage [Malek-windowing-blog]_.

    Args:
        frames  (numpy.ndarray) : array including the overlapping frames.
        frame_len         (int) : frame length.
        win_type          (str) : type of window to use.
                                  (Default is "hamming").

    Returns:
        (numpy.ndarray) : windowed frames.

    References:
        .. [Malek-windowing-blog] : Malek, A. Specctral leakage, 2022.03.13,
                                   https://superkogito.github.io/blog/2020/03/13/spectral_leakage_and_windowing.html

    """
    return {
        "hanning": np.hanning(frame_len) * frames,
        "bartlet": np.bartlett(frame_len) * frames,
        "kaiser": np.kaiser(frame_len, beta=14) * frames,
        "blackman": np.blackman(frame_len) * frames,
        "hamming": np.hamming(frame_len) * frames,
    }[win_type]
