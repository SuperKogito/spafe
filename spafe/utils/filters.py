"""

- Description : Filter utils implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy import signal
from typing_extensions import Literal

ScaleType = Literal["ascendant", "descendant", "constant"]


def rasta_filter(x: np.ndarray) -> np.ndarray:
    """
    Implementing the RASTA filter as in [Ellis-plp]_.

    Args:
        x (numpy.ndarray) : input signal.

    Returns:
        (numpy.ndarray) : filtered signal.

    Note:
        - default filter is single pole at 0.94
        - rows of x = number of frames
        - cols of x = critical bands
    """
    numer = np.arange(-2, 3)
    numer = (-1 * numer) / np.sum(numer**2)
    denom = np.array([1, -0.94])

    z = signal.lfilter_zi(numer, 1)
    y = np.zeros((x.shape))

    for i in range(x.shape[0]):
        # FIR for initial state response compuation
        y1, z = signal.lfilter(numer, 1, x[i, 0:4], axis=0, zi=z * x[i, 0])
        y1 = y1 * 0

        # IIR
        y2, _ = signal.lfilter(numer, denom, x[i, 4 : x.shape[1]], axis=0, zi=z)
        y[i, :] = np.append(y1, y2)
    return y


def scale_fbank(scale: ScaleType, nfilts: int) -> np.ndarray:
    """
    Generate scaling vector.

    Args:
        scale  (str) : type of scaling.
        nfilts (int) : number of filters.

    Returns:
        (numpy.ndarray) : scaling vector.

    Note:
        .. math::
            ascendant  : \\frac{1}{nfilts} \\times [ 1, ..., i, ..., nfilts]

            descendant : \\frac{1}{nfilts} \\times [ nfilts, ..., i, ..., 1]
    """
    return {
        "ascendant": np.array([i / nfilts for i in range(1, nfilts + 1)]).reshape(
            nfilts, 1
        ),
        "descendant": np.array([i / nfilts for i in range(nfilts, 0, -1)]).reshape(
            nfilts, 1
        ),
        "constant": np.ones(shape=(nfilts, 1)),
    }[scale]
