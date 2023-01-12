"""

- Description : Power-Normalized Cepstral Coefficients (PNCCs) extraction algorithm implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy.signal import lfilter
from typing_extensions import Literal

NormalizationType = Literal["mvn", "ms", "vn", "mn"]


def normalize_ceps(
    x: np.ndarray, normalization_type: NormalizationType = "mvn"
) -> np.ndarray:
    """
    Apply normalization to array.

    Args:
        x           (numpy.ndarray) : array of information.
        normalizaation_type (str) : type of normalization to apply:

    Returns:
        (numpy.ndarray) normalized array.

    Note:
        possible options for normalization_type are:

        * "mvn" : Mean Variance Normalisation.
            .. math::
                x^{\\prime}=\\frac{x-\\operatorname{average}(x)}{\\operatorname{std}(x)}

        * "ms" : Mean Substraction: Centering.
            .. math::
                x^{\\prime} = x - \\operatorname{average}(x)

        * "vn" : Variance Normalisation: Standardization.
            .. math::
                x^{\\prime} = \\frac{x}{\\operatorname{std}(x)}

        * "mn" : Mean normalization.
            .. math::
                x^{\\prime} = \\frac{x - \\operatorname{average}(x)}{ \\max(x) - \\min(x)}

        where :math:`\\operatorname{std}(x)` is the standard deviation.
    """
    return {
        "mvn": (x - np.mean(x, axis=0)) / np.std(x),
        "ms": x - np.mean(x, axis=0),
        "vn": x / np.std(x),
        "mn": (x - np.mean(x)) / (np.max(x) - np.min(x)),
    }[normalization_type]


def lifter_ceps(ceps: np.ndarray, lift: int = 3) -> np.ndarray:
    """
    Apply a cepstral lifter the the matrix of cepstra. This has the effect of
    increasing the magnitude of the high frequency DCT coeffs. the liftering is
    implemented as in [Ellis-plp]_.

    Args:
        ceps (numpy.ndarray) : the matrix of mel-cepstra, will be numframes * numcep in size.
        lift           (int) : the liftering coefficient to use. (Default is 3).

    Returns:
        (numpy.ndarray) liftered cepstra.

    Note:
        - The liftering is applied to matrix of cepstra (one per column).
        - If the lift is positive (Use values smaller than 10 for meaningful results), then
          the liftering uses the exponent. However, if the lift is negative (Use integers), then
          the sine curve liftering is used.

    References:
        .. [Ellis-plp] : Ellis, D. P. W., 2005, PLP and RASTA and MFCC, and inversion in Matlab,
                     <http://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/>
    """
    if lift == 0 or lift > 10:
        return ceps

    elif lift > 0:
        lift_vec = np.array([1] + [i**lift for i in range(1, ceps.shape[1])])
        lift_mat = np.diag(lift_vec)
        return np.dot(ceps, lift_mat)

    else:
        lift = int(-1 * lift)
        lift_vec = 1 + (lift / 2.0) * np.sin(
            np.pi * np.arange(1, 1 + ceps.shape[1]) / lift
        )
        return ceps * lift_vec


def deltas(x: np.ndarray, w: int = 9) -> np.ndarray:
    """
    Calculate the deltas (derivatives) of an input sequence with a W-points
    window (W odd, default 9) using a simple linear slope. This mirrors the delta
    calculation performed in feacalc etc. Each row of X is filtered separately.

    Args:
        x (numpy.ndarray) : input sequence
        w           (int) : window size to use in the derivatives calculation.
                            (Default is 9).

    Returns:
        (numpy.ndarray) 2d-arrays containing the derivatives values.
    """
    _, cols = x.shape
    hlen = np.floor(w / 2)
    win = np.arange(hlen, -(hlen + 1), -1, dtype="float32")

    xx = np.append(
        np.append(np.tile(x[:, 0], (int(hlen), 1)).T, x, axis=1),
        np.tile(x[:, cols - 1], (int(hlen), 1)).T,
        axis=1,
    )

    deltas = lfilter(win, 1, xx, axis=1)[:, int(2 * hlen) : int(2 * hlen + cols)]
    return deltas
