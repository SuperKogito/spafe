import scipy
import numpy as np


def cmn(x):
    """
    Mean normalization.

    Args:
        x (array) : input data.

    Returns:
        array with the mean normalized data.
    """
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


def cms(x):
    """
    Mean Substraction: Centering

    Args:
        x (array) : input data.

    Returns:
        array with the centered data.
    """
    return x - np.mean(x, axis=0)


def cvn(x):
    """
    Variance Normalisation: Standardization

    Args:
        x (array) : input data.

    Returns:
        array with the variance normalized data.
    """
    return x / np.std(x)


def cmvn(x):
    """
    Mean Variance Normalisation

    Args:
        x (array) : input data.

    Returns:
        array with the mean and variance normalized data.
    """
    return cvn(cms(x))


def _helper_idx(i, start, stop, step, dct_type):
    """
    Helper function to compute the cosine vector used in the dct.

    Args:
        i      (float) : angle index.
        start    (int) : start of the angles index interval.
                         Default start value is 0.
        stop     (int) : End of the angles index interval. The interval does not include this value.
        step     (int) : Spacing between values.
                         Default is 1.
        dct_type (int) : the discrete cosine transform type.
                         Default is 2.

    Returns:
        array containing the cosine values.
    """
    r = np.arange(start, stop, step)

    if dct_type not in [2, 3]:
        r = r.T
    return np.cos(np.pi * i * r / (stop + start - 1))


def _helper_mat(K, ncep, start, stop, step, dct_type):
    """
    Helper function to compute the discrete cosine tranform and the inverse
    discrete cosine transform coefficients.

    Args:
        K      (float) : multiplication coefficient.
        ncep     (int) : number of cepstrals.
                         Default is 9.
        start    (int) : start of the angles index interval.
                         Default start value is 0.
        stop     (int) : End of the angles index interval. The interval does not include this value.
        step     (int) : Spacing between values.
                         Default is 1.
        dct_type (int) : the discrete cosine transform type.
                         Default is 2.

    Returns:
        2d-array containing the dct coefficients.
    """
    mat = [
        K *
        _helper_idx(i=i, start=start, stop=stop, step=step, dct_type=dct_type)
        for i in range(ncep)
    ]
    return np.array(mat)


def cep2spec(cep, ncep, nfreq, dct_type=2):
    """
    Reverse the cepstrum to recover a spectrum.

    Args:
        cep    (array) : cepstral data to convert to spectral data.
        ncep     (int) : number of cepstrals.
        nfreq    (int) : number of points to reconstruct in spectrum.
        dct_type (int) : the discrete cosine transform type.
                         Default is 2.

    Returns
         2d-array spec, spectrum matrix
         2d-array idctm, the IDCT matrix that spec was multiplied by to give cep.
    """
    _, ncol = cep.shape

    dctm = np.zeros((ncep, nfreq))
    idctm = np.zeros((nfreq, ncep))

    if dct_type == 2 or dct_type == 3:
        dctm[0:ncep, :] = _helper_mat(np.sqrt(2 / nfreq),
                                      ncep,
                                      start=1,
                                      stop=2 * nfreq,
                                      step=2,
                                      dct_type=dct_type)

        if dct_type == 2:
            dctm[0, :] = dctm[0, :] / np.sqrt(2)
        else:
            dctm[0, :] = dctm[0, :] / 2

        idctm = dctm.T

    elif dct_type == 4:
        idctm[:, 0:ncep] = _helper_mat(2,
                                       ncep,
                                       start=1,
                                       stop=nfreq + 1,
                                       step=None,
                                       dct_type=dct_type).T
        idctm[:, 0:ncep] = idctm[:, 0:ncep] / 2

    else:
        idctm[:, 0:ncep] = _helper_mat(2,
                                       ncep,
                                       start=0,
                                       stop=nfreq,
                                       step=None,
                                       dct_type=dct_type).T
        idctm[:, [0, -1]] = idctm[:, [0, -1]] / 2

    spec = np.exp(np.matmul(idctm, cep))
    return spec, idctm


def deltas(x, w=9):
    """
    Calculate the deltas (derivatives) of an input sequence with a W-points
    window (W odd, default 9) using a simple linear slope. This mirrors the delta
    calculation performed in feacalc etc. Each row of X is filtered separately.

    Args:
        x (array) : input sequence
        w   (int) : window size to use in the derivatives calculation.
                    Default is 9.

    Returns:
        2d-arrays containing the derivatives values.
    """
    _, cols = x.shape
    hlen = np.floor(w / 2)
    win = np.arange(hlen, -(hlen + 1), -1, dtype='float32')

    xx = np.append(np.append(np.tile(x[:, 0], (int(hlen), 1)).T, x, axis=1),
                   np.tile(x[:, cols - 1], (int(hlen), 1)).T,
                   axis=1)
    from scipy.signal import lfilter
    deltas = lfilter(win, 1, xx, axis=1)[:, int(2 * hlen):int(2 * hlen + cols)]
    return deltas


def spec2cep(spec, ncep=9, dct_type=2):
    """
    Calculate cepstra from spectral samples (in columns of spec).

    Args:
        spec   (array) : spectral data to convert to cepstral.
        ncep     (int) : number of cepstrals.
                         Default is 9.
        dct_type (int) : the discrete cosine transform type.
                         Default is 2.

    Returns
         2d-array ncep, cepstral rows
         2d-array dctm, the DCT matrix that spec was multiplied by to give cep.
    """
    nrow, _ = spec.shape[0], spec.shape[1]
    dctm = np.zeros((ncep, nrow))

    if dct_type == 2 or dct_type == 3:
        dctm[:ncep, :] = _helper_mat(np.sqrt(2 / nrow),
                                     ncep,
                                     start=1,
                                     stop=2 * nrow,
                                     step=2,
                                     dct_type=dct_type)

        if dct_type == 2:
            dctm[0, :] = (dctm[0, :] / np.sqrt(2))

    elif dct_type == 4:
        dctm[:ncep, :] = _helper_mat(2,
                                     ncep,
                                     start=1,
                                     stop=nrow + 1,
                                     step=None,
                                     dct_type=dct_type)
        dctm[:ncep, 0] = dctm[:ncep, 0] + 1
        dctm[:ncep, int(nrow -
                        1)] = dctm[:ncep, int(nrow - 1)] * (-1)**np.arange(
                            0, ncep)
        dctm = np.divide(dctm, 2 * (nrow + 1))

    else:
        dctm[:ncep, :] = _helper_mat((nrow - 1)**-1,
                                     ncep,
                                     start=0,
                                     stop=nrow,
                                     step=None,
                                     dct_type=dct_type)

        dctm[:, 0] = dctm[:, 0] / 2
        dctm[:, int(nrow - 1)] = dctm[:, int(nrow - 1)] / 2

    cep = np.matmul(dctm, np.log(spec + 1e-8))

    return cep, dctm


def lifter_ceps(cepstra, L=22):
    """
    Apply a cepstral lifter the the matrix of cepstra. This has the effect of
    increasing the magnitude of the high frequency DCT coeffs.

    Args:
        cepstra (np.array) : the matrix of mel-cepstra, will be numframes * numcep in size.
        L            (int) : the liftering coefficient to use. Default is 22. L <= 0 disables lifter.

    Returns:
        liftered cepstra.
    """
    if L > 0:
        _, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra
