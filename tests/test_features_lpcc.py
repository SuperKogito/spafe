import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.lpc import lpc, lpcc, lpc2spec
from spafe.utils.cepstral import cms, cmvn, lifter_ceps


@pytest.fixture
def sig():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def fs():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[0]


@pytest.mark.parametrize('num_ceps', [13, 17])
def test_lpc2spec(sig, fs, num_ceps):
    """
    test LPC features module for the following:
        - check that the returned number of cepstrums is correct.
    """
    # compute lpcs and lsps
    lpcs = lpc(sig=sig, fs=fs, num_ceps=num_ceps)

    # checks for lpc2spec
    specs_from_lpc = lpc2spec(lpcs)
    np.testing.assert_almost_equal(specs_from_lpc[1], specs_from_lpc[2])
    np.testing.assert_equal(
        np.any(np.not_equal(lpc2spec(lpcs, FMout=True)[1], specs_from_lpc[2])),
        True)


@pytest.mark.parametrize('num_ceps', [13, 17])
def test_lpc(sig, fs, num_ceps):
    """
    test LPC features module for the following:
        - check that the returned number of cepstrums is correct.
    """
    # compute lpcs and lsps
    lpcs = lpc(sig=sig, fs=fs, num_ceps=num_ceps)

    # assert number of returned cepstrum coefficients
    if not lpcs.shape[1] == num_ceps:
        raise AssertionError


@pytest.mark.parametrize('num_ceps', [13, 17])
@pytest.mark.parametrize('lifter', [0])
@pytest.mark.parametrize('normalize', [False])
def test_lpcc(sig, fs, num_ceps, lifter, normalize):
    """
    test LPCC features module for the following:
        - check that the returned number of cepstrums is correct.
        - check normalization.
        - check liftering.
    """
    lpccs = lpcc(sig=sig,
                 fs=fs,
                 num_ceps=num_ceps,
                 lifter=lifter,
                 normalize=normalize)
    # assert number of returned cepstrum coefficients
    if not lpccs.shape[1] == num_ceps:
        raise AssertionError

    # TO FIX: normalize
    if normalize:
        np.testing.assert_almost_equal(
            lpccs,
            cmvn(
                cms(
                    lpcc(sig=sig,
                         fs=fs,
                         num_ceps=num_ceps,
                         lifter=lifter,
                         normalize=False))), 0)
    else:
        # TO FIX: lifter
        if lifter > 0:
            np.testing.assert_almost_equal(
                lpccs,
                lifter_ceps(
                    lpcc(sig=sig,
                         fs=fs,
                         num_ceps=num_ceps,
                         lifter=False,
                         normalize=normalize), lifter), 0)
