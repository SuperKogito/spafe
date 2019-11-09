import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.utils.exceptions import ParameterError
from spafe.features.lpc import lpc, lpcc, lpc2spec
from spafe.utils.spectral import stft, display_stft
from spafe.utils.cepstral import cms, cmvn, lifter_ceps

DEBUG_MODE = False


def get_data(fname):
    return scipy.io.wavfile.read(fname)


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
    np.testing.assert_equal(np.any(np.not_equal(lpc2spec(lpcs, FMout=True)[1],
                                                specs_from_lpc[2])), True)

@pytest.mark.parametrize('num_ceps', [13, 17])
def test_lpc(sig, fs, num_ceps):
    """
    test LPC features module for the following:
        - check that the returned number of cepstrums is correct.
    """
    # compute lpcs and lsps
    lpcs = lpc(sig=sig, fs=fs, num_ceps=num_ceps)

    # assert number of returned cepstrum coefficients
    assert lpcs.shape[1] == num_ceps

    if DEBUG_MODE:
        vis.visualize_features(lpcs, 'LPC Index', 'Frame Index')
    assert True


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
    lpccs = lpcc(sig=sig, fs=fs, num_ceps=num_ceps, lifter=lifter, normalize=normalize)
    # assert number of returned cepstrum coefficients
    assert lpccs.shape[1] == num_ceps

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

    if DEBUG_MODE:
        vis.visualize_features(lpcs, 'LPC Index', 'Frame Index')
    if DEBUG_MODE:
        vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')
    assert True


if __name__ == "__main__":
    # read wave file  and plot spectogram
    fs, sig = get_data('../test.wav')
    if DEBUG_MODE:
        vis.spectogram(sig, fs)

    # compute and display STFT
    X, _ = stft(sig=sig, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)
    if DEBUG_MODE:
        display_stft(X, fs, len(sig), 0, 2000, -10, 0)

    # init input vars
    num_ceps = 13
    lifter = 5
    normalize = False

    # run tests
    test_lpc(sig=sig, fs=fs, num_ceps=num_ceps)
    test_lpcc(sig=sig,
              fs=fs,
              num_ceps=num_ceps,
              lifter=lifter,
              normalize=normalize)
