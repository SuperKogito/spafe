import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.lpc import lpc, lpcc
from spafe.utils.exceptions import ParameterError
from spafe.utils.cepstral import cms, cmvn, lifter_ceps
from spafe.utils.spectral import stft, display_stft

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


@pytest.mark.parametrize('num_ceps', [13, 19, 26])
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


@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('lifter', [0, 5])
@pytest.mark.parametrize('normalize', [False, True])
def test_lpcc(sig, fs, num_ceps, lifter, normalize):
    """
    test LPCC features module for the following:
        - check that the returned number of cepstrums is correct.
        - check normalization.
        - check liftering.
    """
    lpccs = lpcc(sig=sig, fs=fs, lifter=lifter, normalize=normalize)
    # assert number of returned cepstrum coefficients
    assert lpccs.shape[1] == num_ceps

    # check normalize
    if normalize:
        np.testing.assert_almost_equal(
            lpccs,
            cmvn(
                cms(
                    lpcc(sig=sig,
                         fs=fs,
                         num_ceps=num_ceps,
                         lifter=lifter,
                         normalize=False))), 3)
    else:
        # check lifter
        if lifter > 0:
            np.testing.assert_almost_equal(
                lpccs,
                lifter_ceps(
                    lpcc(sig=sig,
                         fs=fs,
                         num_ceps=num_ceps,
                         lifter=False,
                         normalize=normalize), lifter), 3)

    if DEBUG_MODE:
        vis.visualize_features(lpcs, 'LPC Index', 'Frame Index')
    if DEBUG_MODE:
        vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')
    assert True


if __name__ == "__main__":
    # read wave file  and plot spectogram
    fs, sig = get_data('../test21.wav')
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
