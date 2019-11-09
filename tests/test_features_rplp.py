import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.rplp import rplp, plp
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


@pytest.mark.test_id(207)
@pytest.mark.parametrize('num_ceps', [13, 19])
def test_rplp(sig, fs, num_ceps):
    """
    test RPLP features module for the following:
        - check that the returned number of cepstrums is correct.
    """
    # compute plps
    plps = plp(sig, fs, num_ceps)

    # assert number of returned cepstrum coefficients
    assert plps.shape[1] == num_ceps

    if DEBUG_MODE:
        vis.visualize_features(plps, 'PLP Coefficient Index', 'Frame Index')

    # compute bfccs
    rplps = rplp(sig, fs, num_ceps)

    # assert number of returned cepstrum coefficients
    assert rplps.shape[1] == num_ceps

    if DEBUG_MODE:
        vis.visualize_features(rplps, 'RPLP Coefficient Index', 'Frame Index')
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

    # run tests
    test_rplp(sig=sig, fs=fs, num_ceps=num_ceps)
