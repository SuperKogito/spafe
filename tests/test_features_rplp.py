import pytest
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.rplp import rplp, plp
from spafe.utils.spectral import stft, display_stft


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
@pytest.mark.parametrize('pre_emph', [False, True])
@pytest.mark.parametrize('modelorder', [0, 13])
def test_rplp(sig, fs, num_ceps, pre_emph, modelorder):
    """
    test RPLP features module for the following:
        - check that the returned number of cepstrums is correct.
    """
    # compute plps
    plps = plp(sig, fs, num_ceps, pre_emph)

    # assert number of returned cepstrum coefficients
    if not plps.shape[1] == num_ceps:
        raise AssertionError

    # compute bfccs
    rplps = rplp(sig, fs, num_ceps, pre_emph, modelorder)

    # assert number of returned cepstrum coefficients
    if not rplps.shape[1] == num_ceps:
        raise AssertionError
