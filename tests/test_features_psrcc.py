import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.psrcc import psrcc
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


@pytest.mark.test_id(202)
@pytest.mark.parametrize('num_ceps', [13, 26])
@pytest.mark.parametrize('nfilts', [32, 48])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('low_freq', [0, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('dct_type', [1, 2, 4])
@pytest.mark.parametrize('use_energy', [False, True])
@pytest.mark.parametrize('lifter', [0, 5])
@pytest.mark.parametrize('normalize', [False, True])
def test_psrcc(sig, fs, num_ceps, nfilts, nfft, low_freq, high_freq, dct_type,
               use_energy, lifter, normalize):
    """
    test PSRCC features module for the following:
        - check if ParameterErrors are raised for:
                - nfilts < num_ceps
                - negative low_freq value
                - high_freq > fs / 2
        - check that the returned number of cepstrums is correct.
        - check the use energy functionality.
        - check normalization.
        - check liftering.
    """

    # check error for number of filters is smaller than number of cepstrums
    with pytest.raises(ParameterError):
        psrccs = psrcc(sig=sig,
                       fs=fs,
                       num_ceps=num_ceps,
                       nfilts=num_ceps - 1,
                       nfft=nfft,
                       low_freq=low_freq,
                       high_freq=high_freq)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        psrccs = psrcc(sig=sig,
                       fs=fs,
                       num_ceps=num_ceps,
                       nfilts=nfilts,
                       nfft=nfft,
                       low_freq=-5,
                       high_freq=high_freq)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        psrccs = psrcc(sig=sig,
                       fs=fs,
                       num_ceps=num_ceps,
                       nfilts=nfilts,
                       nfft=nfft,
                       low_freq=low_freq,
                       high_freq=16000)

    # compute features
    psrccs = psrcc(sig=sig,
                   fs=fs,
                   num_ceps=num_ceps,
                   nfilts=nfilts,
                   nfft=nfft,
                   low_freq=low_freq,
                   high_freq=high_freq,
                   dct_type=dct_type,
                   use_energy=use_energy,
                   lifter=lifter,
                   normalize=normalize)

    # assert number of returned cepstrum coefficients
    assert psrccs.shape[1] == num_ceps

    # check use energy
    if use_energy:
        psrccs_energy = psrccs[:, 0]
        gfccs_energy = psrcc(sig=sig,
                             fs=fs,
                             num_ceps=num_ceps,
                             nfilts=nfilts,
                             nfft=nfft,
                             low_freq=low_freq,
                             high_freq=high_freq,
                             dct_type=dct_type,
                             use_energy=use_energy,
                             lifter=lifter,
                             normalize=normalize)[:, 0]

        np.testing.assert_almost_equal(psrccs_energy, gfccs_energy, 3)

    # check normalize
    if normalize:
        np.testing.assert_almost_equal(
            psrccs,
            cmvn(
                cms(
                    psrcc(sig=sig,
                          fs=fs,
                          num_ceps=num_ceps,
                          nfilts=nfilts,
                          nfft=nfft,
                          low_freq=low_freq,
                          high_freq=high_freq,
                          dct_type=dct_type,
                          use_energy=use_energy,
                          lifter=lifter,
                          normalize=False))), 3)
    else:
        # check lifter
        if lifter > 0:
            np.testing.assert_almost_equal(
                psrccs,
                lifter_ceps(
                    psrcc(sig=sig,
                          fs=fs,
                          num_ceps=num_ceps,
                          nfilts=nfilts,
                          nfft=nfft,
                          low_freq=low_freq,
                          high_freq=high_freq,
                          dct_type=dct_type,
                          use_energy=use_energy,
                          lifter=False,
                          normalize=normalize), lifter), 3)

    if DEBUG_MODE:
        vis.visualize_features(psrccs, 'PSRCC Index', 'Frame Index')
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
    low_freq = 0
    high_freq = 2000
    nfilts = 24
    nfft = 512
    dct_type = 2,
    use_energy = False,
    lifter = 5
    normalize = False

    # run tests for debug mode (with visualization)
    test_psrcc(sig=sig,
               fs=fs,
               num_ceps=num_ceps,
               nfilts=nfilts,
               nfft=nfft,
               low_freq=low_freq,
               high_freq=high_freq,
               dct_type=dct_type,
               use_energy=use_energy,
               lifter=lifter,
               normalize=normalize)
