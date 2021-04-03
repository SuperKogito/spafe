import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.msrcc import msrcc
from spafe.utils.exceptions import ParameterError
from spafe.utils.cepstral import cms, cmvn, lifter_ceps


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
@pytest.mark.parametrize('pre_emph', [False, True])
@pytest.mark.parametrize('nfilts', [32, 48])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('low_freq', [0, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('dct_type', [1, 2, 4])
@pytest.mark.parametrize('use_energy', [False, True])
@pytest.mark.parametrize('lifter', [0, 5])
@pytest.mark.parametrize('normalize', [False, True])
def test_msrcc(sig, fs, num_ceps, pre_emph, nfilts, nfft, low_freq, high_freq, dct_type,
               use_energy, lifter, normalize):
    """
    test MSRCC features module for the following:
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
        msrccs = msrcc(sig=sig,
                       fs=fs,
                       num_ceps=num_ceps,
                       nfilts=num_ceps - 1,
                       nfft=nfft,
                       low_freq=low_freq,
                       high_freq=high_freq)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        msrccs = msrcc(sig=sig,
                       fs=fs,
                       num_ceps=num_ceps,
                       nfilts=nfilts,
                       nfft=nfft,
                       low_freq=-5,
                       high_freq=high_freq)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        msrccs = msrcc(sig=sig,
                       fs=fs,
                       num_ceps=num_ceps,
                       nfilts=nfilts,
                       nfft=nfft,
                       low_freq=low_freq,
                       high_freq=16000)

    # compute features
    msrccs = msrcc(sig=sig,
                   fs=fs,
                   num_ceps=num_ceps,
                   pre_emph=pre_emph,
                   nfilts=nfilts,
                   nfft=nfft,
                   low_freq=low_freq,
                   high_freq=high_freq,
                   dct_type=dct_type,
                   use_energy=use_energy,
                   lifter=lifter,
                   normalize=normalize)

    # assert number of returned cepstrum coefficients
    if not msrccs.shape[1] == num_ceps:
        raise AssertionError

    # check use energy
    if use_energy:
        msrccs_energy = msrccs[:, 0]
        xfccs_energy = msrcc(sig=sig,
                             fs=fs,
                             num_ceps=num_ceps,
                             pre_emph=pre_emph,
                             nfilts=nfilts,
                             nfft=nfft,
                             low_freq=low_freq,
                             high_freq=high_freq,
                             dct_type=dct_type,
                             use_energy=use_energy,
                             lifter=lifter,
                             normalize=normalize)[:, 0]

        np.testing.assert_almost_equal(msrccs_energy, xfccs_energy, 3)

    # check normalize
    if normalize:
        np.testing.assert_almost_equal(
            msrccs,
            cmvn(
                cms(
                    msrcc(sig=sig,
                          fs=fs,
                          num_ceps=num_ceps,
                          pre_emph=pre_emph,
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
                msrccs,
                lifter_ceps(
                    msrcc(sig=sig,
                          fs=fs,
                          num_ceps=num_ceps,
                          pre_emph=pre_emph,
                          nfilts=nfilts,
                          nfft=nfft,
                          low_freq=low_freq,
                          high_freq=high_freq,
                          dct_type=dct_type,
                          use_energy=use_energy,
                          lifter=False,
                          normalize=normalize), lifter), 3)
