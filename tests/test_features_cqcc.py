import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.cqcc import cqcc
from spafe.utils.exceptions import ParameterError
from spafe.utils.cepstral import normalize_ceps, lifter_ceps


@pytest.mark.test_id(202)
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
@pytest.mark.parametrize("num_ceps", [13, 28])
@pytest.mark.parametrize("pre_emph", [False, True])
@pytest.mark.parametrize("nfft", [1024])
@pytest.mark.parametrize("low_freq", [50])
@pytest.mark.parametrize("high_freq", [4000])
@pytest.mark.parametrize("dct_type", [1, 2, 4])
@pytest.mark.parametrize("lifter", [None, 0.7, -7])
@pytest.mark.parametrize("normalize", [None, "mvn", "ms"])
def test_cqcc(
    sig,
    fs,
    num_ceps,
    pre_emph,
    nfft,
    low_freq,
    high_freq,
    dct_type,
    lifter,
    normalize,
):
    """
    test cqcc features module for the following:
        - check if ParameterErrors are raised for:
                - nfilts < num_ceps
                - negative low_freq value
                - high_freq > fs / 2
        - check that the returned number of cepstrums is correct.
        - check normalization.
        - check liftering.
    """

    # check error for number of filters is smaller than number of cepstrums
    with pytest.raises(ParameterError):
        cqccs = cqcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            nfft=nfft,
            low_freq=low_freq,
            high_freq=fs,
        )

    # compute features
    cqccs = cqcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        pre_emph=pre_emph,
        nfft=nfft,
        low_freq=low_freq,
        high_freq=high_freq,
        dct_type=dct_type,
        lifter=lifter,
        normalize=normalize,
    )

    # assert number of returned cepstrum coefficients
    if not cqccs.shape[1] == num_ceps:
        raise AssertionError

    # check normalize
    if normalize:
        np.testing.assert_array_almost_equal(
            cqccs,
            normalize_ceps(
                cqcc(
                    sig=sig,
                    fs=fs,
                    num_ceps=num_ceps,
                    pre_emph=pre_emph,
                    nfft=nfft,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    dct_type=dct_type,
                    lifter=lifter,
                    normalize=None,
                ),
                normalize,
            ),
            3,
        )
    else:
        # check lifter
        if lifter:
            np.testing.assert_array_almost_equal(
                cqccs,
                lifter_ceps(
                    cqcc(
                        sig=sig,
                        fs=fs,
                        num_ceps=num_ceps,
                        pre_emph=pre_emph,
                        nfft=nfft,
                        low_freq=low_freq,
                        high_freq=high_freq,
                        dct_type=dct_type,
                        lifter=None,
                        normalize=normalize,
                    ),
                    lifter,
                ),
                3,
            )
