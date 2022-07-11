import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.ngcc import ngcc
from spafe.utils.exceptions import ParameterError
from spafe.utils.cepstral import normalize_ceps, lifter_ceps
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks


@pytest.mark.test_id(210)
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
@pytest.mark.parametrize("num_ceps", [13, 28])
@pytest.mark.parametrize("pre_emph", [False, True])
@pytest.mark.parametrize("nfilts", [128])
@pytest.mark.parametrize("nfft", [1024])
@pytest.mark.parametrize("low_freq", [50])
@pytest.mark.parametrize("high_freq", [2000])
@pytest.mark.parametrize("dct_type", [1, 2, 4])
@pytest.mark.parametrize("use_energy", [False, True])
@pytest.mark.parametrize("lifter", [None, 0.7, -7])
@pytest.mark.parametrize("normalize", [None, "mvn", "ms"])
def test_ngcc(
    sig,
    fs,
    num_ceps,
    pre_emph,
    nfilts,
    nfft,
    low_freq,
    high_freq,
    dct_type,
    use_energy,
    lifter,
    normalize,
):
    """
    test NGCC features module for the following:
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
        ngccs = ngcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            nfilts=num_ceps - 1,
            nfft=nfft,
            low_freq=low_freq,
            high_freq=high_freq,
        )

    # compute features
    ngccs = ngcc(
        sig=sig,
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
        normalize=normalize,
    )

    # assert number of returned cepstrum coefficients
    if not ngccs.shape[1] == num_ceps:
        raise AssertionError

    # check use energy
    if use_energy:
        ngccs_energy = ngccs[:, 0]
        xfccs_energy = ngcc(
            sig=sig,
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
            normalize=normalize,
        )[:, 0]

        np.testing.assert_array_almost_equal(ngccs_energy, xfccs_energy, 3)

    # check normalize
    if normalize:
        np.testing.assert_array_almost_equal(
            ngccs,
            normalize_ceps(
                ngcc(
                    sig=sig,
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
                ngccs,
                lifter_ceps(
                    ngcc(
                        sig=sig,
                        fs=fs,
                        num_ceps=num_ceps,
                        pre_emph=pre_emph,
                        nfilts=nfilts,
                        nfft=nfft,
                        low_freq=low_freq,
                        high_freq=high_freq,
                        dct_type=dct_type,
                        use_energy=use_energy,
                        lifter=None,
                        normalize=normalize,
                    ),
                    lifter,
                ),
                3,
            )

    # check predifined fbanks features
    fbanks, _ = gammatone_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    predifined_fbanks_feats = ngcc(
        sig=sig,
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
        normalize=normalize,
        fbanks=fbanks,
    )

    np.testing.assert_array_almost_equal(ngccs, predifined_fbanks_feats, 3)
