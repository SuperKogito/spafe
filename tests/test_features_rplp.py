import pytest
import numpy as np
from spafe.features.rplp import rplp, plp
from spafe.fbanks.bark_fbanks import bark_filter_banks
from spafe.utils.cepstral import normalize_ceps, lifter_ceps


@pytest.mark.test_id(212)
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
@pytest.mark.parametrize("order", [13, 22])
@pytest.mark.parametrize("pre_emph", [False, True])
@pytest.mark.parametrize("lifter", [None, 0.7, -7])
@pytest.mark.parametrize("normalize", [None, "mvn", "ms"])
def test_rplp(
    sig,
    fs,
    order,
    pre_emph,
    lifter,
    normalize,
):
    """
    test RPLP features module for the following:
        - check that the returned number of cepstrums is correct.
    """
    # compute plps
    plps = plp(
        sig=sig,
        fs=fs,
        order=order,
        pre_emph=pre_emph,
        lifter=lifter,
        normalize=normalize,
    )

    # assert number of returned cepstrum coefficients
    if not plps.shape[1] == order:
        raise AssertionError

    # Test normalize
    if normalize:
        np.testing.assert_array_almost_equal(
            plps,
            normalize_ceps(
                plp(
                    sig=sig,
                    fs=fs,
                    order=order,
                    pre_emph=pre_emph,
                    lifter=lifter,
                    normalize=None,
                ),
                normalize,
            ),
            3,
        )
    else:
        # Test lifter
        if lifter:
            np.testing.assert_array_almost_equal(
                plps,
                lifter_ceps(
                    plp(
                        sig=sig,
                        fs=fs,
                        order=order,
                        pre_emph=pre_emph,
                        lifter=None,
                        normalize=normalize,
                    ),
                    lifter,
                ),
                3,
            )

    # compute bfccs
    rplps = rplp(
        sig=sig,
        fs=fs,
        order=order,
        pre_emph=pre_emph,
        lifter=lifter,
        normalize=normalize,
    )
    # assert number of returned cepstrum coefficients
    if not rplps.shape[1] == order:
        raise AssertionError

    # check predifined fbanks features
    fbanks, _ = bark_filter_banks(fs=fs)

    predifined_fbanks_feats = rplp(
        sig=sig,
        fs=fs,
        order=order,
        pre_emph=pre_emph,
        lifter=lifter,
        normalize=normalize,
        fbanks=fbanks,
    )

    np.testing.assert_array_almost_equal(rplps, predifined_fbanks_feats, 3)

    predifined_fbanks_feats = plp(
        sig=sig,
        fs=fs,
        order=order,
        pre_emph=pre_emph,
        lifter=lifter,
        normalize=normalize,
        fbanks=fbanks,
    )

    np.testing.assert_array_almost_equal(plps, predifined_fbanks_feats, 3)
