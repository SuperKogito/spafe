import pytest
import numpy as np
from spafe.features.lpc import lpc, lpcc
from spafe.utils.cepstral import normalize_ceps, lifter_ceps


@pytest.mark.test_id(205)
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
@pytest.mark.parametrize("order", [13, 28])
@pytest.mark.parametrize("pre_emph", [False, True])
def test_lpc(
    sig,
    fs,
    order,
    pre_emph,
):
    """
    test LPC features module for the following:
        - check that the returned number of cepstrums is correct.
    """
    # compute lpcs and lsps
    lpcs, _ = lpc(sig=sig, fs=fs, order=order, pre_emph=pre_emph)

    # assert number of returned cepstrum coefficients
    if not lpcs.shape[1] == order:
        raise AssertionError


@pytest.mark.test_id(206)
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
@pytest.mark.parametrize("order", [13, 28])
@pytest.mark.parametrize("pre_emph", [False, True])
@pytest.mark.parametrize("lifter", [None, 0.7, -7])
@pytest.mark.parametrize("normalize", [None, "mvn", "ms"])
def test_lpcc(
    sig,
    fs,
    order,
    pre_emph,
    lifter,
    normalize,
):
    """
    test LPCC features module for the following:
        - check that the returned number of cepstrums is correct.
        - check normalization.
        - check liftering.
    """
    lpccs = lpcc(
        sig=sig,
        fs=fs,
        order=order,
        pre_emph=pre_emph,
        lifter=lifter,
        normalize=normalize,
    )

    # assert number of returned cepstrum coefficients
    if not lpccs.shape[1] == order:
        raise AssertionError

    # Test normalize
    if normalize:
        np.testing.assert_array_almost_equal(
            lpccs,
            normalize_ceps(
                lpcc(
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
                lpccs,
                lifter_ceps(
                    lpcc(
                        sig=sig,
                        fs=fs,
                        order=order,
                        pre_emph=pre_emph,
                        lifter=None,
                        normalize=normalize,
                    ),
                    lifter,
                ),
                0,
            )
