import pytest
from spafe.utils import vis
from spafe.fbanks import bark_fbanks
from spafe.utils.exceptions import ParameterError


@pytest.mark.test_id(102)
@pytest.mark.parametrize("nfilts", [24])
@pytest.mark.parametrize("nfft", [256])
@pytest.mark.parametrize("fs", [8000, 16000])
@pytest.mark.parametrize("low_freq", [50])
@pytest.mark.parametrize("high_freq", [2000])
@pytest.mark.parametrize("scale", ["ascendant", "descendant", "constant"])
def test_barkfbanks(nfilts, nfft, fs, low_freq, high_freq, scale):
    """
    test bark filter banks module for the following:
        - check if filter banks have the correct shape.
        - check if parameter errors are raised for low_freq.
        - check if parameter errors are raised for high_freq.
    """
    # compute the bark filter banks
    bark_filbanks, _ = bark_fbanks.bark_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale,
    )

    # assert that the filter bank shape is correct given nfilts and nfft
    if not bark_filbanks.shape == (nfilts, nfft // 2 + 1):
        raise AssertionError

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        bark_filbanks, _ = bark_fbanks.bark_filter_banks(
            nfilts=nfilts, nfft=nfft, fs=fs, low_freq=-5, high_freq=high_freq
        )

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        bark_filbanks, _ = bark_fbanks.bark_filter_banks(
            nfilts=nfilts, nfft=nfft, fs=fs, low_freq=low_freq, high_freq=fs * 2
        )
