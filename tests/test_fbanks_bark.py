import pytest
from spafe.utils import vis
from spafe.fbanks import bark_fbanks
from spafe.utils.exceptions import ParameterError, ErrorMsgs

DEBUG_MODE = False


@pytest.mark.test_id(102)
@pytest.mark.parametrize('nfilts', [12, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('fs', [8000, 16000])
@pytest.mark.parametrize('low_freq', [0, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('scale', ["ascendant", "descendant", "constant"])
def test_barkfbanks(nfilts, nfft, fs, low_freq, high_freq, scale):
    """
    test bark filter banks module for the following:
        - check if filterbanks have the correct shape.
        - check if parameter errors are raised for low_freq.
        - check if parameter errors are raised for high_freq.
    """
    # compute the bark filterbanks
    bark_filbanks = bark_fbanks.bark_filter_banks(nfilts=nfilts,
                                                  nfft=nfft,
                                                  fs=fs,
                                                  low_freq=low_freq,
                                                  high_freq=high_freq,
                                                  scale=scale)

    # assert that the filterbank shape is correct given nfilts and nfft
    assert bark_filbanks.shape == (nfilts, nfft // 2 + 1)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        bark_filbanks = bark_fbanks.bark_filter_banks(nfilts=nfilts,
                                                      nfft=nfft,
                                                      fs=fs,
                                                      low_freq=-5,
                                                      high_freq=high_freq)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        bark_filbanks = bark_fbanks.bark_filter_banks(nfilts=nfilts,
                                                      nfft=nfft,
                                                      fs=fs,
                                                      low_freq=low_freq,
                                                      high_freq=16000)

    # plot filter banks
    if DEBUG_MODE:
        vis.visualize_fbanks(bark_filbanks, "Amplitude", "Frequency (Hz)")
    assert True


if __name__ == "__main__":
    # init vars
    nfilts = 48
    nfft = 512
    fs = 16000
    low_freq = 0
    high_freq = 8000
    scale = "descendant"

    # run tests
    test_barkfbanks(nfilts, nfft, fs, low_freq, high_freq, scale)
