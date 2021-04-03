import pytest
from spafe.utils import vis
from spafe.fbanks import mel_fbanks
from spafe.utils.exceptions import ParameterError


@pytest.mark.test_id(101)
@pytest.mark.parametrize('nfilts', [12, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('fs', [8000, 16000])
@pytest.mark.parametrize('low_freq', [0, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('scale', ["ascendant", "descendant", "constant"])
def test_melfbanks(nfilts, nfft, fs, low_freq, high_freq, scale):
    """
    test mel filter banks module for the following:
        - check if filterbanks have the correct shape.
        - check if parameter errors are raised for low_freq.
        - check if parameter errors are raised for high_freq.
    """
    # compute the Mel filterbanks
    mel_filbanks = mel_fbanks.mel_filter_banks(nfilts=nfilts,
                                               nfft=nfft,
                                               fs=fs,
                                               low_freq=low_freq,
                                               high_freq=high_freq,
                                               scale=scale)

    # assert that the filterbank shape is correct given nfilts and nfft
    if not mel_filbanks.shape == (nfilts, nfft // 2 + 1):
        raise AssertionError

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        mel_filbanks = mel_fbanks.mel_filter_banks(nfilts=nfilts,
                                                   nfft=nfft,
                                                   fs=fs,
                                                   low_freq=-5,
                                                   high_freq=high_freq)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        mel_filbanks = mel_fbanks.mel_filter_banks(nfilts=nfilts,
                                                   nfft=nfft,
                                                   fs=fs,
                                                   low_freq=low_freq,
                                                   high_freq=16000)


@pytest.mark.test_id(105)
@pytest.mark.parametrize('nfilts', [12, 18, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('fs', [8000, 16000])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('scale', ["ascendant", "descendant", "constant"])
def test_imelfbanks(nfilts, nfft, fs, low_freq, high_freq, scale):
    """
    test inverse mel filter banks module for the following:
        - check if filterbanks have the correct shape.
        - check if parameter errors are raised for low_freq.
        - check if parameter errors are raised for high_freq.
    """
    # compute the inverse mel filterbanks
    imel_filbanks = mel_fbanks.inverse_mel_filter_banks(nfilts=nfilts,
                                                        nfft=nfft,
                                                        fs=fs,
                                                        low_freq=low_freq,
                                                        high_freq=high_freq,
                                                        scale=scale)

    # assert that the filterbank shape is correct given nfilts and nfft
    if not imel_filbanks.shape == (nfilts, nfft // 2 + 1):
        raise AssertionError

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        imel_filbanks = mel_fbanks.inverse_mel_filter_banks(
            nfilts=nfilts, nfft=nfft, fs=fs, low_freq=-5, high_freq=high_freq)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        imel_filbanks = mel_fbanks.inverse_mel_filter_banks(nfilts=nfilts,
                                                            nfft=nfft,
                                                            fs=fs,
                                                            low_freq=low_freq,
                                                            high_freq=16000)
