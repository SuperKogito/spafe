import pytest
from spafe.utils import vis
from spafe.fbanks import mel_fbanks
from spafe.fbanks import bark_fbanks
from spafe.fbanks import linear_fbanks
from spafe.fbanks import gammatone_fbanks
from spafe.utils.exceptions import ParameterError, ErrorMsgs

DEBUG_MODE = False


@pytest.mark.test_id(101)
@pytest.mark.parametrize('nfilts', [12, 18, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('fs', [8000, 16000])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
def test_melfbanks(nfilts, nfft, fs, low_freq, high_freq):
    # compute the Mel, Bark and Gammaton filterbanks
    mel_filbanks = mel_fbanks.mel_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq)

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute the Mel, Bark and Gammaton filterbanks
        mel_filbanks = mel_fbanks.mel_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=16000)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        # compute the Mel, Bark and Gammaton filterbanks
        mel_filbanks = mel_fbanks.mel_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=-5,
            high_freq=high_freq)

    # plot the Mel filter banks
    if DEBUG_MODE:
        vis.visualize_fbanks(mel_filbanks, "Amplitude", "Frequency (Hz)")
    assert True


@pytest.mark.test_id(102)
@pytest.mark.parametrize('nfilts', [12, 18, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('fs', [8000, 16000])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
def test_barkfbanks(nfilts, nfft, fs, low_freq, high_freq):
    # compute the Mel, Bark and Gammaton filterbanks
    bark_filbanks = bark_fbanks.bark_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq)
    # plot the Bark filter banks
    if DEBUG_MODE:
        vis.visualize_fbanks(bark_filbanks, "Amplitude", "Frequency (Hz)")
    assert True


@pytest.mark.test_id(103)
@pytest.mark.parametrize('nfilts', [12, 18, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('fs', [8000, 16000])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
def test_linfbanks(nfilts, nfft, fs, low_freq, high_freq):
    # compute the Mel, Bark and Gammaton filterbanks
    lin_filbanks = linear_fbanks.linear_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq)

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute the Mel, Bark and Gammaton filterbanks
        lin_filbanks = linear_fbanks.linear_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=16000)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        # compute the Mel, Bark and Gammaton filterbanks
        lin_filbanks = linear_fbanks.linear_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=-5,
            high_freq=high_freq)

    # plot the Linear filter banks
    if DEBUG_MODE:
        vis.visualize_fbanks(lin_filbanks, "Amplitude", "Frequency (Hz)")
    assert True


@pytest.mark.test_id(104)
@pytest.mark.parametrize('nfilts', [12, 18, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('fs', [8000, 16000])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
def test_gamma_fbanks(nfilts, nfft, fs, low_freq, high_freq):
    # compute the Mel, Bark and Gammaton filterbanks
    gamma_filbanks = gammatone_fbanks.gammatone_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq)

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute the Mel, Bark and Gammaton filterbanks
        gamma_filbanks = gammatone_fbanks.gammatone_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=16000)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        # compute the Mel, Bark and Gammaton filterbanks
        gamma_filbanks = gammatone_fbanks.gammatone_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=-5,
            high_freq=high_freq)


    # plot the Gammatone filter banks
    if DEBUG_MODE:
        vis.visualize_fbanks(gamma_filbanks, "Amplitude", "Frequency (Hz)")
    assert True


@pytest.mark.test_id(105)
@pytest.mark.parametrize('nfilts', [12, 18, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
@pytest.mark.parametrize('fs', [8000, 16000])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
def test_imelfbanks(nfilts, nfft, fs, low_freq, high_freq):
    # compute the Mel, Bark and Gammaton filterbanks
    imel_filbanks = mel_fbanks.inverse_mel_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq)

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute the Mel, Bark and Gammaton filterbanks
        imel_filbanks = mel_fbanks.inverse_mel_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=low_freq,
            high_freq=16000)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        # compute the Mel, Bark and Gammaton filterbanks
        imel_filbanks = mel_fbanks.inverse_mel_filter_banks(
            nfilts=nfilts,
            nfft=nfft,
            fs=fs,
            low_freq=-5,
            high_freq=high_freq)


    # plot the inverse Mel filter banks
    if DEBUG_MODE:
        vis.visualize_fbanks(imel_filbanks, "Amplitude", "Frequency (Hz)")
    assert True


if __name__ == "__main__":
    test_melfbanks()
    test_barkfbanks()
    test_linfbanks()
    test_gamma_fbanks()
    test_imelfbanks()
