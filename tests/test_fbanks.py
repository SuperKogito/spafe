import pytest
from spafe.utils import vis
from spafe.fbanks import mel_fbanks
from spafe.fbanks import bark_fbanks
from spafe.fbanks import linear_fbanks
from spafe.fbanks import gammatone_fbanks


@pytest.mark.test_id(101)
def test_melfbanks():
    # compute the Mel, Bark and Gammaton filterbanks
    mel_filbanks = mel_fbanks.mel_filter_banks(nfilts=24, nfft=512, fs=16000)
    # plot the Mel filter banks
    vis.visualize_fbanks(mel_filbanks, "Amplitude", "Frequency (Hz)", True)
    assert True

@pytest.mark.test_id(102)
def test_barkfbanks():
    # compute the Mel, Bark and Gammaton filterbanks
    bark_filbanks = bark_fbanks.bark_filter_banks(nfilts=24, nfft=512, fs=16000)
    # plot the Bark filter banks
    vis.visualize_fbanks(bark_filbanks, "Amplitude", "Frequency (Hz)", True)
    assert True
    
@pytest.mark.test_id(103)
def test_linfbanks():
    # compute the Mel, Bark and Gammaton filterbanks
    lin_filbanks = linear_fbanks.linear_filter_banks(nfilts=24, nfft=512, fs=16000)
    # plot the Linear filter banks
    vis.visualize_fbanks(lin_filbanks, "Amplitude", "Frequency (Hz)", True)
    assert True 
    
@pytest.mark.test_id(104)
def test_gamma_fbanks():
    # compute the Mel, Bark and Gammaton filterbanks
    gamma_filbanks = gammatone_fbanks.gammatone_filter_banks(nfilts=24, nfft=512, fs=16000)
    # plot the Gammatone filter banks
    vis.visualize_fbanks(gamma_filbanks, "Amplitude", "Frequency (Hz)", True)
    assert True 

@pytest.mark.test_id(105)
def test_imelfbanks():
    # compute the Mel, Bark and Gammaton filterbanks
    imel_filbanks = mel_fbanks.inverse_mel_filter_banks(nfilts=24, nfft=512, fs=16000)
    # plot the inverse Mel filter banks
    vis.visualize_fbanks(imel_filbanks, "Amplitude", "Frequency (Hz)", True)
    assert True
    
if __name__ == "__main__":
    test_melfbanks()
    test_barkfbanks()
    test_linfbanks()    
    test_gamma_fbanks()    
    test_imelfbanks()
