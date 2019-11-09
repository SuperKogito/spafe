"""
based on: https://github.com/scoreur/cqt/blob/master/cqt.py
"""
import scipy
import pytest
import numpy as np
from mock import patch
import scipy.io.wavfile
import matplotlib.pyplot as plt
from spafe.features.spfeats import extract_feats
from spafe.frequencies.dominant_frequencies import DominantFrequenciesExtractor
from spafe.frequencies.fundamental_frequencies import FundamentalFrequenciesExtractor


DEBUG_MODE = False


def get_data(fname):
    return scipy.io.wavfile.read(fname)

@pytest.fixture
def sig():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def fs():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[0]

@patch("matplotlib.pyplot.show")
def test_dom_freqs(sig, fs):
    """
    test the computation of dominant frequencies
    """
    # test dominant frequencies extraction
    dom_freqs_extractor = DominantFrequenciesExtractor(debug=True)
    dom_freqs = dom_freqs_extractor.main(sig=sig, fs=fs)
    assert True

@patch("matplotlib.pyplot.show")
def test_fund_freqs(sig, fs):
    """
    test the computation of fundamental frequencies.
    """
    #  test fundamental frequencies extraction
    fund_freqs_extractor = FundamentalFrequenciesExtractor(debug=True)
    pitches, harmonic_rates, argmins, times = fund_freqs_extractor.main(sig=sig, fs=fs)

@patch("matplotlib.pyplot.show")
def test_extract_feats(mock_show, sig, fs):
    """
    test the computations of spectral features.
    """
    spectral_features = extract_feats(sig=sig, fs=fs)
    print(len(spectral_features))

    # general stats
    assert len(spectral_features) == 33
    assert spectral_features["duration"] == (len(sig) / float(fs))
    for k, v in spectral_features.items():
        assert v is not None


if __name__ == "__main__":
    fs, sig = scipy.io.wavfile.read('../test.wav')
    # run tests
    test_dom_freqs(sig, fs)
    test_fund_freqs(sig, fs)
    test_extract_feats(sig, fs)
