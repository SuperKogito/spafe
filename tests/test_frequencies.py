"""
based on: https://github.com/scoreur/cqt/blob/master/cqt.py
"""
import scipy
import pytest
from mock import patch
import scipy.io.wavfile
from spafe.features.spfeats import extract_feats
from spafe.frequencies.dominant_frequencies import DominantFrequenciesExtractor
from spafe.frequencies.fundamental_frequencies import FundamentalFrequenciesExtractor

DEBUG_MODE = False


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


@patch("matplotlib.pyplot.show")
def test_fund_freqs(sig, fs):
    """
    test the computation of fundamental frequencies.
    """
    #  test fundamental frequencies extraction
    fund_freqs_extractor = FundamentalFrequenciesExtractor(debug=True)
    pitches, harmonic_rates, argmins, times = fund_freqs_extractor.main(
        sig=sig, fs=fs)


@patch("matplotlib.pyplot.show")
def test_extract_feats(mock_show, sig, fs):
    """
    test the computations of spectral features.
    """
    spectral_features = extract_feats(sig=sig, fs=fs)

    # general stats
    if not len(spectral_features) == 33:
        raise AssertionError

    if not spectral_features["duration"] == (len(sig) / float(fs)):
        raise AssertionError

    for k, v in spectral_features.items():
        if v is None:
            raise AssertionError


if __name__ == "__main__":
    fs, sig = scipy.io.wavfile.read('../test.wav')
    # run tests
    test_dom_freqs(sig, fs)
    test_fund_freqs(sig, fs)
    test_extract_feats(sig, fs)
