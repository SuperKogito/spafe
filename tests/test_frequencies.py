import pytest
from mock import patch

from spafe.features.spfeats import extract_feats
from spafe.frequencies.dominant_frequencies import get_dominant_frequencies
from spafe.frequencies.fundamental_frequencies import compute_yin


@pytest.mark.test_id(301)
@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
def test_dom_freqs(mock_show, sig, fs):
    """
    test the computation of dominant frequencies
    """
    # test dominant frequencies extraction
    dom_freqs = get_dominant_frequencies(
        sig=sig,
        fs=fs,
        butter_filter=False,
        lower_cutoff=50,
        upper_cutoff=3000,
        nfft=512,
        win_len=0.025,
        win_hop=0.01,
        win_type="hamming",
    )
    # assert is not None
    if dom_freqs is None:
        raise AssertionError


@pytest.mark.test_id(302)
@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
def test_fund_freqs(mock_show, sig, fs):
    """
    test the computation of fundamental frequencies.
    """
    #  test fundamental frequencies extraction
    duration = len(sig) / fs
    harmonic_threshold = 0.85

    pitches, harmonic_rates, argmins, times = compute_yin(
        sig,
        fs,
        win_len=0.030,
        win_hop=0.015,
        low_freq=50,
        high_freq=500,
        harmonic_threshold=harmonic_threshold,
    )

    # assert is not None
    if pitches is None:
        raise AssertionError


@pytest.mark.test_id(303)
@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
def test_extract_feats(mock_show, sig, fs):
    """
    test the computations of spectral features.
    """
    spectral_features = extract_feats(sig=sig, fs=fs)

    # general stats
    if not len(spectral_features) == 12:
        raise AssertionError

    for _, v in spectral_features.items():
        if v is None:
            raise AssertionError
