import spafe
import pytest
import numpy as np
from mock import patch
from spafe.utils import vis
from spafe.features.lfcc import lfcc, linear_spectrogram
from spafe.fbanks.linear_fbanks import linear_filter_banks
from spafe.utils.exceptions import assert_function_availability


@pytest.mark.test_id(418)
def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert_function_availability(hasattr(spafe.utils.vis, "show_fbanks"))
    assert_function_availability(hasattr(spafe.utils.vis, "show_features"))
    assert_function_availability(hasattr(spafe.utils.vis, "show_spectrogram"))


@pytest.mark.test_id(419)
@patch("matplotlib.pyplot.show")
def test_show_fbanks(mock_show):
    # init var
    fs = 8000
    nfilt = 32
    nfft = 1024

    # compute freqs for xaxis
    lhz_freqs = np.linspace(0, fs / 2, nfft // 2 + 1)

    # ascendant linear fbank
    linear_fbanks_mat, lin_freqs = linear_filter_banks(
        nfilts=nfilt, nfft=nfft, fs=fs, high_freq=fs / 2
    )

    # visualize fbanks
    vis.show_fbanks(
        linear_fbanks_mat,
        lin_freqs,
        lhz_freqs,
        "Linear Filter Bank",
        ylabel="Weight",
        x1label="Frequency / Hz",
        figsize=(14, 5),
        fb_type="lin",
    )


@pytest.mark.test_id(420)
@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
def test_show_spectrogram(mock_show, sig, fs):
    # compute spectrogram
    lSpec, lfreqs = linear_spectrogram(sig, fs=fs)

    # visualize spectrogram
    vis.show_spectrogram(
        lSpec.T,
        fs,
        xmin=0,
        xmax=len(sig) / fs,
        ymin=0,
        ymax=(fs / 2) / 1000,
        dbf=80.0,
        xlabel="Time (s)",
        ylabel="Frequency (kHz)",
        title="Linear spectrogram (dB)",
        cmap="jet",
    )


@pytest.mark.test_id(421)
@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
def test_show_features(mock_show, sig, fs):
    # compute lfccs
    lfccs = lfcc(sig, fs=fs)

    # visualize features
    vis.show_features(
        lfccs, "Linear Frequency Cepstral Coefﬁcients", "LFCC Index", "Frame Index"
    )
