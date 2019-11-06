import scipy
import pytest
import numpy as np
from mock import patch
from spafe.utils import vis
import matplotlib.pyplot as plt
from spafe.features.lfcc import lfcc
from spafe.fbanks import linear_fbanks


@pytest.fixture
def sig():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def fs():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[0]


@patch("matplotlib.pyplot.show")
def test_visualize_fbanks(mock_show):
    # compute filterbanks
    lin_filbanks = linear_fbanks.linear_filter_banks()
    vis.visualize_fbanks(fbanks=lin_filbanks,
                         ylabel="Amplitude",
                         xlabel="Frequency (Hz)")
    return plt.gcf()


@patch("matplotlib.pyplot.show")
def test_visualize_features(sig, fs):
    lfccs = lfcc(sig=sig, fs=fs)
    vis.visualize_features(feats=lfccs,
                           ylabel='LFCC Index',
                           xlabel='Frame Index',
                           cmap='viridis')
    return plt.gcf()


@patch("matplotlib.pyplot.show")
def test_plot(mock_show):
    y = np.arange(10)
    vis.plot(y=y, ylabel="y", xlabel="x")
    return plt.gcf()


@patch("matplotlib.pyplot.show")
def test_spectogram(sig, fs):
    vis.spectogram(sig, fs)
    return plt.gcf()
