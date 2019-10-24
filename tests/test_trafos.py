"""
based on: https://github.com/scoreur/cqt/blob/master/cqt.py
"""
import scipy
import pytest
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from spafe.utils.spectral import (cqt, stft, istft, display_stft)

DEBUG_MODE = False


def get_data(fname):
    return scipy.io.wavfile.read(fname)


@pytest.fixture
def x():
    __EXAMPLE_FILE = 'test21.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def sig():
    __EXAMPLE_FILE = 'test21.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def fs():
    __EXAMPLE_FILE = 'test21.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[0]


@pytest.mark.test_id(301)
def test_stft(x, fs):
    # compute and display STFT
    X, _ = stft(sig=x, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)
    if DEBUG_MODE:
        display_stft(X, fs, len(x), 0, 2000, -10, 0, True)


@pytest.mark.test_id(302)
def test_istft(x, fs):
    # compute and display STFT
    X, x_pad = stft(sig=x, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)

    # inverse STFT
    y = istft(X=X, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)
    yr = np.real(y)

    # plot results
    if DEBUG_MODE:
        # check that ISTFT actually inverts the STFT
        t = np.arange(0, x_pad.size) / fs
        diff = x_pad / np.abs(x_pad).max() - yr[:x_pad.size] / np.abs(
            yr[:x_pad.size]).max()

        # plot the original and the difference of the original from the reconstruction
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(t, diff, 'r-')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Reconstruction Error value")
        ax.set_title("Reconstruction Error")
        plt.show()


@pytest.mark.test_id(303)
def test_cqt(sig, fs):
    """
    Test constant Q-transform function.
    """
    xcq = cqt(sig=sig, fs=fs, low_freq=40, high_freq=22050, b=12)
    ampxcq = np.abs(xcq)**2
    if DEBUG_MODE:
        plt.plot(ampxcq)
        plt.show()


if __name__ == "__main__":
    fs, x = get_data('../test21.wav')
    # run tests
    test_stft(x, fs)
    test_istft(x, fs)
