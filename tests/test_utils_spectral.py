"""
based on: https://github.com/scoreur/cqt/blob/master/cqt.py
"""
import spafe
import scipy
import pytest
import numpy as np
from mock import patch
import scipy.io.wavfile
import matplotlib.pyplot as plt
from spafe.utils.spectral import (cqt, stft, istft, display_stft)

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

def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert hasattr(spafe.utils.spectral, 'cqt')
    assert hasattr(spafe.utils.spectral, 'pre_process_x')
    assert hasattr(spafe.utils.spectral, 'stft')
    assert hasattr(spafe.utils.spectral, 'compute_stft')
    assert hasattr(spafe.utils.spectral, 'istft')
    assert hasattr(spafe.utils.spectral, 'normalize_window')
    assert hasattr(spafe.utils.spectral, 'display_stft')
    assert hasattr(spafe.utils.spectral, 'power_spectrum')
    assert hasattr(spafe.utils.spectral, 'rfft')
    assert hasattr(spafe.utils.spectral, 'dct')
    assert hasattr(spafe.utils.spectral, 'powspec')
    assert hasattr(spafe.utils.spectral, 'lifter')
    assert hasattr(spafe.utils.spectral, 'audspec')
    assert hasattr(spafe.utils.spectral, 'postaud')
    assert hasattr(spafe.utils.spectral, 'invpostaud')
    assert hasattr(spafe.utils.spectral, 'invpowspec')
    assert hasattr(spafe.utils.spectral, 'invaudspec')

@pytest.mark.test_id(301)
def test_stft(sig, fs):
    """
    test STFT computations.
    """
    # compute and display STFT
    X, _ = stft(sig=sig, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)
    if DEBUG_MODE:
        display_stft(X, fs, len(sig), 0, 2000, -10, 0, True)


@pytest.mark.test_id(302)
def test_istft(sig, fs):
    """
    test inverse STFT computations.
    """
    # compute and display STFT
    X, x_pad = stft(sig=sig, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)

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
    test constant Q-transform function.
    """
    xcq = cqt(sig=sig, fs=fs, low_freq=40, high_freq=22050, b=12)
    ampxcq = np.abs(xcq)**2
    if DEBUG_MODE:
        plt.plot(ampxcq)
        plt.show()

@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize('low_freq', [0, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('normalize', [False, True])
def test_display(sig, fs, low_freq, high_freq, normalize):
    """
    test display STFT.
    """
    # compute and display STFT
    X, x_pad = stft(sig=sig, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)
    display_stft(X,
                 fs,
                 len_sig=len(sig),
                 low_freq=low_freq,
                 high_freq=high_freq,
                 min_db=-10,
                 max_db=0,
                 normalize=normalize)

if __name__ == "__main__":
    fs, x = get_data('../test.wav')
    # run tests
    test_stft(x, fs)
    test_istft(x, fs)
