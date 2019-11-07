import spafe
import pytest
import numpy as np
from spafe.utils.preprocessing import (zero_handling, pre_emphasis, framing, windowing)

def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert hasattr(spafe.utils.preprocessing, 'zero_handling')
    assert hasattr(spafe.utils.preprocessing, 'pre_emphasis')
    assert hasattr(spafe.utils.preprocessing, 'framing')
    assert hasattr(spafe.utils.preprocessing, 'windowing')


@pytest.mark.parametrize('x', [np.arange(4)])
def test_zero_handling(x):
    """
    test zero handling and check if it handles all zero values correctly
    """
    y = zero_handling(x=x)
    # to be implemented
    assert True

@pytest.mark.parametrize('sig', [np.arange(5)])
@pytest.mark.parametrize('pre_emph_coeff', [0.97])
def test_pre_emphasis(sig, pre_emph_coeff):
    """
    test if pre_emphasis is applied correctly.
    """
    precomputed_result = np.array([0.  , 1.  , 1.03, 1.06, 1.09])
    np.testing.assert_almost_equal(pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff),
                                    precomputed_result, 3)


@pytest.mark.parametrize('sig', [np.arange(16*3)])
@pytest.mark.parametrize('fs', [16000])
@pytest.mark.parametrize('win_len', [0.001])
@pytest.mark.parametrize('win_hop', [0.001, 0.0005])
def test_framing(sig, fs, win_len, win_hop):
    """
    test if overlapped frames are correctly generated.
    """
    frames, frames_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)
    assert frames.shape == (2 * int(win_len / win_hop), 16)
    assert frames_length ==  win_len * fs


@pytest.mark.parametrize('frames', [[[7, 8 ,9], [1, 2, 3]]])
@pytest.mark.parametrize('frame_len', [3])
def test_windowing(frames, frame_len, win_type="hamming"):
    """
    test if windowing is applied correctly.
    """
    windows = windowing(frames, frame_len, win_type="hamming")
    for window in windows:
        assert window[0] < 0.9 and window[-1] < 0.99
