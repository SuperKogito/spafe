import spafe
import pytest
import numpy as np
from spafe.utils.exceptions import ParameterError, assert_function_availability
from spafe.utils.preprocessing import (
    zero_handling,
    pre_emphasis,
    framing,
    windowing,
)


@pytest.mark.test_id(412)
def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert_function_availability(hasattr(spafe.utils.preprocessing, "zero_handling"))
    assert_function_availability(hasattr(spafe.utils.preprocessing, "pre_emphasis"))
    assert_function_availability(hasattr(spafe.utils.preprocessing, "framing"))
    assert_function_availability(hasattr(spafe.utils.preprocessing, "windowing"))


@pytest.mark.test_id(413)
@pytest.mark.parametrize("x", [np.arange(4)])
def test_zero_handling(x):
    """
    test zero handling and check if it handles all zero values correctly
    """
    y = zero_handling(x=x)
    # check if log can be computed without a problem
    log_y = np.log(y)


@pytest.mark.test_id(414)
@pytest.mark.parametrize("sig", [np.arange(5)])
@pytest.mark.parametrize("pre_emph_coeff", [0.97])
def test_pre_emphasis(sig, pre_emph_coeff):
    """
    test if pre_emphasis is applied correctly.
    """
    precomputed_result = np.array([0.0, 1.0, 1.03, 1.06, 1.09])
    np.testing.assert_array_almost_equal(
        pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff), precomputed_result, 3
    )


@pytest.mark.test_id(415)
@pytest.mark.parametrize("sig", [np.arange(16000 * 3)])  # 3 seconds
@pytest.mark.parametrize("fs", [16000])
@pytest.mark.parametrize("win_len", [1, 2, 0.025, 0.030])
@pytest.mark.parametrize("win_hop", [1, 1, 0.025, 0.015])
def test_framing(sig, fs, win_len, win_hop):
    """
    test if overlapped frames are correctly generated.
    """
    if win_len < win_hop:
        # check error for number of filters is smaller than number of cepstrums
        with pytest.raises(ParameterError):
            frames, frame_length = framing(
                sig=sig, fs=fs, win_len=win_len, win_hop=win_hop
            )

    else:
        frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

        # check frames length
        if not frame_length == win_len * fs:
            raise AssertionError

        # check signal was not cropping
        if len(sig) > len(frames) * frame_length:
            raise AssertionError


@pytest.mark.test_id(416)
@pytest.mark.parametrize("frames", [[[7, 8, 9], [1, 2, 3]]])
@pytest.mark.parametrize("frame_len", [3])
@pytest.mark.parametrize(
    "win_type", ["hamming", "hanning", "bartlet", "kaiser", "blackman"]
)
def test_windowing(frames, frame_len, win_type):
    """
    test if windowing is applied correctly.
    """
    windows = windowing(frames, frame_len, win_type)
    for window in windows:
        if not window[0] < 0.0 and window[-1] < 0.0:
            raise AssertionError
