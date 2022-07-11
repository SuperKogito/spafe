import scipy
import pytest
import numpy as np
import scipy.io.wavfile


@pytest.fixture
def sig():
    __EXAMPLE_FILE = "test.wav"
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def fs():
    __EXAMPLE_FILE = "test.wav"
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[0]
