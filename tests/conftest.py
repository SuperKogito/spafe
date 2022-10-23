from pathlib import Path

import pytest
import scipy
import scipy.io.wavfile

data_folder = Path(__file__).parent / "data"


@pytest.fixture
def sig():
    __EXAMPLE_FILE = data_folder / "test.wav"
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def fs():
    __EXAMPLE_FILE = data_folder / "test.wav"
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[0]
