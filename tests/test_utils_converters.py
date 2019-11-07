import spafe
import pytest
import numpy as np
from spafe.utils.exceptions import ParameterError, ErrorMsgs
from spafe.utils.converters import (hz2erb, erb2hz, hz2bark, bark2hz, hz2mel,
                                    mel2hz, fft2hz, fft2erb, erb2fft, hz2fft,
                                    fft2bark, bark2fft)


@pytest.fixture
def fhz():
    return np.array([
        20., 160., 394., 670., 1000., 1420., 1900., 2450., 3120., 4000., 5100.,
        6600., 9000., 14000.
    ])


@pytest.fixture
def ferb():
    return np.array([
        26.85878, 41.97024, 67.227966, 97.01913, 132.639, 177.97338, 229.7841,
        289.15055, 361.46968, 456.456, 575.1889, 737.0974, 996.151, 1535.846
    ])


@pytest.fixture
def fbark():
    return np.array([
        0.2, 1.582, 3.701, 5.769, 7.703, 9.579, 11.219, 12.688, 14.106, 15.575,
        17.02, 18.559, 20.414, 23.061
    ])


@pytest.fixture
def fmel():
    return np.array([
        31.748, 231.994, 503.221, 756.76, 999.986, 1248.812, 1478.826,
        1695.086, 1912.425, 2146.065, 2383.066, 2642.293, 2962.643, 3431.159
    ])


@pytest.fixture
def fix_fft():
    return np.array([
        0.64125, 5.13, 12.632625, 21.481875, 32.0625, 45.52875, 60.91875,
        78.553125, 100.035, 128.25, 163.51875, 211.6125, 288.5625, 448.875
    ])

def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert hasattr(spafe.utils.converters, 'hz2erb')
    assert hasattr(spafe.utils.converters, 'erb2hz')
    assert hasattr(spafe.utils.converters, 'hz2bark')
    assert hasattr(spafe.utils.converters, 'bark2hz')
    assert hasattr(spafe.utils.converters, 'hz2mel')
    assert hasattr(spafe.utils.converters, 'mel2hz')
    assert hasattr(spafe.utils.converters, 'hz2fft')
    assert hasattr(spafe.utils.converters, 'fft2hz')
    assert hasattr(spafe.utils.converters, 'fft2erb')
    assert hasattr(spafe.utils.converters, 'erb2fft')
    assert hasattr(spafe.utils.converters, 'fft2bark')
    assert hasattr(spafe.utils.converters, 'bark2fft')


def test_hz2erb(fhz, ferb):
    np.testing.assert_almost_equal(hz2erb(fhz), ferb, 0)


def test_erb2hz(ferb, fhz):
    np.testing.assert_almost_equal(erb2hz(ferb), fhz, 0)


def test_hz2bark(fhz, fbark):
    np.testing.assert_almost_equal(hz2bark(fhz), fbark, 0)


def test_bark2hz(fbark, fhz):
    np.testing.assert_almost_equal(bark2hz(fbark), fhz, 0)

@pytest.mark.parametrize('htk', [0, 1])
def test_hz2mel(fhz, fmel, htk):
    if htk == 0:
        fhz = np.round(mel2hz(fmel, htk))
    np.testing.assert_almost_equal(hz2mel(fhz, htk), fmel, 0)

@pytest.mark.parametrize('htk', [0, 1])
def test_mel2hz(fmel, fhz, htk):
    # TO REVISE
    if htk == 0:
        fmel = np.round(hz2mel(fhz, htk))
        fhz  = np.round(mel2hz(fmel, htk))
    np.testing.assert_almost_equal(mel2hz(fmel, htk), fhz, 0)


def test_hz2fft(fhz, fix_fft):
    np.testing.assert_almost_equal(hz2fft(fhz), fix_fft, 0)


def test_fft2hz(fix_fft, fhz):
    np.testing.assert_almost_equal(fft2hz(fix_fft), fhz, 0)


def test_fft2erb(fix_fft, ferb):
    np.testing.assert_almost_equal(fft2erb(fix_fft), ferb, 0)


def test_erb2fft(fix_fft, ferb):
    np.testing.assert_almost_equal(erb2fft(ferb), fix_fft, 0)


def test_fft2bark(fix_fft, fbark):
    np.testing.assert_almost_equal(fft2bark(fix_fft), fbark, 0)


def test_bark2fft(fix_fft, fbark):
    np.testing.assert_almost_equal(bark2fft(fbark), fix_fft, 0)
