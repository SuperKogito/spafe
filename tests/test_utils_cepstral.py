import spafe
import pytest
import numpy as np
from spafe.utils.cepstral import cmn, cms, cvn, cmvn
from spafe.utils.exceptions import assert_function_availability


@pytest.fixture
def x():
    return np.round(np.random.normal(loc=0.0, scale=1.0, size=100), 3)


def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert_function_availability(hasattr(spafe.utils.cepstral, 'cmn'))
    assert_function_availability(hasattr(spafe.utils.cepstral, 'cms'))
    assert_function_availability(hasattr(spafe.utils.cepstral, 'cvn'))
    assert_function_availability(hasattr(spafe.utils.cepstral, 'cmvn'))
    assert_function_availability(hasattr(spafe.utils.cepstral, '_helper_idx'))
    assert_function_availability(hasattr(spafe.utils.cepstral, '_helper_mat'))
    assert_function_availability(hasattr(spafe.utils.cepstral, 'cep2spec'))
    assert_function_availability(hasattr(spafe.utils.cepstral, 'deltas'))
    assert_function_availability(hasattr(spafe.utils.cepstral, 'spec2cep'))
    assert_function_availability(hasattr(spafe.utils.cepstral, 'lifter_ceps'))


def test_cmn(x):
    # To improve
    x = np.round(np.random.normal(loc=0.0, scale=1.0, size=100), 3)
    y = cmn(x)
    np.testing.assert_almost_equal(y,
                                   (x - np.mean(x)) / (np.max(x) - np.min(x)),
                                   0)


def test_cms(x):
    x = np.round(np.random.normal(loc=0.0, scale=1.0, size=100), 3)
    y = cms(x)
    np.testing.assert_almost_equal(y, x, 0)


def test_cvn(x):
    # To improve
    x = np.round(np.random.normal(loc=0.0, scale=1.0, size=100), 3)
    y = cvn(x)
    np.testing.assert_almost_equal(y, x / np.std(x), 0)


def test_cmvn(x):
    # To improve
    x = np.round(np.random.normal(loc=0.0, scale=1.0, size=100), 3)
    y = cmvn(x)
    np.testing.assert_almost_equal(y, cvn(cms(x)), 0)
