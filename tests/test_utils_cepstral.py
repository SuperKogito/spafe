import spafe
import pytest
import numpy as np
from spafe.utils.exceptions import assert_function_availability
from spafe.utils.cepstral import (cmn, cms, cvn, cmvn, spec2cep, cep2spec,
                                  deltas, lifter_ceps)


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

# to investigate and fix
@pytest.mark.xfail
@pytest.mark.parametrize('w', [0, 2])
def test_deltas(w):
    features = np.linspace((1, 2), (10, 20), 10)
    expected_result = {0 : np.linspace((0, 0), ( 0, 0), 10),
                       2 : np.linspace((1, 1), (10,10), 10) }
    delta_features = deltas(features, w)
    np.testing.assert_almost_equal(delta_features, expected_result[w], 1)

# to investigate and fix
@pytest.mark.xfail
@pytest.mark.parametrize('ncep', [9, 12])
@pytest.mark.parametrize('dct_type', [1, 2, 4])
def test_specs_and_ceps(ncep, dct_type):
    x = np.linspace((1, 2, 3), (10, 20, 30), 5)
    ceps, _ = spec2cep(spec=x, ncep=ncep, dct_type=dct_type)
    specs, _ = cep2spec(cep=ceps, ncep=ncep, nfreq=x.shape[0], dct_type=dct_type)
    np.testing.assert_almost_equal(x, specs, 3)
    
# to investigate and fix
@pytest.mark.xfail
@pytest.mark.parametrize('lifter_coefficient', [0, 22])
def test_lifter_ceps(lifter_coefficient):
    cepstra = np.linspace((1, 2), (10, 20), 10)
    liftered_cepstra = lifter_ceps(cepstra, lifter_coefficient)

    # check when liftering should be applied
    if lifter_coefficient > 0 :
        n = np.arange(np.shape(cepstra)[1])
        lift = 1 + (lifter_coefficient / 2.) * np.sin(np.pi * n / lifter_coefficient)
        np.testing.assert_almost_equal(liftered_cepstra, cepstra * lift, 3)

    # check when liftering is not supposed to be applied
    if lifter_coefficient == 0 :
        np.testing.assert_almost_equal(liftered_cepstra, cepstra, 3)
