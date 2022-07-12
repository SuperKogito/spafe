import spafe
import pytest
import numpy as np
from spafe.utils.exceptions import assert_function_availability
from spafe.utils.cepstral import (
    normalize_ceps,
    deltas,
    lifter_ceps,
)


@pytest.mark.test_id(401)
def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert_function_availability(hasattr(spafe.utils.cepstral, "normalize_ceps"))
    assert_function_availability(hasattr(spafe.utils.cepstral, "deltas"))
    assert_function_availability(hasattr(spafe.utils.cepstral, "lifter_ceps"))


@pytest.mark.test_id(402)
@pytest.mark.parametrize("normalize", ["mvn", "ms", "vn", "mn"])
def test_normalize_ceps(normalize):
    x = np.round(np.random.normal(loc=0.0, scale=1.0, size=100), 3)
    y0 = normalize_ceps(x, normalize)

    if normalize == "mvn":
        y1 = (x - np.mean(x, axis=0)) / np.std(x)
    elif normalize == "ms":
        y1 = x - np.mean(x, axis=0)
    elif normalize == "vn":
        y1 = x / np.std(x)
    elif normalize == "mn":
        y1 = (x - np.mean(x)) / (np.max(x) - np.min(x))

    np.testing.assert_array_almost_equal(y0, y1, 1)


@pytest.mark.test_id(403)
@pytest.mark.parametrize("w", [0, 2])
def test_deltas(w):
    features = np.linspace((1, 2), (10, 20), 10)
    expected_result = {
        0: np.linspace((0, 0), (0, 0), 10),
        2: np.linspace((1, 1), (10, 10), 10),
    }
    delta_features = deltas(features, w)
    np.testing.assert_array_almost_equal(delta_features, expected_result[w], 1)


@pytest.mark.test_id(404)
@pytest.mark.parametrize("lifter_coefficient", [0, 0.7, -7, -22])
def test_lifter_ceps(lifter_coefficient):
    cepstra = np.linspace((1, 2), (10, 20), 10)
    liftered_cepstra = lifter_ceps(cepstra, lifter_coefficient)

    # check when liftering is not supposed to be applied
    if lifter_coefficient == 0:
        np.testing.assert_array_almost_equal(liftered_cepstra, cepstra, 3)

    # check when liftering should be applied
    elif lifter_coefficient > 0:
        lift_vec = np.array(
            [1] + [i**lifter_coefficient for i in range(1, cepstra.shape[1])]
        )
        lift_mat = np.diag(lift_vec)
        y1 = np.dot(cepstra, lift_mat)

    else:
        lifter_coefficient = int(-1 * lifter_coefficient)
        lift_vec = 1 + (lifter_coefficient / 2.0) * np.sin(
            np.pi * np.arange(1, 1 + cepstra.shape[1]) / lifter_coefficient
        )
        y1 = cepstra * lift_vec
        np.testing.assert_array_almost_equal(liftered_cepstra, y1, 3)
