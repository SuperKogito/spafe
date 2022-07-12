import spafe
import pytest
import numpy as np
from spafe.utils.exceptions import ParameterError
from spafe.utils.exceptions import assert_function_availability
from spafe.utils.converters import (
    hz2erb,
    erb2hz,
    hz2bark,
    bark2hz,
    hz2mel,
    mel2hz,
)


@pytest.mark.test_id(212)
def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert_function_availability(hasattr(spafe.utils.converters, "hz2erb"))
    assert_function_availability(hasattr(spafe.utils.converters, "erb2hz"))
    assert_function_availability(hasattr(spafe.utils.converters, "hz2bark"))
    assert_function_availability(hasattr(spafe.utils.converters, "bark2hz"))
    assert_function_availability(hasattr(spafe.utils.converters, "hz2mel"))
    assert_function_availability(hasattr(spafe.utils.converters, "mel2hz"))


@pytest.mark.test_id(405)
@pytest.mark.parametrize("approach", ["Glasberg"])
def test_hz2erb(approach):
    erb_freqs = [freq for freq in range(0, 30, 2)]
    hz_freqs = [erb2hz(freq, approach) for freq in erb_freqs]
    np.testing.assert_array_almost_equal(
        [hz2erb(freq, approach) for freq in hz_freqs], erb_freqs, 3
    )


@pytest.mark.test_id(406)
@pytest.mark.parametrize("approach", ["Glasberg"])
def test_erb2hz(approach):
    hz_freqs = [freq for freq in range(0, 30, 2)]
    erb_freqs = [hz2erb(freq, approach) for freq in hz_freqs]
    np.testing.assert_array_almost_equal(
        [erb2hz(freq, approach) for freq in erb_freqs], hz_freqs, 3
    )


@pytest.mark.test_id(407)
@pytest.mark.parametrize(
    "approach", ["Tjomov", "Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"]
)
def test_hz2bark(approach):
    hz_freqs = [freq for freq in range(0, 30, 2)]
    bark_freqs = [hz2bark(freq, approach) for freq in hz_freqs]
    np.testing.assert_array_almost_equal(
        [hz2bark(freq, approach) for freq in hz_freqs], bark_freqs, 3
    )


@pytest.mark.test_id(408)
@pytest.mark.parametrize(
    "approach", ["Tjomov", "Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"]
)
def test_bark2hz(approach):
    bark_freqs = [freq for freq in range(0, 30, 2)]
    hz_freqs = [bark2hz(freq, approach) for freq in bark_freqs]
    np.testing.assert_array_almost_equal(
        [bark2hz(freq, approach) for freq in bark_freqs], hz_freqs, 3
    )


@pytest.mark.test_id(409)
@pytest.mark.parametrize("approach", ["Oshaghnessy", "Lindsay"])
def test_hz2mel(approach):
    hz_freqs = [freq for freq in range(0, 30, 2)]
    mel_freqs = [hz2mel(freq, approach) for freq in hz_freqs]
    np.testing.assert_array_almost_equal(
        [hz2mel(freq, approach) for freq in hz_freqs], mel_freqs, 3
    )


@pytest.mark.test_id(410)
@pytest.mark.parametrize("approach", ["Oshaghnessy", "Lindsay"])
def test_mel2hz(approach):
    bark_freqs = [freq for freq in range(0, 30, 2)]
    hz_freqs = [mel2hz(freq, approach) for freq in bark_freqs]
    np.testing.assert_array_almost_equal(
        [mel2hz(freq, approach) for freq in bark_freqs], hz_freqs, 3
    )
