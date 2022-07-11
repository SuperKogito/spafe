import spafe
import pytest
from mock import patch
from spafe.utils.exceptions import assert_function_availability
from spafe.utils.spectral import compute_constant_qtransform


@pytest.mark.test_id(417)
def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert_function_availability(
        hasattr(spafe.utils.spectral, "compute_constant_qtransform")
    )
