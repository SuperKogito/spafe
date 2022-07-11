import pytest
import warnings
import numpy as np
import spafe.utils.filters
from spafe.utils.exceptions import assert_function_availability

DEBUG_MODE = False
warnings.filterwarnings("ignore")


@pytest.mark.test_id(411)
def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert_function_availability(hasattr(spafe.utils.filters, "rasta_filter"))
