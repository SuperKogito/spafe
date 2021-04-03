"""
Exception classes for Spafe
"""

ErrorMsgs = {
    "low_freq": "minimal frequency cannot be less than zero.",
    "high_freq":
    "maximum frequency cannot be greater than half sampling frequency.",
    "nfft": "size of the FFT must be an integer.",
    "nfilts": "number of filters must be bigger than number of cepstrums",
    "win_len_win_hop_comparison": "window's length has to be larger than the window's hop"
}


class SpafeError(Exception):
    """
    The root spafe exception class
    """


class ParameterError(SpafeError):
    """
    Exception class for mal-formed inputs
    """


def assert_function_availability(hasattr_output):
    # raise assertion error if function is not availible
    if not hasattr_output:
        raise AssertionError
