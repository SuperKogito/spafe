"""

- Description : Exception classes for Spafe implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""

ErrorMsgs = {
    "low_freq": "minimal frequency cannot be less than zero.",
    "high_freq": "maximum frequency cannot be greater than half sampling frequency.",
    "nfft": "size of the FFT must be an integer.",
    "nfilts": "number of filters must be bigger than number of cepstrums",
    "win_len_win_hop_comparison": "window's length has to be larger than the window's hop",
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
