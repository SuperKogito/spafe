import numpy as np 


def fft_spectrum(frames, fft_points=512):
    """This function computes the one-dimensional n-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT). Please refer to
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html
    for further details.
    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    Returns:
            array: The fft spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x FFT_LENGTH.
    """
    SPECTRUM_VECTOR = np.fft.rfft(frames, n=fft_points, axis=-1, norm=None)
    return np.absolute(SPECTRUM_VECTOR)


def power_spectrum(frames, fft_points=512):
    """Power spectrum of each frame.
    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
    Returns:
            array: The power spectrum.
            If frames is an num_frames x sample_per_frame matrix, output
            will be num_frames x fft_length.
    """
    return 1.0 / fft_points * np.square(fft_spectrum(frames, fft_points))


def log_power_spectrum(frames, fft_points=512, normalize=True):
    """Log power spectrum of each frame in frames.
    Args:
        frames (array): The frame array in which each row is a frame.
        fft_points (int): The length of FFT. If fft_length is greater than
            frame_len, the frames will be zero-padded.
        normalize (bool): If normalize=True, the log power spectrum
            will be normalized.
    Returns:
           array: The power spectrum - If frames is an
           num_frames x sample_per_frame matrix, output will be
           num_frames x fft_length.
    """
    power_spec = power_spectrum(frames, fft_points)
    power_spec[power_spec <= 1e-20] = 1e-20
    log_power_spec = 10 * np.log10(power_spec)
    if normalize:
        return log_power_spec - np.max(log_power_spec)
    else:
        return log_power_spec