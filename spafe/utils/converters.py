import numpy as np


def hz2erb(f):
    """
    Convert Hz frequencies to Bark.

    Args:
        f (np.array) : input frequencies [Hz].

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return 24.7 * (4.37 * (f / 1000) + 1)

def erb2hz(fe):
    """
    Convert Bark frequencies to Hz.

    Args:
        fb (np.array) : input frequencies [Bark].

    Returns:
        (np.array)  : frequencies in Hz [Hz].
    """
    return ((fe/24.7) - 1) * (1000. / 4.37)

def fft2erb(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.

    Args:
        fft (np.array) : fft bin numbers.

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return hz2erb((fft * fs) / (nfft + 1))

def erb2fft(fb, fs=16000, nfft=512):
    """
    Convert Bark frequencies to fft bins.

    Args:
        fb (np.array): frequencies in Bark [Bark].

    Returns:
        (np.array) : fft bin numbers.
    """
    return (nfft + 1) * erb2hz(fb) / fs

def hz2bark(f):
    """
    Convert Hz frequencies to Bark.

    Args:
        f (np.array) : input frequencies [Hz].

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return 6. * np.arcsinh(f / 600. )

def bark2hz(fb):
    """
    Convert Bark frequencies to Hz.

    Args:
        fb (np.array) : input frequencies [Bark].

    Returns:
        (np.array)  : frequencies in Hz [Hz].
    """
    return 600. * np.sinh( fb / 6.)

def fft2hz(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.

    Args:
        fft (np.array) : fft bin numbers.

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return (fft * fs) / (nfft + 1)

def hz2fft(fb, fs=16000, nfft=512):
    """
    Convert Bark frequencies to fft bins.

    Args:
        fb (np.array): frequencies in Bark [Bark].

    Returns:
        (np.array) : fft bin numbers.
    """
    return (nfft + 1) * fb / fs

def fft2bark(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.

    Args:
        fft (np.array) : fft bin numbers.

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return hz2bark((fft * fs) / (nfft + 1))

def bark2fft(fb, fs=16000, nfft=512):
    """
    Convert Bark frequencies to fft bins.

    Args:
        fb (np.array): frequencies in Bark [Bark].

    Returns:
        (np.array) : fft bin numbers.
    """
    return (nfft + 1) * bark2hz(fb) / fs

def hz2mel(hz):
    """
    Convert a value in Hertz to Mels

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.)

def mel2hz(mel):
    """
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10**(mel / 2595.0) - 1)
