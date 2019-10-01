import numpy




def freq2bark(f):
    return 7.*np.log(f/650.+np.sqrt(np.power(1.+(f/650.), 2.)))


def bark2freq(b):
    return 650.*np.sinh(b/7.)
    
def hz2mel(hz):
    """
    Convert a value in Hertz to Mels

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.)

def mel2hz(mel):
    """
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10**(mel / 2595.0) - 1)
