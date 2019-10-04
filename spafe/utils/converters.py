import numpy as np 


def hz2erb(f):
    """
    Convert Hz frequencies to Bark.
    
    Args:
    -----
    f (np.array) : input frequencies [Hz].
    
    Returns:
    --------
    fb (np.array): frequencies in Bark [Bark].
    """
    return 24.7 * (4.37 * (f / 1000) + 1)  

def erb2hz(fe):
    """
    Convert Bark frequencies to Hz.
    
    Args:
    -----
    fb (np.array) : input frequencies [Bark].
    
    Returns:
    --------
    f (np.array)  : frequencies in Hz [Hz].
    """
    return ((fe/24.7) - 1) * (1000. / 4.37)

def fft2erb(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.
    
    Args:
    -----
    fft (np.array) : fft bin numbers.
    
    Returns:
    --------
    fb (np.array): frequencies in Bark [Bark].
    """
    return hz2erb((fft * fs) / (nfft + 1))

def erb2fft(fb, fs=16000, nfft=512):   
    """
    Convert Bark frequencies to fft bins.
    
    Args:
    -----
    fb (np.array): frequencies in Bark [Bark].
    
    Returns:
    --------
    fft (np.array) : fft bin numbers.
    """
    return (nfft + 1) * erb2hz(fb) / fs



def hz2bark(f):
    """
    Convert Hz frequencies to Bark.
    
    Args:
    -----
    f (np.array) : input frequencies [Hz].
    
    Returns:
    --------
    fb (np.array): frequencies in Bark [Bark].
    """
    return 6. * np.arcsinh(f / 600. )  

def bark2hz(fb):
    """
    Convert Bark frequencies to Hz.
    
    Args:
    -----
    fb (np.array) : input frequencies [Bark].
    
    Returns:
    --------
    f (np.array)  : frequencies in Hz [Hz].
    """
    return 600. * np.sinh( fb / 6.)

def fft2hz(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.
    
    Args:
    -----
    fft (np.array) : fft bin numbers.
    
    Returns:
    --------
    fb (np.array): frequencies in Bark [Bark].
    """
    return (fft * fs) / (nfft + 1)

def hz2fft(fb, fs=16000, nfft=512):   
    """
    Convert Bark frequencies to fft bins.
    
    Args:
    -----
    fb (np.array): frequencies in Bark [Bark].
    
    Returns:
    --------
    fft (np.array) : fft bin numbers.
    """
    return (nfft + 1) * fb / fs
