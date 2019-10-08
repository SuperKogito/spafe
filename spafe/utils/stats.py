import numpy as np


NFFT = 512


def fft(frames, nfft = NFFT):
    """
    compute the fourrier transform of a certain signal frames.
    """
    return np.fft.rfft(frames, nfft)

def psd(fft_vec,  nfft = NFFT):
    """
    compute the power spectrum density.
    # Magnitude of the FFT = np.abs(fft_vec)
    # Power Spectrum       = (1/nfft) * (magnitude**2)  
    """
    return (1.0 / nfft) * (np.abs(fft_vec) ** 2) 

def cepstral_analysis(sig):
    """
    Do cepstral analysis.
    """
    return np.fft.ifft(np.log(np.abs(np.fft.fft(sig))))

def mn(x):
    """
    Mean normalization.
    """
    return (x - np.mean(x)) / (np.max(x) - np.min(x))

def ms(x):
    """ 
    Mean Substraction: Centering
    """
    return x - np.mean(x, axis=0)
  
def vn(x):
    """ 
    Variance Normalisation: Standdization
    """
    return x / np.std(x)  

def mvn(x):
    """ 
    Mean Variance Normalisation 
    """
    return vn(ms(x))  

def min_max_normalization(vec):
    """ 
    Min Max Normalisation 
    """
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))

def rescale_to_ab_range(vec, a, b):
    """
    transfer vector elements from an interval [x,y] to interval [a,b]
    """
    return a + ((vec - np.min(vec))* (b - a)) / (np.max(vec) - np.min(vec))

def dft(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform.
    """
    return np.fft.fft2(a, s, axes, norm)

def idft(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.
    """
    return np.fft.ifft2(a, s, axes, norm)

def rms(x):
    """
    compute the root mean square
    """
    return np.sqrt(np.sum(x**2)/len(x))

# Source of interpolation function - https://gist.github.com/255291
def parabolic(f, x):
	"""
     Quadratic interpolation for estimating the true position of an
	inter-sample maximum when nearby samples are known.
 	f is a vector and x is an index for that vector.
	Returns (vx, vy), the coordinates of the vertex of a parabola that goes
	through point x and its two neighbors.
	"""
	xv = float(1/2 * (f[x-1] - f[x+1] + 1) / (f[x-1] - 2 * f[x] + f[x+1]) + x)
	yv = float(f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x))
	return (xv, yv)