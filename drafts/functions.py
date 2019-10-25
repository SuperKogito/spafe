import numpy as np
from scipy.fftpack import fft2, ifft2

NFFT = 512

def fft(frames, nfft = NFFT):
    return np.fft.rfft(frames, nfft)

def psd(fft_vec,  nfft = NFFT):
    # Magnitude of the FFT
    magnitude_frames = np.absolute(fft_vec)
    # Power Spectrum
    power_frames = ((1.0 / nfft) * ((magnitude_frames) ** 2))
    return power_frames

def ceil2(x):
    x = int(x)
    assert x > 0
    return 2**(x - 1).bit_length()

def pad2(x):
    return np.r_[x, np.zeros(ceil2(len(x)) - len(x), x.dtype)]

def fcs(s):  # fast cepstrum
    return np.fft.ifft(np.log(np.fft.fft(s)))


def ifcs(s):  # inverted fast cepstrum
    return np.fft.fft(np.exp(np.fft.ifft(s)))


def mcs(s):  # magnitude (actually power)
    return (np.abs(np.fft.ifft(np.log(np.abs(np.fft.fft(s))**2)))**2)[:len(s) // 2]


def clipdb(s, cutoff=-100):
    as_ = np.abs(s)
    mas = np.max(as_)
    if mas == 0 or cutoff >= 0:
        return s
    thresh = mas*10**(cutoff/20)
    return np.where(as_ < thresh, thresh, s)


def fold(r):
    # via https://ccrma.stanford.edu/~jos/fp/Matlab_listing_fold_m.html
    # Fold left wing of vector in "FFT buffer format" onto right wing
    # J.O. Smith, 1982-2002
    n = len(r)
    if n < 3:
        rw = r
    elif n % 2 == 1:
        nt = (n + 1)//2
        rf = r[1:nt] + np.conj(r[-1:nt-1:-1])
        rw = np.r_[r[0], rf, np.zeros(n-nt)]
    else:
        nt = n//2
        rf = np.r_[r[1:nt], 0] + np.conj(r[-1:nt-1:-1])
        rw = np.r_[r[0], rf, np.zeros(n-nt-1)]
    return rw


def minphase(s, pad=True):
    # via https://ccrma.stanford.edu/~jos/fp/Matlab_listing_mps_m.html
    # TODO: oversampling
    if pad:
        s = pad2(s)
    cepstrum = np.fft.ifft(np.log(clipdb(np.fft.fft(s), -100)))
    signal = np.real(np.fft.ifft(np.exp(np.fft.fft(fold(cepstrum)))))
    return signal

def dft(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform.
    """
    return fft2(a, s, axes, norm)

def idft(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.
    """
    return ifft2(a, s, axes, norm)

def equalizeAudio():
    # Grabs the inDir and outDir variable from the mail file
    global inDir
    global outDir

    audioLevels = []
    audioFiles = []

def findAudioLevel():
    # Adds the current songs audio level to an array
    audioLevels.append(song.dBFS)

def findAverageLevel():
    # Resets value for each time is has to do it
    averageAudioLevel = 0

    # Adds all decibel levels up
    for x in audioLevels:
        averageAudioLevel += x

    # Divides them by total number of files
    return averageAudioLevel / len(audioLevels)

def normalizeAudio(song):
    # Finds the difference for apply_gain to use
    dBDifference = song.dBFS - averageAudioLevel

    # Debug
    #print("Difference: ", dBDifference)

    # Removes the difference from the audio level to set it to the average
    return song - dBDifference




from scipy.signal.filter_design import bilinear
from scipy.signal import lfilter

def A_weighting_filter(fs):
    """construct an a-weighting filter at the specified samplerate
    from here: http://www.mathworks.com/matlabcentral/fileexchange/69
    """
    f1, f2, f3, f4, A1000 = 20.598997, 107.65265, 737.86223, 12194.217, 1.9997

    NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000/20)), 0, 0, 0, 0]
    DENs = np.convolve([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                       [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2], mode='full')
    DENs = np.convolve(np.convolve(DENs, [1, 2 * np.pi * f3], mode='full'),
                       [1, 2 * np.pi * f2], mode='full')

    return bilinear(NUMs, DENs, fs)

#functions for going from stereo to mono
def to_mono(stereo):
    return stereo.set_channels(1)


#functions for normalizing
def normalize(sound, target_dBFS):
    delta = target_dBFS - sound.dBFS
    return sound.apply_gain(delta)


def A_weight(sig, fs):
    B, A = A_weighting_filter(fs)
    return lfilter(B, A, sig)

def analyze(sig, fs, enc):
    props = {}
    props['sig']      =  sig
    props['fs']       = fs
    props['enc']      = enc
    props['dc']       = np.mean(sig)
    props['peak']     = np.max(np.abs(sig))
    props['rms']      = rms(sig)
    props['crest']    = props['peak']/props['rms']
    weighted_sig      = A_weight(sig, fs)
    props['weighted'] = rms(weighted_sig)
    return props




def hamming(N):
    return np.array([0.54 -0.46 * np.cos(2 * np.pi * n /(N-1)) for n in range(N)])

def intensity(s):
#Computes the intensity, a.k.a the absolute value of the point of the wave furthest away from 0.
    M=0
    for i in range(len(s)):
        if(abs(float(s[i]))>M):
            M=abs(float(s[i]))
    return M



def normalize(file,noise_threshold=-80):
#Normalizes the total sound wave so that amplitude is equal throughout the sound wave.
    name=file[:len(file)-4]
    noise_threshold=pow(10.0,(-6+noise_threshold)/20.0)
    #convert the decibel threshold to linear representation.
    #-6 decibels are added because the average section of the chunk is multiplied by 0.5 when the chunks are pieced back together.
    sound_wave=read(file)[1]
    padding=list([0]*689)
    sound_wave=np.concatenate((list(padding),sound_wave,list(padding)))
    outsound=list([0]*(len(sound_wave)))
    #outsound is initialized to a array of 0s.
    i=0
    current_chunk_intensity=0
    maximum_intensity=0
    #maximum intensity of all chunks is calculated.
    while(i+1378<len(sound_wave)):
        current_chunk_intensity=intensity(sound_wave[i:i+1378])
        if(current_chunk_intensity>maximum_intensity):
            maximum_intensity=current_chunk_intensity
        i=i+689
    i=0
    #all chunks brought back to the same intensity
    print('Normalization of submitted wav file :')
    while(i+1378<len(sound_wave)):
        if(i%137800==0):
            print(str(i//689)+'/'+str(len(sound_wave)//689)+' chunks normalized.')
        last_chunk_intensity=current_chunk_intensity
        current_chunk_intensity=intensity(sound_wave[i:i+1378])
        for j in range(1378):
            if(0.5*(last_chunk_intensity+current_chunk_intensity)<maximum_intensity*noise_threshold):
                pass
            else:
                outsound[i+j]+=float(((0.5-0.5*math.cos(2*math.pi*j/1378))*float(sound_wave[i+j]))/(last_chunk_intensity+current_chunk_intensity))
        i=i+689
    outsound_scaled=outsound/np.max(np.abs(outsound))
    #scaled up so that the point of the wave with biggest absolute value is at 1 or -1.
    outsound_cropped=[]
    #when scaled up this way, a few points are quite far from the area that contains most of the wave.
    #There extreme values are cropped out by multiplying the wave by 4/3 and clipping whatever outside of [-1,1].
    for i in range(len(outsound_scaled)):
        s=(4/3)*outsound_scaled[i]
        if(s>1):
            s=1
        elif(s<-1):
            s=-1
        else:
            pass
        outsound_cropped.append(s)
    write(name+'_optimized.wav',44100,np.asarray(outsound_cropped))




#-----------------------------Helper Functions------------------------------#
# Note: wavread and _raw_data helper functions were taken from Eric Humphrey's
# Python tutorial given on behalf of MARL (Music and Auditory Research Lab),
# Friday 4/27/2012

def wavread(fin):
	""" Read in an Audio file using the wave library """
	wfile = wave.open(fin,'rb')
	x_raw = wfile.readframes(wfile.getnframes())
	x = _rawdata_to_array(x_raw, wfile.getnchannels(), wfile.getsampwidth())
	fs = wfile.getframerate()
	wfile.close()
	return x, float(fs)

def _rawdata_to_array(data, channels, bytedepth):
	"""
	Convert packed byte string into usable numpy arrays
	Returns
	-------
	frame : nd.array of floats
	    array with shape (N,channels), normalized to [-1.0, 1.0]
	"""

	if data is None:
		return None

	N = len(data) / float(channels) / float(bytedepth)
	frame = np.array(struct.unpack('%dh' % N * channels, data)) / (2.0 ** (8 * bytedepth - 1))
	return frame.reshape([N, channels])

def pickWinType(winType, N):
	""" Allow the user to pick a window type"""
	# Select window type
	if winType is "bartlett":
		window = np.bartlett(N)
	elif winType is "blackman":
		window = np.blackman(N)
	elif winType is "hamming":
		window = np.hamming(N)
	elif winType is "hanning":
		window = np.hanning(N)
	else:
		window = None

		return window

# Source of interpolation function - https://gist.github.com/255291
def parabolic(f, x):
	"""Quadratic interpolation for estimating the true position of an
	inter-sample maximum when nearby samples are known.
 	f is a vector and x is an index for that vector.
	Returns (vx, vy), the coordinates of the vertex of a parabola that goes
	through point x and its two neighbors.
	"""
	xv = float(1/2 * (f[x-1] - f[x+1] + 1) / (f[x-1] - 2 * f[x] + f[x+1]) + x)
	yv = float(f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x))
	return (xv, yv)

def split_seq(seq,size):
	""" Split up seq in pieces of size """
	return [seq[i:i+size] for i in range(0, len(seq), size)]

def convertToMono(x):
	""" Take a single channel from a stereo signal """
	if x.ndim == 2: # Stereo
		# Limit to one channel
		x = (x[:,0])
	elif x.ndim == 1: # Mono
		x = x
	else:
		raise ValueError("Input of wrong shape")
	return x

def decibels(x):
	""" Return value in decibles """
	return 20.0 * np.log10(x + eps)

def normalize(x):
	""" Normailize values between -1 and 1"""
	return x / np.abs(x).max()




import numpy as np


NFFT = 512


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

    Args:
        f (array) : input vector.
        x   (int) : index for that vector.

    Returns:
        tuple (vx, vy), the coordinates of the vertex of a parabola that goes
        through point x and its two neighbors.
	"""
	xv = float(1/2 * (f[x-1] - f[x+1] + 1) / (f[x-1] - 2 * f[x] + f[x+1]) + x)
	yv = float(f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x))
	return (xv, yv)
