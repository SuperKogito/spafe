import numpy as np
from scipy.fftpack import fft2, ifft2 


def cms(mfccs):
    # Cepstral Mean Substraction: Centering 
    ms_mfccs = mfccs - np.mean(mfccs, axis=0)
    return ms_mfccs  
  
def cvn(mfccs):
    # Cepstral Variance Normalisation: Standdization
    vn_mfccs = mfccs / np.std(mfccs)
    return vn_mfccs  

def cmvn(mfccs):
    # Cepstral Mean Variance Normalisation
    ms_mfccs = cms(mfccs)
    vn_mfccs = (ms_mfccs)
    return vn_mfccs  

def min_max_normalization(vec):
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))

def rescale_to_ab_range(vec, r):
    [a, b] = r
    return a + ((vec - np.min(vec))* (b - a)) / (np.max(vec) - np.min(vec))

def mean_normalization(vec):
    return (vec - np.mean(vec)) / (np.max(vec) - np.min(vec))

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

def rms(a):
    return np.sqrt(np.sum(a**2)/len(a))


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
    print("Difference: ", dBDifference)

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
