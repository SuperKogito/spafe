import librosa
import numpy as np
from scipy.io import wavfile
from scipy.stats import gmean
import matplotlib.pyplot as plt



class Signal:
    def __init__(self, wave_file):
        self.centroid()                       # Spectral Centroid
        self.chroma()                         # Chroma vector
        self.crest()                          # Spectral Crest Factor
        self.flatness()                       # Spectral Flatness
        self.idct()                           # Inverse DCT
        self.ifft()                           # Inverse FFT
        self.kurtosis()                       # Spectral Kurtosis
        self.mean()                           # Spectral Mean
        self.mfcc2()                          # MFCC (vectorized implementation)
        self.plot()                           # Plot using matplotlib
        self.rolloff()                        # Spectral Rolloff
        self.skewness()                       # Spectral Skewness
        self.spread()                         # Spectral Spread
        self.variance()                       # Spectral Variance
        self.cqt()                            # Constant Q Transform
        self.dct()                            # Discrete Cosine Transform
        self.energy(windowSize = 256)         # Energy
        self.play()                           # Playback using pyAudio
        self.plot()                           # Plot using matplotlib
        self.rms()                            # Root-mean-squared amplitude
        self.zcr()                            # Zero-crossing raate
        self.lpc()                            # LPC, with order = len(self)-1
        self.lpcc()                           # LPCC, with order = len(self)-1
        self.lsp()                            # LSP/LSF, with order = len(fixedFrames[0])-1
        
        self.duration                         # length of signal
        self.meanfreq                         # mean frequency (in kHz)
        self.sd                               # standard deviation of frequency
        self.median                           # median frequency (in kHz)
        self.q25                              # first quantile (in kHz)
        self.q75                              # third quantile (in kHz)
        self.IQR                              # interquantile range (in kHz)
        self.skew                             # skewness (see note in specprop description)
        self.kurt                             # kurtosis (see note in specprop description)
        self.sp.ent                           # spectral entropy
        self.sfm                              # spectral flatness
        self.mode                             # mode frequency
        self.centroid                         # frequency centroid (see specprop)
        self.peakf                            # peak frequency (frequency with highest energy)
        self.meanfun                          # average of fundamental frequency measured across acoustic signal
        self.minfun                           # minimum fundamental frequency measured across acoustic signal
        self.maxfun                           # maximum fundamental frequency measured across acoustic signal
        self.meandom                          # average of dominant frequency measured across acoustic signal
        self.mindom                           # minimum of dominant frequency measured across acoustic signal
        self.maxdom                           # maximum of dominant frequency measured across acoustic signal
        self.dfrange                          # range of dominant frequency measured across acoustic signal
        self.modindx                          # modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
    