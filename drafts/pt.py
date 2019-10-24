import librosa
import librosa.filters

import scipy
import warnings
from scipy import signal
import scipy.fftpack as fft
from spafe.utils import levinsondr
from spafe.features.ceps import deltas, spec2cep, cep2spec
from spafe.features.lpc import  dolpc, lpc2cep, lpc2spec, lpc
from spafe.features import rasta
import numpy as np
import matplotlib.pyplot as plt

from spafe.utils.converters import hz2bark, bark2hz, hz2mel, mel2hz, fft2melmx, fft2barkmx
from spafe.utils.spectral import powspec, lifter, audspec, postaud, invpostaud, invpowspec, invaudspec
from spafe.features.mfcc import mfcc, melfcc, invmelfcc, imfcc, mfe

# reference
# https://labrosa.ee.columbia.edu/matlab/rastamat/
# test
if __name__ == '__main__':
    from spafe.utils import vis 
    sig, fs = librosa.load("test.wav")
   
    # rastaplp
    from spafe.features.rplp import rastaplp
    rasta_plp = rastaplp(sig)
    vis.visualize_features(rasta_plp, "R_PLP", "")

    # first compute power spectrum
    p_spectrum, _ = powspec(sig, fs, 0.040, 0.020)
    # next group to critical bands
    aspectrum = audspec(p_spectrum, fs)
    nbands = aspectrum.shape[0]
    dorasta = 0
    if dorasta:
        # put in log domain
        nl_aspectrum = np.log(aspectrum)
        # next do rasta filtering
        ras_nl_aspectrum = rasta.rasta_filter(nl_aspectrum)
        # do inverse log
        aspectrum = np.exp(ras_nl_aspectrum)

    postspectrum, _ = postaud(aspectrum, fs / 2)
    lpcas = dolpc(postspectrum);
    lpccs = lpc2cep(lpcas);
    # visualize the results
    vis.visualize_features(lpcas.T, 'LPCa Coefficient Index','Frame Index')
    vis.visualize_features(lpccs.T, 'LPCC Coefficient Index','Frame Index')

    # first compute power spectrum
    p_spectrum, _ = powspec(sig, fs, 0.040, 0.020)
    # next group to critical bands
    aspectrum = audspec(p_spectrum, fs)
    nbands = aspectrum.shape[0]
    dorasta = 1
    if dorasta:
        # put in log domain
        nl_aspectrum = np.log(aspectrum)
        # next do rasta filtering
        ras_nl_aspectrum = rasta.rasta_filter(nl_aspectrum)
        # do inverse log
        aspectrum = np.exp(ras_nl_aspectrum)

    postspectrum, _ = postaud(aspectrum, fs / 2)
    lpcas = dolpc(postspectrum);
    lpccs = lpc2cep(lpcas);
    # visualize the results
    vis.visualize_features(lpcas.T, 'LPCa Coefficient Index','Frame Index')
    vis.visualize_features(lpccs.T, 'LPCC Coefficient Index','Frame Index')