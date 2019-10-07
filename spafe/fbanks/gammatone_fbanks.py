################################################################################
#                      Gammatone-filter-banks implementation
################################################################################
"""
based on https://github.com/mcusi/gammatonegram/
"""
import numpy as np

# Slaney's ERB Filter constants
EarQ  = 9.26449 
minBW = 24.7 

def generate_center_frequencies(fl, fh, nfilt):
    # init vars
    m  = np.array(range(nfilt)) + 2
    c  = EarQ * minBW
    M  = nfilt

    # compute center frequencies
    cfreqs = (fh + c) * np.exp((m/M) * np.log((fl + c) / (fh + c))) - c
    return cfreqs[::-1]

def compute_gain(fcs, B, wT, T):
    # pre-computations for simplification
    K    = np.exp(B * T)
    Cos  = np.cos(2 * fcs * np.pi * T)
    Sin  = np.sin(2 * fcs * np.pi * T)
    Smax = np.sqrt(3 + 2**(3/2))
    Smin = np.sqrt(3 - 2**(3/2))
        
    # define A matrix rows
    A11 = (Cos + Smax * Sin) / K
    A12 = (Cos - Smax * Sin) / K
    A13 = (Cos + Smin * Sin) / K
    A14 = (Cos - Smin * Sin) / K
    
    # Compute gain (vectorized)
    A     = np.array([A11, A12, A13, A14])
    Kj    = np.exp(1j*wT)
    Kjmat = np.array([Kj, Kj, Kj, Kj]).T
    G     =  2 * T * Kjmat * (A.T - Kjmat)
    Coe   = -2 / K**2 - 2*Kj**2 +  2*(1 + Kj**2) / K
    Gain  = np.abs(G[:,0] * G[:,1] * G[:,2] * G[:, 3] * Coe**-4)
    return A, Gain

def gammatone_filter_banks(nfilts=20, nfft=512, fs=16000, fmin=0, fmax=10000, order=1):
    # define custom difference func
    Dif = lambda u, a: u - a.reshape(nfilts, 1) 

    # init vars
    fbank  = np.zeros([nfilts, nfft])
    width  = 1.0 
    maxlen = nfft // 2 + 2
    T      = 1 / fs
    n      = 4
    u      = np.exp(1j * 2 * np.pi * np.array(range(nfft // 2 + 1)) / nfft)
    idx    = range(nfft // 2 + 1)  
   
    # computer center frequencies, convert to ERB scale and compute bandwidths
    fcs  = generate_center_frequencies(fmin, fmax, nfilts)
    ERB  = width * ((fcs / EarQ)**order + minBW**order)**(1 / order)
    B    = 1.019 * 2 * np.pi * ERB
    
    # compute input vars
    wT   = 2 * fcs * np.pi * T  
    pole = np.exp(1j*wT) / np.exp(B*T)
    
    # compute gain and A matrix 
    A, Gain = compute_gain(fcs, B, wT, T) 
    
    # compute fbank    
    fbank[:, idx] = (
                      (T**4 / Gain.reshape(nfilts, 1)) * 
                      np.abs(Dif(u, A[0]) * Dif(u, A[1]) * Dif(u, A[2]) * Dif(u, A[3])) * 
                      np.abs(Dif(u, pole) * Dif(u, pole.conj()))**(-n)
                    )           
    # make sure all filters has max value = 1.0
    fbs = np.array([ f/np.max(f) for f in fbank[:, range(maxlen)] ])
    return fbs
