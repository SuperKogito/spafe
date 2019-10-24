import numpy as np
from ..utils import spectral as spec
from ..utils.filters import rasta_filter
from ..utils.cepstral import cms, cmvn, lifter_ceps
from ..features.lpc import do_lpc, lpc2cep, lpc2spec


def plp(sig,
        fs,
        num_ceps=13,
        win_time=0.025,
        hop_time=0.010,
        do_rasta=False,
        modelorder=13,
        normalize=0):
    return rastaplp(x=sig,
                    fs=fs,
                    win_time=win_time,
                    hop_time=hop_time,
                    do_rasta=do_rasta,
                    modelorder=num_ceps - 1,
                    normalize=normalize)


def rplp(sig,
         fs,
         num_ceps=13,
         win_time=0.025,
         hop_time=0.010,
         do_rasta=True,
         normalize=0):
    return rastaplp(x=sig,
                    fs=fs,
                    win_time=win_time,
                    hop_time=hop_time,
                    do_rasta=do_rasta,
                    modelorder=num_ceps - 1,
                    normalize=normalize)


def rastaplp(x,
             fs=16000,
             win_time=0.025,
             hop_time=0.010,
             do_rasta=True,
             modelorder=13,
             normalize=0):
    """
    %[cepstra, spectra, lpcas] = rastaplp(samples, sr, do_rasta, modelorder)
    %
    % cheap version of log rasta with fixed parameters
    %
    % output is matrix of features, row = feature, col = frame
    %
    % sr is sampling rate of samples, defaults to 8000
    % do_rasta defaults to 1; if 0, just calculate PLP
    % modelorder is order of PLP model, defaults to 8.  0 -> no PLP
    %
    % rastaplp(d, sr, 0, 12) is pretty close to the unix command line
    % feacalc -dith -delta 0 -ras no -plp 12 -dom cep ...
    % except during very quiet areas, where our approach of adding noise
    % in the time domain is different from rasta's approach
    %
    % 2003-04-12 dpwe@ee.columbia.edu after shire@icsi.berkeley.edu's version

    """
    # first compute power spectrum
    p_spectrum, _ = spec.powspec(x, fs, win_time, hop_time)

    # next group to critical bands
    aspectrum = spec.audspec(p_spectrum, fs)
    nbands = aspectrum.shape[0]

    if do_rasta:
        # put in log domain
        nl_aspectrum = np.log(aspectrum)
        # next do rasta filtering
        ras_nl_aspectrum = rasta_filter(nl_aspectrum)
        # do inverse log
        aspectrum = np.exp(ras_nl_aspectrum)

    postspectrum, _ = spec.postaud(aspectrum, fs / 2)

    lpcas = do_lpc(postspectrum, modelorder)
    cepstra = lpc2cep(lpcas, modelorder + 1)

    if modelorder > 0:
        lpcas = do_lpc(postspectrum, modelorder)
        cepstra = lpc2cep(lpcas, modelorder + 1)
        spectra, F, M = lpc2spec(lpcas, nbands)
    else:
        spectra = postspectrum
        cepstra = ceps.spec2cep(spectra)

    cepstra = spec.lifter(cepstra, 0.6)
    # normalize
    if normalize == "cms":
        cepstra = cmvn(cms(cepstra))

    return cepstra.T
