import numpy as np
from ..utils import spectral as spec
from ..utils.filters import rasta_filter
from ..utils.cepstral import cms, cmvn, lifter_ceps
from ..features.lpc import do_lpc, lpc2cep, lpc2spec


def plp(sig,
        fs,
        num_ceps=13,
        pre_emph=0,
        pre_emph_coeff=0.97,
        win_len=0.025,
        win_hop=0.010,
        modelorder=13,
        normalize=0):
    """
    compute plps.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        modelorder       (int) : order of the model / number of cepstra. 0 -> no PLP.
                                 Default is 13.
        normalize        (int) : if True apply normalization.
                                 Default is 0.
    Returns:
        plps.
    """
    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    return rastaplp(x=sig,
                    fs=fs,
                    win_len=win_len,
                    win_hop=win_hop,
                    do_rasta=False,
                    modelorder=num_ceps - 1,
                    normalize=normalize)


def rplp(sig,
         fs,
         num_ceps=13,
         pre_emph=0,
         pre_emph_coeff=0.97,
         win_len=0.025,
         win_hop=0.010,
         normalize=0):
    """
    compute rasta plps.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        normalize        (int) : if True apply normalization.
                                 Default is 0.
    Returns:
        rasta plps.
    """
    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    return rastaplp(x=sig,
                    fs=fs,
                    win_len=win_len,
                    win_hop=win_hop,
                    do_rasta=True,
                    modelorder=num_ceps - 1,
                    normalize=normalize)


def rastaplp(x,
             fs=16000,
             win_len=0.025,
             win_hop=0.010,
             do_rasta=True,
             modelorder=13,
             normalize=0):
    """
    compute rasta Perceptual Linear Prediction coefficients (rasta plp) [cepstra, spectra, lpcas] = rastaplp(samples, sr, do_rasta, modelorder)

    Args:
        x        (array) : signal array.
        fs         (int) : sampling rate.
                           Default is 1600.
        win_len  (float) : length of window.
                           Default is 0.025,
        win_hop  (float) : window hop.
                           Default is 0.010,
        do_rasta  (bool) : if True apply rasta filtering. If False PLP is calculated.
                           Default is True,
        modelorder (int) : order of the model / number of cepstra. 0 -> no PLP.
                           Default is 13,
        normalize (int) : if True apply normalization.
                           Default is 0
    Returns:
        PLP or RPLP coefficients. Matrix of features, row = feature, col = frame.
    """
    # first compute power spectrum
    p_spectrum, _ = spec.powspec(x, fs, win_len, win_hop)

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
